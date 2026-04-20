"""
Ray-Based Sound Propagation
Implements acoustic ray tracing for sound propagation simulation.
"""

import numpy as np
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass
import threading
from concurrent.futures import ThreadPoolExecutor

from .core import SimulationConfig
from .room import Room, Surface
from .source import SoundSource
from .receiver import Receiver


@dataclass
class Ray:
    """Represents an acoustic ray."""
    origin: np.ndarray
    direction: np.ndarray
    energy: float
    path: List[Tuple[np.ndarray, Surface, float]]  # (position, surface, energy_loss)
    num_reflections: int = 0
    
    def copy(self) -> "Ray":
        """Create a copy of the ray."""
        return Ray(
            origin=self.origin.copy(),
            direction=self.direction.copy(),
            energy=self.energy,
            path=self.path.copy(),
            num_reflections=self.num_reflections
        )


@dataclass
class RayHit:
    """Represents a ray-surface intersection."""
    point: np.ndarray
    surface: Surface
    distance: float
    incident_angle: float
    energy_at_hit: float


class RayTracer:
    """
    Acoustic ray tracer for sound propagation simulation.
    
    Ray tracing is a geometric acoustics method that models sound as rays
    emanating from sources and following paths determined by reflection
    and diffraction at surfaces.
    
    Example:
        >>> tracer = RayTracer(config)
        >>> hits = tracer.trace_rays(room, source, num_rays=5000)
        >>> rir = tracer.compute_rir_from_hits(hits, receiver)
    """
    
    def __init__(self, config: SimulationConfig):
        """Initialize ray tracer."""
        self.config = config
        self.rays: List[Ray] = []
        self.hits: List[RayHit] = []
        
    def trace_rays(
        self,
        room: Room,
        source: SoundSource,
        num_rays: Optional[int] = None,
        max_reflections: Optional[int] = None,
        energy_threshold: Optional[float] = None
    ) -> List[RayHit]:
        """
        Trace rays from source through the room.
        
        Args:
            room: Room geometry
            source: Sound source
            num_rays: Number of rays to trace
            max_reflections: Maximum reflections per ray
            energy_threshold: Minimum energy to continue tracing
            
        Returns:
            List of ray-surface hits
        """
        num_rays = num_rays or self.config.num_rays
        max_reflections = max_reflections or self.config.max_reflections
        energy_threshold = energy_threshold or self.config.ray_energy_threshold
        
        # Generate initial rays
        self.rays = self._generate_rays(source.position, num_rays)
        
        # Apply source directivity
        if source.directivity is not None:
            self._apply_directivity(source)
            
        # Trace each ray
        self.hits = []
        
        for ray in self.rays:
            self._trace_ray(ray, room, max_reflections, energy_threshold)
            
        return self.hits
        
    def _generate_rays(self, origin: np.ndarray, num_rays: int) -> List[Ray]:
        """Generate initial ray directions from source."""
        rays = []
        
        # Use Fibonacci sphere distribution for uniform coverage
        phi = np.pi * (3.0 - np.sqrt(5.0))  # Golden angle
        
        for i in range(num_rays):
            y = 1 - (i / float(num_rays - 1)) * 2  # y goes from 1 to -1
            radius = np.sqrt(1 - y * y)  # Radius at y
            
            theta = phi * i
            
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            
            direction = np.array([x, y, z])
            direction = direction / np.linalg.norm(direction)
            
            ray = Ray(
                origin=origin.copy(),
                direction=direction,
                energy=1.0,
                path=[]
            )
            rays.append(ray)
            
        return rays
        
    def _apply_directivity(self, source: SoundSource) -> None:
        """Apply source directivity to ray energies."""
        directivity = source.directivity
        
        for ray in self.rays:
            # Compute angle from source axis
            angle = np.arccos(np.clip(
                np.dot(ray.direction, directivity.axis),
                -1, 1
            ))
            
            # Get directivity response
            response = directivity.get_response(np.array([0]), np.array([angle]))[0]
            ray.energy *= response
            
    def _trace_ray(
        self,
        ray: Ray,
        room: Room,
        max_reflections: int,
        energy_threshold: float
    ) -> None:
        """Trace a single ray through the room."""
        current_origin = ray.origin.copy()
        current_direction = ray.direction.copy()
        current_energy = ray.energy
        path = []
        
        for _ in range(max_reflections):
            if current_energy < energy_threshold:
                break
                
            # Find nearest intersection
            hit = self._find_nearest_hit(current_origin, current_direction, room)
            
            if hit is None:
                break
                
            # Record hit
            hit.energy_at_hit = current_energy
            self.hits.append(hit)
            path.append((hit.point.copy(), hit.surface, current_energy))
            
            # Apply surface absorption
            frequency = 1000  # Use mid-frequency absorption
            absorption = hit.surface.material.absorption(frequency)
            current_energy *= (1 - absorption)
            
            # Reflect ray
            incident = -current_direction
            normal = hit.surface.normal
            
            # Ensure normal points away from incident direction
            if np.dot(incident, normal) > 0:
                normal = -normal
                
            # Specular reflection
            current_direction = current_direction - 2 * np.dot(current_direction, normal) * normal
            
            # Move origin slightly along direction to avoid self-intersection
            current_origin = hit.point + current_direction * 1e-6
            
    def _find_nearest_hit(
        self,
        origin: np.ndarray,
        direction: np.ndarray,
        room: Room
    ) -> Optional[RayHit]:
        """Find the nearest surface hit."""
        min_distance = float('inf')
        hit = None
        
        for surface in room.surfaces:
            t, point = self._ray_surface_intersection(origin, direction, surface)
            
            if t is not None and t < min_distance and t > 1e-4:
                min_distance = t
                
                # Compute incident angle
                incident_angle = np.arccos(np.clip(
                    -np.dot(direction, surface.normal),
                    -1, 1
                ))
                
                hit = RayHit(
                    point=point,
                    surface=surface,
                    distance=t,
                    incident_angle=incident_angle,
                    energy_at_hit=1.0
                )
                
        return hit
        
    def _ray_surface_intersection(
        self,
        origin: np.ndarray,
        direction: np.ndarray,
        surface: Surface
    ) -> Tuple[Optional[float], Optional[np.ndarray]]:
        """Compute ray-surface intersection."""
        normal = surface.normal
        denom = np.dot(direction, normal)
        
        if abs(denom) < 1e-10:
            return None, None
            
        # Find intersection with plane
        d = -np.dot(normal, surface.vertices[0])
        t = -(np.dot(normal, origin) + d) / denom
        
        if t < 1e-4:
            return None, None
            
        point = origin + t * direction
        
        # Check if point is on surface polygon
        if surface.point_on_surface(point):
            return t, point
            
        return None, None
        
    def compute_rir(
        self,
        room: Room,
        source: SoundSource,
        receiver: Receiver,
        max_order: Optional[int] = None
    ) -> np.ndarray:
        """
        Compute RIR using ray tracing.
        
        Args:
            room: Room geometry
            source: Sound source
            receiver: Receiver position
            max_order: Maximum reflection order
            
        Returns:
            Room impulse response
        """
        n_samples = int(self.config.duration * self.config.sample_rate)
        rir = np.zeros(n_samples)
        
        max_order = max_order or self.config.max_reflections
        
        # Direct sound
        direct_dist = np.linalg.norm(receiver.position - source.position)
        direct_delay = int(direct_dist / self.config.speed_of_sound * self.config.sample_rate)
        
        if direct_delay < n_samples:
            # Apply source directivity for direct sound
            azimuth = np.arctan2(
                receiver.position[1] - source.position[1],
                receiver.position[0] - source.position[0]
            )
            elevation = np.arcsin(
                (receiver.position[2] - source.position[2]) / direct_dist
            )
            
            if source.directivity is not None:
                direct_gain = source.directivity.get_response(
                    np.array([azimuth]),
                    np.array([elevation])
                )[0]
            else:
                direct_gain = 1.0
                
            # Inverse square law
            direct_gain *= 1.0 / (4 * np.pi * direct_dist**2 + 1e-10)
            
            rir[direct_delay] = direct_gain
            
        # Trace rays and accumulate energy at receiver
        hits = self.trace_rays(
            room, source,
            num_rays=self.config.num_rays,
            max_reflections=max_order
        )
        
        # For each hit, check if ray passes near receiver
        receiver_radius = 0.1  # Effective receiver size in meters
        
        for hit in hits:
            # Check if receiver is near the ray path
            dist_to_receiver = self._distance_point_to_line(
                receiver.position, hit.point, hit.point + hit.direction * 0.5
            )
            
            if dist_to_receiver < receiver_radius:
                # Compute arrival time
                path_length = np.linalg.norm(hit.point - source.position)
                delay = int(path_length / self.config.speed_of_sound * self.config.sample_rate)
                
                if delay < n_samples:
                    # Energy contribution
                    dist_from_hit = np.linalg.norm(hit.point - receiver.position)
                    propagation_loss = 1.0 / (4 * np.pi * dist_from_hit**2 + 1e-10)
                    
                    # Accumulate energy
                    rir[delay] += hit.energy_at_hit * propagation_loss * hit.distance * 0.1
                    
        # Normalize
        max_val = np.max(np.abs(rir))
        if max_val > 1e-10:
            rir = rir / max_val
            
        return rir
        
    def _distance_point_to_line(
        self,
        point: np.ndarray,
        line_start: np.ndarray,
        line_end: np.ndarray
    ) -> float:
        """Compute minimum distance from point to line segment."""
        line_vec = line_end - line_start
        point_vec = point - line_start
        
        line_len = np.linalg.norm(line_vec)
        if line_len < 1e-10:
            return np.linalg.norm(point - line_start)
            
        line_unitvec = line_vec / line_len
        proj_length = np.dot(point_vec, line_unitvec)
        
        if proj_length < 0:
            return np.linalg.norm(point - line_start)
        elif proj_length > line_len:
            return np.linalg.norm(point - line_end)
        else:
            nearest = line_start + proj_length * line_unitvec
            return np.linalg.norm(point - nearest)
            
    def compute_energy_histogram(
        self,
        time_window: float = 0.05,
        num_bins: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute energy histogram of ray hits over time.
        
        Args:
            time_window: Time window in seconds
            num_bins: Number of histogram bins
            
        Returns:
            Tuple of (time_bins, energy_per_bin)
        """
        if not self.hits:
            return np.array([]), np.array([])
            
        # Compute arrival times
        times = []
        energies = []
        
        for hit in self.hits:
            for i, (point, surface, energy) in enumerate(hit.point):
                path_length = np.linalg.norm(point - hit.surface.vertices[0])
                arrival_time = path_length / self.config.speed_of_sound
                times.append(arrival_time)
                energies.append(energy)
                
        times = np.array(times)
        energies = np.array(energies)
        
        # Create histogram
        bins = np.linspace(0, time_window, num_bins + 1)
        hist, bin_edges = np.histogram(times, bins=bins, weights=energies)
        
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        return bin_centers, hist


class AdaptiveRayTracer(RayTracer):
    """
    Adaptive ray tracer with density control.
    
    Automatically adjusts ray density based on acoustic significance.
    """
    
    def __init__(self, config: SimulationConfig):
        """Initialize adaptive ray tracer."""
        super().__init__(config)
        self.initial_density = config.num_rays
        self.min_density = 100
        self.max_density = 20000
        
    def trace_rays_adaptive(
        self,
        room: Room,
        source: SoundSource,
        receiver: Receiver,
        target_error: float = 0.05
    ) -> List[RayHit]:
        """
        Trace rays with adaptive density.
        
        Args:
            room: Room geometry
            source: Sound source
            receiver: Receiver position
            target_error: Target relative error in RIR estimation
            
        Returns:
            List of ray hits
        """
        current_density = self.initial_density
        
        # Initial trace
        hits = self.trace_rays(room, source, num_rays=current_density)
        
        # Estimate error from hit distribution
        error = self._estimate_error(hits, receiver)
        
        # Iteratively refine
        while error > target_error and current_density < self.max_density:
            current_density = min(current_density * 2, self.max_density)
            hits = self.trace_rays(room, source, num_rays=current_density)
            error = self._estimate_error(hits, receiver)
            
        return hits
        
    def _estimate_error(self, hits: List[RayHit], receiver: Receiver) -> float:
        """Estimate error in RIR from ray density."""
        if len(hits) < 10:
            return 1.0
            
        # Simple error estimate based on hit density near receiver
        receiver_radius = 0.2
        
        near_hits = sum(
            1 for hit in hits
            if np.linalg.norm(hit.point - receiver.position) < receiver_radius
        )
        
        # Normalize by expected hits
        expected_hits = self.config.num_rays * 0.01  # Rough estimate
        relative_error = 1.0 / np.sqrt(max(near_hits, 1))
        
        return min(relative_error, 1.0)


class BiangularRayTracer(RayTracer):
    """
    Bi-directional ray tracer.
    
    Traces rays from both source and receiver for improved efficiency.
    """
    
    def trace_bidirectional(
        self,
        room: Room,
        source: SoundSource,
        receiver: Receiver,
        num_source_rays: int,
        num_receiver_rays: int
    ) -> List[RayHit]:
        """
        Trace rays bidirectionally and connect at surfaces.
        
        Args:
            room: Room geometry
            source: Sound source
            receiver: Receiver
            num_source_rays: Rays from source
            num_receiver_rays: Rays from receiver
            
        Returns:
            List of connected path hits
        """
        # Trace from source
        source_hits = self.trace_rays(room, source, num_rays=num_source_rays)
        
        # Create reverse tracer for receiver
        reverse_tracer = RayTracer(self.config)
        reverse_source = SoundSource(
            position=receiver.position,
            name="reverse"
        )
        receiver_hits = reverse_tracer.trace_rays(
            room, reverse_source, num_rays=num_receiver_rays
        )
        
        # Match hits to find common surfaces
        connected_hits = []
        
        for s_hit in source_hits:
            for r_hit in receiver_hits:
                if s_hit.surface == r_hit.surface:
                    connected_hits.append(s_hit)
                    break
                    
        return connected_hits
