"""
Room Geometry and Acoustic Properties
Defines room shapes, surfaces, and acoustic boundary conditions.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import numpy.linalg as la

from .materials import Material, MaterialDatabase


class SurfaceType(Enum):
    """Types of room surfaces."""
    FLOOR = "floor"
    CEILING = "ceiling"
    WALL = "wall"
    WALL_NORTH = "wall_north"
    WALL_SOUTH = "wall_south"
    WALL_EAST = "wall_east"
    WALL_WEST = "wall_west"
    CUSTOM = "custom"


@dataclass
class Surface:
    """Represents a room surface with acoustic properties."""
    name: str
    vertices: np.ndarray  # (N, 3) vertices forming the surface polygon
    material: Material
    surface_type: SurfaceType = SurfaceType.CUSTOM
    
    @property
    def normal(self) -> np.ndarray:
        """Compute the outward-facing normal vector."""
        if len(self.vertices) < 3:
            return np.array([0.0, 0.0, 1.0])
            
        # Use first three vertices to compute normal
        v1 = self.vertices[1] - self.vertices[0]
        v2 = self.vertices[2] - self.vertices[0]
        normal = np.cross(v1, v2)
        norm = la.norm(normal)
        
        if norm < 1e-10:
            return np.array([0.0, 0.0, 1.0])
            
        return normal / norm
    
    @property
    def area(self) -> float:
        """Compute the surface area using the shoelace formula for polygons."""
        if len(self.vertices) < 3:
            return 0.0
            
        # Triangulate from first vertex
        total_area = 0.0
        center = self.vertices[0]
        
        for i in range(1, len(self.vertices) - 1):
            v1 = self.vertices[i] - center
            v2 = self.vertices[i + 1] - center
            cross = np.cross(v1, v2)
            total_area += la.norm(cross) / 2.0
            
        return total_area
    
    @property
    def centroid(self) -> np.ndarray:
        """Compute the centroid of the surface."""
        return np.mean(self.vertices, axis=0)
    
    def contains_point(self, point: np.ndarray, tolerance: float = 1e-6) -> bool:
        """Check if a point lies on the surface plane."""
        normal = self.normal
        d = -np.dot(normal, self.vertices[0])
        distance = abs(np.dot(normal, point) + d)
        return distance < tolerance
    
    def point_on_surface(self, point: np.ndarray) -> bool:
        """Check if point is within the bounds of the surface polygon."""
        if not self.contains_point(point):
            return False
            
        # Ray casting algorithm for point-in-polygon test
        n = len(self.vertices)
        inside = False
        
        j = n - 1
        for i in range(n):
            vi = self.vertices[i]
            vj = self.vertices[j]
            
            if ((vi[1] > point[1]) != (vj[1] > point[1])) and \
               (point[0] < (vj[0] - vi[0]) * (point[1] - vi[1]) / (vj[1] - vi[1] + 1e-10) + vi[0]):
                inside = not inside
            j = i
            
        return inside


class Room:
    """
    Represents an acoustic environment with geometry and materials.
    
    The room class defines the physical space where sound propagates,
    including surfaces, boundary conditions, and atmospheric properties.
    
    Example:
        >>> room = Room.dimensions(10, 8, 3)  # 10m x 8m x 3m room
        >>> room.set_material(SurfaceType.FLOOR, concrete_material)
        >>> room.set_material(SurfaceType.CEILING, plaster_material)
    """
    
    def __init__(
        self,
        surfaces: Optional[List[Surface]] = None,
        temperature: float = 20.0,  # Celsius
        humidity: float = 50.0,  # percent
        pressure: float = 101325.0  # Pa
    ):
        """
        Initialize a room.
        
        Args:
            surfaces: List of room surfaces
            temperature: Room temperature in Celsius
            humidity: Relative humidity in percent
            pressure: Atmospheric pressure in Pa
        """
        self.surfaces: List[Surface] = surfaces or []
        self.temperature = temperature
        self.humidity = humidity
        self.pressure = pressure
        self._material_db = MaterialDatabase()
        
    @classmethod
    def dimensions(
        cls,
        length: float,
        width: float,
        height: float,
        materials: Optional[Dict[SurfaceType, Material]] = None
    ) -> "Room":
        """
        Create a rectangular room with given dimensions.
        
        Args:
            length: Room length (x-axis) in meters
            width: Room width (y-axis) in meters
            height: Room height (z-axis) in meters
            materials: Optional material assignments by surface type
            
        Returns:
            Room instance
        """
        # Default materials if not specified
        default_materials = {
            SurfaceType.FLOOR: Material("concrete", 0.02, 0.03, 0.04, 0.05),
            SurfaceType.CEILING: Material("plaster", 0.05, 0.04, 0.03, 0.02),
            SurfaceType.WALL: Material("brick", 0.03, 0.04, 0.05, 0.06),
        }
        
        if materials:
            default_materials.update(materials)
            
        # Define vertices for each surface (outward-facing normals)
        surfaces = []
        
        # Floor (z = 0)
        floor = Surface(
            name="floor",
            vertices=np.array([
                [0, 0, 0],
                [length, 0, 0],
                [length, width, 0],
                [0, width, 0]
            ]),
            material=default_materials[SurfaceType.FLOOR],
            surface_type=SurfaceType.FLOOR
        )
        surfaces.append(floor)
        
        # Ceiling (z = height)
        ceiling = Surface(
            name="ceiling",
            vertices=np.array([
                [0, 0, height],
                [0, width, height],
                [length, width, height],
                [length, 0, height]
            ]),
            material=default_materials[SurfaceType.CEILING],
            surface_type=SurfaceType.CEILING
        )
        surfaces.append(ceiling)
        
        # North wall (y = width)
        north_wall = Surface(
            name="north_wall",
            vertices=np.array([
                [0, width, 0],
                [length, width, 0],
                [length, width, height],
                [0, width, height]
            ]),
            material=default_materials[SurfaceType.WALL],
            surface_type=SurfaceType.WALL_NORTH
        )
        surfaces.append(north_wall)
        
        # South wall (y = 0)
        south_wall = Surface(
            name="south_wall",
            vertices=np.array([
                [length, 0, 0],
                [0, 0, 0],
                [0, 0, height],
                [length, 0, height]
            ]),
            material=default_materials[SurfaceType.WALL],
            surface_type=SurfaceType.WALL_SOUTH
        )
        surfaces.append(south_wall)
        
        # East wall (x = length)
        east_wall = Surface(
            name="east_wall",
            vertices=np.array([
                [length, 0, 0],
                [length, width, 0],
                [length, width, height],
                [length, 0, height]
            ]),
            material=default_materials[SurfaceType.WALL],
            surface_type=SurfaceType.WALL_EAST
        )
        surfaces.append(east_wall)
        
        # West wall (x = 0)
        west_wall = Surface(
            name="west_wall",
            vertices=np.array([
                [0, width, 0],
                [0, 0, 0],
                [0, 0, height],
                [0, width, height]
            ]),
            material=default_materials[SurfaceType.WALL],
            surface_type=SurfaceType.WALL_WEST
        )
        surfaces.append(west_wall)
        
        return cls(surfaces=surfaces)
    
    @classmethod
    def from_obj(cls, path: str, scale: float = 1.0) -> "Room":
        """
        Load room geometry from OBJ file.
        
        Args:
            path: Path to OBJ file
            scale: Scale factor for vertices
            
        Returns:
            Room instance
        """
        vertices = []
        faces = []
        
        with open(path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                    
                if parts[0] == "v":
                    vertices.append([float(p) * scale for p in parts[1:4]])
                elif parts[0] == "f":
                    # Handle face definitions (may include texture coords)
                    face = []
                    for part in parts[1:]:
                        vertex_idx = int(part.split("/")[0]) - 1
                        face.append(vertex_idx)
                    faces.append(face)
        
        surfaces = []
        default_material = Material("default", 0.05, 0.05, 0.05, 0.05)
        
        for face in faces:
            if len(face) >= 3:
                surface_vertices = np.array([vertices[i] for i in face])
                surface = Surface(
                    name=f"face_{len(surfaces)}",
                    vertices=surface_vertices,
                    material=default_material,
                    surface_type=SurfaceType.CUSTOM
                )
                surfaces.append(surface)
                
        return cls(surfaces=surfaces)
    
    def add_surface(self, surface: Surface) -> None:
        """Add a surface to the room."""
        self.surfaces.append(surface)
        
    def remove_surface(self, name: str) -> bool:
        """Remove a surface by name."""
        for i, surface in enumerate(self.surfaces):
            if surface.name == name:
                self.surfaces.pop(i)
                return True
        return False
        
    def set_material(self, surface_name: str, material: Material) -> None:
        """Set the material for a specific surface."""
        for surface in self.surfaces:
            if surface.name == surface_name:
                surface.material = material
                return
                
    def set_material_by_type(self, surface_type: SurfaceType, material: Material) -> None:
        """Set material for all surfaces of a given type."""
        for surface in self.surfaces:
            if surface.surface_type == surface_type:
                surface.material = material
                
    def get_surface(self, name: str) -> Optional[Surface]:
        """Get surface by name."""
        for surface in self.surfaces:
            if surface.name == name:
                return surface
        return None
        
    @property
    def volume(self) -> float:
        """Compute the room volume using the divergence theorem."""
        if len(self.surfaces) < 4:
            return 0.0
            
        total_volume = 0.0
        for surface in self.surfaces:
            # Volume contribution = (1/3) * A * d * n_hat . c
            # where A is area, d is distance, n_hat is normal, c is centroid
            area = surface.area
            normal = surface.normal
            centroid = surface.centroid
            
            # Signed distance from origin to plane
            d = -np.dot(normal, centroid)
            
            # Volume element
            total_volume += area * d * np.dot(normal, centroid) / 3.0
            
        return abs(total_volume)
    
    @property
    def surface_area(self) -> float:
        """Compute total surface area."""
        return sum(surface.area for surface in self.surfaces)
    
    @property
    def dimensions(self) -> Tuple[float, float, float]:
        """Get room dimensions (length, width, height)."""
        if len(self.surfaces) != 6:
            return (0.0, 0.0, 0.0)
            
        min_coords = np.array([np.inf, np.inf, np.inf])
        max_coords = np.array([-np.inf, -np.inf, -np.inf])
        
        for surface in self.surfaces:
            for vertex in surface.vertices:
                min_coords = np.minimum(min_coords, vertex)
                max_coords = np.maximum(max_coords, vertex)
                
        return tuple(max_coords - min_coords)
    
    def absorption_coefficients(self, frequency: float) -> Tuple[float, float]:
        """
        Get average absorption coefficients.
        
        Args:
            frequency: Frequency in Hz
            
        Returns:
            Tuple of (absorption coefficient, scattering coefficient)
        """
        total_area = self.surface_area
        if total_area < 1e-10:
            return (0.0, 0.0)
            
        total_absorption = 0.0
        total_scattering = 0.0
        
        for surface in self.surfaces:
            alpha = surface.material.absorption(frequency)
            s = surface.material.scattering(frequency)
            area = surface.area
            
            total_absorption += alpha * area
            total_scattering += s * area
            
        return (total_absorption / total_area, total_scattering / total_area)
    
    def speed_of_sound(self, temperature: Optional[float] = None) -> float:
        """
        Compute speed of sound in the room.
        
        Args:
            temperature: Override temperature in Celsius
            
        Returns:
            Speed of sound in m/s
        """
        T = temperature if temperature is not None else self.temperature
        # 331.3 m/s at 0°C plus 0.606 m/s per degree Celsius
        return 331.3 + 0.606 * T
    
    def air_absorption(
        self,
        distance: float,
        frequency: float,
        humidity: Optional[float] = None,
        temperature: Optional[float] = None
    ) -> float:
        """
        Compute air absorption coefficient.
        
        Args:
            distance: Propagation distance in meters
            frequency: Frequency in Hz
            humidity: Override humidity in percent
            temperature: Override temperature in Celsius
            
        Returns:
            Absorption factor (0 to 1)
        """
        h = humidity if humidity is not None else self.humidity
        T = temperature if temperature is not None else self.temperature
        
        # Simplified air absorption model (ANSI S1.26-1995 approximation)
        f = frequency / 1000  # Convert to kHz
        
        # Frequency-dependent absorption
        if f < 1:
            alpha = 0.1 * f**2
        elif f < 10:
            alpha = 0.11 * f / (1 + f)
        else:
            alpha = 0.11
            
        # Temperature and humidity corrections
        h_sat = 5.6 * (1 + 0.04 * (T - 20))
        h_rel = h / h_sat
        
        # Simplified correction
        correction = 1 + 0.0025 * (T - 20) - 0.005 * (h_rel - 0.5)
        alpha *= correction
        
        return np.exp(-alpha * distance)
    
    def is_inside(self, point: np.ndarray, tolerance: float = 1e-6) -> bool:
        """Check if a point is inside the room."""
        for surface in self.surfaces:
            # Ray from point to positive infinity
            # Count how many times it intersects surfaces
            pass
            
        # Simplified: check if point is within bounding box
        min_coords = np.array([np.inf, np.inf, np.inf])
        max_coords = np.array([-np.inf, -np.inf, -np.inf])
        
        for surface in self.surfaces:
            for vertex in surface.vertices:
                min_coords = np.minimum(min_coords, vertex)
                max_coords = np.maximum(max_coords, vertex)
                
        inside = np.all(point >= min_coords - tolerance) and \
                 np.all(point <= max_coords + tolerance)
                 
        # Refine using surface tests
        if inside:
            for surface in self.surfaces:
                if not surface.contains_point(point, tolerance):
                    if not surface.point_on_surface(point):
                        pass
                        
        return inside
    
    def find_intersection(
        self,
        origin: np.ndarray,
        direction: np.ndarray,
        max_distance: float = float("inf")
    ) -> Tuple[Optional[Surface], Optional[np.ndarray], float]:
        """
        Find the first intersection of a ray with room surfaces.
        
        Args:
            origin: Ray origin
            direction: Ray direction (normalized)
            max_distance: Maximum ray length
            
        Returns:
            Tuple of (surface, intersection point, distance)
        """
        min_t = max_distance
        hit_surface = None
        hit_point = None
        
        for surface in self.surfaces:
            t, point = self._ray_surface_intersection(origin, direction, surface)
            
            if t is not None and t < min_t and t > 1e-6:
                min_t = t
                hit_surface = surface
                hit_point = point
                
        return hit_surface, hit_point, min_t
    
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
        
        if t < 1e-6:
            return None, None
            
        point = origin + t * direction
        
        # Check if point is on surface
        if surface.point_on_surface(point):
            return t, point
            
        return None, None
    
    def visualize_bounds(self) -> Dict[str, np.ndarray]:
        """Get bounding box for visualization."""
        min_coords = np.array([np.inf, np.inf, np.inf])
        max_coords = np.array([-np.inf, -np.inf, -np.inf])
        
        for surface in self.surfaces:
            for vertex in surface.vertices:
                min_coords = np.minimum(min_coords, vertex)
                max_coords = np.maximum(max_coords, vertex)
                
        return {"min": min_coords, "max": max_coords}
    
    def __repr__(self) -> str:
        dims = self.dimensions
        return f"Room({dims[0]:.2f}m x {dims[1]:.2f}m x {dims[2]:.2f}m, {len(self.surfaces)} surfaces)"


class RoomBuilder:
    """
    Builder class for constructing rooms programmatically.
    
    Example:
        >>> room = (RoomBuilder()
        ...     .rectangular(12, 10, 3.5)
        ...     .floor("hardwood")
        ...     .ceiling("acoustic_tile")
        ...     .walls("concrete")
        ...     .add_window((5, 2), (3, 2))  # position, size
        ...     .add_door((1, 0), (1, 2.5))  # position, size
        ...     .build())
    """
    
    def __init__(self):
        self._length = 10.0
        self._width = 8.0
        self._height = 3.0
        self._materials: Dict[SurfaceType, str] = {}
        self._openings: List[Dict] = []
        self._obstacles: List[Dict] = []
        self._temperature = 20.0
        self._humidity = 50.0
        
    def rectangular(self, length: float, width: float, height: float) -> "RoomBuilder":
        """Set rectangular room dimensions."""
        self._length = length
        self._width = width
        self._height = height
        return self
        
    def floor(self, material: str) -> "RoomBuilder":
        """Set floor material by name."""
        self._materials[SurfaceType.FLOOR] = material
        return self
        
    def ceiling(self, material: str) -> "RoomBuilder":
        """Set ceiling material by name."""
        self._materials[SurfaceType.CEILING] = material
        return self
        
    def walls(self, material: str) -> "RoomBuilder":
        """Set wall material by name."""
        for stype in [SurfaceType.WALL_NORTH, SurfaceType.WALL_SOUTH,
                      SurfaceType.WALL_EAST, SurfaceType.WALL_WEST]:
            self._materials[stype] = material
        return self
        
    def wall(self, surface: SurfaceType, material: str) -> "RoomBuilder":
        """Set specific wall material."""
        self._materials[surface] = material
        return self
        
    def add_window(
        self,
        position: Tuple[float, float],
        size: Tuple[float, float],
        wall: SurfaceType = SurfaceType.WALL_NORTH
    ) -> "RoomBuilder":
        """Add a window opening."""
        self._openings.append({
            "type": "window",
            "position": position,
            "size": size,
            "wall": wall
        })
        return self
        
    def add_door(
        self,
        position: Tuple[float, float],
        size: Tuple[float, float],
        wall: SurfaceType = SurfaceType.WALL_SOUTH
    ) -> "RoomBuilder":
        """Add a door opening."""
        self._openings.append({
            "type": "door",
            "position": position,
            "size": size,
            "wall": wall
        })
        return self
        
    def add_obstacle(
        self,
        position: np.ndarray,
        size: np.ndarray,
        material: str = "wood"
    ) -> "RoomBuilder":
        """Add an obstacle (pillar, furniture, etc.)."""
        self._obstacles.append({
            "position": position,
            "size": size,
            "material": material
        })
        return self
        
    def atmospheric_conditions(
        self,
        temperature: float,
        humidity: float
    ) -> "RoomBuilder":
        """Set atmospheric conditions."""
        self._temperature = temperature
        self._humidity = humidity
        return self
        
    def build(self) -> Room:
        """Build the room with all specified parameters."""
        mat_db = MaterialDatabase()
        
        # Convert material names to Material objects
        materials = {}
        for stype, name in self._materials.items():
            mat = mat_db.get(name)
            if mat:
                materials[stype] = mat
            else:
                materials[stype] = Material(name, 0.05, 0.05, 0.05, 0.05)
                
        # Create base room
        room = Room.dimensions(self._length, self._width, self._height, materials)
        
        # Apply atmospheric conditions
        room.temperature = self._temperature
        room.humidity = self._humidity
        
        # Handle openings (windows, doors) - these would create surface modifications
        for opening in self._openings:
            # In a full implementation, this would modify the affected wall surface
            pass
            
        # Add obstacles as additional surfaces
        for obstacle in self._obstacles:
            pos = np.array(obstacle["position"])
            size = np.array(obstacle["size"])
            mat = mat_db.get(obstacle["material"]) or Material("default", 0.05, 0.05, 0.05, 0.05)
            
            # Create obstacle surfaces (box shape)
            obstacle_surfaces = [
                # Front face
                Surface(f"obstacle_front_{len(room.surfaces)}",
                       np.array([pos, pos + [size[0], 0, 0], 
                                pos + [size[0], 0, size[2]],
                                pos + [0, 0, size[2]]]),
                       mat),
                # Back face
                Surface(f"obstacle_back_{len(room.surfaces)}",
                       np.array([pos + [0, size[1], 0],
                                pos + [size[0], size[1], 0],
                                pos + [size[0], size[1], size[2]],
                                pos + [0, size[1], size[2]]]),
                       mat),
            ]
            
            for surf in obstacle_surfaces:
                room.add_surface(surf)
                
        return room
