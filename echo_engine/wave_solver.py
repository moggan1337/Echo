"""
Wave Equation Solver
Finite-difference time-domain (FDTD) methods for wave-based acoustics.
"""

import numpy as np
from typing import Optional, Tuple, Callable
from dataclasses import dataclass
import warnings

from .core import SimulationConfig
from .room import Room
from .source import SoundSource
from .receiver import Receiver


@dataclass
class GridPoint:
    """Represents a point on the simulation grid."""
    x: int
    y: int
    z: int
    is_boundary: bool = False
    absorption: float = 0.0
    material_id: int = 0


class WaveEquationSolver:
    """
    Wave equation solver using FDTD methods.
    
    Solves the acoustic wave equation in enclosed spaces, providing
    accurate simulation of wave phenomena like diffraction and resonance.
    
    Example:
        >>> solver = WaveEquationSolver(config)
        >>> solver.setup_grid(room, resolution=0.02)
        >>> solver.add_source(source)
        >>> solver.add_receiver(receiver)
        >>> pressure_field = solver.solve(duration=2.0)
    """
    
    def __init__(self, config: SimulationConfig):
        """Initialize wave equation solver."""
        self.config = config
        self.grid: Optional[np.ndarray] = None
        self.velocity_x: Optional[np.ndarray] = None
        self.velocity_y: Optional[np.ndarray] = None
        self.velocity_z: Optional[np.ndarray] = None
        self.grid_spacing: float = config.grid_resolution
        self.sources: list = []
        self.receivers: list = []
        self._time: float = 0.0
        
        # Courant-Friedrichs-Lewy (CFL) stability condition
        self.courant_number: float = 0.3
        
    def setup_grid(
        self,
        room: Room,
        resolution: Optional[float] = None
    ) -> None:
        """
        Set up the simulation grid based on room geometry.
        
        Args:
            room: Room geometry
            resolution: Grid resolution in meters
        """
        resolution = resolution or self.grid_spacing
        
        # Get room bounds
        bounds = room.visualize_bounds()
        min_coords = bounds["min"]
        max_coords = bounds["max"]
        
        # Calculate grid dimensions
        dims = max_coords - min_coords
        self.grid_shape = (
            int(dims[0] / resolution) + 1,
            int(dims[1] / resolution) + 1,
            int(dims[2] / resolution) + 1
        )
        
        # Store origin offset
        self.origin = min_coords
        
        # Initialize pressure field
        self.grid = np.zeros(self.grid_shape)
        self.grid_prev = np.zeros(self.grid_shape)
        
        # Initialize particle velocity fields
        self.velocity_x = np.zeros(self.grid_shape)
        self.velocity_y = np.zeros(self.grid_shape)
        self.velocity_z = np.zeros(self.grid_shape)
        
        # Create boundary conditions
        self._setup_boundaries(room, resolution)
        
        # Store resolution
        self.grid_spacing = resolution
        
        # Calculate time step from CFL condition
        self.dt = self.courant_number * resolution / self.config.speed_of_sound
        
    def _setup_boundaries(self, room: Room, resolution: float) -> None:
        """Set up absorbing boundary conditions at room surfaces."""
        # Create boundary mask
        self.boundary_mask = np.zeros(self.grid_shape, dtype=bool)
        self.absorption_map = np.zeros(self.grid_shape)
        
        for surface in room.surfaces:
            # Get surface normal and position
            normal = surface.normal
            centroid = surface.centroid
            
            # Determine which axis the surface is perpendicular to
            abs_normal = np.abs(normal)
            axis = np.argmax(abs_normal)
            
            # Determine which side of the grid
            if normal[axis] > 0:
                grid_coord = self.grid_shape[axis] - 1
            else:
                grid_coord = 0
                
            # Mark boundary
            slices = [slice(None), slice(None), slice(None)]
            slices[axis] = slice(grid_coord, grid_coord + 1)
            self.boundary_mask[tuple(slices)] = True
            
            # Set absorption coefficients
            for freq in [250, 500, 1000, 2000, 4000]:
                alpha = surface.material.absorption(freq)
                self.absorption_map[tuple(slices)] = alpha
                
    def add_source(self, source: SoundSource) -> Tuple[int, int, int]:
        """
        Add a source to the simulation.
        
        Args:
            source: Sound source
            
        Returns:
            Grid coordinates of source
        """
        # Convert world position to grid coordinates
        grid_pos = self._world_to_grid(source.position)
        
        self.sources.append({
            "position": grid_pos,
            "signal": source.signal,
            "directivity": source.directivity,
            "world_position": source.position.copy()
        })
        
        return grid_pos
        
    def add_receiver(self, receiver: Receiver) -> Tuple[int, int, int]:
        """
        Add a receiver to the simulation.
        
        Args:
            receiver: Sound receiver
            
        Returns:
            Grid coordinates of receiver
        """
        grid_pos = self._world_to_grid(receiver.position)
        
        self.receivers.append({
            "position": grid_pos,
            "world_position": receiver.position.copy(),
            "samples": []
        })
        
        return grid_pos
        
    def _world_to_grid(self, world_pos: np.ndarray) -> Tuple[int, int, int]:
        """Convert world coordinates to grid indices."""
        grid_pos = ((np.array(world_pos) - self.origin) / self.grid_spacing).astype(int)
        
        # Clamp to grid bounds
        grid_pos = np.clip(grid_pos, 0, np.array(self.grid_shape) - 1)
        
        return tuple(grid_pos)
        
    def _grid_to_world(self, grid_pos: Tuple[int, int, int]) -> np.ndarray:
        """Convert grid indices to world coordinates."""
        return np.array(grid_pos) * self.grid_spacing + self.origin
        
    def solve(
        self,
        duration: Optional[float] = None,
        num_steps: Optional[int] = None
    ) -> np.ndarray:
        """
        Solve the wave equation for specified duration.
        
        Args:
            duration: Simulation duration in seconds
            num_steps: Number of time steps (overrides duration)
            
        Returns:
            Final pressure field
        """
        duration = duration or self.config.duration
        
        if num_steps is None:
            num_steps = int(duration / self.dt)
            
        # Main FDTD loop
        for step in range(num_steps):
            self._update_step(step)
            
            # Record receiver samples
            if step % 100 == 0:  # Downsample for storage
                for receiver in self.receivers:
                    pos = receiver["position"]
                    if all(0 <= p < s for p, s in zip(pos, self.grid_shape)):
                        receiver["samples"].append(self.grid[pos])
                        
        return self.grid
        
    def _update_step(self, step: int) -> None:
        """Perform one time step of the FDTD update."""
        c = self.config.speed_of_sound
        dx = self.grid_spacing
        dt = self.dt
        
        # Courant numbers
        Sx = c * dt / dx
        Sy = c * dt / dx
        Sz = c * dt / dx
        
        # Store previous pressure
        self.grid_prev[:] = self.grid
        
        # Update velocity fields (half step)
        for i in range(1, self.grid_shape[0] - 1):
            for j in range(1, self.grid_shape[1] - 1):
                for k in range(1, self.grid_shape[2] - 1):
                    if self.boundary_mask[i, j, k]:
                        continue
                        
                    # Velocity update (central difference)
                    self.velocity_x[i, j, k] += -Sx * (self.grid_prev[i+1, j, k] - self.grid_prev[i-1, j, k]) / (2 * dx)
                    self.velocity_y[i, j, k] += -Sy * (self.grid_prev[i, j+1, k] - self.grid_prev[i, j-1, k]) / (2 * dx)
                    self.velocity_z[i, j, k] += -Sz * (self.grid_prev[i, j, k+1] - self.grid_prev[i, j, k-1]) / (2 * dx)
                    
        # Update pressure (full step)
        for i in range(1, self.grid_shape[0] - 1):
            for j in range(1, self.grid_shape[1] - 1):
                for k in range(1, self.grid_shape[2] - 1):
                    if self.boundary_mask[i, j, k]:
                        continue
                        
                    # Pressure update
                    self.grid[i, j, k] = self.grid_prev[i, j, k] - c * dt * (
                        (self.velocity_x[i+1, j, k] - self.velocity_x[i-1, j, k]) / (2 * dx) +
                        (self.velocity_y[i, j+1, k] - self.velocity_y[i, j-1, k]) / (2 * dx) +
                        (self.velocity_z[i, j, k+1] - self.velocity_z[i, j, k-1]) / (2 * dx)
                    )
                    
        # Apply absorbing boundary conditions
        self._apply_abc()
        
        # Inject source signals
        self._inject_sources(step)
        
        # Advance time
        self._time += dt
        
    def _apply_abc(self) -> None:
        """Apply simple absorbing boundary conditions."""
        alpha = self.absorption_map
        
        # Simple impedance matching ABC
        self.grid[self.boundary_mask] *= (1 - alpha[self.boundary_mask] * 0.5)
        
    def _inject_sources(self, step: int) -> None:
        """Inject source signals into the grid."""
        t = step * self.dt
        
        for source in self.sources:
            pos = source["position"]
            
            if source["signal"] is not None:
                # Get signal value at current time
                sample_idx = int(t * self.config.sample_rate)
                if sample_idx < len(source["signal"]):
                    signal_val = source["signal"][sample_idx]
                else:
                    signal_val = 0.0
            else:
                # Default to impulse
                signal_val = 1.0 if step == 0 else 0.0
                
            # Inject with Gaussian spatial window
            self._inject_at_point(pos, signal_val, radius=2)
            
    def _inject_at_point(
        self,
        pos: Tuple[int, int, int],
        amplitude: float,
        radius: int = 1
    ) -> None:
        """Inject signal with Gaussian spatial distribution."""
        i0, j0, k0 = pos
        
        for di in range(-radius, radius + 1):
            for dj in range(-radius, radius + 1):
                for dk in range(-radius, radius + 1):
                    i, j, k = i0 + di, j0 + dj, k0 + dk
                    
                    if (0 <= i < self.grid_shape[0] and
                        0 <= j < self.grid_shape[1] and
                        0 <= k < self.grid_shape[2] and
                        not self.boundary_mask[i, j, k]):
                        
                        # Gaussian window
                        r2 = di**2 + dj**2 + dk**2
                        window = np.exp(-r2 / (2 * (radius/2)**2))
                        
                        self.grid[i, j, k] += amplitude * window
                        
    def compute_rir(
        self,
        room: Room,
        source: SoundSource,
        receiver: Receiver
    ) -> np.ndarray:
        """
        Compute RIR using wave-based simulation.
        
        Args:
            room: Room geometry
            source: Sound source
            receiver: Sound receiver
            
        Returns:
            Room impulse response
        """
        # Setup grid if needed
        if self.grid is None:
            self.setup_grid(room)
            
        # Clear previous sources and receivers
        self.sources = []
        self.receivers = []
        
        # Add new source and receiver
        self.add_source(source)
        self.add_receiver(receiver)
        
        # Clear receiver samples
        for rec in self.receivers:
            rec["samples"] = []
            
        # Run simulation with impulse source
        original_signal = source.signal
        source.signal = None  # Impulse
        self.sources[0]["signal"] = None
        
        num_samples = int(self.config.duration * self.config.sample_rate)
        num_steps = int(self.config.duration / self.dt)
        
        for step in range(num_steps):
            self._update_step(step)
            
            # Record at every sample interval
            sample_interval = int(1 / (self.dt * self.config.sample_rate))
            if step % sample_interval == 0:
                pos = self.receivers[0]["position"]
                if all(0 <= p < s for p, s in zip(pos, self.grid_shape)):
                    self.receivers[0]["samples"].append(self.grid[pos])
                    
        # Restore source signal
        source.signal = original_signal
        
        # Build RIR from samples
        rir = np.array(self.receivers[0]["samples"])
        
        # Pad or trim to exact length
        if len(rir) < num_samples:
            rir = np.pad(rir, (0, num_samples - len(rir)))
        else:
            rir = rir[:num_samples]
            
        # Normalize
        if np.max(np.abs(rir)) > 1e-10:
            rir = rir / np.max(np.abs(rir))
            
        return rir
        
    def get_pressure_field(self) -> np.ndarray:
        """Get current pressure field."""
        return self.grid.copy() if self.grid is not None else np.array([])
        
    def get_velocity_field(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get current velocity fields."""
        return (
            self.velocity_x.copy() if self.velocity_x is not None else np.array([]),
            self.velocity_y.copy() if self.velocity_y is not None else np.array([]),
            self.velocity_z.copy() if self.velocity_z is not None else np.array([])
        )


class PseudoSpectralSolver(WaveEquationSolver):
    """
    Pseudo-spectral solver using FFT for faster computation.
    
    More efficient for large domains but requires periodic boundaries.
    """
    
    def __init__(self, config: SimulationConfig):
        """Initialize pseudo-spectral solver."""
        super().__init__(config)
        self.kx: Optional[np.ndarray] = None
        self.ky: Optional[np.ndarray] = None
        self.kz: Optional[np.ndarray] = None
        self.wave_numbers: Optional[np.ndarray] = None
        
    def setup_grid(self, room: Room, resolution: Optional[float] = None) -> None:
        """Set up grid with FFT precomputation."""
        super().setup_grid(room, resolution)
        
        # Precompute wave numbers
        nx, ny, nz = self.grid_shape
        
        kx = 2 * np.pi * np.fft.fftfreq(nx, self.grid_spacing)
        ky = 2 * np.pi * np.fft.fftfreq(ny, self.grid_spacing)
        kz = 2 * np.pi * np.fft.fftfreq(nz, self.grid_spacing)
        
        self.kx, self.ky, self.kz = np.meshgrid(kx, ky, kz, indexing='ij')
        self.wave_numbers = np.sqrt(self.kx**2 + self.ky**2 + self.kz**2)
        
    def _update_step(self, step: int) -> None:
        """Perform pseudo-spectral time step."""
        c = self.config.speed_of_sound
        
        # Transform to wavenumber domain
        p_hat = np.fft.fftn(self.grid)
        
        # Apply dispersion relation
        # For pseudo-spectral, we use exact wave equation solution
        omega = c * self.wave_numbers
        exp_factor = np.exp(1j * omega * self.dt)
        
        # Update in wavenumber domain
        p_hat = p_hat * exp_factor
        
        # Transform back
        self.grid = np.real(np.fft.ifftn(p_hat))
        
        # Inject sources
        self._inject_sources(step)
        
        self._time += self.dt


class PMLSolver(WaveEquationSolver):
    """
    FDTD solver with Perfectly Matched Layer (PML) absorbing boundaries.
    
    Provides superior absorption at boundaries compared to simple ABC.
    """
    
    def __init__(self, config: SimulationConfig):
        """Initialize PML solver."""
        super().__init__(config)
        self.pml_thickness = config.pml_thickness
        self.pml_profile: Optional[np.ndarray] = None
        
    def setup_grid(self, room: Room, resolution: Optional[float] = None) -> None:
        """Set up grid with PML."""
        super().setup_grid(room, resolution)
        
        # Create PML profile
        self._setup_pml()
        
    def _setup_pml(self) -> None:
        """Set up PML absorption profile."""
        nx, ny, nz = self.grid_shape
        d = self.pml_thickness / self.grid_spacing
        
        self.pml_sigma = np.zeros(self.grid_shape)
        
        # Profile in each dimension
        for i in range(nx):
            dist_x = min(i, nx - 1 - i)
            if dist_x < d:
                sigma = ((d - dist_x) / d) ** 2
                self.pml_sigma[i, :, :] = max(self.pml_sigma[i, :, :], sigma)
                
        for j in range(ny):
            dist_y = min(j, ny - 1 - j)
            if dist_y < d:
                sigma = ((d - dist_y) / d) ** 2
                self.pml_sigma[:, j, :] = max(self.pml_sigma[:, j, :], sigma)
                
        for k in range(nz):
            dist_z = min(k, nz - 1 - k)
            if dist_z < d:
                sigma = ((d - dist_z) / d) ** 2
                self.pml_sigma[:, :, k] = max(self.pml_sigma[:, :, k], sigma)
                
        # Scale by frequency
        self.pml_sigma *= 2 * np.pi * 1000 / (self.config.speed_of_sound * 343)
        
    def _update_step(self, step: int) -> None:
        """Perform FDTD step with PML."""
        c = self.config.speed_of_sound
        dx = self.grid_spacing
        dt = self.dt
        
        # Store previous pressure
        self.grid_prev[:] = self.grid
        
        # Update with PML damping
        damping = np.exp(-self.pml_sigma * dt)
        
        # Simplified PML update
        self.grid = self.grid * damping
        
        # Add standard FDTD update
        for i in range(1, self.grid_shape[0] - 1):
            for j in range(1, self.grid_shape[1] - 1):
                for k in range(1, self.grid_shape[2] - 1):
                    if self.boundary_mask[i, j, k]:
                        continue
                        
                    laplacian = (
                        (self.grid_prev[i+1, j, k] - 2*self.grid_prev[i, j, k] + self.grid_prev[i-1, j, k]) +
                        (self.grid_prev[i, j+1, k] - 2*self.grid_prev[i, j, k] + self.grid_prev[i, j-1, k]) +
                        (self.grid_prev[i, j, k+1] - 2*self.grid_prev[i, j, k] + self.grid_prev[i, j, k-1])
                    ) / (dx**2)
                    
                    self.grid[i, j, k] += c**2 * dt**2 * laplacian
                    
        # Inject sources
        self._inject_sources(step)
        
        self._time += self.dt
