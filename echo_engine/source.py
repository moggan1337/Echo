"""
Sound Source Modeling
Includes source directivity patterns and signal generation.
"""

import numpy as np
from typing import Optional, Callable, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import scipy.signal as signal


class DirectivityPattern(Enum):
    """Common directivity patterns."""
    OMNIDIRECTIONAL = "omnidirectional"
    CARDIOID = "cardioid"
    HYPERCARDIOID = "hypercardioid"
    SUBROCARDIOID = "subcardioid"
    FIGURE_EIGHT = "figure_eight"
    CARDIOID_90 = "cardioid_90"
    HEMISPHERE = "hemisphere"
    CONE = "cone"
    CUSTOM = "custom"


@dataclass
class DirectivityData:
    """Custom directivity measurement data."""
    frequencies: np.ndarray
    azimuths: np.ndarray
    elevations: np.ndarray
    magnitudes: np.ndarray  # (freq, az, el)
    
    def interpolate(self, azimuth: float, elevation: float, freq: float) -> float:
        """Interpolate directivity at given angle and frequency."""
        from scipy.interpolate import RegularGridInterpolator
        
        # Find nearest frequency
        freq_idx = np.argmin(np.abs(self.frequencies - freq))
        
        # Interpolate in angular domain
        interp_az = RegularGridInterpolator(
            (self.elevations, self.azimuths),
            self.magnitudes[freq_idx],
            method="linear",
            bounds_error=False,
            fill_value=0.0
        )
        
        return float(interp_az([elevation, azimuth]))


class SourceDirectivity:
    """
    Sound source directivity model.
    
    Directivity describes how the radiation pattern of a sound source varies
    with direction. This affects how sound is perceived at different positions
    around the source.
    
    Example:
        >>> # Cardioid microphone pointing in +X direction
        >>> directivity = SourceDirectivity.cardioid(axis=(1, 0, 0))
        >>> response = directivity.get_response(angles)  # Get response at angles
    """
    
    def __init__(
        self,
        pattern: DirectivityPattern = DirectivityPattern.OMNIDIRECTIONAL,
        custom_function: Optional[Callable[[float, float], float]] = None,
        axis: Tuple[float, float, float] = (1.0, 0.0, 0.0),
        angle_range: float = 180.0,
        front_attenuation: float = 0.0
    ):
        """
        Initialize source directivity.
        
        Args:
            pattern: Directivity pattern type
            custom_function: Custom directivity function (azimuth, elevation) -> gain
            axis: Main axis of radiation (for directional patterns)
            angle_range: Opening angle for cone patterns
            front_attenuation: Front-to-back ratio in dB (for cardioid patterns)
        """
        self.pattern = pattern
        self.custom_function = custom_function
        self.axis = np.array(axis) / np.linalg.norm(axis)
        self.angle_range = angle_range
        self.front_attenuation = front_attenuation
        
        # Precompute pattern parameters
        self._init_pattern_params()
        
    def _init_pattern_params(self) -> None:
        """Initialize pattern-specific parameters."""
        if self.pattern == DirectivityPattern.CARDIOID:
            # Cardioid: R = 0.5(1 + cos θ)
            self._pattern_func = lambda theta: 0.5 * (1 + np.cos(theta))
        elif self.pattern == DirectivityPattern.HYPERCARDIOID:
            # Hypercardioid: R = 0.25(1 + 3cos θ)
            self._pattern_func = lambda theta: 0.25 * (1 + 3 * np.cos(theta))
        elif self.pattern == DirectivityPattern.SUBROCARDIOID:
            # Subcardioid: R = 0.5 + 0.5cos θ
            self._pattern_func = lambda theta: 0.5 + 0.5 * np.cos(theta)
        elif self.pattern == DirectivityPattern.FIGURE_EIGHT:
            # Figure-8: R = |cos θ|
            self._pattern_func = lambda theta: np.abs(np.cos(theta))
        elif self.pattern == DirectivityPattern.HEMISPHERE:
            # Hemisphere: 1 for front hemisphere, 0 for back
            self._pattern_func = lambda theta: np.where(theta <= np.pi/2, 1.0, 0.0)
        elif self.pattern == DirectivityPattern.CARDIOID_90:
            # 90-degree cardioid
            self._pattern_func = lambda theta: np.where(
                theta <= np.pi/2,
                1.0,
                0.5 * (1 + np.cos(theta))
            )
        else:  # OMNIDIRECTIONAL or CUSTOM
            self._pattern_func = lambda theta: 1.0
            
    @classmethod
    def omnidirectional(cls) -> "SourceDirectivity":
        """Create an omnidirectional source."""
        return cls(DirectivityPattern.OMNIDIRECTIONAL)
        
    @classmethod
    def cardioid(
        cls,
        axis: Tuple[float, float, float] = (1.0, 0.0, 0.0),
        front_attenuation: float = 0.0
    ) -> "SourceDirectivity":
        """
        Create a cardioid (heart-shaped) directivity.
        
        Args:
            axis: Main axis of sensitivity
            front_attenuation: Front-to-back ratio in dB
        """
        return cls(
            DirectivityPattern.CARDIOID,
            axis=axis,
            front_attenuation=front_attenuation
        )
        
    @classmethod
    def hypercardioid(cls, axis: Tuple[float, float, float] = (1.0, 0.0, 0.0)) -> "SourceDirectivity":
        """Create a hypercardioid directivity (more directional than cardioid)."""
        return cls(DirectivityPattern.HYPERCARDIOID, axis=axis)
        
    @classmethod
    def figure_eight(cls, axis: Tuple[float, float, float] = (1.0, 0.0, 0.0)) -> "SourceDirectivity":
        """Create a figure-8 (bidirectional) directivity."""
        return cls(DirectivityPattern.FIGURE_EIGHT, axis=axis)
        
    @classmethod
    def cone(
        cls,
        axis: Tuple[float, float, float],
        half_angle: float = 45.0
    ) -> "SourceDirectivity":
        """
        Create a cone-shaped directivity.
        
        Args:
            axis: Axis of the cone
            half_angle: Half-angle of cone in degrees
        """
        return cls(
            DirectivityPattern.CONE,
            axis=axis,
            angle_range=half_angle * 2
        )
        
    @classmethod
    def from_measurement(
        cls,
        frequencies: np.ndarray,
        azimuths: np.ndarray,
        elevations: np.ndarray,
        magnitudes: np.ndarray
    ) -> "SourceDirectivity":
        """
        Create directivity from measurement data.
        
        Args:
            frequencies: Frequency values in Hz
            azimuths: Azimuth angles in radians
            elevations: Elevation angles in radians
            magnitudes: Directivity magnitudes in dB
        """
        directivity_data = DirectivityData(
            frequencies=frequencies,
            azimuths=azimuths,
            elevations=elevations,
            magnitudes=magnitudes
        )
        
        directivity = cls(DirectivityPattern.CUSTOM)
        directivity._custom_data = directivity_data
        directivity.custom_function = lambda az, el, f=0, d=directivity_data: d.interpolate(az, el, f)
        
        return directivity
        
    @classmethod
    def from_file(cls, path: str) -> "SourceDirectivity":
        """Load directivity from file (e.g., SOFA format)."""
        # Placeholder for SOFA file loading
        # In practice, would use pysofaconventions
        return cls.omnidirectional()
        
    def get_response(self, azimuth: np.ndarray, elevation: np.ndarray) -> np.ndarray:
        """
        Get directivity response at given angles.
        
        Args:
            azimuth: Azimuth angles in radians (0 = axis direction)
            elevation: Elevation angles in radians
            
        Returns:
            Directivity gain (linear scale)
        """
        # Compute angle from main axis
        # Convert to spherical coordinates relative to axis
        cos_angle = np.cos(elevation) * np.cos(azimuth)
        angle = np.arccos(np.clip(cos_angle, -1, 1))
        
        if self.pattern == DirectivityPattern.CUSTOM and self.custom_function:
            if hasattr(self, "_custom_data"):
                # Frequency-dependent custom directivity
                results = []
                for az, el in zip(azimuth, elevation):
                    # Use first frequency as default
                    results.append(self.custom_function(az, el, self._custom_data.frequencies[0]))
                return np.array(results)
            else:
                return np.array([self.custom_function(az, el) for az, el in zip(azimuth, elevation)])
                
        return self._pattern_func(angle)
        
    def get_filter(self, receiver_pos: np.ndarray) -> np.ndarray:
        """
        Get frequency-domain filter for a receiver position.
        
        Args:
            receiver_pos: Receiver position relative to source
            
        Returns:
            Filter coefficients
        """
        # Compute direction to receiver
        direction = receiver_pos / (np.linalg.norm(receiver_pos) + 1e-10)
        
        # Compute angle from main axis
        cos_angle = np.dot(direction, self.axis)
        angle = np.arccos(np.clip(cos_angle, -1, 1))
        
        # Get magnitude response
        magnitude = self._pattern_func(angle)
        
        # Create simple lowpass-based filter (placeholder for actual freq response)
        # In practice, would use measured or modeled frequency response
        return np.array([magnitude])
        
    def plot_pattern(
        self,
        num_points: int = 360,
        elevation: float = 0.0,
        ax=None
    ) -> None:
        """Plot the directivity pattern (requires matplotlib)."""
        try:
            import matplotlib.pyplot as plt
            from matplotlib import cm
            
            azimuths = np.linspace(0, 2 * np.pi, num_points)
            response = self.get_response(azimuths, np.full_like(azimuths, elevation))
            
            if ax is None:
                fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
                
            ax.plot(azimuths, response)
            ax.fill(azimuths, response, alpha=0.3)
            ax.set_title(f"{self.pattern.value} Directivity Pattern")
            
        except ImportError:
            print("matplotlib not available for plotting")


class SoundSource:
    """
    Represents a sound source in the acoustic simulation.
    
    A sound source is defined by its position, directivity pattern,
    and the audio signal it emits.
    
    Example:
        >>> source = SoundSource(
        ...     position=[3.0, 4.0, 1.5],
        ...     directivity=SourceDirectivity.cardioid(axis=(0, 1, 0)),
        ...     signal=audio_signal,
        ...     name="Piano"
        ... )
    """
    
    def __init__(
        self,
        position: Union[np.ndarray, Tuple[float, float, float]],
        signal: Optional[np.ndarray] = None,
        directivity: Optional[SourceDirectivity] = None,
        name: str = "Source",
        signal_amplitude: float = 1.0,
        signal_offset: float = 0.0
    ):
        """
        Initialize a sound source.
        
        Args:
            position: Source position (x, y, z) in meters
            signal: Audio signal (optional, can be set later)
            directivity: Source directivity pattern
            name: Source name for identification
            signal_amplitude: Signal amplitude scaling
            signal_offset: Signal DC offset
        """
        self.position = np.array(position, dtype=float)
        self.signal = signal
        self.name = name
        self.signal_amplitude = signal_amplitude
        self.signal_offset = signal_offset
        
        if directivity is None:
            self.directivity = SourceDirectivity.omnidirectional()
        else:
            self.directivity = directivity
            
        # Dynamic properties
        self.velocity = np.zeros(3)
        self.acceleration = np.zeros(3)
        self._position_history: list = []
        
    @property
    def signal_with_gain(self) -> Optional[np.ndarray]:
        """Get signal with amplitude and offset applied."""
        if self.signal is None:
            return None
        return self.signal * self.signal_amplitude + self.signal_offset
        
    def set_signal(self, signal: np.ndarray) -> None:
        """Set the source audio signal."""
        self.signal = signal
        
    def set_position(self, position: Union[np.ndarray, Tuple[float, float, float]]) -> None:
        """Update source position."""
        old_position = self.position.copy()
        self.position = np.array(position, dtype=float)
        
        # Update velocity based on position change
        if len(self._position_history) > 0:
            dt = 1.0 / 48000  # Assume 48kHz sample rate
            self.velocity = (self.position - old_position) / dt
            
    def update_position(self, position: np.ndarray, dt: float) -> None:
        """
        Update position with time step (for moving sources).
        
        Args:
            position: New position
            dt: Time step in seconds
        """
        self._position_history.append(self.position.copy())
        if len(self._position_history) > 100:
            self._position_history.pop(0)
            
        old_position = self.position.copy()
        self.position = position
        self.velocity = (position - old_position) / (dt + 1e-10)
        
    def get_velocity(self) -> np.ndarray:
        """Get current velocity."""
        return self.velocity.copy()
        
    def get_acceleration(self) -> np.ndarray:
        """Estimate acceleration from velocity history."""
        if len(self._position_history) < 2:
            return np.zeros(3)
            
        velocities = np.diff(self._position_history, axis=0)
        if len(velocities) > 0:
            return np.mean(velocities, axis=0)
        return np.zeros(3)
        
    def distance_to(self, point: np.ndarray) -> float:
        """Compute distance to a point."""
        return float(np.linalg.norm(self.position - np.array(point)))
        
    def direction_to(self, point: np.ndarray) -> np.ndarray:
        """Get direction vector to a point (normalized)."""
        direction = np.array(point) - self.position
        norm = np.linalg.norm(direction)
        if norm < 1e-10:
            return np.zeros(3)
        return direction / norm
        
    def angle_to(self, point: np.ndarray) -> float:
        """Get angle to a point relative to source axis."""
        direction = self.direction_to(point)
        return float(np.arccos(np.clip(np.dot(direction, self.directivity.axis), -1, 1)))
        
    def get_signal_at_receiver(
        self,
        receiver_pos: np.ndarray,
        distance: float,
        speed_of_sound: float = 343.0,
        sample_rate: float = 48000.0
    ) -> Tuple[np.ndarray, float]:
        """
        Get the source signal as received at a receiver position.
        
        Args:
            receiver_pos: Receiver position
            distance: Distance to receiver
            speed_of_sound: Speed of sound in m/s
            sample_rate: Audio sample rate
            
        Returns:
            Tuple of (delayed signal, delay in samples)
        """
        if self.signal is None:
            return np.array([]), 0
            
        # Apply directivity
        angle = self.angle_to(receiver_pos)
        azimuth = np.arctan2(
            receiver_pos[1] - self.position[1],
            receiver_pos[0] - self.position[0]
        )
        elevation = np.arcsin(
            (receiver_pos[2] - self.position[2]) / (distance + 1e-10)
        )
        
        directivity_gain = self.directivity.get_response(
            np.array([azimuth]),
            np.array([elevation])
        )[0]
        
        # Apply distance attenuation (inverse square law + air absorption)
        distance_attenuation = 1.0 / (4 * np.pi * distance**2 + 1e-10)
        
        # Total gain
        gain = directivity_gain * distance_attenuation
        
        # Time delay
        delay_samples = int(distance / speed_of_sound * sample_rate)
        
        # Apply gain to signal
        output_signal = self.signal_with_gain * gain if self.signal_with_gain is not None else np.array([])
        
        return output_signal, delay_samples
        
    def generate_impulse(
        self,
        duration: float,
        sample_rate: float = 48000.0,
        peak_freq: float = 1000.0
    ) -> np.ndarray:
        """
        Generate an impulse signal.
        
        Args:
            duration: Duration in seconds
            sample_rate: Sample rate in Hz
            peak_freq: Center frequency for bandpass
            
        Returns:
            Impulse signal
        """
        n_samples = int(duration * sample_rate)
        impulse = np.zeros(n_samples)
        impulse[0] = 1.0
        
        # Apply bandpass filter
        if peak_freq < sample_rate / 2:
            nyquist = sample_rate / 2
            low = max(peak_freq * 0.5 / nyquist, 0.01)
            high = min(peak_freq * 2.0 / nyquist, 0.99)
            b, a = signal.butter(4, [low, high], btype="band")
            self.signal = signal.filtfilt(b, a, impulse)
        else:
            self.signal = impulse
            
        return self.signal
        
    def generate_sweep(
        self,
        duration: float,
        sample_rate: float = 48000.0,
        freq_start: float = 20.0,
        freq_end: float = 20000.0,
        sweep_type: str = "exponential"
    ) -> np.ndarray:
        """
        Generate a frequency sweep signal.
        
        Args:
            duration: Duration in seconds
            sample_rate: Sample rate in Hz
            freq_start: Start frequency in Hz
            freq_end: End frequency in Hz
            sweep_type: "linear" or "exponential"
            
        Returns:
            Sweep signal
        """
        t = np.linspace(0, duration, int(duration * sample_rate))
        
        if sweep_type == "exponential":
            sweep = np.sin(
                2 * np.pi * freq_start * duration *
                (np.exp(t / duration * np.log(freq_end / freq_start)) - 1) /
                np.log(freq_end / freq_start)
            )
        else:
            freq_range = freq_end - freq_start
            phase = 2 * np.pi * (freq_start * t + 0.5 * freq_range / duration * t**2)
            sweep = np.sin(phase)
            
        self.signal = sweep
        return self.signal
        
    def generate_noise(
        self,
        duration: float,
        sample_rate: float = 48000.0,
        noise_type: str = "white",
        amplitude: float = 0.1
    ) -> np.ndarray:
        """
        Generate noise signal.
        
        Args:
            duration: Duration in seconds
            sample_rate: Sample rate in Hz
            noise_type: "white", "pink", "brown"
            amplitude: Output amplitude
            
        Returns:
            Noise signal
        """
        n_samples = int(duration * sample_rate)
        
        if noise_type == "white":
            noise = np.random.randn(n_samples)
        elif noise_type == "pink":
            # Pink noise via Voss-McCartney algorithm
            noise = self._generate_pink_noise(n_samples)
        else:  # brown
            noise = self._generate_brown_noise(n_samples)
            
        self.signal = noise * amplitude
        return self.signal
        
    def _generate_pink_noise(self, n_samples: int) -> np.ndarray:
        """Generate pink noise."""
        white = np.random.randn(n_samples)
        
        # Simple pink noise approximation via IIR
        b = np.array([0.02109238, 0.07113478, 0.68873558])
        a = np.array([1.0, -0.50934324, 0.14068827])
        
        return signal.lfilter(b, a, white)
        
    def _generate_brown_noise(self, n_samples: int) -> np.ndarray:
        """Generate brown (red) noise."""
        white = np.random.randn(n_samples)
        
        # Brown noise via integration
        brown = np.cumsum(white)
        brown = brown - np.mean(brown)
        
        return brown / (np.max(np.abs(brown)) + 1e-10)
        
    def __repr__(self) -> str:
        return f"SoundSource('{self.name}' at {self.position})"
