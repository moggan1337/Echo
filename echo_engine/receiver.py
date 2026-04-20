"""
Sound Receiver and HRTF Processing
Handles listener modeling and head-related transfer functions.
"""

import numpy as np
from typing import Optional, Tuple, Union, List
from dataclasses import dataclass
import scipy.signal as signal


@dataclass
class ListenerConfig:
    """Configuration for listener/receiver properties."""
    head_radius: float = 0.0875  # Average human head radius in meters
    torso_width: float = 0.3     # Shoulder width
    torso_depth: float = 0.15    # Shoulder depth
    ear_height: float = 0.1      # Height of ears above head center
    pinna_angle: float = 15.0    # Pinna angle in degrees
    
    # ITD parameters
    use_antipodal_ITD: bool = True  # Use Woodworth's formula
    
    # HRTF parameters
    hrtf_dataset: str = "CIPIC"  # CIPIC, KEMAR, IRC, etc.
    hrtf_subject: Optional[int] = None


class HRTFProcessor:
    """
    Head-Related Transfer Function processor.
    
    HRTF models how the head, torso, and pinnae affect sound before it
    reaches the eardrums. This enables spatial audio rendering for
    binaural playback.
    
    Example:
        >>> hrtf = HRTFProcessor()
        >>> left_ear, right_ear = hrtf.apply_hrtf(signal, azimuth, elevation)
    """
    
    def __init__(
        self,
        sample_rate: float = 48000.0,
        config: Optional[ListenerConfig] = None
    ):
        """
        Initialize HRTF processor.
        
        Args:
            sample_rate: Audio sample rate in Hz
            config: Listener configuration
        """
        self.sample_rate = sample_rate
        self.config = config or ListenerConfig()
        
        # Precompute HRTF filters (simplified model)
        self._init_hrtf_model()
        
        # Load default HRTF dataset (simplified)
        self._load_default_hrtf()
        
    def _init_hrtf_model(self) -> None:
        """Initialize simplified HRTF model parameters."""
        # Frequency bands for HRTF (critical bands)
        self.frequencies = np.array([
            250, 500, 1000, 2000, 3000, 4000, 6000, 8000, 10000, 12000
        ])
        
        # Head shadowing coefficients (simplified)
        self.head_shadow = self._compute_head_shadow()
        
        # Pinna filtering coefficients
        self.pinna_filter = self._compute_pinna_filter()
        
    def _compute_head_shadow(self) -> np.ndarray:
        """
        Compute head shadowing effect.
        
        Based on Rayleigh's model for head diffraction.
        """
        k = 2 * np.pi * self.frequencies / 343.0  # Wave number
        a = self.config.head_radius
        
        # Diffraction around sphere (Rayleigh model)
        # For angles: 0 = directly ahead, 90 = 90 degrees from source
        shadow = np.zeros((len(self.frequencies), 181))
        
        for i, freq in enumerate(self.frequencies):
            ka = k[i] * a
            if ka < 0.5:
                # Low frequency: minimal shadowing
                shadow[i, :] = 1.0
            elif ka > 10:
                # High frequency: significant shadowing
                for angle in range(181):
                    theta = np.radians(angle)
                    if theta < np.pi / 2:
                        shadow[i, angle] = 1.0 - 0.5 * np.cos(theta)
                    else:
                        shadow[i, angle] = 0.5 + 0.5 * np.cos(theta)
            else:
                # Mid frequencies: interpolation
                for angle in range(181):
                    theta = np.radians(angle)
                    factor = np.exp(-ka * (1 + np.cos(theta)) / 2)
                    shadow[i, angle] = 0.5 + 0.5 * factor
                    
        return shadow
        
    def _compute_pinna_filter(self) -> np.ndarray:
        """
        Compute pinna filtering effect.
        
        Pinnae create spectral notches and peaks that help localize
        sound in the vertical dimension.
        """
        # Simplified pinna model with characteristic frequency notches
        notch_freqs = [3000, 5000, 8000, 10000]  # Hz
        notch_depths = [6, 10, 8, 12]  # dB
        notch_widths = [500, 1000, 1500, 2000]  # Hz
        
        # Create filter response
        pinna = np.ones(len(self.frequencies))
        
        for notch_freq, notch_depth, notch_width in zip(
            notch_freqs, notch_depths, notch_widths
        ):
            idx = np.argmin(np.abs(self.frequencies - notch_freq))
            for i, freq in enumerate(self.frequencies):
                dist = abs(freq - notch_freq)
                if dist < notch_width:
                    attenuation = notch_depth * (1 - dist / notch_width)**2
                    pinna[i] *= 10**(-attenuation / 20)
                    
        return pinna
        
    def _load_default_hrtf(self) -> None:
        """Load default HRTF dataset (simplified synthetic HRTF)."""
        # Create synthetic HRTF based on acoustic model
        self.hrtf_data = self._generate_synthetic_hrtf()
        
    def _generate_synthetic_hrtf(self) -> dict:
        """
        Generate synthetic HRTF based on acoustic model.
        
        This creates a basic HRTF set without requiring external data.
        """
        # Spherical head model with pinna
        n_azimuths = 72  # 5-degree resolution
        n_elevations = 19  # -40 to +90 degrees
        
        azimuths = np.linspace(0, 360, n_azimuths, endpoint=False)
        elevations = np.linspace(-40, 90, n_elevations)
        
        # ILD (Interaural Level Difference) model
        ild = np.zeros((n_elevations, n_azimuths, len(self.frequencies)))
        
        # ITD (Interaural Time Difference) model
        itd = np.zeros((n_elevations, n_azimuths))
        
        c = 343.0  # Speed of sound
        a = self.config.head_radius
        
        for i_el, elev in enumerate(elevations):
            for i_az, az in enumerate(azimuths):
                # ITD using Woodworth's formula
                theta = np.radians(90 - elev)  # Angle from median plane
                phi = np.radians(az)
                
                if self.config.use_antipodal_ITD:
                    # Woodworth's formula
                    itd[i_el, i_az] = (3 * a / c) * (
                        0.5 * np.sin(theta) + 
                        np.cos(theta) * np.arcsin(np.sin(theta) / 2)
                    ) * np.cos(phi)
                else:
                    # Simple model
                    itd[i_el, i_az] = (a / c) * np.sin(theta) * np.cos(phi)
                    
                # ILD based on head shadowing
                # Ipsilateral ear (same side as source)
                ipsi_az = az
                # Contralateral ear (opposite side)
                contra_az = (az + 180) % 360
                
                for i_f, freq in enumerate(self.frequencies):
                    # Find angle index for shadowing
                    az_idx = int(ipsi_az / 5) % 72
                    az_idx_contra = int(contra_az / 5) % 72
                    
                    shadow_ipsi = self.head_shadow[i_f, min(az_idx, 180)]
                    shadow_contra = self.head_shadow[i_f, min(az_idx_contra, 180)]
                    
                    # ILD in dB
                    ild[i_el, i_az, i_f] = 20 * np.log10(
                        shadow_ipsi / (shadow_contra + 1e-10) + 1e-10
                    )
                    
        return {
            "azimuths": azimuths,
            "elevations": elevations,
            "frequencies": self.frequencies,
            "ild": ild,
            "itd": itd,
            "sample_rate": self.sample_rate
        }
        
    def load_sofa_file(self, path: str) -> None:
        """
        Load HRTF from SOFA file.
        
        Args:
            path: Path to SOFA file
        """
        # Placeholder for SOFA file loading
        # In practice: use pysofaconventions library
        pass
        
    def get_hrtf(
        self,
        azimuth: float,
        elevation: float,
        ear: str = "both"
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Get HRTF for a specific direction.
        
        Args:
            azimuth: Azimuth angle in degrees (0 = front, 90 = right)
            elevation: Elevation angle in degrees (-90 = below, 0 = horizon, +90 = above)
            ear: "left", "right", or "both"
            
        Returns:
            Tuple of (left HRTF, right HRTF, ITD in seconds)
        """
        # Find nearest indices
        az_data = self.hrtf_data["azimuths"]
        el_data = self.hrtf_data["elevations"]
        
        az_idx = np.argmin(np.abs(az_data - azimuth % 360))
        el_idx = np.argmin(np.abs(el_data - elevation))
        
        # Get ILD
        ild = self.hrtf_data["ild"][el_idx, az_idx, :]
        
        # Get ITD
        itd = self.hrtf_data["itd"][el_idx, az_idx]
        
        # Create frequency response
        # Reference level at 1kHz
        ref_idx = np.argmin(np.abs(self.frequencies - 1000))
        ref_level = ild[ref_idx] if ild[ref_idx] != 0 else 1.0
        
        # Normalize and apply pinna
        hrtf_left = 10**(ild / 20) * self.pinna_filter
        hrtf_right = 10**(-ild / 20) * self.pinna_filter
        
        if ear == "left":
            return hrtf_left, np.array([]), itd
        elif ear == "right":
            return np.array([]), hrtf_right, itd
        else:
            return hrtf_left, hrtf_right, itd
            
    def apply_hrtf(
        self,
        signal: np.ndarray,
        azimuth: float,
        elevation: float = 0.0,
        ear: str = "both"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply HRTF to a mono signal for spatial rendering.
        
        Args:
            signal: Input mono signal
            azimuth: Azimuth angle in degrees
            elevation: Elevation angle in degrees
            ear: "left", "right", or "both"
            
        Returns:
            Tuple of (left channel, right channel)
        """
        hrtf_left, hrtf_right, itd = self.get_hrtf(azimuth, elevation, ear)
        
        # Interpolate to full frequency range
        full_freq = np.linspace(20, 20000, len(signal) // 2)
        hrtf_left_interp = np.interp(full_freq, self.frequencies, hrtf_left)
        hrtf_right_interp = np.interp(full_freq, self.frequencies, hrtf_right)
        
        # Create frequency domain filters
        hrtf_left_fft = np.concatenate([hrtf_left_interp, hrtf_left_interp[::-1]])
        hrtf_right_fft = np.concatenate([hrtf_right_interp, hrtf_right_interp[::-1]])
        
        # Apply via convolution in frequency domain
        signal_fft = np.fft.rfft(signal)
        
        left_fft = signal_fft * hrtf_left_fft[:len(signal_fft)]
        right_fft = signal_fft * hrtf_right_fft[:len(signal_fft)]
        
        left_channel = np.fft.irfft(left_fft)
        right_channel = np.fft.irfft(right_fft)
        
        # Apply ITD
        itd_samples = int(itd * self.sample_rate)
        if itd > 0:
            # Source to the right: delay left channel
            left_channel = np.roll(left_channel, -itd_samples)
        else:
            # Source to the left: delay right channel
            right_channel = np.roll(right_channel, itd_samples)
            
        return left_channel, right_channel
        
    def render_binaural(
        self,
        signals: List[Tuple[np.ndarray, float, float]],
        head_orientation: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Render multiple sound sources to binaural output.
        
        Args:
            signals: List of (signal, azimuth, elevation) tuples
            head_orientation: Listener head orientation in degrees
            
        Returns:
            Tuple of (left channel, right channel)
        """
        left_out = np.zeros(int(self.sample_rate * 5))  # 5 seconds max
        right_out = np.zeros_like(left_out)
        
        for sig, az, el in signals:
            # Adjust azimuth relative to head orientation
            rel_az = (az - head_orientation) % 360
            
            left, right = self.apply_hrtf(sig, rel_az, el)
            
            # Mix into output
            min_len = min(len(left), len(left_out))
            left_out[:min_len] += left[:min_len]
            right_out[:min_len] += right[:min_len]
            
        return left_out, right_out
        
    def compute_ITD(self, azimuth: float, elevation: float = 0.0) -> float:
        """
        Compute interaural time difference.
        
        Args:
            azimuth: Source azimuth in degrees
            elevation: Source elevation in degrees
            
        Returns:
            ITD in seconds
        """
        c = 343.0
        a = self.config.head_radius
        
        theta = np.radians(90 - elevation)
        phi = np.radians(azimuth)
        
        if self.config.use_antipodal_ITD:
            itd = (3 * a / c) * (
                0.5 * np.sin(theta) + 
                np.cos(theta) * np.arcsin(np.sin(theta) / 2)
            ) * np.cos(phi)
        else:
            itd = (a / c) * np.sin(theta) * np.cos(phi)
            
        return itd
        
    def compute_ILD(self, azimuth: float, elevation: float = 0.0, frequency: float = 1000.0) -> float:
        """
        Compute interaural level difference.
        
        Args:
            azimuth: Source azimuth in degrees
            elevation: Source elevation in degrees
            frequency: Frequency in Hz
            
        Returns:
            ILD in dB
        """
        freq_idx = np.argmin(np.abs(self.frequencies - frequency))
        
        az_idx = int(azimuth / 5) % 72
        
        el_idx = np.argmin(np.abs(self.hrtf_data["elevations"] - elevation))
        
        ild = self.hrtf_data["ild"][el_idx, az_idx, freq_idx]
        
        return ild
        
    def __repr__(self) -> str:
        return f"HRTFProcessor(sample_rate={self.sample_rate}, dataset='{self.config.hrtf_dataset}')"


class Receiver:
    """
    Represents a sound receiver/listener in the acoustic simulation.
    
    A receiver captures sound at a specific position with a specific
    orientation and HRTF processing.
    
    Example:
        >>> receiver = Receiver(
        ...     position=[5.0, 4.0, 1.2],
        ...     orientation=45.0,  # Facing 45 degrees
        ...     name="Listener"
        ... )
    """
    
    def __init__(
        self,
        position: Union[np.ndarray, Tuple[float, float, float]],
        orientation: float = 0.0,
        elevation: float = 0.0,
        name: str = "Receiver",
        config: Optional[ListenerConfig] = None,
        enable_hrtf: bool = True
    ):
        """
        Initialize a receiver.
        
        Args:
            position: Receiver position (x, y, z) in meters
            orientation: Horizontal facing direction in degrees (0 = +X)
            elevation: Vertical tilt in degrees (0 = horizontal)
            name: Receiver name
            config: Listener configuration
            enable_hrtf: Enable HRTF processing
        """
        self.position = np.array(position, dtype=float)
        self.orientation = orientation  # degrees
        self.elevation = elevation      # degrees
        self.name = name
        
        self.config = config or ListenerConfig()
        self.enable_hrtf = enable_hrtf
        
        if enable_hrtf:
            self.hrtf = HRTFProcessor(config=self.config)
        else:
            self.hrtf = None
            
        # Dynamic properties
        self.velocity = np.zeros(3)
        self._orientation_history: list = []
        
    @property
    def orientation_rad(self) -> float:
        """Get orientation in radians."""
        return np.radians(self.orientation)
        
    @property
    def forward_vector(self) -> np.ndarray:
        """Get forward direction vector."""
        az = self.orientation_rad
        el = np.radians(self.elevation)
        return np.array([
            np.cos(el) * np.cos(az),
            np.cos(el) * np.sin(az),
            np.sin(el)
        ])
        
    @property
    def up_vector(self) -> np.ndarray:
        """Get up vector."""
        return np.array([0, 0, 1])
        
    @property
    def right_vector(self) -> np.ndarray:
        """Get right vector."""
        return np.cross(self.up_vector, self.forward_vector)
        
    def set_orientation(self, orientation: float, elevation: Optional[float] = None) -> None:
        """Set receiver orientation."""
        self._orientation_history.append((self.orientation, self.elevation))
        self.orientation = orientation % 360
        if elevation is not None:
            self.elevation = elevation
            
    def rotate(self, delta_yaw: float = 0.0, delta_pitch: float = 0.0) -> None:
        """
        Rotate the receiver.
        
        Args:
            delta_yaw: Horizontal rotation in degrees
            delta_pitch: Vertical rotation in degrees
        """
        self.orientation = (self.orientation + delta_yaw) % 360
        self.elevation = np.clip(self.elevation + delta_pitch, -90, 90)
        
    def direction_from(self, source_pos: np.ndarray) -> Tuple[float, float]:
        """
        Get direction from receiver to a source.
        
        Args:
            source_pos: Source position
            
        Returns:
            Tuple of (azimuth, elevation) in degrees relative to receiver
        """
        source_pos = np.array(source_pos)
        direction = source_pos - self.position
        
        # Transform to receiver-local coordinates
        forward = self.forward_vector
        right = self.right_vector
        up = self.up_vector
        
        # Spherical coordinates in receiver's local frame
        x = np.dot(direction, right)
        y = np.dot(direction, forward)
        z = np.dot(direction, up)
        
        azimuth = np.degrees(np.arctan2(x, y))
        elevation = np.degrees(np.arcsin(z / (np.linalg.norm(direction) + 1e-10)))
        
        return azimuth, elevation
        
    def distance_to(self, point: np.ndarray) -> float:
        """Compute distance to a point."""
        return float(np.linalg.norm(self.position - np.array(point)))
        
    def receive(
        self,
        signal: np.ndarray,
        source_azimuth: float,
        source_elevation: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process incoming sound through receiver.
        
        Args:
            signal: Input audio signal
            source_azimuth: Source azimuth relative to receiver
            source_elevation: Source elevation relative to receiver
            
        Returns:
            Tuple of (left ear, right ear) signals
        """
        if self.enable_hrtf and self.hrtf is not None:
            return self.hrtf.apply_hrtf(signal, source_azimuth, source_elevation)
        else:
            # Return mono signal as both channels
            return signal.copy(), signal.copy()
            
    def set_hrtf_enabled(self, enabled: bool) -> None:
        """Enable or disable HRTF processing."""
        self.enable_hrtf = enabled
        if enabled and self.hrtf is None:
            self.hrtf = HRTFProcessor(config=self.config)
            
    def __repr__(self) -> str:
        return f"Receiver('{self.name}' at {self.position}, facing {self.orientation}°)"
