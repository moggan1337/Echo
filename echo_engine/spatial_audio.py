"""
Spatial Audio Rendering
Binaural and Ambisonics encoding for immersive audio playback.
"""

import numpy as np
from typing import Optional, Tuple, List, Union
import scipy.signal as signal

from .core import SimulationConfig
from .receiver import Receiver, ListenerConfig


class BinauralRenderer:
    """
    Binaural audio renderer.
    
    Converts mono signals to binaural (stereo) format with spatial
    positioning using head-related transfer functions.
    
    Example:
        >>> renderer = BinauralRenderer(config)
        >>> binaural = renderer.render(mono_signal, azimuth=45, elevation=10)
    """
    
    def __init__(self, config: SimulationConfig):
        """Initialize binaural renderer."""
        self.config = config
        self.sample_rate = config.sample_rate
        
        # Load or generate HRTF filters
        self._init_hrtf_filters()
        
    def _init_hrtf_filters(self) -> None:
        """Initialize HRTF filter bank."""
        # Standard HRTFs for common positions
        self.hrtf_angles = np.arange(-180, 180, 15)  # 15-degree resolution
        self.hrtf_filters: dict = {}
        
        # Generate simplified HRTF filters
        for angle in self.hrtf_angles:
            self.hrtf_filters[angle] = self._generate_hrtf_filter(angle)
            
    def _generate_hrtf_filter(self, azimuth: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate HRTF filter coefficients for given azimuth.
        
        Args:
            azimuth: Azimuth angle in degrees
            
        Returns:
            Tuple of (left_filter, right_filter)
        """
        # Simplified HRTF model
        # Real implementation would use measured HRTFs
        
        # Head shadowing delay (interaural time difference)
        itd_samples = int(azimuth * 0.00001 * self.sample_rate)  # Simplified ITD
        
        # Create frequency-shaping filters
        # Low frequencies: less directional
        # High frequencies: more affected by head shadowing
        
        # Left ear
        b_low, a_low = signal.butter(4, 1500 / (self.sample_rate / 2), btype="low")
        b_high, a_high = signal.butter(4, 1500 / (self.sample_rate / 2), btype="high")
        
        # Direction-dependent gain
        az_rad = np.radians(azimuth)
        
        # Left ear gain (ipsilateral for negative azimuth, contralateral for positive)
        if azimuth <= 0:
            left_gain = 1.0
            right_gain = max(0.3, 1.0 - abs(azimuth) / 180)
        else:
            left_gain = max(0.3, 1.0 - abs(azimuth) / 180)
            right_gain = 1.0
            
        # Combine into filter
        left_filter = np.zeros(128)
        right_filter = np.zeros(128)
        
        # Add ITD delay
        if itd_samples > 0:
            right_filter[itd_samples] = 1.0 * right_gain
            left_filter[0] = 1.0 * left_gain
        else:
            left_filter[-itd_samples] = 1.0 * left_gain
            right_filter[0] = 1.0 * right_gain
            
        return left_filter, right_filter
        
    def render(
        self,
        signal: np.ndarray,
        azimuth: float,
        elevation: float = 0.0,
        distance: float = 1.0
    ) -> np.ndarray:
        """
        Render mono signal to binaural with spatial positioning.
        
        Args:
            signal: Input mono signal
            azimuth: Azimuth angle in degrees (-180 to 180)
            elevation: Elevation angle in degrees (-90 to 90)
            distance: Source distance for distance cues
            
        Returns:
            Stereo signal array of shape (2, len(signal))
        """
        # Find nearest HRTF
        nearest_angle = int(round(azimuth / 15) * 15) % 360
        if nearest_angle > 180:
            nearest_angle -= 360
            
        if nearest_angle not in self.hrtf_filters:
            nearest_angle = 0
            
        left_filter, right_filter = self.hrtf_filters[nearest_angle]
        
        # Apply HRTF filters
        left = signalconvolve(signal, left_filter, mode="full")
        right = signal.convolve(signal, right_filter, mode="full")
        
        # Apply distance attenuation
        distance_gain = 1.0 / (distance + 0.1)
        left = left[:len(signal)] * distance_gain
        right = right[:len(signal)] * distance_gain
        
        # Add distance-dependent early reflections (room simulation proxy)
        if distance < 3.0:
            reflection_delay = int(distance * 0.005 * self.sample_rate)
            reflection_gain = 0.2 / (distance + 0.5)
            
            # Simple reflection model
            left_reflected = np.zeros_like(left)
            right_reflected = np.zeros_like(right)
            
            if reflection_delay < len(left):
                left_reflected[reflection_delay:] = left[:-reflection_delay]
                right_reflected[reflection_delay:] = right[:-reflection_delay]
                
            left += left_reflected * reflection_gain
            right += right_reflected * reflection_gain
            
        return np.stack([left, right])
        
    def render_multiple_sources(
        self,
        sources: List[Tuple[np.ndarray, float, float, float]]
    ) -> np.ndarray:
        """
        Render multiple positioned sources to binaural.
        
        Args:
            sources: List of (signal, azimuth, elevation, distance) tuples
            
        Returns:
            Stereo signal
        """
        # Initialize output
        max_length = max(len(s[0]) for s in sources) if sources else 0
        output_left = np.zeros(max_length)
        output_right = np.zeros(max_length)
        
        # Render each source
        for sig, az, el, dist in sources:
            left, right = self.render(sig, az, el, dist)
            
            # Mix
            min_len = min(len(sig), max_length)
            output_left[:min_len] += left[:min_len]
            output_right[:min_len] += right[:min_len]
            
        return np.stack([output_left, output_right])
        
    def apply_head_tracking(
        self,
        binaural: np.ndarray,
        head_orientation: float,
        previous_orientation: float
    ) -> np.ndarray:
        """
        Apply head tracking rotation to binaural signal.
        
        Args:
            binaural: Input binaural signal
            head_orientation: Current head orientation in degrees
            previous_orientation: Previous head orientation in degrees
            
        Returns:
            Head-tracked binaural signal
        """
        # Compute rotation delta
        delta = head_orientation - previous_orientation
        
        if abs(delta) < 0.1:
            return binaural
            
        # Rotate HRTF angles (simplified implementation)
        # In practice, would interpolate between HRTF sets
        azimuth = binaural
        
        return azimuth  # Placeholder


class AmbisonicsEncoder:
    """
    Ambisonics encoder for multi-channel spatial audio.
    
    Supports first-order (FOA), second-order (SOA), and third-order (TOA)
    Ambisonics formats.
    
    Example:
        >>> encoder = AmbisonicsEncoder(config)
        >>> ambi = encoder.encode(mono_signal, azimuth=45, elevation=10, order=1)
    """
    
    # Ambisonics channel ordering (FuMa)
    CHANNEL_NAMES_FOA = ["W", "X", "Y", "Z"]
    CHANNEL_NAMES_SOA = ["R", "S", "T", "U", "V"]  # Second order
    CHANNEL_NAMES_TOA = ["Q", "O", "N", "M", "K", "L", "J", "I", "H"]  # Third order
    
    def __init__(self, config: SimulationConfig):
        """Initialize Ambisonics encoder."""
        self.config = config
        self.sample_rate = config.sample_rate
        
        # Spherical harmonic coefficients
        self._init_spherical_harmonics()
        
    def _init_spherical_harmonics(self) -> None:
        """Precompute spherical harmonic coefficients."""
        # We'll compute these on-the-fly for accuracy
        pass
        
    def spherical_harmonics(
        self,
        order: int,
        degree: int,
        azimuth: float,
        elevation: float
    ) -> float:
        """
        Compute real spherical harmonic.
        
        Args:
            order: Harmonic order (0, 1, 2, 3)
            degree: Degree within order (-order to +order)
            azimuth: Azimuth in radians
            elevation: Elevation in radians
            
        Returns:
            Spherical harmonic value
        """
        import scipy.special as special
        
        # Convert to colatitude
        theta = np.pi / 2 - elevation  # Colatitude
        phi = azimuth
        
        # Real spherical harmonics using scipy
        Y = special.sph_harm(abs(degree), order, phi, theta)
        
        # Convert to real form
        if degree < 0:
            return np.imag(Y) * np.sqrt(2)
        elif degree > 0:
            return np.real(Y) * np.sqrt(2)
        else:
            return np.real(Y)
            
    def encode(
        self,
        signal: np.ndarray,
        receiver: Receiver,
        order: int = 1
    ) -> np.ndarray:
        """
        Encode mono signal to Ambisonics.
        
        Args:
            signal: Input mono signal
            receiver: Receiver with position information
            order: Ambisonics order (1, 2, or 3)
            
        Returns:
            Ambisonics signal array (N_channels, len(signal))
        """
        # Get source direction from receiver perspective
        # For now, assume source is at receiver's forward direction
        azimuth = np.radians(receiver.orientation)
        elevation = np.radians(receiver.elevation)
        
        return self.encode_direction(signal, azimuth, elevation, order)
        
    def encode_direction(
        self,
        signal: np.ndarray,
        azimuth: float,
        elevation: float,
        order: int = 1
    ) -> np.ndarray:
        """
        Encode mono signal to Ambisonics for specific direction.
        
        Args:
            signal: Input mono signal
            azimuth: Azimuth in radians
            elevation: Elevation in radians
            order: Ambisonics order
            
        Returns:
            Ambisonics signal array
        """
        n_samples = len(signal)
        
        if order == 1:
            # First-order Ambisonics (W, X, Y, Z)
            channels = np.zeros((4, n_samples))
            
            # W (omnidirectional)
            channels[0] = signal * 0.707  # 1/sqrt(2) normalization
            
            # X (figure-8, front-back)
            channels[1] = signal * self.spherical_harmonics(1, 1, azimuth, elevation)
            
            # Y (figure-8, left-right)
            channels[2] = signal * self.spherical_harmonics(1, -1, azimuth, elevation)
            
            # Z (figure-8, up-down)
            channels[3] = signal * self.spherical_harmonics(1, 0, azimuth, elevation)
            
        elif order == 2:
            # Second-order Ambisonics (9 channels)
            channels = np.zeros((9, n_samples))
            
            channels[0] = signal * 0.5  # W
            
            # First-order components
            channels[1] = signal * self.spherical_harmonics(1, 1, azimuth, elevation)
            channels[2] = signal * self.spherical_harmonics(1, -1, azimuth, elevation)
            channels[3] = signal * self.spherical_harmonics(1, 0, azimuth, elevation)
            
            # Second-order components
            channels[4] = signal * self.spherical_harmonics(2, 2, azimuth, elevation)  # R
            channels[5] = signal * self.spherical_harmonics(2, 1, azimuth, elevation)  # S
            channels[6] = signal * self.spherical_harmonics(2, -1, azimuth, elevation)  # T
            channels[7] = signal * self.spherical_harmonics(2, 0, azimuth, elevation)  # U
            channels[8] = signal * self.spherical_harmonics(2, -2, azimuth, elevation)  # V
            
        elif order == 3:
            # Third-order Ambisonics (16 channels)
            channels = np.zeros((16, n_samples))
            
            channels[0] = signal * 0.3536  # W
            
            # First-order
            channels[1] = signal * self.spherical_harmonics(1, 1, azimuth, elevation)
            channels[2] = signal * self.spherical_harmonics(1, -1, azimuth, elevation)
            channels[3] = signal * self.spherical_harmonics(1, 0, azimuth, elevation)
            
            # Second-order
            channels[4] = signal * self.spherical_harmonics(2, 2, azimuth, elevation)
            channels[5] = signal * self.spherical_harmonics(2, 1, azimuth, elevation)
            channels[6] = signal * self.spherical_harmonics(2, -1, azimuth, elevation)
            channels[7] = signal * self.spherical_harmonics(2, 0, azimuth, elevation)
            channels[8] = signal * self.spherical_harmonics(2, -2, azimuth, elevation)
            
            # Third-order
            channels[9] = signal * self.spherical_harmonics(3, 3, azimuth, elevation)
            channels[10] = signal * self.spherical_harmonics(3, 2, azimuth, elevation)
            channels[11] = signal * self.spherical_harmonics(3, 1, azimuth, elevation)
            channels[12] = signal * self.spherical_harmonics(3, 0, azimuth, elevation)
            channels[13] = signal * self.spherical_harmonics(3, -1, azimuth, elevation)
            channels[14] = signal * self.spherical_harmonics(3, -2, azimuth, elevation)
            channels[15] = signal * self.spherical_harmonics(3, -3, azimuth, elevation)
            
        else:
            raise ValueError(f"Unsupported order: {order}")
            
        return channels
        
    def decode_to_binaural(
        self,
        ambi_signal: np.ndarray,
        head_orientation: float = 0.0,
        order: int = 1
    ) -> np.ndarray:
        """
        Decode Ambisonics to binaural for headphone playback.
        
        Args:
            ambi_signal: Ambisonics signal (N_channels, N_samples)
            head_orientation: Listener head orientation in radians
            order: Ambisonics order
            
        Returns:
            Stereo signal (2, N_samples)
        """
        n_samples = ambi_signal.shape[1]
        
        # Simple virtual speaker decoding
        # Define virtual speaker positions
        if order == 1:
            speakers_az = np.array([0, 90, 180, 270])  # W, X, Y, Z maps to quadrants
        elif order == 2:
            speakers_az = np.linspace(0, 360, 8, endpoint=False)
        else:
            speakers_az = np.linspace(0, 360, 12, endpoint=False)
            
        speakers_el = np.zeros_like(speakers_az)
        
        # Decode each speaker
        output_left = np.zeros(n_samples)
        output_right = np.zeros(n_samples)
        
        for i, (az, el) in enumerate(zip(speakers_az, speakers_el)):
            if i >= len(ambi_signal):
                break
                
            # Get speaker signal
            speaker_sig = ambi_signal[i]
            
            # Simple stereo pan based on azimuth
            pan = (az - head_orientation) / 180 + 0.5  # 0 to 1
            left_gain = np.sqrt(1 - pan)
            right_gain = np.sqrt(pan)
            
            output_left += speaker_sig * left_gain
            output_right += speaker_sig * right_gain
            
        return np.stack([output_left, output_right])
        
    def encode_multiple_sources(
        self,
        sources: List[Tuple[np.ndarray, float, float]]
    ) -> np.ndarray:
        """
        Encode multiple sources to Ambisonics.
        
        Args:
            sources: List of (signal, azimuth, elevation) tuples
            
        Returns:
            Ambisonics signal array
        """
        if not sources:
            return np.array([])
            
        max_len = max(len(s[0]) for s in sources)
        order = 1  # Determine order based on source count
        
        # Calculate required order
        n_sources = len(sources)
        if n_sources <= 4:
            order = 1
        elif n_sources <= 9:
            order = 2
        else:
            order = 3
            
        n_channels = (order + 1) ** 2
        output = np.zeros((n_channels, max_len))
        
        for sig, az, el in sources:
            encoded = self.encode_direction(sig, az, el, order)
            
            # Mix
            min_len = min(len(sig), max_len)
            for ch in range(len(encoded)):
                output[ch, :min_len] += encoded[ch, :min_len]
                
        return output
        
    def apply_rotation(
        self,
        ambi_signal: np.ndarray,
        yaw: float,
        pitch: float = 0.0,
        roll: float = 0.0,
        order: int = 1
    ) -> np.ndarray:
        """
        Apply rotation to Ambisonics signal.
        
        Args:
            ambi_signal: Input Ambisonics signal
            yaw: Yaw rotation in radians
            pitch: Pitch rotation in radians
            roll: Roll rotation in radians
            order: Ambisonics order
            
        Returns:
            Rotated Ambisonics signal
        """
        # Build rotation matrices for each order
        # Simplified implementation
        
        output = ambi_signal.copy()
        
        if order == 1:
            # First-order rotation (simple)
            cos_yaw = np.cos(yaw)
            sin_yaw = np.sin(yaw)
            
            # Rotate X and Y
            X = ambi_signal[1].copy()
            Y = ambi_signal[2].copy()
            
            output[1] = X * cos_yaw - Y * sin_yaw
            output[2] = X * sin_yaw + Y * cos_yaw
            
        # Higher-order rotations would use Wigner D-matrix
        
        return output


def signalconvolve(signal: np.ndarray, kernel: np.ndarray, mode: str = "full") -> np.ndarray:
    """Convolution helper with scipy backend."""
    return signal.fftconvolve(kernel, mode=mode) if hasattr(signal, 'fftconvolve') else np.convolve(signal, kernel, mode=mode)


class VBAPRenderer:
    """
    Vector Base Amplitude Panning renderer.
    
    Pans mono sources between speaker pairs for multi-channel playback.
    """
    
    def __init__(self, speaker_positions: List[Tuple[float, float]]):
        """
        Initialize VBAP renderer.
        
        Args:
            speaker_positions: List of (azimuth, elevation) tuples for each speaker
        """
        self.speaker_positions = speaker_positions
        self.n_speakers = len(speaker_positions)
        
        # Sort speakers by azimuth
        sorted_speakers = sorted(enumerate(speaker_positions), key=lambda x: x[1][0])
        self.speaker_order = [s[0] for s in sorted_speakers]
        self.speaker_positions_sorted = [s[1] for s in sorted_speakers]
        
    def render(
        self,
        signal: np.ndarray,
        azimuth: float,
        elevation: float = 0.0
    ) -> np.ndarray:
        """
        Render signal using VBAP.
        
        Args:
            signal: Input mono signal
            azimuth: Source azimuth in degrees
            elevation: Source elevation in degrees
            
        Returns:
            Multi-channel signal (n_speakers, n_samples)
        """
        n_samples = len(signal)
        output = np.zeros((self.n_speakers, n_samples))
        
        # Find nearest speaker pair
        az_rad = np.radians(azimuth)
        
        # Find enclosing speakers
        best_pair = None
        best_error = float('inf')
        
        for i in range(self.n_speakers):
            sp1_az = np.radians(self.speaker_positions_sorted[i][0])
            
            for j in range(i + 1, min(i + 2, self.n_speakers)):
                sp2_az = np.radians(self.speaker_positions_sorted[j][0])
                
                # Check if source is between these speakers
                diff = (sp2_az - sp1_az) % (2 * np.pi)
                if diff > np.pi:
                    diff = 2 * np.pi - diff
                    
                # Source direction relative to first speaker
                source_diff = (az_rad - sp1_az) % (2 * np.pi)
                if source_diff > np.pi:
                    source_diff -= 2 * np.pi
                    
                if abs(source_diff) <= diff / 2:
                    error = abs(source_diff - diff / 2)
                    if error < best_error:
                        best_error = error
                        best_pair = (i, j, source_diff / diff if diff > 0 else 0.5)
                        
        if best_pair is None:
            # Default to nearest single speaker
            nearest = min(range(self.n_speakers), 
                         key=lambda i: abs(np.radians(azimuth) - 
                                         np.radians(self.speaker_positions[i][0])))
            output[nearest] = signal
            return output
            
        i, j, t = best_pair
        
        # Compute gains using tangent formula
        g1 = 1.0 / np.tan(np.pi / 2 - (j - i) / 2 * np.pi / self.n_speakers + t * np.pi)
        g2 = 1.0 / np.tan(np.pi / 2 - (j - i) / 2 * np.pi / self.n_speakers - (1 - t) * np.pi)
        
        # Normalize
        norm = np.sqrt(g1**2 + g2**2)
        g1 /= norm
        g2 /= norm
        
        # Apply gains
        output[i] = signal * g1
        output[j] = signal * g2
        
        return output
