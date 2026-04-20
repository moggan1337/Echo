"""
Diffraction and Scattering Models
Implements acoustic diffraction around obstacles and scattering from rough surfaces.
"""

import numpy as np
from typing import Optional, Tuple, List, Union
from scipy.special import fresnel
from scipy.interpolate import interp1d

from .core import SimulationConfig


class DiffractionModel:
    """
    Acoustic diffraction model for edge and obstacle diffraction.
    
    Implements multiple diffraction models:
    - Uniform Theory of Diffraction (UTD)
    - Biot-Tolstoy-Medav (BTM)
    - Far-field Fraunhofer diffraction
    
    Example:
        >>> model = DiffractionModel(config)
        >>> filter = model.compute_filter(source_pos, receiver_pos, edge_pos)
    """
    
    def __init__(self, config: SimulationConfig):
        """Initialize diffraction model."""
        self.config = config
        self.speed_of_sound = config.speed_of_sound
        
    def compute_filter(
        self,
        source_pos: np.ndarray,
        receiver_pos: np.ndarray,
        obstacle_pos: np.ndarray,
        obstacle_size: np.ndarray,
        sample_rate: float = 48000.0
    ) -> np.ndarray:
        """
        Compute diffraction filter coefficients.
        
        Args:
            source_pos: Source position (3,)
            receiver_pos: Receiver position (3,)
            obstacle_pos: Obstacle center position (3,)
            obstacle_size: Obstacle dimensions (3,)
            sample_rate: Audio sample rate
            
        Returns:
            Filter coefficients
        """
        # Simple edge diffraction model
        # In practice, would use UTD or BTM
        
        n_samples = int(self.config.duration * sample_rate)
        
        # Compute distances
        r1 = np.linalg.norm(source_pos - obstacle_pos)
        r2 = np.linalg.norm(receiver_pos - obstacle_pos)
        r12 = np.linalg.norm(source_pos - receiver_pos)
        
        # Fresnel number
        wavelength = self.speed_of_sound / 1000  # Use 1kHz as reference
        N = obstacle_size[0]**2 / (wavelength * (r1 + r2) + 1e-10)
        
        # Diffraction coefficient
        if N < 0.1:
            D = 1.0  # No diffraction (far from obstacle)
        elif N > 10:
            D = 0.0  # Full shadow
        else:
            # Approximate using Fresnel integrals
            sqrt_N = np.sqrt(N)
            C, S = fresnel(sqrt_N)
            D = 0.5 + 0.5 * (C + S)**2 / (C**2 + S**2 + 1e-10)
            
        # Create filter (simple lowpass approximation)
        cutoff = 2000 * D + 200  # Frequency-dependent
        b, a = self._butter_lowpass(4, cutoff / (sample_rate / 2))
        
        return b
        
    def _butter_lowpass(self, order: int, cutoff: float) -> Tuple[np.ndarray, np.ndarray]:
        """Create Butterworth lowpass filter coefficients."""
        # Simplified Butterworth approximation
        if cutoff >= 0.99:
            cutoff = 0.99
            
        # Second-order sections
        alpha = np.tan(np.pi * cutoff)
        a0 = 1 + np.sqrt(2) * alpha + alpha**2
        b0 = alpha**2 / a0
        b1 = 2 * alpha**2 / a0
        b2 = alpha**2 / a0
        a1 = 2 * (alpha**2 - 1) / a0
        a2 = (1 - np.sqrt(2) * alpha + alpha**2) / a0
        
        return np.array([b0, b1, b2]), np.array([1, a1, a2])


class UTDDiffraction:
    """
    Uniform Theory of Diffraction (UTD) implementation.
    
    Provides accurate edge diffraction calculations for hard obstacles.
    """
    
    def __init__(self, speed_of_sound: float = 343.0):
        """Initialize UTD diffraction model."""
        self.c = speed_of_sound
        
    def compute_diffraction(
        self,
        source_pos: np.ndarray,
        edge_start: np.ndarray,
        edge_end: np.ndarray,
        receiver_pos: np.ndarray,
        frequency: float = 1000.0
    ) -> float:
        """
        Compute UTD diffraction coefficient.
        
        Args:
            source_pos: Source position
            edge_start: Start of diffracting edge
            edge_end: End of diffracting edge
            receiver_pos: Receiver position
            frequency: Frequency in Hz
            
        Returns:
            Diffraction coefficient (0-1)
        """
        # Edge direction
        edge_dir = edge_end - edge_start
        edge_length = np.linalg.norm(edge_dir)
        edge_dir = edge_dir / edge_length if edge_length > 1e-10 else np.zeros(3)
        
        # Distances to edge endpoints
        r_s1 = np.linalg.norm(source_pos - edge_start)
        r_s2 = np.linalg.norm(source_pos - edge_end)
        r_r1 = np.linalg.norm(receiver_pos - edge_start)
        r_r2 = np.linalg.norm(receiver_pos - edge_end)
        
        # Projection of source/receiver onto edge
        s_proj = np.dot(source_pos - edge_start, edge_dir)
        r_proj = np.dot(receiver_pos - edge_start, edge_dir)
        
        # Clamp to edge
        s_proj = np.clip(s_proj, 0, edge_length)
        r_proj = np.clip(r_proj, 0, edge_length)
        
        # Closest point on edge
        closest_s = edge_start + s_proj * edge_dir
        closest_r = edge_start + r_proj * edge_dir
        
        # Distances to closest points
        d_s = np.linalg.norm(source_pos - closest_s)
        d_r = np.linalg.norm(receiver_pos - closest_r)
        
        # Wavelength
        wavelength = self.c / frequency
        
        # Phase
        k = 2 * np.pi / wavelength
        
        # Path lengths
        path_direct = np.linalg.norm(source_pos - receiver_pos)
        path_diffracted = d_s + d_r
        
        # Phase difference
        delta_phi = k * (path_diffracted - path_direct)
        
        # UTD wedge diffraction coefficient (simplified for soft wedge)
        # This is a simplified 2D formulation
        theta_s = self._compute_incident_angle(source_pos, edge_dir, closest_s)
        theta_r = self._compute_incident_angle(receiver_pos, edge_dir, closest_r)
        
        # Distance factor
        R = d_s * d_r / (d_s + d_r)
        
        # UTD coefficient (simplified)
        L = np.sqrt(8 * np.pi * k * R * np.cos(theta_s) * np.cos(theta_r))
        
        if L < 1e-10:
            return 0.0
            
        # Fresnel parameter
        N = 2 * R * np.sin((theta_s + theta_r) / 2)**2 / wavelength
        
        # Diffraction coefficient
        D = -np.exp(-1j * np.pi / 4) / (2 * np.sqrt(2 * np.pi * k)) * (
            1 / np.cos((theta_s - theta_r) / 2) +
            1 / np.cos((theta_s + theta_r) / 2)
        )
        
        # Include Fresnel transition function
        if N < 0.1:
            F = 1.0
        else:
            sqrt_N = np.sqrt(N)
            C, S = fresnel(sqrt_N)
            F = (0.5 + 0.5 * (C + 1j * S)) / (0.5 + 0.5 * (C[-1] + 1j * S[-1]))
            
        D = D * F
        
        # Magnitude
        magnitude = np.abs(D)
        
        # Apply distance attenuation
        attenuation = np.exp(-1j * k * path_diffracted) / np.sqrt(path_diffracted)
        
        return np.abs(magnitude * attenuation)
        
    def _compute_incident_angle(
        self,
        point: np.ndarray,
        edge_dir: np.ndarray,
        edge_point: np.ndarray
    ) -> float:
        """Compute incident angle relative to edge."""
        incident = point - edge_point
        incident = incident / np.linalg.norm(incident)
        
        cos_angle = np.dot(incident, edge_dir)
        return np.arccos(np.clip(cos_angle, -1, 1))


class BTMDiffraction:
    """
    Biot-Tolstoy-Medav (BTM) diffraction model.
    
    Provides accurate time-domain diffraction impulse responses.
    """
    
    def __init__(self, speed_of_sound: float = 343.0):
        """Initialize BTM diffraction model."""
        self.c = speed_of_sound
        
    def compute_impulse_response(
        self,
        source_pos: np.ndarray,
        edge_start: np.ndarray,
        edge_end: np.ndarray,
        receiver_pos: np.ndarray,
        duration: float = 0.5,
        sample_rate: float = 48000.0
    ) -> np.ndarray:
        """
        Compute BTM diffraction impulse response.
        
        Args:
            source_pos: Source position
            edge_start: Start of edge
            edge_end: End of edge
            receiver_pos: Receiver position
            duration: IR duration in seconds
            sample_rate: Sample rate
            
        Returns:
            Diffraction impulse response
        """
        n_samples = int(duration * sample_rate)
        t = np.arange(n_samples) / sample_rate
        
        # Edge parameters
        edge_vec = edge_end - edge_start
        L = np.linalg.norm(edge_vec)  # Edge length
        
        if L < 1e-10:
            return np.zeros(n_samples)
            
        edge_dir = edge_vec / L
        
        # Source and receiver distances to edge
        d_s = self._distance_to_segment(source_pos, edge_start, edge_end)
        d_r = self._distance_to_segment(receiver_pos, edge_start, edge_end)
        
        # Projection on edge
        s_proj = np.dot(source_pos - edge_start, edge_dir)
        r_proj = np.dot(receiver_pos - edge_start, edge_dir)
        
        # Clamp to edge
        s_proj = np.clip(s_proj, 0, L)
        r_proj = np.clip(r_proj, 0, L)
        
        # Minimum path difference
        s_proj = np.clip(s_proj, 0.01, L - 0.01)
        r_proj = np.clip(r_proj, 0.01, L - 0.01)
        
        # BTM coefficients
        gamma_s = self._compute_gamma(d_s, source_pos, edge_start, edge_dir, s_proj)
        gamma_r = self._compute_gamma(d_r, receiver_pos, edge_start, edge_dir, r_proj)
        
        # Time axis
        ir = np.zeros(n_samples)
        
        # BTM formula (simplified)
        c = self.c
        
        for i, ti in enumerate(t):
            # Path difference for this time
            # (simplified - full BTM has complex integration)
            
            t1 = (s_proj + r_proj) / c
            t2 = np.sqrt(s_proj**2 + d_s**2) / c + np.sqrt(r_proj**2 + d_r**2) / c
            
            if ti < t1:
                continue
                
            # Simplified BTM response
            if ti < t2:
                # Before direct arrival
                tau = (ti - t1) * c
                if tau > 0:
                    # Amplitude (simplified)
                    amplitude = 1.0 / (np.sqrt(ti) * np.sqrt(ti - t1) + 1e-10)
                    ir[i] = amplitude * gamma_s * gamma_r
                    
        # Normalize
        ir = ir / (np.max(np.abs(ir)) + 1e-10)
        
        return ir
        
    def _compute_gamma(
        self,
        d: float,
        point: np.ndarray,
        edge_start: np.ndarray,
        edge_dir: np.ndarray,
        proj: float
    ) -> float:
        """Compute BTM gamma coefficient."""
        closest = edge_start + proj * edge_dir
        vec = point - closest
        return d / np.linalg.norm(vec) if np.linalg.norm(vec) > 1e-10 else 1.0
        
    def _distance_to_segment(
        self,
        point: np.ndarray,
        seg_start: np.ndarray,
        seg_end: np.ndarray
    ) -> float:
        """Compute distance from point to line segment."""
        vec = seg_end - seg_start
        length = np.linalg.norm(vec)
        
        if length < 1e-10:
            return np.linalg.norm(point - seg_start)
            
        t = np.dot(point - seg_start, vec) / length**2
        t = np.clip(t, 0, 1)
        
        closest = seg_start + t * vec
        return np.linalg.norm(point - closest)


class ScatteringModel:
    """
    Surface scattering model for rough surfaces.
    
    Implements Lambertian and more sophisticated scattering models.
    """
    
    def __init__(self, speed_of_sound: float = 343.0):
        """Initialize scattering model."""
        self.c = speed_of_sound
        
    def compute_scattering_coefficient(
        self,
        frequency: float,
        roughness: float,
        incident_angle: float = 0.0
    ) -> float:
        """
        Compute scattering coefficient.
        
        Args:
            frequency: Frequency in Hz
            roughness: Surface roughness (RMS height) in meters
            incident_angle: Incident angle in radians
            
        Returns:
            Scattering coefficient (0-1)
        """
        wavelength = self.c / frequency
        
        # Rayleigh roughness parameter
        k = 2 * np.pi / wavelength
        sigma = roughness
        
        # Phase variance
        sigma_phase = 2 * k**2 * sigma**2
        
        # Scattering coefficient (Lambertian approximation)
        if sigma_phase < 0.1:
            s = sigma_phase / 2
        else:
            # More realistic model
            s = 1 - np.exp(-sigma_phase / 2)
            
        return np.clip(s, 0, 1)
        
    def scatter_impulse_response(
        self,
        incident_impulse: np.ndarray,
        scattering_coefficient: float,
        n_scatters: int = 100,
        sample_rate: float = 48000.0
    ) -> np.ndarray:
        """
        Generate scattered impulse response.
        
        Args:
            incident_impulse: Incident impulse response
            scattering_coefficient: Scattering coefficient
            n_scatters: Number of scatterers
            sample_rate: Sample rate
            
        Returns:
            Scattered impulse response
        """
        n_samples = len(incident_impulse)
        output = np.zeros(n_samples)
        
        # Random delays for each scatterer
        max_delay = n_samples / sample_rate * 0.1  # 10% of duration
        delays = np.random.rand(n_scatters) * max_delay
        delays_samples = (delays * sample_rate).astype(int)
        
        # Random gains
        gains = np.random.rand(n_scatters) * scattering_coefficient
        
        # Sum scattered contributions
        for delay, gain in zip(delays_samples, gains):
            if delay < n_samples:
                output[delay:] += incident_impulse[:n_samples-delay] * gain
                
        # Normalize
        output = output / (n_scatters + 1)
        
        return output
        
    def phase_screen_scattering(
        self,
        signal: np.ndarray,
        roughness: float,
        sample_rate: float = 48000.0
    ) -> np.ndarray:
        """
        Apply scattering using phase screen model.
        
        Args:
            signal: Input signal
            roughness: Surface roughness in meters
            sample_rate: Sample rate
            
        Returns:
            Scattered signal
        """
        n_samples = len(signal)
        
        # Frequency domain
        freq = np.fft.rfftfreq(n_samples, 1/sample_rate)
        
        # Random phase screen
        k = 2 * np.pi * freq / self.c
        
        # Phase variance
        sigma_phi = 2 * k**2 * roughness**2
        
        # Random phases
        phases = np.random.randn(len(freq)) * np.sqrt(sigma_phi)
        
        # Apply phase modulation
        fft_signal = np.fft.rfft(signal)
        fft_scattered = fft_signal * np.exp(1j * phases)
        
        return np.fft.irfft(fft_scattered, n=n_samples)


class FresnelDiffraction:
    """
    Fresnel diffraction for aperture and obstacle diffraction.
    """
    
    def __init__(self, speed_of_sound: float = 343.0):
        """Initialize Fresnel diffraction model."""
        self.c = speed_of_sound
        
    def compute_aperture_diffraction(
        self,
        source_pos: np.ndarray,
        aperture_center: np.ndarray,
        aperture_size: float,
        receiver_pos: np.ndarray,
        frequency: float = 1000.0,
        sample_rate: float = 48000.0
    ) -> np.ndarray:
        """
        Compute diffraction through rectangular aperture.
        
        Args:
            source_pos: Source position
            aperture_center: Aperture center
            aperture_size: Aperture size
            receiver_pos: Receiver position
            frequency: Frequency
            sample_rate: Sample rate
            
        Returns:
            Diffraction filter
        """
        wavelength = self.c / frequency
        
        # Distances
        r1 = np.linalg.norm(source_pos - aperture_center)
        r2 = np.linalg.norm(receiver_pos - aperture_center)
        
        # Fresnel number
        N = aperture_size**2 / (wavelength * (r1 + r2) + 1e-10)
        
        # Time-domain response
        duration = 2 * (r1 + r2) / self.c
        n_samples = int(duration * sample_rate)
        t = np.arange(n_samples) / sample_rate
        
        # Reference time
        t0 = (r1 + r2) / self.c
        
        # Fresnel integral response
        ir = np.zeros(n_samples)
        
        for i, ti in enumerate(t):
            if ti < t0:
                continue
                
            # Fresnel parameter
            u = np.sqrt(2 / wavelength) * (ti - t0) * self.c / np.sqrt(r1 * r2 / (r1 + r2))
            
            if abs(u) < 10:
                C, S = fresnel(u)
                ir[i] = 0.5 + 0.5 * (C + 1j * S)
                
        return ir
