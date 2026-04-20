"""
Room Impulse Response (RIR) Generation
Computes realistic RIRs using various algorithms.
"""

import numpy as np
from typing import Optional, Tuple, Union
import scipy.signal as signal
from scipy.interpolate import interp1d

from .core import SimulationConfig
from .room import Room
from .source import SoundSource
from .receiver import Receiver
from .materials import MaterialDatabase


class RIRGenerator:
    """
    Generates Room Impulse Responses (RIRs) using multiple methods.
    
    RIRs capture the acoustic characteristics of a room and are essential
    for acoustic simulation and auralization.
    
    Example:
        >>> generator = RIRGenerator(config)
        >>> rir = generator.generate(
        ...     room=room,
        ...     source=source,
        ...     receiver=receiver
        ... )
    """
    
    def __init__(self, config: SimulationConfig):
        """Initialize RIR generator."""
        self.config = config
        self.material_db = MaterialDatabase()
        
    def generate(
        self,
        room: Room,
        source: SoundSource,
        receiver: Receiver,
        method: str = "hybrid"
    ) -> np.ndarray:
        """
        Generate a room impulse response.
        
        Args:
            room: Room geometry and materials
            source: Sound source
            receiver: Sound receiver
            method: Generation method ("ray_tracing", "wave_based", "hybrid")
            
        Returns:
            Room impulse response
        """
        if method == "ray_tracing":
            return self._generate_ray_tracing(room, source, receiver)
        elif method == "wave_based":
            return self._generate_wave_based(room, source, receiver)
        else:  # hybrid
            return self._generate_hybrid(room, source, receiver)
            
    def _generate_ray_tracing(
        self,
        room: Room,
        source: SoundSource,
        receiver: Receiver
    ) -> np.ndarray:
        """Generate RIR using image source method."""
        n_samples = int(self.config.duration * self.config.sample_rate)
        rir = np.zeros(n_samples)
        
        # Direct sound
        direct_dist = np.linalg.norm(receiver.position - source.position)
        direct_delay = int(direct_dist / self.config.speed_of_sound * self.config.sample_rate)
        
        if direct_delay < n_samples:
            rir[direct_delay] = 1.0 / (4 * np.pi * direct_dist**2 + 1e-10)
            
        # Early reflections using image source method
        reflections = self._compute_image_sources(room, source, receiver, max_order=5)
        
        for refl in reflections:
            order, image_pos, surface = refl
            
            # Distance from image source to receiver
            dist = np.linalg.norm(receiver.position - image_pos)
            delay = int(dist / self.config.speed_of_sound * self.config.sample_rate)
            
            if delay < n_samples:
                # Reflection coefficient based on surface material
                alpha = surface.material.absorption(1000) if surface else 0.1
                amplitude = (1 - alpha) / (4 * np.pi * dist**2 + 1e-10)
                
                # Apply temporal decay
                decay = np.exp(-3.0 * order / (self.config.speed_of_sound * 0.5))
                amplitude *= decay
                
                rir[delay] += amplitude
                
        # Apply smoothing to simulate band-limited response
        rir = self._apply_frequency_smoothing(rir)
        
        # Normalize
        rir = rir / (np.max(np.abs(rir)) + 1e-10)
        
        return rir
        
    def _generate_wave_based(
        self,
        room: Room,
        source: SoundSource,
        receiver: Receiver
    ) -> np.ndarray:
        """Generate RIR using statistical model (Sabine/Eyring)."""
        n_samples = int(self.config.duration * self.config.sample_rate)
        
        # Compute reverberation time using Sabine's formula
        V = room.volume
        S = room.surface_area
        
        # Average absorption
        avg_alpha = 0.0
        for surface in room.surfaces:
            avg_alpha += surface.material.absorption(500) * surface.area
        avg_alpha /= S
        
        # Sabine's formula: T60 = 0.161 * V / (S * alpha_bar)
        if avg_alpha > 0.001:
            T60 = 0.161 * V / (S * avg_alpha + 1e-10)
        else:
            T60 = 10.0  # Large value for anechoic
            
        T60 = min(T60, self.config.duration)
        
        # Generate reverberant tail using filtered noise
        t = np.arange(n_samples) / self.config.sample_rate
        
        # Energy decay curve
        energy = np.exp(-6.91 * t / T60)
        
        # Add frequency-dependent characteristics
        noise = np.random.randn(n_samples)
        
        # Apply lowpass filter for more realistic decay
        b, a = signal.butter(4, 4000 / (self.config.sample_rate / 2), btype="low")
        noise_filtered = signal.filtfilt(b, a, noise)
        
        # Create RIR with energy decay
        rir = noise_filtered * np.sqrt(energy)
        
        # Add direct sound
        direct_dist = np.linalg.norm(receiver.position - source.position)
        direct_delay = int(direct_dist / self.config.speed_of_sound * self.config.sample_rate)
        
        if direct_delay < n_samples:
            rir[direct_delay] += 1.0 / (4 * np.pi * direct_dist**2 + 1e-10)
            
        # Normalize
        rir = rir / (np.max(np.abs(rir)) + 1e-10)
        
        return rir
        
    def _generate_hybrid(
        self,
        room: Room,
        source: SoundSource,
        receiver: Receiver
    ) -> np.ndarray:
        """
        Generate RIR using hybrid approach.
        
        Combines deterministic early reflections with statistical late reverb.
        """
        n_samples = int(self.config.duration * self.config.sample_rate)
        
        # Early reflections (deterministic)
        early_rir = self._generate_ray_tracing(room, source, receiver)
        
        # Late reverberation (statistical)
        late_rir = self._generate_wave_based(room, source, receiver)
        
        # Crossfade between early and late
        crossfade_time = 0.05  # 50ms crossfade
        crossfade_samples = int(crossfade_time * self.config.sample_rate)
        
        # Find transition point in early RIR (where energy starts decaying)
        early_energy = np.cumsum(early_rir[::-1]**2)[::-1]
        transition_idx = np.argmax(early_energy < np.max(early_energy) * 0.1)
        
        if transition_idx == 0:
            transition_idx = len(early_rir) // 10
            
        # Apply crossfade
        fade_out = np.ones(transition_idx)
        fade_in = np.zeros_like(fade_out)
        
        if crossfade_samples < transition_idx:
            fade_out[-crossfade_samples:] = np.linspace(1, 0, crossfade_samples)
            fade_in[:crossfade_samples] = np.linspace(0, 1, crossfade_samples)
            
        early_rir[:transition_idx] *= fade_out
        late_rir[:transition_idx] *= fade_in
        
        # Combine
        rir = early_rir + late_rir
        
        # Final normalization
        rir = rir / (np.max(np.abs(rir)) + 1e-10)
        
        return rir
        
    def _compute_image_sources(
        self,
        room: Room,
        source: SoundSource,
        receiver: Receiver,
        max_order: int = 5
    ) -> list:
        """Compute image sources up to specified order."""
        image_sources = []
        
        # For a rectangular room, compute image sources
        dims = room.dimensions
        
        for order in range(1, max_order + 1):
            # Generate all combinations of signs for this order
            signs = self._generate_sign_combinations(3, order)
            
            for sign in signs:
                # Image source position
                image_pos = source.position.copy()
                
                for i, s in enumerate(sign):
                    if s == 1:
                        image_pos[i] = 2 * dims[i] - source.position[i]
                    elif s == -1:
                        image_pos[i] = -source.position[i]
                        
                # Find which surface this image is associated with
                surface = None
                
                # Add to list
                image_sources.append((order, image_pos, surface))
                
        return image_sources
        
    def _generate_sign_combinations(self, n_dims: int, order: int) -> list:
        """Generate sign combinations for image sources."""
        if order == 0:
            return [(0, 0, 0)]
            
        combinations = []
        
        def generate(pos, remaining):
            if remaining == 0:
                combinations.append(tuple(pos))
                return
            for i in range(n_dims):
                for sign in [1, -1]:
                    new_pos = list(pos)
                    new_pos[i] = sign
                    generate(new_pos, remaining - 1)
                    
        generate([0, 0, 0], order)
        
        # Remove duplicates and order-0
        seen = set()
        unique = []
        for combo in combinations:
            if combo not in seen and sum(1 for x in combo if x != 0) == order:
                seen.add(combo)
                unique.append(combo)
                
        return unique
        
    def _apply_frequency_smoothing(self, rir: np.ndarray) -> np.ndarray:
        """Apply frequency-dependent smoothing to RIR."""
        # Simple lowpass to simulate band-limited nature
        cutoff = min(8000, self.config.sample_rate / 2 - 1)
        b, a = signal.butter(2, cutoff / (self.config.sample_rate / 2), btype="low")
        
        smoothed = signal.filtfilt(b, a, rir)
        
        return smoothed
        
    def generate_from_rir_dataset(
        self,
        dataset_path: str,
        source_pos: np.ndarray,
        receiver_pos: np.ndarray
    ) -> np.ndarray:
        """
        Load RIR from dataset and interpolate.
        
        Args:
            dataset_path: Path to RIR dataset
            source_pos: Source position
            receiver_pos: Receiver position
            
        Returns:
            Interpolated RIR
        """
        # Placeholder for dataset loading
        # Would load pre-recorded RIRs and interpolate spatially
        return np.zeros(int(self.config.duration * self.config.sample_rate))
        
    def add_air_absorption(self, rir: np.ndarray, distance: float) -> np.ndarray:
        """Add air absorption to RIR."""
        # Frequency-dependent air absorption
        frequencies = np.fft.rfftfreq(len(rir), 1/self.config.sample_rate)
        
        # Absorption coefficient (dB per meter) - varies with frequency
        # Higher frequencies absorb more
        alpha = 0.1 * (frequencies / 1000)**0.7  # Approximate model
        
        # Total absorption
        total_attenuation = np.exp(-alpha * distance / 8.686)
        
        # Apply in frequency domain
        rir_fft = np.fft.rfft(rir)
        rir_fft *= total_attenuation
        rir = np.fft.irfft(rir_fft, n=len(rir))
        
        return rir
        
    def compute_acoustic_parameters(self, rir: np.ndarray) -> dict:
        """
        Compute standard acoustic parameters from RIR.
        
        Args:
            rir: Room impulse response
            
        Returns:
            Dictionary of acoustic parameters
        """
        params = {}
        
        # Direct sound parameters
        direct_idx = np.argmax(rir)
        direct_level = rir[direct_idx]
        params["direct_level"] = direct_level
        params["direct_delay_ms"] = direct_idx / self.config.sample_rate * 1000
        
        # Energy
        energy = rir**2
        total_energy = np.sum(energy)
        
        # Early energy (0-50ms)
        early_samples = int(0.050 * self.config.sample_rate)
        early_energy = np.sum(energy[:early_samples])
        
        # Late energy (50ms onwards)
        late_energy = np.sum(energy[early_samples:])
        
        # Clarity index (C50, C80)
        params["C50"] = 10 * np.log10(early_energy / (late_energy + 1e-10))
        params["C80"] = 10 * np.log10(
            np.sum(energy[:int(0.080 * self.config.sample_rate)]) /
            (np.sum(energy[int(0.080 * self.config.sample_rate):]) + 1e-10)
        )
        
        # Definition (D50)
        params["D50"] = early_energy / (total_energy + 1e-10)
        
        # Center time
        t = np.arange(len(rir)) / self.config.sample_rate
        params["center_time_ms"] = np.sum(t * energy) / (total_energy + 1e-10) * 1000
        
        # T20, T30, T60 using Schroeder integration
        schroeder = np.cumsum(rir[::-1]**2)[::-1]
        schroeder_db = 10 * np.log10(schroeder / (np.max(schroeder) + 1e-10))
        
        # Find T20 (decay from -5dB to -25dB)
        params["T20"] = self._find_decay_time(schroeder_db, -5, -25)
        
        # Find T30 (decay from -5dB to -35dB)
        params["T30"] = self._find_decay_time(schroeder_db, -5, -35)
        
        # Find T60 (decay from -5dB to -65dB)
        params["T60"] = self._find_decay_time(schroeder_db, -5, -65)
        
        return params
        
    def _find_decay_time(self, schroeder_db: np.ndarray, start_db: float, end_db: float) -> float:
        """Find time for energy to decay from start_db to end_db."""
        max_idx = np.argmax(schroeder_db > start_db)
        
        if max_idx == 0 and schroeder_db[0] <= start_db:
            return 0.0
            
        start_idx = max_idx if schroeder_db[max_idx] > start_db else max_idx + 1
        
        end_mask = schroeder_db[start_idx:] <= end_db
        if np.any(end_mask):
            end_idx = start_idx + np.argmax(end_mask)
        else:
            end_idx = len(schroeder_db) - 1
            
        time_start = start_idx / self.config.sample_rate
        time_end = end_idx / self.config.sample_rate
        
        return (time_end - time_start) * 60 / abs(end_db - start_db)


class RIRProcessor:
    """
    Processes and enhances room impulse responses.
    """
    
    def __init__(self, sample_rate: float = 48000.0):
        """Initialize RIR processor."""
        self.sample_rate = sample_rate
        
    def normalize(
        self,
        rir: np.ndarray,
        target_level: float = -20.0
    ) -> np.ndarray:
        """
        Normalize RIR to target level.
        
        Args:
            rir: Input RIR
            target_level: Target peak level in dB
            
        Returns:
            Normalized RIR
        """
        peak = np.max(np.abs(rir))
        if peak < 1e-10:
            return rir
            
        target_linear = 10**(target_level / 20)
        return rir * (target_linear / peak)
        
    def trim(
        self,
        rir: np.ndarray,
        pre_delay: float = 0.0,
        post_delay: float = 0.1,
        tail_threshold: float = -60.0
    ) -> np.ndarray:
        """
        Trim RIR to reasonable length.
        
        Args:
            rir: Input RIR
            pre_delay: Pre-direct-sound padding in seconds
            post_delay: Minimum tail length in seconds
            tail_threshold: Trim threshold in dB
            
        Returns:
            Trimmed RIR
        """
        # Find direct sound
        direct_idx = np.argmax(np.abs(rir))
        
        # Pre-delay
        pre_samples = int(pre_delay * self.sample_rate)
        
        # Post delay
        post_samples = max(
            int(post_delay * self.sample_rate),
            int((len(rir) - direct_idx) * 0.5)
        )
        
        # Find tail end
        tail_start = direct_idx + post_samples
        if tail_start < len(rir):
            tail_rir = rir[tail_start:]
            tail_energy = np.cumsum(tail_rir[::-1]**2)[::-1]
            tail_db = 10 * np.log10(tail_energy / (np.max(tail_energy) + 1e-10))
            
            tail_end_idx = np.argmax(tail_db < tail_threshold)
            if tail_end_idx > 0:
                post_samples = max(post_samples, tail_end_idx)
                
        # Extract region
        start = max(0, direct_idx - pre_samples)
        end = min(len(rir), direct_idx + post_samples)
        
        return rir[start:end]
        
    def apply_window(
        self,
        rir: np.ndarray,
        window_type: str = "hann"
    ) -> np.ndarray:
        """
        Apply temporal window to RIR.
        
        Args:
            rir: Input RIR
            window_type: Window type ("hann", "hamming", "tukey")
            
        Returns:
            Windowed RIR
        """
        if window_type == "hann":
            window = np.hanning(len(rir))
        elif window_type == "hamming":
            window = np.hamming(len(rir))
        elif window_type == "tukey":
            from scipy.signal import tukey
            window = tukey(len(rir), alpha=0.1)
        else:
            return rir
            
        return rir * window
        
    def deconvolve(
        self,
        signal: np.ndarray,
        rir: np.ndarray,
        method: str = "frequency_domain"
    ) -> np.ndarray:
        """
        Deconvolve RIR from signal (inverse filtering).
        
        Args:
            signal: Input signal
            rir: Room impulse response
            method: Deconvolution method
            
        Returns:
            Deconvolved signal
        """
        if method == "frequency_domain":
            # Frequency domain deconvolution with regularization
            signal_fft = np.fft.rfft(signal)
            rir_fft = np.fft.rfft(rir)
            
            # Regularization to avoid division by zero
            epsilon = np.max(np.abs(rir_fft)) * 0.001
            inverse_fft = np.conj(rir_fft) / (np.abs(rir_fft)**2 + epsilon**2)
            
            result_fft = signal_fft * inverse_fft
            result = np.fft.irfft(result_fft, n=len(signal))
            
            return result
        else:
            raise ValueError(f"Unknown deconvolution method: {method}")
