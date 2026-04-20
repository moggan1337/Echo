"""
Doppler Effect Simulation
Simulates frequency shifts due to relative motion between source and listener.
"""

import numpy as np
from typing import Optional, Tuple, Union, Callable
from scipy.interpolate import interp1d

from .core import SimulationConfig


class DopplerProcessor:
    """
    Doppler effect processor for moving sources and receivers.
    
    Computes time-varying delays and frequency shifts based on
    relative velocity between source and listener.
    
    Example:
        >>> processor = DopplerProcessor(config)
        >>> doppler_signal = processor.process(signal, positions, velocities)
    """
    
    def __init__(self, config: SimulationConfig):
        """Initialize Doppler processor."""
        self.config = config
        self.speed_of_sound = config.speed_of_sound
        self.sample_rate = config.sample_rate
        
    def process(
        self,
        signal: np.ndarray,
        source_position: np.ndarray,
        source_velocity: np.ndarray,
        initial_distance: Optional[float] = None
    ) -> np.ndarray:
        """
        Process signal with Doppler effect from moving source.
        
        Args:
            signal: Input audio signal
            source_position: Source position array (N, 3) over time
            source_velocity: Source velocity array (N, 3) over time
            initial_distance: Initial distance to receiver
            
        Returns:
            Doppler-shifted signal
        """
        n_samples = len(signal)
        n_steps = len(source_position)
        
        # Compute time array
        t = np.arange(n_samples) / self.sample_rate
        
        # Compute distance from source to receiver (assumed at origin)
        distance = np.linalg.norm(source_position, axis=1)
        
        # Compute propagation delay
        delay_samples = distance / self.speed_of_sound * self.sample_rate
        
        # Compute Doppler factor (ratio of received to emitted frequency)
        # v_radial is the component of velocity toward the receiver
        if len(source_velocity) > 1:
            v_radial = np.sum(source_velocity * source_position / 
                            (distance[:, np.newaxis] + 1e-10), axis=1)
        else:
            v_radial = 0.0
            
        doppler_factor = self.speed_of_sound / (self.speed_of_sound - v_radial + 1e-10)
        
        # Apply time warping to simulate Doppler
        output = np.zeros(n_samples)
        
        # For each output sample, find the corresponding input sample
        for i in range(n_samples):
            current_time = t[i]
            
            # Find delay at this time (interpolate)
            time_steps = np.arange(n_steps) / n_steps * n_samples / self.sample_rate
            delay_interp = interp1d(time_steps, delay_samples, 
                                    bounds_error=False, fill_value=delay_samples[-1])
            current_delay = delay_interp(current_time)
            
            # Find input sample
            input_time = current_time - current_delay / self.sample_rate
            input_sample = int(input_time * self.sample_rate)
            
            if 0 <= input_sample < n_samples:
                output[i] = signal[input_sample]
                
        # Apply amplitude modulation due to distance
        dist_interp = interp1d(
            np.arange(n_steps) / n_steps * n_samples / self.sample_rate,
            distance,
            bounds_error=False, fill_value=distance[-1]
        )
        
        # Apply 1/r distance attenuation
        distance_factor = np.zeros(n_samples)
        for i in range(n_samples):
            d = dist_interp(t[i])
            distance_factor[i] = 1.0 / (d + 0.1) if d > 0.1 else 1.0
            
        # Smooth the distance factor to avoid clicks
        window = np.hanning(101)
        window = window / np.sum(window)
        distance_factor = np.convolve(distance_factor, window, mode='same')
        
        output = output * distance_factor * np.max(distance)
        
        return output
        
    def process_moving_receiver(
        self,
        signal: np.ndarray,
        receiver_position: np.ndarray,
        receiver_velocity: np.ndarray,
        source_position: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Process signal with Doppler effect from moving receiver.
        
        Args:
            signal: Input audio signal
            receiver_position: Receiver position array (N, 3)
            receiver_velocity: Receiver velocity array (N, 3)
            source_position: Source position (fixed or moving)
            
        Returns:
            Doppler-shifted signal
        """
        # Moving receiver is equivalent to moving source in opposite direction
        # Just reverse the perspective
        
        n_samples = len(signal)
        n_steps = len(receiver_position)
        t = np.arange(n_samples) / self.sample_rate
        
        # Source assumed at origin unless specified
        if source_position is None:
            source_position = np.zeros((n_steps, 3))
            
        # Distance
        distance = np.linalg.norm(source_position - receiver_position, axis=1)
        
        # Radial velocity (positive = approaching)
        if len(receiver_velocity) > 1:
            direction = (source_position - receiver_position) / (distance[:, np.newaxis] + 1e-10)
            v_radial = np.sum(receiver_velocity * direction, axis=1)
        else:
            v_radial = 0.0
            
        # Doppler shift factor
        doppler_factor = (self.speed_of_sound - v_radial) / self.speed_of_sound
        
        # Time warping
        output = np.zeros(n_samples)
        
        for i in range(n_samples):
            current_time = t[i]
            
            # Current delay
            time_steps = np.arange(n_steps) / n_steps * n_samples / self.sample_rate
            delay_samples = distance / self.speed_of_sound * self.sample_rate
            
            delay_interp = interp1d(time_steps, delay_samples,
                                    bounds_error=False, fill_value=delay_samples[-1])
            current_delay = delay_interp(current_time)
            
            # Doppler shift interpolation
            doppler_interp = interp1d(time_steps, doppler_factor,
                                     bounds_error=False, fill_value=1.0)
            current_doppler = doppler_interp(current_time)
            
            # Find input sample (shifted by Doppler)
            input_time = current_time / current_doppler
            input_sample = int(input_time * self.sample_rate)
            
            if 0 <= input_sample < n_samples:
                output[i] = signal[input_sample]
                
        return output
        
    def process_both_moving(
        self,
        signal: np.ndarray,
        source_position: np.ndarray,
        source_velocity: np.ndarray,
        receiver_position: np.ndarray,
        receiver_velocity: np.ndarray
    ) -> np.ndarray:
        """
        Process signal with both source and receiver moving.
        
        Args:
            signal: Input signal
            source_position: Source positions (N, 3)
            source_velocity: Source velocities (N, 3)
            receiver_position: Receiver positions (N, 3)
            receiver_velocity: Receiver velocities (N, 3)
            
        Returns:
            Doppler-shifted signal
        """
        n_samples = len(signal)
        n_steps = len(source_position)
        t = np.arange(n_samples) / self.sample_rate
        
        # Distance between source and receiver
        distance = np.linalg.norm(source_position - receiver_position, axis=1)
        
        # Relative velocity
        relative_vel = source_velocity - receiver_velocity
        direction = (source_position - receiver_position) / (distance[:, np.newaxis] + 1e-10)
        v_radial = np.sum(relative_vel * direction, axis=1)
        
        # Combined Doppler factor
        doppler_factor = self.speed_of_sound / (self.speed_of_sound - v_radial + 1e-10)
        
        # Time warping with combined effects
        output = np.zeros(n_samples)
        
        for i in range(n_samples):
            current_time = t[i]
            
            # Interpolation helpers
            time_per_step = n_samples / self.sample_rate / n_steps
            time_steps = np.arange(n_steps) * time_per_step
            
            # Delay
            delay_samples = distance / self.speed_of_sound * self.sample_rate
            delay_interp = interp1d(time_steps, delay_samples,
                                    bounds_error=False, fill_value=delay_samples[-1])
            
            # Doppler
            doppler_interp = interp1d(time_steps, doppler_factor,
                                     bounds_error=False, fill_value=1.0)
            
            current_delay = delay_interp(current_time)
            current_doppler = doppler_interp(current_time)
            
            # Effective input time
            # This is a simplified model
            input_time = (current_time - current_delay / self.sample_rate) / current_doppler
            input_sample = int(input_time * self.sample_rate)
            
            if 0 <= input_sample < n_samples:
                output[i] = signal[input_sample]
                
        return output
        
    def apply_vibrato(
        self,
        signal: np.ndarray,
        vibrato_freq: float = 5.0,
        vibrato_depth: float = 0.001
    ) -> np.ndarray:
        """
        Apply vibrato effect (periodic pitch modulation).
        
        Args:
            signal: Input signal
            vibrato_freq: Vibrato frequency in Hz
            vibrato_depth: Pitch deviation in seconds
            
        Returns:
            Signal with vibrato
        """
        n_samples = len(signal)
        t = np.arange(n_samples) / self.sample_rate
        
        # Modulation signal
        modulation = np.sin(2 * np.pi * vibrato_freq * t)
        
        # Time-varying delay
        delay_samples = vibrato_depth * self.sample_rate * modulation
        delay_samples = delay_samples.astype(int)
        
        # Apply delay with interpolation
        output = np.zeros(n_samples)
        for i in range(n_samples):
            delay = delay_samples[i]
            if 0 <= i - delay < n_samples:
                output[i] = signal[i - delay]
            elif i - delay >= n_samples:
                output[i] = signal[-1] if n_samples > 0 else 0
                
        return output
        
    def apply_tremolo(
        self,
        signal: np.ndarray,
        tremolo_freq: float = 5.0,
        tremolo_depth: float = 0.5
    ) -> np.ndarray:
        """
        Apply tremolo effect (periodic amplitude modulation).
        
        Args:
            signal: Input signal
            tremolo_freq: Tremolo frequency in Hz
            tremolo_depth: Amplitude modulation depth (0-1)
            
        Returns:
            Signal with tremolo
        """
        t = np.arange(len(signal)) / self.sample_rate
        
        # Modulation envelope
        modulation = 1 - tremolo_depth * (1 - np.cos(2 * np.pi * tremolo_freq * t)) / 2
        
        return signal * modulation


class DopplerResampler:
    """
    High-quality Doppler effect using variable-rate resampling.
    """
    
    def __init__(self, config: SimulationConfig):
        """Initialize Doppler resampler."""
        self.config = config
        self.speed_of_sound = config.speed_of_sound
        self.sample_rate = config.sample_rate
        
    def resample_with_doppler(
        self,
        signal: np.ndarray,
        velocities: np.ndarray,
        distances: np.ndarray,
        direction: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Resample signal with Doppler shift.
        
        Args:
            signal: Input signal
            velocities: Radial velocities (positive = approaching)
            distances: Distances at each sample
            direction: Direction vectors for spatial effects
            
        Returns:
            Resampled signal
        """
        n_samples = len(signal)
        
        # Compute instantaneous frequency multiplier
        # f_received = f_source * (c / (c - v))
        freq_multiplier = self.speed_of_sound / (self.speed_of_sound - velocities + 1e-10)
        
        # Compute cumulative phase for resampling
        phase = np.cumsum(freq_multiplier)
        
        # Normalize phase
        phase = phase - phase[0]
        phase = phase / phase[-1] * (n_samples - 1)
        
        # Resample
        output = np.interp(np.arange(n_samples), phase, signal)
        
        # Apply amplitude correction
        amplitude_factor = np.sqrt(distances[0] / (distances + 1e-10))
        amplitude_factor = np.clip(amplitude_factor, 0.1, 10.0)
        
        return output * amplitude_factor
        
    def create_warped_signal(
        self,
        signal: np.ndarray,
        warp_function: Callable[[float], float]
    ) -> np.ndarray:
        """
        Create warped signal using arbitrary time warp function.
        
        Args:
            signal: Input signal
            warp_function: Function mapping output time to input time
            
        Returns:
            Warped signal
        """
        n_samples = len(signal)
        t_out = np.arange(n_samples) / self.sample_rate
        
        # Compute input times
        t_in = warp_function(t_out)
        
        # Clamp to valid range
        t_in = np.clip(t_in, 0, (n_samples - 1) / self.sample_rate)
        
        # Convert to samples
        samples_in = t_in * self.sample_rate
        
        # Interpolate
        output = np.interp(samples_in, np.arange(n_samples), signal)
        
        return output


def doppler_shift_factor(velocity: float, speed_of_sound: float = 343.0) -> float:
    """
    Compute Doppler shift factor from velocity.
    
    Args:
        velocity: Radial velocity in m/s (positive = source approaching)
        speed_of_sound: Speed of sound in m/s
        
    Returns:
        Frequency ratio (f_received / f_source)
    """
    return speed_of_sound / (speed_of_sound - velocity + 1e-10)


def frequency_from_velocity(
    source_freq: float,
    velocity: float,
    speed_of_sound: float = 343.0
) -> float:
    """
    Compute received frequency accounting for Doppler.
    
    Args:
        source_freq: Source frequency in Hz
        velocity: Radial velocity (positive = approaching)
        speed_of_sound: Speed of sound in m/s
        
    Returns:
        Received frequency in Hz
    """
    return source_freq * doppler_shift_factor(velocity, speed_of_sound)
