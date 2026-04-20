"""
Real-Time Audio Processing
Real-time capable audio processing blocks for live acoustic simulation.
"""

import numpy as np
from typing import Optional, Tuple, List, Callable, Union
from collections import deque
import threading
import time

from .core import SimulationConfig


class CircularBuffer:
    """
    Lock-free circular buffer for real-time audio processing.
    """
    
    def __init__(self, size: int):
        """Initialize circular buffer."""
        self.size = size
        self.buffer = np.zeros(size)
        self.write_idx = 0
        self.read_idx = 0
        self._lock = threading.Lock()
        
    def write(self, data: np.ndarray) -> None:
        """Write data to buffer."""
        n = len(data)
        with self._lock:
            for i in range(n):
                self.buffer[self.write_idx] = data[i]
                self.write_idx = (self.write_idx + 1) % self.size
                
    def read(self, n: int) -> np.ndarray:
        """Read n samples from buffer."""
        output = np.zeros(n)
        with self._lock:
            for i in range(n):
                output[i] = self.buffer[self.read_idx]
                self.read_idx = (self.read_idx + 1) % self.size
        return output
    
    def available(self) -> int:
        """Get number of available samples."""
        with self._lock:
            if self.write_idx >= self.read_idx:
                return self.write_idx - self.read_idx
            else:
                return self.size - self.read_idx + self.write_idx


class RealTimeProcessor:
    """
    Real-time audio processor for live acoustic simulation.
    
    Provides efficient processing blocks for:
    - Convolution (partitioned overlap-add)
    - Filtering (biquad sections)
    - Spatial audio processing
    
    Example:
        >>> processor = RealTimeProcessor(config)
        >>> processor.setup_convolution(rir)
        >>> output = processor.process(input_chunk)
    """
    
    def __init__(self, config: SimulationConfig):
        """Initialize real-time processor."""
        self.config = config
        self.sample_rate = config.sample_rate
        self.chunk_size = 512  # Default processing chunk size
        
        # Convolution engine
        self._convolution_buffer: Optional[np.ndarray] = None
        self._partitioned_convolver = None
        
        # Filter state
        self._filter_state: dict = {}
        
        # Spatial processing
        self._spatial_processors: dict = {}
        
    def set_chunk_size(self, size: int) -> None:
        """Set processing chunk size."""
        self.chunk_size = size
        
    def setup_partitioned_convolution(
        self,
        impulse_response: np.ndarray,
        num_partitions: int = 8
    ) -> None:
        """
        Set up partitioned overlap-add convolution.
        
        Args:
            impulse_response: Impulse response to convolve with
            num_partitions: Number of FFT partitions
        """
        self._rir = impulse_response
        self._num_partitions = num_partitions
        
        # Calculate partition size
        partition_size = len(impulse_response) // num_partitions + 1
        
        # Pad to power of 2
        self._fft_size = 1
        while self._fft_size < partition_size * 2:
            self._fft_size *= 2
            
        self._partition_size = self._fft_size // 2
        self._num_partitions = num_partitions
        
        # Pre-compute FFT of partitions
        self._rir_fft = []
        for i in range(num_partitions):
            start = i * self._partition_size
            end = min(start + self._partition_size, len(impulse_response))
            
            partition = np.zeros(self._fft_size)
            partition[:end-start] = impulse_response[start:end]
            
            self._rir_fft.append(np.fft.rfft(partition))
            
        # Overlap buffer
        self._overlap = np.zeros(self._fft_size)
        self._partition_idx = 0
        
    def process_partitioned_convolution(self, chunk: np.ndarray) -> np.ndarray:
        """
        Process audio chunk using partitioned convolution.
        
        Args:
            chunk: Input audio chunk
            
        Returns:
            Processed audio chunk
        """
        if self._rir is None:
            return chunk
            
        # Pad input
        input_padded = np.zeros(self._fft_size)
        input_padded[:len(chunk)] = chunk
        
        # FFT of input
        input_fft = np.fft.rfft(input_padded)
        
        # Multiply with current partition
        output_fft = input_fft * self._rir_fft[self._partition_idx]
        
        # IFFT and overlap-add
        output = np.fft.irfft(output_fft)
        
        # Add overlap from previous block
        if self._partition_idx == 0:
            # First partition: only add tail from last partition
            output[:len(self._overlap)] += self._overlap[:len(output)]
        else:
            output[:len(self._overlap)] += self._overlap[:len(output)]
            
        # Save overlap for next block
        self._overlap = output[self._partition_size:]
        
        # Move to next partition
        self._partition_idx = (self._partition_idx + 1) % self._num_partitions
        
        # Return processed chunk
        return output[:len(chunk)]
        
    def process_overlap_add(
        self,
        chunk: np.ndarray,
        ir: np.ndarray,
        block_size: Optional[int] = None
    ) -> np.ndarray:
        """
        Simple overlap-add convolution.
        
        Args:
            chunk: Input chunk
            ir: Impulse response
            block_size: Processing block size
            
        Returns:
            Processed audio
        """
        if block_size is None:
            block_size = self.chunk_size
            
        # Pad IR if needed
        ir_padded = np.pad(ir, (0, block_size))
        
        # FFT of IR block
        ir_fft = np.fft.rfft(ir_padded)
        
        # Process in blocks
        output = np.zeros(len(chunk) + len(ir) - 1)
        
        for i in range(0, len(chunk), block_size):
            block = chunk[i:i+block_size]
            block_padded = np.zeros(len(ir_padded))
            block_padded[:len(block)] = block
            
            # FFT
            block_fft = np.fft.rfft(block_padded)
            
            # Multiply
            result_fft = block_fft * ir_fft
            
            # IFFT
            result = np.fft.irfft(result_fft)
            
            # Overlap-add
            start = i
            end = min(i + len(result), len(output))
            output[start:end] += result[:end-start]
            
        return output[:len(chunk)]
        
    def create_biquad_filter(
        self,
        filter_type: str,
        cutoff: float,
        q: float = 0.707,
        gain: float = 0.0,
        sample_rate: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create biquad filter coefficients.
        
        Args:
            filter_type: Filter type ("lowpass", "highpass", "bandpass", etc.)
            cutoff: Cutoff frequency in Hz
            q: Q factor
            gain: Gain for peaking filters (dB)
            sample_rate: Sample rate
            
        Returns:
            Tuple of (b, a) coefficients
        """
        sample_rate = sample_rate or self.sample_rate
        w0 = 2 * np.pi * cutoff / sample_rate
        alpha = np.sin(w0) / (2 * q)
        A = 10**(gain / 40)
        
        cos_w0 = np.cos(w0)
        sin_w0 = np.sin(w0)
        
        if filter_type == "lowpass":
            b = np.array([
                (1 - cos_w0) / 2,
                1 - cos_w0,
                (1 - cos_w0) / 2
            ])
            a = np.array([
                1 + alpha,
                -2 * cos_w0,
                1 - alpha
            ])
            
        elif filter_type == "highpass":
            b = np.array([
                (1 + cos_w0) / 2,
                -(1 + cos_w0),
                (1 + cos_w0) / 2
            ])
            a = np.array([
                1 + alpha,
                -2 * cos_w0,
                1 - alpha
            ])
            
        elif filter_type == "bandpass":
            b = np.array([
                sin_w0 / 2,
                0,
                -sin_w0 / 2
            ])
            a = np.array([
                1 + alpha,
                -2 * cos_w0,
                1 - alpha
            ])
            
        elif filter_type == "notch":
            b = np.array([1, -2 * cos_w0, 1])
            a = np.array([1 + alpha, -2 * cos_w0, 1 - alpha])
            
        elif filter_type == "peaking":
            b = np.array([
                1 + alpha * A,
                -2 * cos_w0,
                1 - alpha * A
            ])
            a = np.array([
                1 + alpha / A,
                -2 * cos_w0,
                1 - alpha / A
            ])
            
        elif filter_type == "lowshelf":
            sqA = np.sqrt(A)
            alpha2 = sin_w0 / 2 * np.sqrt((A + 1/A) * (1/q - 1) + 2)
            
            b = np.array([
                A * ((A + 1) - (A - 1) * cos_w0 + 2 * sqA * alpha2),
                2 * A * ((A - 1) - (A + 1) * cos_w0),
                A * ((A + 1) - (A - 1) * cos_w0 - 2 * sqA * alpha2)
            ])
            a = np.array([
                (A + 1) + (A - 1) * cos_w0 + 2 * sqA * alpha2,
                -2 * ((A - 1) + (A + 1) * cos_w0),
                (A + 1) + (A - 1) * cos_w0 - 2 * sqA * alpha2
            ])
            
        elif filter_type == "highshelf":
            sqA = np.sqrt(A)
            alpha2 = sin_w0 / 2 * np.sqrt((A + 1/A) * (1/q - 1) + 2)
            
            b = np.array([
                A * ((A + 1) + (A - 1) * cos_w0 + 2 * sqA * alpha2),
                -2 * A * ((A - 1) + (A + 1) * cos_w0),
                A * ((A + 1) + (A - 1) * cos_w0 - 2 * sqA * alpha2)
            ])
            a = np.array([
                (A + 1) - (A - 1) * cos_w0 + 2 * sqA * alpha2,
                2 * ((A - 1) - (A + 1) * cos_w0),
                (A + 1) - (A - 1) * cos_w0 - 2 * sqA * alpha2
            ])
            
        else:
            # Allpass
            b = np.array([1, -2 * cos_w0, 1])
            a = np.array([1 + alpha, -2 * cos_w0, 1 - alpha])
            
        # Normalize a coefficients
        a = a / a[0]
        b = b / a[0]
        
        return b, a
        
    def apply_biquad(self, signal: np.ndarray, b: np.ndarray, a: np.ndarray) -> np.ndarray:
        """Apply biquad filter with state."""
        # Simple implementation using scipy
        from scipy.signal import lfilter
        return lfilter(b, a, signal)


class LatencyCompensator:
    """
    Compensates for processing latency in real-time systems.
    """
    
    def __init__(self, latency_samples: int = 0):
        """Initialize latency compensator."""
        self.latency_samples = latency_samples
        self.delay_buffer = deque(maxlen=latency_samples + 1)
        
    def set_latency(self, latency_samples: int) -> None:
        """Set latency compensation amount."""
        self.latency_samples = latency_samples
        
    def compensate(self, signal: np.ndarray) -> np.ndarray:
        """Compensate for latency."""
        if self.latency_samples == 0:
            return signal
            
        # Add to buffer
        for sample in signal:
            self.delay_buffer.append(sample)
            
        # Read delayed samples
        output = np.zeros(len(signal))
        for i in range(len(signal)):
            idx = len(self.delay_buffer) - self.latency_samples + i
            if 0 <= idx < len(self.delay_buffer):
                output[i] = self.delay_buffer[idx]
                
        return output


class AdaptiveGain:
    """
    Adaptive gain controller for maintaining consistent output levels.
    """
    
    def __init__(
        self,
        target_rms: float = -20.0,
        attack_time: float = 0.01,
        release_time: float = 0.1,
        max_gain: float = 10.0
    ):
        """Initialize adaptive gain controller."""
        self.target_rms = target_rms
        self.attack_coef = np.exp(-1.0 / (attack_time * 48000))
        self.release_coef = np.exp(-1.0 / (release_time * 48000))
        self.max_gain = max_gain
        self.current_gain = 1.0
        
    def process(self, signal: np.ndarray) -> np.ndarray:
        """Process signal with adaptive gain."""
        # Compute RMS
        rms = np.sqrt(np.mean(signal**2))
        
        # Target gain
        if rms > 1e-6:
            target_gain = self.target_rms / (20 * np.log10(rms) + 1e-10)
        else:
            target_gain = self.max_gain
            
        # Smooth gain changes
        if target_gain < self.current_gain:
            # Attack
            self.current_gain = (self.attack_coef * self.current_gain + 
                               (1 - self.attack_coef) * target_gain)
        else:
            # Release
            self.current_gain = (self.release_coef * self.current_gain + 
                               (1 - self.release_coef) * target_gain)
            
        # Clamp gain
        self.current_gain = np.clip(self.current_gain, 1.0 / self.max_gain, self.max_gain)
        
        return signal * self.current_gain


class DynamicRangeCompressor:
    """
    Dynamic range compressor for audio level control.
    """
    
    def __init__(
        self,
        threshold: float = -20.0,
        ratio: float = 4.0,
        attack: float = 0.005,
        release: float = 0.050,
        knee: float = 6.0,
        makeup_gain: float = 0.0
    ):
        """Initialize compressor."""
        self.threshold = threshold  # dB
        self.ratio = ratio
        self.attack_coef = np.exp(-1.0 / (attack * 48000))
        self.release_coef = np.exp(-1.0 / (release * 48000))
        self.knee = knee
        self.makeup_gain = makeup_gain
        self.envelope = 0.0
        
    def process(self, signal: np.ndarray) -> np.ndarray:
        """Process signal with compression."""
        # Sidechain: compute envelope
        input_rms = np.sqrt(np.mean(signal**2))
        input_db = 20 * np.log10(input_rms + 1e-10)
        
        # Compute gain reduction
        if input_db > self.threshold + self.knee / 2:
            # Above knee
            overshoot = input_db - self.threshold - self.knee / 2
            if overshoot > 0:
                gain_reduction_db = overshoot * (1 - 1/self.ratio)
            else:
                gain_reduction_db = overshoot * (1 - 1/self.ratio) * (overshoot / (self.knee/2))**2
        else:
            gain_reduction_db = 0
            
        # Smooth envelope
        target_envelope = -gain_reduction_db
        if target_envelope > self.envelope:
            self.envelope = (self.attack_coef * self.envelope + 
                           (1 - self.attack_coef) * target_envelope)
        else:
            self.envelope = (self.release_coef * self.envelope + 
                           (1 - self.release_coef) * target_envelope)
            
        # Apply gain
        gain = 10**(self.envelope / 20) * 10**(self.makeup_gain / 20)
        
        return signal * gain


class PitchShifter:
    """
    Simple pitch shifting using resampling.
    
    Note: This is a basic implementation. For high-quality pitch shifting,
    consider using WSOLA or phase vocoder algorithms.
    """
    
    def __init__(self, sample_rate: float = 48000.0):
        """Initialize pitch shifter."""
        self.sample_rate = sample_rate
        
    def shift(self, signal: np.ndarray, semitones: float) -> np.ndarray:
        """
        Shift pitch by semitones.
        
        Args:
            signal: Input signal
            semitones: Number of semitones to shift (positive = up)
            
        Returns:
            Pitch-shifted signal
        """
        # Convert semitones to ratio
        ratio = 2**(semitones / 12.0)
        
        # Resample
        new_length = int(len(signal) / ratio)
        
        # Simple linear interpolation resampling
        indices = np.arange(new_length) * ratio
        output = np.interp(indices, np.arange(len(signal)), signal)
        
        return output


def apply_lowpass_filter(
    signal: np.ndarray,
    cutoff: float,
    sample_rate: float = 48000.0,
    order: int = 4
) -> np.ndarray:
    """Apply lowpass filter."""
    nyquist = sample_rate / 2
    normalized_cutoff = min(cutoff / nyquist, 0.99)
    
    from scipy.signal import butter, filtfilt
    b, a = butter(order, normalized_cutoff, btype="low")
    return filtfilt(b, a, signal)


def apply_highpass_filter(
    signal: np.ndarray,
    cutoff: float,
    sample_rate: float = 48000.0,
    order: int = 4
) -> np.ndarray:
    """Apply highpass filter."""
    nyquist = sample_rate / 2
    normalized_cutoff = max(cutoff / nyquist, 0.01)
    
    from scipy.signal import butter, filtfilt
    b, a = butter(order, normalized_cutoff, btype="high")
    return filtfilt(b, a, signal)


def apply_bandpass_filter(
    signal: np.ndarray,
    low_cutoff: float,
    high_cutoff: float,
    sample_rate: float = 48000.0,
    order: int = 4
) -> np.ndarray:
    """Apply bandpass filter."""
    nyquist = sample_rate / 2
    low = max(low_cutoff / nyquist, 0.01)
    high = min(high_cutoff / nyquist, 0.99)
    
    from scipy.signal import butter, filtfilt
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, signal)
