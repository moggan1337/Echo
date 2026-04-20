"""
Core Acoustic Simulation Engine
The main engine that orchestrates all acoustic simulation components.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path

from .room import Room
from .source import SoundSource
from .receiver import Receiver
from .materials import MaterialDatabase
from .rir import RIRGenerator
from .raytracer import RayTracer
from .wave_solver import WaveEquationSolver
from .spatial_audio import BinauralRenderer, AmbisonicsEncoder
from .doppler import DopplerProcessor
from .diffraction import DiffractionModel
from .realtime import RealTimeProcessor


class SimulationMethod(Enum):
    """Available acoustic simulation methods."""
    RAY_TRACING = "ray_tracing"
    WAVE_BASED = "wave_based"
    HYBRID = "hybrid"
    RADIOSITY = "radiosity"


class OutputFormat(Enum):
    """Output audio format options."""
    MONO = "mono"
    STEREO = "stereo"
    BINAURAL = "binaural"
    QUAD = "quad"
    SURROUND_5_1 = "surround_5_1"
    SURROUND_7_1 = "surround_7_1"
    AMBISONICS_FIRST_ORDER = "ambisonics_foa"
    AMBISONICS_SECOND_ORDER = "ambisonics_soa"
    AMBISONICS_THIRD_ORDER = "ambisonics_toa"


@dataclass
class SimulationConfig:
    """Configuration for acoustic simulation."""
    # Simulation parameters
    method: SimulationMethod = SimulationMethod.HYBRID
    sample_rate: int = 48000
    duration: float = 2.0  # seconds
    speed_of_sound: float = 343.0  # m/s
    
    # Ray tracing parameters
    num_rays: int = 5000
    max_reflections: int = 10
    ray_energy_threshold: float = 0.001
    
    # Wave-based parameters
    grid_resolution: float = 0.05  # meters
    pml_thickness: float = 0.5  # perfectly matched layer thickness
    
    # Frequency parameters
    min_frequency: float = 20.0  # Hz
    max_frequency: float = 20000.0  # Hz
    frequency_bands: int = 32
    
    # Output parameters
    output_format: OutputFormat = OutputFormat.BINAURAL
    bit_depth: int = 24
    
    # Performance parameters
    num_threads: int = 4
    enable_gpu: bool = False
    cache_intermediate: bool = True
    
    # Quality presets
    quality_preset: str = "high"  # low, medium, high, ultra
    
    @classmethod
    def from_preset(cls, preset: str) -> "SimulationConfig":
        """Create configuration from quality preset."""
        presets = {
            "low": cls(
                num_rays=1000, max_reflections=5,
                frequency_bands=8, grid_resolution=0.1
            ),
            "medium": cls(
                num_rays=3000, max_reflections=7,
                frequency_bands=16, grid_resolution=0.07
            ),
            "high": cls(
                num_rays=5000, max_reflections=10,
                frequency_bands=32, grid_resolution=0.05
            ),
            "ultra": cls(
                num_rays=10000, max_reflections=15,
                frequency_bands=64, grid_resolution=0.025
            ),
        }
        return presets.get(preset, presets["high"])


class AcousticEngine:
    """
    Main acoustic simulation engine that coordinates all simulation components.
    
    The engine provides a unified interface for acoustic simulation including:
    - Ray-based sound propagation
    - Wave equation solving
    - Room impulse response generation
    - Spatial audio rendering
    
    Example:
        >>> config = SimulationConfig.from_preset("high")
        >>> engine = AcousticEngine(config)
        >>> engine.set_room(room)
        >>> engine.add_source(source)
        >>> engine.add_receiver(receiver)
        >>> rir = engine.compute_rir()
        >>> output = engine.render_audio(input_signal)
    """
    
    def __init__(self, config: Optional[SimulationConfig] = None):
        """Initialize the acoustic engine with configuration."""
        self.config = config or SimulationConfig()
        self.room: Optional[Room] = None
        self.sources: List[SoundSource] = []
        self.receivers: List[Receiver] = []
        
        # Initialize components
        self.material_db = MaterialDatabase()
        self.rir_generator = RIRGenerator(self.config)
        self.ray_tracer = RayTracer(self.config)
        self.wave_solver = WaveEquationSolver(self.config)
        self.binaural_renderer = BinauralRenderer(self.config)
        self.ambisonics_encoder = AmbisonicsEncoder(self.config)
        self.doppler_processor = DopplerProcessor(self.config)
        self.diffraction_model = DiffractionModel(self.config)
        self.realtime_processor = RealTimeProcessor(self.config)
        
        # Simulation state
        self._cached_rirs: Dict[str, np.ndarray] = {}
        self._simulation_time: float = 0.0
        self._is_initialized: bool = False
        
    def set_room(self, room: Room) -> None:
        """Set the room geometry and acoustic properties."""
        self.room = room
        self._invalidate_cache()
        
    def add_source(self, source: SoundSource) -> int:
        """Add a sound source to the simulation."""
        self.sources.append(source)
        self._invalidate_cache()
        return len(self.sources) - 1
        
    def remove_source(self, index: int) -> None:
        """Remove a sound source by index."""
        if 0 <= index < len(self.sources):
            self.sources.pop(index)
            self._invalidate_cache()
            
    def add_receiver(self, receiver: Receiver) -> int:
        """Add a receiver to the simulation."""
        self.receivers.append(receiver)
        self._invalidate_cache()
        return len(self.receivers) - 1
        
    def remove_receiver(self, index: int) -> None:
        """Remove a receiver by index."""
        if 0 <= index < len(self.receivers):
            self.receivers.pop(index)
            self._invalidate_cache()
            
    def _invalidate_cache(self) -> None:
        """Invalidate cached simulation results."""
        self._cached_rirs.clear()
        self._is_initialized = False
        
    def _check_initialization(self) -> None:
        """Check if engine is properly initialized."""
        if self.room is None:
            raise RuntimeError("Room must be set before simulation")
        if not self.sources:
            raise RuntimeError("At least one source must be added")
        if not self.receivers:
            raise RuntimeError("At least one receiver must be added")
        self._is_initialized = True
        
    def compute_rir(
        self,
        source_idx: int = 0,
        receiver_idx: int = 0,
        use_cache: bool = True
    ) -> np.ndarray:
        """
        Compute the room impulse response between source and receiver.
        
        Args:
            source_idx: Index of the source
            receiver_idx: Index of the receiver
            use_cache: Whether to use cached results
            
        Returns:
            Room impulse response as numpy array
        """
        self._check_initialization()
        
        cache_key = f"rir_{source_idx}_{receiver_idx}"
        
        if use_cache and cache_key in self._cached_rirs:
            return self._cached_rirs[cache_key]
            
        source = self.sources[source_idx]
        receiver = self.receivers[receiver_idx]
        
        if self.config.method == SimulationMethod.RAY_TRACING:
            rir = self.ray_tracer.compute_rir(
                self.room, source, receiver
            )
        elif self.config.method == SimulationMethod.WAVE_BASED:
            rir = self.wave_solver.compute_rir(
                self.room, source, receiver
            )
        else:  # HYBRID
            # Combine ray tracing (early reflections) with wave-based (late reverberation)
            early_rir = self.ray_tracer.compute_rir(
                self.room, source, receiver, max_order=3
            )
            late_rir = self.wave_solver.compute_rir(
                self.room, source, receiver
            )
            # Crossfade at the mixing point
            fade_samples = int(self.config.sample_rate * 0.05)
            fade_in = np.linspace(0, 1, fade_samples)
            fade_out = 1 - fade_in
            mixing_point = len(early_rir) - fade_samples
            rir = np.concatenate([
                early_rir[:mixing_point],
                early_rir[mixing_point:] * fade_out + late_rir[mixing_point:] * fade_in,
                late_rir[mixing_point + fade_samples:]
            ])
            
        self._cached_rirs[cache_key] = rir
        return rir
        
    def compute_all_rirs(self) -> Dict[Tuple[int, int], np.ndarray]:
        """
        Compute room impulse responses for all source-receiver pairs.
        
        Returns:
            Dictionary mapping (source_idx, receiver_idx) to RIR array
        """
        self._check_initialization()
        rirs = {}
        for si, source in enumerate(self.sources):
            for ri, receiver in enumerate(self.receivers):
                rirs[(si, ri)] = self.compute_rir(si, ri)
        return rirs
        
    def apply_rir(self, signal: np.ndarray, rir: np.ndarray) -> np.ndarray:
        """
        Apply a room impulse response to an input signal.
        
        Args:
            signal: Input audio signal
            rir: Room impulse response
            
        Returns:
            Convolved output signal
        """
        from scipy.signal import fftconvolve
        return fftconvolve(signal, rir, mode="full")[:len(signal)]
        
    def render_audio(
        self,
        input_signals: Union[np.ndarray, List[np.ndarray]],
        source_indices: Optional[List[int]] = None
    ) -> np.ndarray:
        """
        Render audio with acoustic simulation applied.
        
        Args:
            input_signals: Input audio signal(s)
            source_indices: Indices of sources to render (default: all)
            
        Returns:
            Rendered audio in the configured output format
        """
        self._check_initialization()
        
        if not isinstance(input_signals, list):
            input_signals = [input_signals]
            
        if source_indices is None:
            source_indices = list(range(len(input_signals)))
            
        # Compute RIRs for all active sources
        rendered_signals = []
        
        for sig_idx, src_idx in enumerate(source_indices):
            if src_idx >= len(self.sources):
                continue
                
            source = self.sources[src_idx]
            signal = input_signals[sig_idx] if sig_idx < len(input_signals) else input_signals[0]
            
            for ri, receiver in enumerate(self.receivers):
                rir = self.compute_rir(src_idx, ri)
                
                # Apply source directivity
                if source.directivity is not None:
                    directivity_filter = source.directivity.get_filter(receiver.position)
                    rir = np.convolve(rir, directivity_filter, mode="same")
                
                # Convolve signal with RIR
                rendered = self.apply_rir(signal, rir)
                rendered_signals.append(rendered)
                
        # Combine all rendered signals
        if not rendered_signals:
            return np.zeros(int(self.config.sample_rate * self.config.duration))
            
        output = np.sum(rendered_signals, axis=0)
        
        # Apply output format processing
        if self.config.output_format == OutputFormat.BINAURAL:
            output = self.binaural_renderer.render(output, self.receivers[0])
        elif self.config.output_format in [
            OutputFormat.AMBISONICS_FIRST_ORDER,
            OutputFormat.AMBISONICS_SECOND_ORDER,
            OutputFormat.AMBISONICS_THIRD_ORDER
        ]:
            order = self._get_ambisonics_order()
            output = self.ambisonics_encoder.encode(
                output, self.receivers[0], order=order
            )
            
        return output
        
    def _get_ambisonics_order(self) -> int:
        """Get ambisonics order from output format."""
        order_map = {
            OutputFormat.AMBISONICS_FIRST_ORDER: 1,
            OutputFormat.AMBISONICS_SECOND_ORDER: 2,
            OutputFormat.AMBISONICS_THIRD_ORDER: 3,
        }
        return order_map.get(self.config.output_format, 1)
        
    def simulate_doppler(
        self,
        source_position: np.ndarray,
        source_velocity: np.ndarray,
        signal: np.ndarray
    ) -> np.ndarray:
        """
        Apply Doppler shift to simulate moving source.
        
        Args:
            source_position: Source position over time (N, 3)
            source_velocity: Source velocity over time (N, 3)
            signal: Input audio signal
            
        Returns:
            Doppler-shifted audio signal
        """
        return self.doppler_processor.process(
            signal, source_position, source_velocity
        )
        
    def compute_diffraction(
        self,
        source_pos: np.ndarray,
        receiver_pos: np.ndarray,
        obstacle_pos: np.ndarray,
        obstacle_size: np.ndarray
    ) -> np.ndarray:
        """
        Compute diffraction around obstacles.
        
        Args:
            source_pos: Source position
            receiver_pos: Receiver position
            obstacle_pos: Obstacle position
            obstacle_size: Obstacle dimensions
            
        Returns:
            Diffraction filter coefficients
        """
        return self.diffraction_model.compute_filter(
            source_pos, receiver_pos, obstacle_pos, obstacle_size,
            self.config.sample_rate
        )
        
    def analyze_acoustics(self) -> Dict:
        """
        Perform comprehensive acoustic analysis of the configured environment.
        
        Returns:
            Dictionary containing acoustic metrics
        """
        self._check_initialization()
        
        metrics = {
            "room_volume": self.room.volume,
            "room_surface_area": self.room.surface_area,
            "reverberation_time": {},
            "clarity_index": {},
            "definition": {},
            "center_time": {},
        }
        
        # Compute T60 for each source-receiver pair
        for si, source in enumerate(self.sources):
            for ri, receiver in enumerate(self.receivers):
                rir = self.compute_rir(si, ri)
                
                # T60 estimation using Schroeder integration
                t60 = self._estimate_t60(rir)
                metrics["reverberation_time"][f"src{si}_recv{ri}"] = t60
                
                # Clarity index (C50, C80)
                metrics["clarity_index"][f"src{si}_recv{ri}"] = {
                    "C50": self._compute_clarity(rir, 50),
                    "C80": self._compute_clarity(rir, 80),
                }
                
                # Definition (D50)
                metrics["definition"][f"src{si}_recv{ri}"] = self._compute_definition(rir)
                
                # Center time
                metrics["center_time"][f"src{si}_recv{ri}"] = self._compute_center_time(rir)
                
        return metrics
        
    def _estimate_t60(self, rir: np.ndarray) -> float:
        """Estimate T60 reverberation time from RIR."""
        from scipy.signal import sosfilt, butter
        
        # Energy envelope via Schroeder integration
        energy = np.cumsum(rir[::-1]**2)[::-1]
        energy = np.maximum(energy, 1e-10)
        energy_db = 10 * np.log10(energy / np.max(energy))
        
        # Find decay curve
        max_idx = np.argmax(energy_db)
        if max_idx < len(energy_db) - 1:
            decay_curve = energy_db[max_idx:]
            
            # Linear regression for decay rate
            from scipy.stats import linregress
            valid_mask = decay_curve > (decay_curve[0] - 35)
            if np.sum(valid_mask) > 10:
                slope, intercept, _, _, _ = linregress(
                    np.arange(np.sum(valid_mask)),
                    decay_curve[valid_mask]
                )
                t60 = -60 / slope if slope < 0 else np.inf
                return t60 / self.config.sample_rate
                
        return 0.0
        
    def _compute_clarity(self, rir: np.ndarray, t_ms: float) -> float:
        """Compute clarity index (early vs late energy)."""
        t_samples = int(t_ms * self.config.sample_rate / 1000)
        early_energy = np.sum(rir[:t_samples]**2)
        late_energy = np.sum(rir[t_samples:]**2) + 1e-10
        return 10 * np.log10(early_energy / late_energy)
        
    def _compute_definition(self, rir: np.ndarray) -> float:
        """Compute definition (D50)."""
        return 10**(self._compute_clarity(rir, 50) / 10) / (
            10**(self._compute_clarity(rir, 50) / 10) + 1
        )
        
    def _compute_center_time(self, rir: np.ndarray) -> float:
        """Compute center time of RIR energy."""
        energy = rir**2
        times = np.arange(len(rir)) / self.config.sample_rate
        total_energy = np.sum(energy) + 1e-10
        return np.sum(times * energy) / total_energy
        
    def export_config(self, path: Union[str, Path]) -> None:
        """Export simulation configuration to JSON file."""
        config_dict = {
            "method": self.config.method.value,
            "sample_rate": self.config.sample_rate,
            "duration": self.config.duration,
            "speed_of_sound": self.config.speed_of_sound,
            "num_rays": self.config.num_rays,
            "max_reflections": self.config.max_reflections,
            "ray_energy_threshold": self.config.ray_energy_threshold,
            "grid_resolution": self.config.grid_resolution,
            "min_frequency": self.config.min_frequency,
            "max_frequency": self.config.max_frequency,
            "frequency_bands": self.config.frequency_bands,
            "output_format": self.config.output_format.value,
            "quality_preset": self.config.quality_preset,
        }
        
        with open(path, "w") as f:
            json.dump(config_dict, f, indent=2)
            
    @classmethod
    def load_config(cls, path: Union[str, Path]) -> SimulationConfig:
        """Load simulation configuration from JSON file."""
        with open(path, "r") as f:
            config_dict = json.load(f)
            
        config = SimulationConfig()
        config.method = SimulationMethod(config_dict.get("method", "hybrid"))
        config.sample_rate = config_dict.get("sample_rate", 48000)
        config.duration = config_dict.get("duration", 2.0)
        config.speed_of_sound = config_dict.get("speed_of_sound", 343.0)
        config.num_rays = config_dict.get("num_rays", 5000)
        config.max_reflections = config_dict.get("max_reflections", 10)
        config.ray_energy_threshold = config_dict.get("ray_energy_threshold", 0.001)
        config.grid_resolution = config_dict.get("grid_resolution", 0.05)
        config.min_frequency = config_dict.get("min_frequency", 20.0)
        config.max_frequency = config_dict.get("max_frequency", 20000.0)
        config.frequency_bands = config_dict.get("frequency_bands", 32)
        config.output_format = OutputFormat(config_dict.get("output_format", "binaural"))
        config.quality_preset = config_dict.get("quality_preset", "high")
        
        return config
        
    def reset(self) -> None:
        """Reset the engine state."""
        self._cached_rirs.clear()
        self._simulation_time = 0.0
        self._is_initialized = False
