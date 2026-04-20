"""
Echo - Acoustic Simulation Engine
A comprehensive acoustic simulation framework for realistic sound propagation,
room acoustics, and spatial audio rendering.
"""

__version__ = "1.0.0"
__author__ = "Moggan1337"

from .core import AcousticEngine, SimulationConfig
from .room import Room, RoomBuilder
from .source import SoundSource, SourceDirectivity
from .receiver import Receiver, HRTFProcessor
from .materials import MaterialDatabase, Material
from .rir import RIRGenerator
from .raytracer import RayTracer
from .wave_solver import WaveEquationSolver
from .spatial_audio import BinauralRenderer, AmbisonicsEncoder
from .doppler import DopplerProcessor
from .diffraction import DiffractionModel
from .realtime import RealTimeProcessor

__all__ = [
    "AcousticEngine",
    "SimulationConfig",
    "Room",
    "RoomBuilder",
    "SoundSource",
    "SourceDirectivity",
    "Receiver",
    "HRTFProcessor",
    "MaterialDatabase",
    "Material",
    "RIRGenerator",
    "RayTracer",
    "WaveEquationSolver",
    "BinauralRenderer",
    "AmbisonicsEncoder",
    "DopplerProcessor",
    "DiffractionModel",
    "RealTimeProcessor",
]
