"""
Echo Engine Test Suite
"""

import pytest
import numpy as np
import sys
sys.path.insert(0, "../")

from echo_engine import (
    AcousticEngine, SimulationConfig, SimulationMethod,
    Room, RoomBuilder, SurfaceType,
    SoundSource, SourceDirectivity,
    Receiver, ListenerConfig,
    MaterialDatabase, Material,
    OutputFormat
)


class TestSimulationConfig:
    """Test simulation configuration."""
    
    def test_default_config(self):
        config = SimulationConfig()
        assert config.sample_rate == 48000
        assert config.duration == 2.0
        assert config.method == SimulationMethod.HYBRID
        
    def test_preset_configs(self):
        for preset in ["low", "medium", "high", "ultra"]:
            config = SimulationConfig.from_preset(preset)
            assert config is not None
            assert config.quality_preset == preset
            
    def test_export_import_config(self, tmp_path):
        config = SimulationConfig.from_preset("high")
        config_path = tmp_path / "config.json"
        
        # Test export (manual implementation needed)
        config_dict = {
            "method": config.method.value,
            "sample_rate": config.sample_rate,
            "duration": config.duration,
        }
        
        import json
        with open(config_path, "w") as f:
            json.dump(config_dict, f)
            
        with open(config_path, "r") as f:
            loaded = json.load(f)
            
        assert loaded["sample_rate"] == 48000


class TestRoom:
    """Test room geometry."""
    
    def test_rectangular_room(self):
        room = Room.dimensions(10, 8, 3)
        assert len(room.surfaces) == 6
        
        dims = room.dimensions
        assert dims == (10, 8, 3)
        
    def test_room_volume(self):
        room = Room.dimensions(5, 4, 3)
        expected_volume = 5 * 4 * 3  # 60 m³
        assert abs(room.volume - expected_volume) < 1.0
        
    def test_room_surface_area(self):
        room = Room.dimensions(5, 4, 3)
        expected_area = 2 * (5*4 + 4*3 + 5*3)  # 94 m²
        assert abs(room.surface_area - expected_area) < 1.0
        
    def test_room_builder(self):
        room = (RoomBuilder()
                .rectangular(12, 10, 3.5)
                .floor("hardwood")
                .ceiling("acoustic_tile")
                .walls("concrete")
                .build())
                
        assert len(room.surfaces) == 6
        
    def test_point_inside_room(self):
        room = Room.dimensions(10, 8, 3)
        point = np.array([5, 4, 1.5])
        # Simple check - should be inside bounding box
        assert room.is_inside(point)


class TestMaterial:
    """Test acoustic materials."""
    
    def test_material_absorption(self):
        mat = Material(
            name="test",
            absorption_125=0.1,
            absorption_250=0.2,
            absorption_500=0.3,
            absorption_1000=0.4,
            absorption_2000=0.5,
            absorption_4000=0.6
        )
        
        # Test interpolation at mid frequencies
        alpha_300 = mat.absorption(300)
        assert 0.2 <= alpha_300 <= 0.3
        
    def test_material_reflection(self):
        mat = Material(name="test", absorption_500=0.3)
        refl = mat.reflection(500)
        assert 0.0 <= refl <= 1.0
        
    def test_material_database(self):
        db = MaterialDatabase()
        
        concrete = db.get("concrete")
        assert concrete is not None
        assert concrete.name == "concrete"
        
        glass = db.get("glass")
        assert glass is not None
        
    def test_material_search(self):
        db = MaterialDatabase()
        results = db.search("carpet")
        assert len(results) >= 2  # Should find carpet_heavy, carpet_light


class TestSoundSource:
    """Test sound sources."""
    
    def test_source_creation(self):
        source = SoundSource(position=[3, 4, 1.5])
        assert np.allclose(source.position, [3, 4, 1.5])
        
    def test_source_signal_generation(self):
        source = SoundSource(position=[0, 0, 0])
        
        # Test impulse
        impulse = source.generate_impulse(duration=0.5)
        assert len(impulse) > 0
        assert np.max(np.abs(impulse)) > 0
        
        # Test sweep
        sweep = source.generate_sweep(duration=0.5, freq_start=100, freq_end=1000)
        assert len(sweep) > 0
        
    def test_source_directivity(self):
        directivity = SourceDirectivity.cardioid(axis=(1, 0, 0))
        
        # Test response at various angles
        azimuths = np.linspace(-np.pi, np.pi, 10)
        elevations = np.zeros(10)
        
        response = directivity.get_response(azimuths, elevations)
        assert len(response) == 10
        assert np.all(response >= 0)


class TestReceiver:
    """Test receivers."""
    
    def test_receiver_creation(self):
        receiver = Receiver(position=[5, 4, 1.2])
        assert np.allclose(receiver.position, [5, 4, 1.2])
        
    def test_receiver_orientation(self):
        receiver = Receiver(position=[0, 0, 0], orientation=45)
        assert receiver.orientation == 45
        
        forward = receiver.forward_vector
        assert len(forward) == 3
        
    def test_direction_from(self):
        receiver = Receiver(position=[5, 4, 1.2], orientation=0)
        source_pos = np.array([3, 4, 1.2])
        
        az, el = receiver.direction_from(source_pos)
        assert isinstance(az, (int, float, np.floating))
        assert isinstance(el, (int, float, np.floating))


class TestAcousticEngine:
    """Test main acoustic engine."""
    
    def test_engine_initialization(self):
        config = SimulationConfig.from_preset("low")
        engine = AcousticEngine(config)
        assert engine is not None
        
    def test_engine_setup(self):
        config = SimulationConfig.from_preset("low")
        engine = AcousticEngine(config)
        
        room = Room.dimensions(8, 6, 3)
        engine.set_room(room)
        
        source = SoundSource(position=[2, 3, 1.5])
        engine.add_source(source)
        
        receiver = Receiver(position=[5, 3, 1.2])
        engine.add_receiver(receiver)
        
        assert engine.room is not None
        assert len(engine.sources) == 1
        assert len(engine.receivers) == 1
        
    def test_engine_rir_computation(self):
        config = SimulationConfig.from_preset("low")
        config.duration = 0.5
        engine = AcousticEngine(config)
        
        room = Room.dimensions(8, 6, 3)
        engine.set_room(room)
        
        source = SoundSource(position=[2, 3, 1.5])
        engine.add_source(source)
        
        receiver = Receiver(position=[5, 3, 1.2])
        engine.add_receiver(receiver)
        
        rir = engine.compute_rir()
        assert len(rir) > 0


class TestHRTF:
    """Test HRTF processing."""
    
    def test_hrtf_creation(self):
        from echo_engine.receiver import HRTFProcessor
        
        hrtf = HRTFProcessor()
        assert hrtf is not None
        
    def test_hrtf_response(self):
        from echo_engine.receiver import HRTFProcessor
        
        hrtf = HRTFProcessor()
        
        left, right, itd = hrtf.get_hrtf(azimuth=30, elevation=0)
        assert len(left) > 0
        assert len(right) > 0


class TestDoppler:
    """Test Doppler effect."""
    
    def test_doppler_processor(self):
        from echo_engine.doppler import DopplerProcessor
        
        config = SimulationConfig()
        doppler = DopplerProcessor(config)
        assert doppler is not None
        
    def test_doppler_factor(self):
        from echo_engine.doppler import doppler_shift_factor
        
        # Stationary source
        factor = doppler_shift_factor(0)
        assert abs(factor - 1.0) < 0.01
        
        # Approaching source
        factor = doppler_shift_factor(10)
        assert factor > 1.0
        
        # Receding source
        factor = doppler_shift_factor(-10)
        assert factor < 1.0


class TestRealTimeProcessor:
    """Test real-time processing."""
    
    def test_biquad_coefficients(self):
        from echo_engine.realtime import RealTimeProcessor
        
        config = SimulationConfig()
        processor = RealTimeProcessor(config)
        
        b, a = processor.create_biquad_filter("lowpass", 1000)
        assert len(b) == 3
        assert len(a) == 3
        
    def test_partitioned_convolution_setup(self):
        from echo_engine.realtime import RealTimeProcessor
        
        config = SimulationConfig()
        processor = RealTimeProcessor(config)
        
        # Create simple IR
        ir = np.random.randn(1000)
        processor.setup_partitioned_convolution(ir, num_partitions=8)
        
        assert processor._rir is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
