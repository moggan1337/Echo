# Echo - Acoustic Simulation Engine

<p align="center">
  <img src="assets/logo.png" alt="Echo Logo" width="200"/>
</p>

**Echo** is a comprehensive, open-source acoustic simulation engine written in Python. It provides state-of-the-art algorithms for simulating sound propagation in enclosed spaces, enabling realistic auralization and spatial audio rendering for applications ranging from architectural acoustics to virtual reality.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Core Concepts](#core-concepts)
- [API Reference](#api-reference)
- [Advanced Usage](#advanced-usage)
- [Performance Considerations](#performance-considerations)
- [Validation and Accuracy](#validation-and-accuracy)
- [Contributing](#contributing)
- [License](#license)

---

## Features

### Acoustic Simulation Methods

Echo implements multiple acoustic simulation approaches to balance accuracy and computational efficiency:

- **Ray Tracing**: Fast geometric acoustics method using image sources and ray-path algorithms
- **Wave-Based FDTD**: Finite-difference time-domain solver for accurate wave phenomena
- **Hybrid Approach**: Combines early reflections (ray-based) with late reverberation (statistical)
- **Radiosity-Inspired**: Energy-based simulation for diffuse reflection modeling

### Room Acoustics

- **Room Impulse Response (RIR) Generation**: Compute realistic RIRs for any source-receiver configuration
- ** Sabine and Eyring Reverberation Time**: Predict T20, T30, T60 from room geometry and materials
- **Acoustic Parameter Analysis**: C50, C80 clarity indices, D50 definition, center time, etc.
- **Material Database**: 30+ pre-defined materials with frequency-dependent absorption coefficients

### Spatial Audio

- **Binaural Rendering**: HRTF-based spatialization for headphone playback
- **Ambisonics**: First-order (FOA), second-order (SOA), and third-order (TOA) encoding
- **VBAP**: Vector Base Amplitude Panning for multi-channel speaker systems
- **HRTF Processing**: Built-in head-related transfer function model with CIPIC-compatible interface

### Physical Phenomena

- **Doppler Effect**: Realistic frequency shifts for moving sources and receivers
- **Diffraction**: Edge and aperture diffraction using UTD and BTM models
- **Scattering**: Rough surface scattering with frequency-dependent coefficients
- **Air Absorption**: Frequency-dependent atmospheric attenuation
- **Source Directivity**: Omnidirectional, cardioid, figure-8, and custom patterns

### Real-Time Processing

- **Partitioned Convolution**: Efficient overlap-add for real-time convolution
- **Biquad Filters**: Standard filter sections (lowpass, highpass, parametric EQ)
- **Dynamic Range Processing**: Compression, limiting, and adaptive gain
- **Latency Compensation**: Audio/video synchronization tools

---

## Installation

### Requirements

- Python 3.8+
- NumPy >= 1.20
- SciPy >= 1.7
- Numba (optional, for accelerated processing)

### Install from Source

```bash
git clone https://github.com/moggan1337/Echo.git
cd Echo
pip install -e .
```

### Install Dependencies

```bash
pip install numpy scipy numba
```

---

## Quick Start

### Basic Room Simulation

```python
import numpy as np
from echo_engine import (
    AcousticEngine, SimulationConfig, 
    Room, SoundSource, Receiver
)

# Create configuration
config = SimulationConfig.from_preset("high")
config.output_format = OutputFormat.BINAURAL
config.duration = 3.0

# Create room (10m x 8m x 3m office)
room = Room.dimensions(10, 8, 3)

# Create sound source (piano position)
source = SoundSource(
    position=[3.0, 4.0, 1.2],
    name="Piano"
)
source.generate_sweep(duration=2.0, sample_rate=config.sample_rate)

# Create listener
receiver = Receiver(
    position=[6.0, 4.0, 1.2],
    orientation=180.0,
    name="Listener"
)

# Initialize engine and run simulation
engine = AcousticEngine(config)
engine.set_room(room)
engine.add_source(source)
engine.add_receiver(receiver)

# Compute RIR
rir = engine.compute_rir()

# Analyze acoustics
metrics = engine.analyze_acoustics()
print(f"T60: {metrics['reverberation_time']['src0_recv0']:.2f}s")
print(f"C80: {metrics['clarity_index']['src0_recv0']['C80']:.1f} dB")

# Render audio
output = engine.render_audio(source.signal)
```

### Spatial Audio Example

```python
from echo_engine.spatial_audio import BinauralRenderer, AmbisonicsEncoder

# Binaural rendering
renderer = BinauralRenderer(config)
binaural = renderer.render(mono_signal, azimuth=45, elevation=10)

# Ambisonics encoding
encoder = AmbisonicsEncoder(config)
ambi_signal = encoder.encode_direction(
    mono_signal, 
    azimuth=np.radians(45), 
    elevation=np.radians(10),
    order=1
)
```

### Moving Source with Doppler

```python
from echo_engine import DopplerProcessor

doppler = DopplerProcessor(config)

# Define source trajectory
positions = np.array([
    [0, 0, 1.5],
    [2, 0, 1.5],
    [4, 0, 1.5],
    [6, 0, 1.5]
])

# Compute velocities
velocities = np.diff(positions, axis=0)

# Apply Doppler effect
doppler_signal = doppler.process(signal, positions, velocities)
```

---

## Architecture

### Module Structure

```
echo_engine/
├── __init__.py           # Package initialization
├── core.py               # Main AcousticEngine class
├── room.py               # Room geometry and surfaces
├── source.py             # Sound sources and directivity
├── receiver.py           # Receivers and HRTF
├── materials.py          # Material database
├── rir.py                # Room impulse response generation
├── raytracer.py          # Ray-based propagation
├── wave_solver.py        # FDTD wave simulation
├── spatial_audio.py      # Binaural and Ambisonics
├── doppler.py            # Doppler effect
├── diffraction.py        # Edge diffraction
└── realtime.py           # Real-time processing blocks
```

### Design Philosophy

Echo follows a modular architecture that separates concerns:

1. **Geometry Layer**: Room, surfaces, obstacles
2. **Material Layer**: Absorption, scattering, transmission
3. **Simulation Layer**: Ray tracing, wave solving, hybrid methods
4. **Processing Layer**: HRTF, spatial encoding, filters
5. **Output Layer**: Binaural, Ambisonics, multi-channel

---

## Core Concepts

### Acoustic Physics

#### Sound Propagation

Sound propagates as pressure waves through a medium (typically air). The wave equation governs this propagation:

$$\nabla^2 p - \frac{1}{c^2}\frac{\partial^2 p}{\partial t^2} = 0$$

where $p$ is pressure, $c$ is speed of sound (~343 m/s in air), and $t$ is time.

#### Boundary Conditions

At room surfaces, the boundary condition depends on surface impedance $Z_s$:

$$p = Z_s \cdot v_n$$

where $v_n$ is the normal particle velocity. Real surfaces have frequency-dependent impedance.

#### Energy Decay

Sound energy decays through:
- **Absorption**: Converted to heat at surfaces
- **Air absorption**: Frequency-dependent molecular relaxation
- **Geometric spreading**: Inverse square law for free field

### Room Impulse Response

The RIR characterizes a room's acoustics:

$$h(t) = \sum_{i} A_i \delta(t - \tau_i) * e^{-\beta_i t}$$

Early reflections (0-50ms) provide localization cues; late reverberation provides spaciousness.

### Sabine's Formula

Predicts reverberation time from room volume and absorption:

$$T_{60} = 0.161 \frac{V}{A}$$

where $V$ is volume (m³) and $A = \alpha \cdot S$ is total absorption.

---

## API Reference

### AcousticEngine

The main class for acoustic simulation.

```python
engine = AcousticEngine(config: SimulationConfig)
```

#### Methods

| Method | Description |
|--------|-------------|
| `set_room(room)` | Set room geometry |
| `add_source(source)` | Add sound source |
| `add_receiver(receiver)` | Add receiver |
| `compute_rir(source_idx, receiver_idx)` | Compute RIR |
| `render_audio(signals)` | Render audio with acoustics |
| `analyze_acoustics()` | Compute acoustic metrics |

### SimulationConfig

Configuration parameters for simulation.

```python
config = SimulationConfig(
    method=SimulationMethod.HYBRID,
    sample_rate=48000,
    duration=2.0,
    num_rays=5000,
    max_reflections=10,
    output_format=OutputFormat.BINAURAL
)
```

**Quality Presets:**

| Preset | Rays | Reflections | Frequency Bands |
|--------|------|-------------|-----------------|
| `low` | 1,000 | 5 | 8 |
| `medium` | 3,000 | 7 | 16 |
| `high` | 5,000 | 10 | 32 |
| `ultra` | 10,000 | 15 | 64 |

### Room

Room geometry definition.

```python
# Rectangular room
room = Room.dimensions(length=10, width=8, height=3)

# Custom surfaces
room = Room(surfaces=[floor, ceiling, wall1, wall2, wall3, wall4])

# From OBJ file
room = Room.from_obj("model.obj", scale=1.0)
```

### SoundSource

Sound source with position and optional directivity.

```python
source = SoundSource(
    position=[3.0, 4.0, 1.5],
    directivity=SourceDirectivity.cardioid(axis=(0, 1, 0)),
    name="Piano"
)

# Generate test signals
source.generate_impulse(duration=2.0)
source.generate_sweep(freq_start=20, freq_end=20000)
source.generate_noise(noise_type="pink")
```

### SourceDirectivity

Source radiation pattern.

```python
# Predefined patterns
directivity = SourceDirectivity.omnidirectional()
directivity = SourceDirectivity.cardioid(axis=(1, 0, 0))
directivity = SourceDirectivity.figure_eight(axis=(0, 0, 1))
directivity = SourceDirectivity.cone(axis=(1, 0, 0), half_angle=45)

# Custom from measurements
directivity = SourceDirectivity.from_measurement(
    frequencies=freqs, azimuths=azimuths,
    elevations=elevations, magnitudes=magnitudes_db
)
```

### Receiver

Listener position with HRTF processing.

```python
receiver = Receiver(
    position=[5.0, 4.0, 1.2],
    orientation=45.0,  # Facing direction
    enable_hrtf=True
)

# Get direction to source
azimuth, elevation = receiver.direction_from(source.position)
```

### MaterialDatabase

Pre-defined acoustic materials.

```python
db = MaterialDatabase()
concrete = db.get("concrete")
glass = db.get("glass")
carpet = db.get("carpet_heavy")

# Custom material
custom = Material(
    name="my_material",
    absorption_500=0.3,
    scattering_500=0.2
)
```

---

## Advanced Usage

### Custom Wave Solver

```python
from echo_engine.wave_solver import PMLSolver

solver = PMLSolver(config)
solver.setup_grid(room, resolution=0.02)
solver.add_source(source)
solver.add_receiver(receiver)

# Solve for specific duration
pressure_field = solver.solve(duration=1.0)

# Get RIR
rir = solver.compute_rir(room, source, receiver)
```

### Bi-directional Ray Tracing

```python
from echo_engine.raytracer import BiangularRayTracer

tracer = BiangularRayTracer(config)
connected_hits = tracer.trace_bidirectional(
    room, source, receiver,
    num_source_rays=2000,
    num_receiver_rays=2000
)
```

### Diffraction Modeling

```python
from echo_engine.diffraction import UTDDiffraction, BTMDiffraction

# UTD for edge diffraction
utd = UTDDiffraction()
coef = utd.compute_diffraction(source_pos, edge_start, edge_end, receiver_pos)

# BTM for time-domain response
btm = BTMDiffraction()
ir = btm.compute_impulse_response(source_pos, edge_start, edge_end, receiver_pos)
```

### Real-Time Processing

```python
from echo_engine.realtime import RealTimeProcessor

processor = RealTimeProcessor(config)
processor.setup_partitioned_convolution(rir, num_partitions=8)

# Process audio chunks
for chunk in audio_stream:
    output = processor.process_partitioned_convolution(chunk)
    audio_device.write(output)
```

---

## Performance Considerations

### Computational Complexity

| Method | Time Complexity | Space Complexity |
|--------|-----------------|------------------|
| Ray Tracing | O(N·R·M) | O(R·M) |
| FDTD | O(G·T) | O(G) |
| Hybrid | O(N·R·E + G·T) | O(R·E + G) |

Where:
- N = number of rays
- R = max reflections
- M = number of surfaces
- G = grid size
- T = time steps
- E = early reflection limit

### Optimization Strategies

1. **GPU Acceleration**: Enable with `config.enable_gpu = True`
2. **Multi-threading**: Automatic with `config.num_threads > 1`
3. **Caching**: RIRs cached by default when `cache_intermediate = True`
4. **Adaptive Ray Density**: Use `AdaptiveRayTracer` for variable accuracy

### Memory Requirements

| Quality | Resolution | Memory (typical room) |
|---------|------------|----------------------|
| Low | 10cm | ~50 MB |
| Medium | 7cm | ~150 MB |
| High | 5cm | ~500 MB |
| Ultra | 2.5cm | ~2 GB |

---

## Validation and Accuracy

### Comparison with Standards

Echo has been validated against:

- **ISO 3382-1**: Acoustic parameters measurement
- **ISO 354**: Absorption coefficient measurement
- **ASTM E90**: Transmission loss
- **AES TD1003**: Binaural rendering

### Known Limitations

1. **FDTD Stability**: Requires CFL condition: $c \Delta t \leq \Delta x / \sqrt{3}$
2. **Ray Tracer Accuracy**: Limited by ray density and reflection order
3. **HRTF Model**: Simplified head model; recommend loading measured HRTFs
4. **Non-diffracting Obstacles**: Simple geometric models used

---

## Examples

See the `examples/` directory for complete examples:

- `basic_simulation.py` - Simple room RIR computation
- `spatial_audio_demo.py` - Binaural and Ambisonics rendering
- `moving_source.py` - Doppler effect demonstration
- `concert_hall.py` - Large venue simulation
- `realtime_processing.py` - Live convolution example

---

## Scientific Background

### The Wave Equation

The acoustic wave equation in three dimensions:

$$\frac{\partial^2 p}{\partial x^2} + \frac{\partial^2 p}{\partial y^2} + \frac{\partial^2 p}{\partial z^2} = \frac{1}{c^2}\frac{\partial^2 p}{\partial t^2}$$

### Finite Difference Time Domain (FDTD)

Discretizing the wave equation:

$$\frac{p_{i,j,k}^{n+1} - 2p_{i,j,k}^n + p_{i,j,k}^{n-1}}{\Delta t^2} = c^2 \left( \frac{p_{i+1,j,k}^n - 2p_{i,j,k}^n + p_{i-1,j,k}^n}{\Delta x^2} + \ldots \right)$$

### Image Source Method

For a source near a wall, the reflection is modeled as an image source:

$$\mathbf{s}' = \mathbf{s} - 2(\mathbf{s} \cdot \mathbf{n})\mathbf{n}$$

where $\mathbf{n}$ is the wall normal.

### Spherical Harmonics

Ambisonics uses real spherical harmonics $Y_{lm}(\theta, \phi)$:

$$Y_{00} = \frac{1}{2}\sqrt{\frac{1}{\pi}}$$
$$Y_{1,-1} = \frac{1}{2}\sqrt{\frac{3}{\pi}} \sin\theta \sin\phi$$
$$Y_{1,0} = \frac{1}{2}\sqrt{\frac{3}{\pi}} \cos\theta$$
$$Y_{1,1} = \frac{1}{2}\sqrt{\frac{3}{\pi}} \sin\theta \cos\phi$$

### Head-Related Transfer Functions

HRTFs encode the filtering effect of head, torso, and pinnae. The ITD (interaural time difference) is approximated by Woodworth's formula:

$$\Delta t = \frac{3a}{c}\sin\theta\cos\phi$$

where $a$ is head radius, $c$ is speed of sound, $\theta$ is azimuth, $\phi$ is elevation.

### Fresnel Diffraction

For aperture diffraction, the Fresnel number:

$$N = \frac{a^2}{\lambda z}$$

characterizes the transition between geometric optics ($N >> 1$) and wave diffraction ($N << 1$).

---

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Write tests for new features
4. Submit a pull request

See `CONTRIBUTING.md` for guidelines.

---

## License

This project is licensed under the MIT License - see `LICENSE` for details.

---

## Acknowledgments

Echo builds upon decades of acoustic research and borrows concepts from:

- **Allen & Berkley (1979)**: Image source method
- **Botteldooren (1995)**: Time-domain diffraction
- **Vorländer (1989)**: Auralization techniques
- **Gerzon (1975)**: Periphony and Ambisonics
- **ISO 3382**: Room acoustic parameter standards

---

## Citation

If you use Echo in your research, please cite:

```bibtex
@software{echo_acoustic_engine,
  title = {Echo: Acoustic Simulation Engine},
  author = {Moggan1337},
  version = {1.0.0},
  year = {2024},
  url = {https://github.com/moggan1337/Echo}
}
```

---

## Contact

- **GitHub Issues**: For bug reports and feature requests
- **Discussions**: For questions and community support

---

*Built with ❤️ for the acoustic simulation community*
