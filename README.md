# Echo - Acoustic Simulation Engine

[![CI](https://github.com/moggan1337/Echo/actions/workflows/ci.yml/badge.svg)](https://github.com/moggan1337/Echo/actions/workflows/ci.yml)

<p align="center">
  <img src="assets/logo.png" alt="Echo Logo" width="200"/>
</p>

**Echo** is a comprehensive, open-source acoustic simulation engine written in Python. It provides state-of-the-art algorithms for simulating sound propagation in enclosed spaces, enabling realistic auralization and spatial audio rendering for applications ranging from architectural acoustics to virtual reality.

## 🎬 Demo
![Echo Demo](demo.gif)

*Acoustic simulation with ray-traced sound propagation*

## Screenshots
| Component | Preview |
|-----------|---------|
| Room Model | ![room](screenshots/room-model.png) |
| Sound Paths | ![paths](screenshots/sound-paths.png) |
| Impulse Response | ![ir](screenshots/impulse-response.png) |

## Visual Description
Room model displays 3D geometry with material annotations. Sound paths visualize acoustic rays bouncing off surfaces with color-coded reflections. Impulse response shows frequency-dependent decay curves.

---


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

### Example: Concert Hall Simulation

This example demonstrates simulating a large venue like a concert hall:

```python
# Concert hall dimensions: ~50m x 30m x 20m
hall = Room.dimensions(50, 30, 20)

# Apply concert hall materials
materials = {
    SurfaceType.FLOOR: Material("gymnasium_floor", 0.05, 0.10, 0.15, 0.20),
    SurfaceType.CEILING: Material("concrete", 0.02, 0.02, 0.03, 0.04),
    SurfaceType.WALL: Material("plaster_smooth", 0.02, 0.03, 0.03, 0.04),
}

# Set up sources (orchestra)
orchestra = SoundSource(position=[25, 15, 3], name="Orchestra")
orchestra.generate_sweep(duration=5.0)

# Set up receiver (audience position)
audience = Receiver(position=[25, 20, 1.2], orientation=0)

# Run simulation
config = SimulationConfig.from_preset("ultra")
config.duration = 4.0  # Longer RIR for large space
engine = AcousticEngine(config)
engine.set_room(hall)
engine.add_source(orchestra)
engine.add_receiver(audience)

rir = engine.compute_rir()
metrics = engine.analyze_acoustics()
```

### Example: Open Office Acoustic Analysis

Simulating a modern open-plan office with partitions:

```python
# Office layout with obstacles
from echo_engine.room import RoomBuilder

office = (RoomBuilder()
          .rectangular(20, 15, 3)
          .floor("carpet_light")
          .ceiling("acoustic_tile")
          .walls("plaster_smooth")
          .build())

# Add desk partitions as obstacles
office.add_obstacle(
    position=np.array([5, 5, 0]),
    size=np.array([1.2, 0.6, 1.1]),
    material="chipboard"
)

# Source at one end
speaker = SoundSource(
    position=[2, 7.5, 1.5],
    directivity=SourceDirectivity.cardioid(axis=(1, 0, 0))
)
speaker.generate_sweep(duration=2.0)

# Receiver in work area
worker = Receiver(position=[10, 7.5, 1.2])
```

### Example: Anechoic Chamber Comparison

Comparing room acoustics with an anechoic reference:

```python
# Real room
real_room = Room.dimensions(8, 6, 3)
real_room.set_material("floor", "concrete")
real_room.set_material("ceiling", "concrete")
real_room.set_material("walls", "concrete")

# Compute RIR for real room
engine_real = AcousticEngine(config)
engine_real.set_room(real_room)
rir_real = engine_real.compute_rir()

# Anechoic (open window) room for reference
anechoic = Room.dimensions(8, 6, 3)
anechoic.set_material("floor", "open_window")
anechoic.set_material("ceiling", "open_window")
anechoic.set_material("walls", "open_window")

# Compute RIR for anechoic room
engine_anechoic = AcousticEngine(config)
engine_anechoic.set_room(anechoic)
rir_anechoic = engine_anechoic.compute_rir()

# Analyze difference
real_t60 = compute_rt60(rir_real)
print(f"Real room T60: {real_t60:.2f}s")
print(f"Anechoic T60: {compute_rt60(rir_anechoic):.2f}s")
```

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

## Detailed Theory

### Geometric Acoustics

Geometric acoustics approximates sound as rays traveling in straight lines, reflecting off surfaces according to the law of reflection:

$$\theta_i = \theta_r$$

where $\theta_i$ is the angle of incidence and $\theta_r$ is the angle of reflection, measured from the surface normal.

#### Energy at Receiver

The acoustic energy arriving at a receiver from a ray path is:

$$E = \frac{E_0}{4\pi r^2} \prod_{i=1}^{n} (1 - \alpha_i) \cdot D(\theta_i)$$

where:
- $E_0$ is the source energy
- $r$ is the path length
- $\alpha_i$ is the absorption coefficient at the $i$-th reflection
- $D(\theta_i)$ is the source directivity factor

### Wave-Based Methods

Wave-based methods solve the acoustic wave equation directly, capturing phenomena that geometric acoustics cannot:

- **Diffraction**: Bending of waves around edges
- **Resonance**: Standing waves in enclosed spaces
- **Interference**: Constructive and destructive superposition

#### FDTD Algorithm

The explicit FDTD update for pressure:

$$p^{n+1}_{i,j,k} = 2p^n_{i,j,k} - p^{n-1}_{i,j,k} + \left(\frac{c\Delta t}{\Delta x}\right)^2 \nabla^2 p^n$$

Stability requires the Courant-Friedrichs-Lewy (CFL) condition:

$$\Delta t \leq \frac{\Delta x}{c\sqrt{3}}$$

#### Perfectly Matched Layer (PML)

PML boundaries absorb outgoing waves without reflection:

$$\frac{\partial p}{\partial t} + \sigma(x) p = -\frac{\partial u}{\partial x}$$

where $\sigma(x)$ is the absorption profile increasing toward the boundary.

### Room Acoustic Parameters

#### Reverberation Time (T60)

The time for sound to decay by 60 dB:

$$T_{60} = \frac{24 \ln(10)}{c \bar{\alpha}}$$

(Sabin) or:

$$T_{60} = \frac{55.2 V}{c S \ln(1-\bar{\alpha})}$$

(Eyring, for small absorption).

#### Clarity Index (C50, C80)

Ratio of early to late energy:

$$C_{t} = 10 \log_{10}\left(\frac{\int_0^t h^2(t) dt}{\int_t^\infty h^2(t) dt}\right)$$

C50 relates to speech intelligibility; C80 to musical clarity.

#### Definition (D50)

Percentage of early energy:

$$D_{50} = \frac{\int_0^{50ms} h^2(t) dt}{\int_0^{\infty} h^2(t) dt}$$

Values > 50% indicate good speech intelligibility.

#### Center Time (Ts)

First moment of the squared impulse response:

$$T_s = \frac{\int_0^{\infty} t \cdot h^2(t) dt}{\int_0^{\infty} h^2(t) dt}$$

Lower values indicate clearer sound.

### Spherical Harmonics Theory

Ambisonics encodes sound using spherical harmonic basis functions.

#### Real Spherical Harmonics

For order $n$ and degree $m$:

$$Y_{nm}(\theta, \phi) = \begin{cases}
\sqrt{2} N_{nm} \sin(|m|\phi) P_n^{|m|}(\cos\theta) & m < 0 \\
N_{n0} P_n^0(\cos\theta) & m = 0 \\
\sqrt{2} N_{nm} \cos(|m|\phi) P_n^{|m|}(\cos\theta) & m > 0
\end{cases}$$

where $P_n^m$ are associated Legendre polynomials and $N_{nm}$ is normalization.

#### Encoding

The Ambisonics signal for order $n$:

$$B_{nm} = \int_0^{2\pi} \int_0^{\pi} p(\theta, \phi) Y_{nm}^*(\theta, \phi) \sin\theta \, d\theta \, d\phi$$

### HRTF Fundamentals

Head-related transfer functions capture the spectral filtering of the head, torso, and pinnae.

#### Interaural Level Difference (ILD)

The level difference between ears due to head shadowing:

$$ILD(f, \theta) = 20 \log_{10}\frac{|H_{left}(f, \theta)|}{|H_{right}(f, \theta)|}$$

At high frequencies (> 2 kHz), ILD provides localization cues.

#### Interaural Time Difference (ITD)

The phase/time difference between ears:

$$ITD = \frac{a}{c}(\sin\theta + \cos\theta \cdot \arcsin(\frac{\sin\theta}{2}))$$

where $a$ is head radius and $c$ is speed of sound.

#### Pinna Cues

The pinnae create spectral notches and peaks that vary with elevation:

- Concha resonance: ~5 kHz
- Cavity resonance: ~10 kHz
- Timbral notch: 6-12 kHz (varies with elevation)

---

## Troubleshooting

### Common Issues

#### Out of Memory Errors

**Problem**: FDTD simulations require large amounts of memory.

**Solution**:
- Reduce grid resolution (increase `grid_resolution` in config)
- Use ray tracing instead of wave-based methods
- Enable chunked processing with streaming

```python
# Reduce memory usage
config = SimulationConfig.from_preset("medium")
config.grid_resolution = 0.1  # 10cm instead of 5cm
```

#### Slow Simulation Times

**Problem**: Simulation takes too long.

**Solution**:
- Use quality presets that match your needs
- Enable GPU acceleration if available
- Cache RIRs when computing multiple receivers

```python
# Faster simulation
config = SimulationConfig.from_preset("low")  # Quick preview
# Then upgrade for final render
config = SimulationConfig.from_preset("high")
```

#### Unstable FDTD Simulation

**Problem**: Results contain NaN or infinity values.

**Solution**:
- Reduce time step (increase CFL number)
- Check that grid resolution is uniform
- Verify material properties are physical

```python
# Fix stability
config = SimulationConfig()
config.courant_number = 0.2  # Reduce from default 0.3
```

#### Poor HRTF Quality

**Problem**: Binaural audio sounds unnatural.

**Solution**:
- Use measured HRTFs instead of synthetic model
- Load SOFA files for personalized HRTFs

```python
from echo_engine.receiver import HRTFProcessor
hrtf = HRTFProcessor()
hrtf.load_sofa_file("personal_hrtf.sofa")
```

### Performance Tips

1. **Use Quality Presets**: Match preset to development stage
2. **Cache RIRs**: Don't recompute unchanged configurations
3. **Batch Processing**: Process multiple sources together
4. **Parallel Execution**: Use multiple threads for ray tracing
5. **GPU Acceleration**: Enable with `config.enable_gpu = True`

### Debug Mode

Enable verbose output for troubleshooting:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Now run simulation with debug output
engine = AcousticEngine(config)
```

---

## Frequently Asked Questions

**Q: Can Echo simulate outdoor acoustics?**
A: Yes, but the room model must be modified. Use a ground plane with appropriate absorption and set room boundaries far from the source.

**Q: How accurate is the FDTD solver?**
A: FDTD provides physically accurate results limited by grid resolution. Use at least 10 points per wavelength for accurate high-frequency response.

**Q: Can I use custom HRTF measurements?**
A: Yes, load SOFA format files using `HRTFProcessor.load_sofa_file()`.

**Q: What's the maximum room size?**
A: Practically limited by memory. A 20m room at 5cm resolution needs ~64M grid points (~500MB).

**Q: Does Echo support non-rectangular rooms?**
A: Yes, define arbitrary polygon surfaces. Complex geometries may need mesh preprocessing.

**Q: Can I simulate moving sources?**
A: Yes, use the `DopplerProcessor` or update source positions between frames in animation.

**Q: How do I validate simulation results?**
A: Compare with measured RIRs, use standard test cases (ISO 3382), and verify against Sabine/Eyring predictions.

---

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Write tests for new features
4. Submit a pull request

### Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/Echo.git
cd Echo

# Install in development mode
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=echo_engine --cov-report=html
```

### Code Style

- Follow PEP 8
- Use type hints where possible
- Write docstrings for all public functions
- Add tests for new features

### Pull Request Guidelines

- Reference related issues
- Update documentation as needed
- Ensure all tests pass
- Add entry to CHANGELOG.md

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
