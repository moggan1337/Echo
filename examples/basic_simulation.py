"""
Echo - Basic Acoustic Simulation Example

This example demonstrates the fundamental usage of the Echo acoustic
simulation engine to compute a room impulse response and analyze room acoustics.
"""

import numpy as np
import matplotlib.pyplot as plt
from echo_engine import (
    AcousticEngine, SimulationConfig, SimulationMethod, OutputFormat,
    Room, RoomBuilder,
    SoundSource, SourceDirectivity,
    Receiver,
    MaterialDatabase
)


def main():
    print("Echo - Basic Acoustic Simulation")
    print("=" * 50)
    
    # ============================================
    # Step 1: Create Simulation Configuration
    # ============================================
    print("\n[1/6] Creating simulation configuration...")
    
    config = SimulationConfig.from_preset("medium")
    config.sample_rate = 48000
    config.duration = 2.0  # 2 second RIR
    config.output_format = OutputFormat.BINAURAL
    config.method = SimulationMethod.HYBRID
    
    print(f"    Sample rate: {config.sample_rate} Hz")
    print(f"    Duration: {config.duration} s")
    print(f"    Quality preset: {config.quality_preset}")
    
    # ============================================
    # Step 2: Define Room Geometry
    # ============================================
    print("\n[2/6] Defining room geometry...")
    
    # Method 1: Simple rectangular room
    # room = Room.dimensions(length=12, width=10, height=3.5)
    
    # Method 2: Using RoomBuilder for more control
    room = (RoomBuilder()
            .rectangular(12, 10, 3.5)
            .floor("hardwood_floor")
            .ceiling("acoustic_tile")
            .walls("plaster_smooth")
            .atmospheric_conditions(temperature=20, humidity=50)
            .build())
    
    print(f"    Room dimensions: {room.dimensions}")
    print(f"    Room volume: {room.volume:.1f} m³")
    print(f"    Surface area: {room.surface_area:.1f} m²")
    
    # ============================================
    # Step 3: Create Sound Source
    # ============================================
    print("\n[3/6] Creating sound source...")
    
    # Source with cardioid directivity (typical speaker)
    source = SoundSource(
        position=[3.0, 5.0, 1.5],  # Near front wall, elevated
        name="Piano",
        directivity=SourceDirectivity.cardioid(axis=(0, -1, 0))
    )
    
    # Generate a frequency sweep signal for testing
    source.generate_sweep(
        duration=1.0,
        sample_rate=config.sample_rate,
        freq_start=20,
        freq_end=20000,
        sweep_type="exponential"
    )
    
    print(f"    Position: {source.position}")
    print(f"    Directivity: {source.directivity.pattern.value}")
    print(f"    Signal duration: {len(source.signal) / config.sample_rate:.2f} s")
    
    # ============================================
    # Step 4: Create Receiver (Listener)
    # ============================================
    print("\n[4/6] Creating receiver...")
    
    receiver = Receiver(
        position=[6.0, 5.0, 1.2],  # Center of room, seated height
        orientation=0,  # Facing front (+X direction)
        elevation=0,
        name="Listener",
        enable_hrtf=True
    )
    
    print(f"    Position: {receiver.position}")
    print(f"    Orientation: {receiver.orientation}°")
    
    # ============================================
    # Step 5: Initialize and Run Simulation
    # ============================================
    print("\n[5/6] Running acoustic simulation...")
    
    # Create engine and configure
    engine = AcousticEngine(config)
    engine.set_room(room)
    engine.add_source(source)
    engine.add_receiver(receiver)
    
    # Compute room impulse response
    print("    Computing RIR...")
    rir = engine.compute_rir()
    print(f"    RIR length: {len(rir)} samples ({len(rir)/config.sample_rate:.2f} s)")
    
    # Analyze acoustics
    print("    Analyzing acoustic parameters...")
    metrics = engine.analyze_acoustics()
    
    # ============================================
    # Step 6: Results and Visualization
    # ============================================
    print("\n[6/6] Results:")
    print("-" * 50)
    
    # Extract metrics
    t60_key = list(metrics["reverberation_time"].keys())[0]
    c80_key = list(metrics["clarity_index"].keys())[0]
    d50_key = list(metrics["definition"].keys())[0]
    center_key = list(metrics["center_time"].keys())[0]
    
    print(f"\n  Room Parameters:")
    print(f"    Volume: {room.volume:.1f} m³")
    print(f"    Surface Area: {room.surface_area:.1f} m²")
    print(f"    Temperature: {room.temperature}°C")
    print(f"    Humidity: {room.humidity}%")
    
    print(f"\n  Acoustic Metrics:")
    print(f"    T60 (reverberation time): {metrics['reverberation_time'][t60_key]:.2f} s")
    print(f"    C80 (clarity, 80ms): {metrics['clarity_index'][c80_key]['C80']:.1f} dB")
    print(f"    C50 (clarity, 50ms): {metrics['clarity_index'][c80_key]['C50']:.1f} dB")
    print(f"    D50 (definition): {metrics['definition'][d50_key]*100:.1f}%")
    print(f"    Center time: {metrics['center_time'][center_key]*1000:.1f} ms")
    
    # ============================================
    # Visualization
    # ============================================
    try:
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Echo Acoustic Simulation Results", fontsize=14, fontweight='bold')
        
        # Plot 1: RIR
        t = np.arange(len(rir)) / config.sample_rate
        axes[0, 0].plot(t, rir, 'b-', linewidth=0.5)
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Amplitude')
        axes[0, 0].set_title('Room Impulse Response')
        axes[0, 0].set_xlim([0, min(config.duration, 0.5)])
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: RIR envelope (Schroeder curve)
        energy = np.cumsum(rir[::-1]**2)[::-1]
        energy_db = 10 * np.log10(energy / (np.max(energy) + 1e-10))
        axes[0, 1].plot(t, energy_db, 'r-', linewidth=1)
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('Energy (dB)')
        axes[0, 1].set_title('Energy Decay Curve (Schroeder)')
        axes[0, 1].set_xlim([0, min(config.duration, 1.0)])
        axes[0, 1].set_ylim([-80, 5])
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Frequency response
        fftrir = np.fft.rfft(rir)
        freq = np.fft.rfftfreq(len(rir), 1/config.sample_rate)
        axes[1, 0].semilogx(freq, 20*np.log10(np.abs(fftrir) + 1e-10), 'g-', linewidth=0.5)
        axes[1, 0].set_xlabel('Frequency (Hz)')
        axes[1, 0].set_ylabel('Magnitude (dB)')
        axes[1, 0].set_title('RIR Frequency Response')
        axes[1, 0].set_xlim([20, 20000])
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Metrics summary
        axes[1, 1].axis('off')
        
        summary_text = f"""
        Acoustic Analysis Summary
        ─────────────────────────
        
        Room: {room.dimensions[0]:.1f} × {room.dimensions[1]:.1f} × {room.dimensions[2]:.1f} m
        
        Source: {source.name}
          Position: {source.position}
          Directivity: {source.directivity.pattern.value}
        
        Receiver: {receiver.name}
          Position: {receiver.position}
          Orientation: {receiver.orientation}°
        
        Acoustic Metrics:
          T60: {metrics['reverberation_time'][t60_key]:.2f} s
          C80: {metrics['clarity_index'][c80_key]['C80']:.1f} dB
          D50: {metrics['definition'][d50_key]*100:.1f}%
          Center Time: {metrics['center_time'][center_key]*1000:.1f} ms
        """
        
        axes[1, 1].text(0.1, 0.9, summary_text, transform=axes[1, 1].transAxes,
                        fontsize=10, verticalalignment='top',
                        fontfamily='monospace')
        
        plt.tight_layout()
        plt.savefig('acoustic_simulation_results.png', dpi=150, bbox_inches='tight')
        print("\n  Visualization saved to 'acoustic_simulation_results.png'")
        
    except ImportError:
        print("\n  (matplotlib not available for visualization)")
    
    print("\n" + "=" * 50)
    print("Simulation complete!")
    
    return {
        "rir": rir,
        "metrics": metrics,
        "room": room,
        "source": source,
        "receiver": receiver
    }


if __name__ == "__main__":
    results = main()
