"""
Acoustic Material Database
Contains absorption coefficients, scattering coefficients, and
transmission properties for common building materials.
"""

import numpy as np
from typing import Dict, Optional, Callable, Tuple, List
from dataclasses import dataclass
import json
from pathlib import Path


@dataclass
class Material:
    """
    Represents an acoustic material with frequency-dependent properties.
    
    Materials define how sound interacts with surfaces through:
    - Absorption: Sound energy converted to heat
    - Scattering: Random redirection of sound
    - Transmission: Sound passing through the material
    
    Example:
        >>> concrete = Material(
        ...     name="concrete",
        ...     absorption_125=0.01,
        ...     absorption_250=0.01,
        ...     absorption_500=0.02,
        ...     absorption_1000=0.02,
        ...     absorption_2000=0.03,
        ...     absorption_4000=0.04
        ... )
    """
    name: str
    
    # Absorption coefficients at octave bands
    absorption_125: float = 0.05  # 125 Hz
    absorption_250: float = 0.05  # 250 Hz
    absorption_500: float = 0.05  # 500 Hz
    absorption_1000: float = 0.05  # 1 kHz
    absorption_2000: float = 0.05  # 2 kHz
    absorption_4000: float = 0.05  # 4 kHz
    
    # Scattering coefficient (can be frequency-dependent)
    scattering_500: float = 0.05
    scattering_1000: float = 0.05
    scattering_2000: float = 0.05
    
    # Transmission properties
    transmission_125: float = 0.0
    transmission_250: float = 0.0
    transmission_500: float = 0.0
    transmission_1000: float = 0.0
    transmission_2000: float = 0.0
    transmission_4000: float = 0.0
    
    # Physical properties
    density: float = 100.0  # kg/m³
    thickness: float = 0.1  # meters
    porosity: float = 0.0  # 0-1
    flow_resistivity: float = 10000.0  # Rayls/m (for porous materials)
    
    # Description
    description: str = ""
    
    # Reference
    source: str = "generic"
    
    def __post_init__(self):
        """Validate material properties."""
        for attr in ['absorption_125', 'absorption_250', 'absorption_500', 
                     'absorption_1000', 'absorption_2000', 'absorption_4000']:
            value = getattr(self, attr)
            if not 0 <= value <= 1:
                setattr(self, attr, np.clip(value, 0, 1))
                
        for attr in ['scattering_500', 'scattering_1000', 'scattering_2000',
                     'transmission_125', 'transmission_250', 'transmission_500',
                     'transmission_1000', 'transmission_2000', 'transmission_4000']:
            value = getattr(self, attr)
            if not 0 <= value <= 1:
                setattr(self, attr, np.clip(value, 0, 1))
    
    @property
    def absorption_bands(self) -> np.ndarray:
        """Get absorption coefficients as array [125, 250, 500, 1000, 2000, 4000] Hz."""
        return np.array([
            self.absorption_125,
            self.absorption_250,
            self.absorption_500,
            self.absorption_1000,
            self.absorption_2000,
            self.absorption_4000
        ])
    
    @property
    def frequencies(self) -> np.ndarray:
        """Standard octave band frequencies."""
        return np.array([125, 250, 500, 1000, 2000, 4000])
    
    def absorption(self, frequency: float) -> float:
        """
        Get absorption coefficient at a specific frequency.
        
        Args:
            frequency: Frequency in Hz
            
        Returns:
            Absorption coefficient (0-1)
        """
        return np.interp(
            frequency,
            self.frequencies,
            self.absorption_bands,
            left=self.absorption_125,
            right=self.absorption_4000
        )
    
    def scattering(self, frequency: float) -> float:
        """
        Get scattering coefficient at a specific frequency.
        
        Args:
            frequency: Frequency in Hz
            
        Returns:
            Scattering coefficient (0-1)
        """
        # Interpolate between defined bands
        scatters = np.array([
            self.scattering_500,
            self.scattering_1000,
            self.scattering_2000
        ])
        freqs = np.array([500, 1000, 2000])
        
        if frequency < 500:
            return self.scattering_500
        elif frequency > 2000:
            return self.scattering_2000
        else:
            return np.interp(frequency, freqs, scatters)
    
    def transmission(self, frequency: float) -> float:
        """
        Get transmission coefficient at a specific frequency.
        
        Args:
            frequency: Frequency in Hz
            
        Returns:
            Transmission coefficient (0-1)
        """
        trans = np.array([
            self.transmission_125,
            self.transmission_250,
            self.transmission_500,
            self.transmission_1000,
            self.transmission_2000,
            self.transmission_4000
        ])
        
        return np.interp(
            frequency,
            self.frequencies,
            trans,
            left=self.transmission_125,
            right=self.transmission_4000
        )
    
    def reflection(self, frequency: float) -> float:
        """
        Get reflection coefficient at a specific frequency.
        
        Reflection = 1 - Absorption - Transmission
        
        Args:
            frequency: Frequency in Hz
            
        Returns:
            Reflection coefficient (0-1)
        """
        alpha = self.absorption(frequency)
        tau = self.transmission(frequency)
        return max(0, 1 - alpha - tau)
    
    def get_filter_coefficients(
        self,
        frequency: float,
        sample_rate: float = 48000.0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get IIR filter coefficients for material frequency response.
        
        Args:
            frequency: Center frequency
            sample_rate: Sample rate
            
        Returns:
            Tuple of (b, a) filter coefficients
        """
        # Simple resonant model for material absorption
        alpha = self.absorption(frequency)
        
        # Resonant frequency
        w0 = 2 * np.pi * frequency
        Q = 1 / (alpha + 0.01)  # Quality factor based on absorption
        
        # Lowpass filter approximating absorption
        b = [1.0 - alpha]
        a = [1.0, alpha - 1.0]
        
        return np.array(b), np.array(a)
    
    def to_dict(self) -> Dict:
        """Convert material to dictionary."""
        return {
            "name": self.name,
            "absorption": {
                "125": self.absorption_125,
                "250": self.absorption_250,
                "500": self.absorption_500,
                "1000": self.absorption_1000,
                "2000": self.absorption_2000,
                "4000": self.absorption_4000
            },
            "scattering": {
                "500": self.scattering_500,
                "1000": self.scattering_1000,
                "2000": self.scattering_2000
            },
            "transmission": {
                "125": self.transmission_125,
                "250": self.transmission_250,
                "500": self.transmission_500,
                "1000": self.transmission_1000,
                "2000": self.transmission_2000,
                "4000": self.transmission_4000
            },
            "physical": {
                "density": self.density,
                "thickness": self.thickness,
                "porosity": self.porosity,
                "flow_resistivity": self.flow_resistivity
            },
            "description": self.description,
            "source": self.source
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "Material":
        """Create material from dictionary."""
        absorption = data.get("absorption", {})
        scattering = data.get("scattering", {})
        transmission = data.get("transmission", {})
        physical = data.get("physical", {})
        
        return cls(
            name=data.get("name", "Unknown"),
            absorption_125=absorption.get("125", 0.05),
            absorption_250=absorption.get("250", 0.05),
            absorption_500=absorption.get("500", 0.05),
            absorption_1000=absorption.get("1000", 0.05),
            absorption_2000=absorption.get("2000", 0.05),
            absorption_4000=absorption.get("4000", 0.05),
            scattering_500=scattering.get("500", 0.05),
            scattering_1000=scattering.get("1000", 0.05),
            scattering_2000=scattering.get("2000", 0.05),
            transmission_125=transmission.get("125", 0.0),
            transmission_250=transmission.get("250", 0.0),
            transmission_500=transmission.get("500", 0.0),
            transmission_1000=transmission.get("1000", 0.0),
            transmission_2000=transmission.get("2000", 0.0),
            transmission_4000=transmission.get("4000", 0.0),
            density=physical.get("density", 100.0),
            thickness=physical.get("thickness", 0.1),
            porosity=physical.get("porosity", 0.0),
            flow_resistivity=physical.get("flow_resistivity", 10000.0),
            description=data.get("description", ""),
            source=data.get("source", "generic")
        )
    
    def __repr__(self) -> str:
        return f"Material('{self.name}', α@1kHz={self.absorption_1000:.2f})"


class MaterialDatabase:
    """
    Database of common acoustic materials.
    
    Contains pre-defined materials with measured acoustic properties
    from standards organizations (ISO 11654, ASTM E90, etc.).
    
    Example:
        >>> db = MaterialDatabase()
        >>> concrete = db.get("concrete")
        >>> carpet = db.get("carpet_heavy")
        >>> glass = db.get("glass")
    """
    
    def __init__(self, custom_materials: Optional[Dict[str, Material]] = None):
        """Initialize the material database."""
        self._materials: Dict[str, Material] = {}
        self._initialize_standard_materials()
        
        if custom_materials:
            self._materials.update(custom_materials)
    
    def _initialize_standard_materials(self) -> None:
        """Initialize standard acoustic materials."""
        
        # Concrete and Masonry
        self.add(Material(
            name="concrete",
            absorption_125=0.01, absorption_250=0.01, absorption_500=0.02,
            absorption_1000=0.02, absorption_2000=0.03, absorption_4000=0.04,
            scattering_500=0.1, scattering_1000=0.1, scattering_2000=0.15,
            density=2300, thickness=0.15,
            description="Poured concrete, unpolished",
            source="ISO 11654"
        ))
        
        self.add(Material(
            name="concrete_block_heavy",
            absorption_125=0.02, absorption_250=0.02, absorption_500=0.03,
            absorption_1000=0.03, absorption_2000=0.04, absorption_4000=0.05,
            scattering_500=0.15, scattering_1000=0.15, scattering_2000=0.2,
            density=2100, thickness=0.1,
            description=" heavyweight concrete block, unfinished",
            source="ISO 11654"
        ))
        
        self.add(Material(
            name="brick",
            absorption_125=0.02, absorption_250=0.03, absorption_500=0.04,
            absorption_1000=0.05, absorption_2000=0.06, absorption_4000=0.07,
            scattering_500=0.12, scattering_1000=0.12, scattering_2000=0.15,
            density=1800, thickness=0.1,
            description="Face brick",
            source="ISO 11654"
        ))
        
        self.add(Material(
            name="glass",
            absorption_125=0.03, absorption_250=0.03, absorption_500=0.03,
            absorption_1000=0.02, absorption_2000=0.02, absorption_4000=0.02,
            scattering_500=0.05, scattering_1000=0.05, scattering_2000=0.05,
            transmission_125=0.2, transmission_250=0.15, transmission_500=0.1,
            transmission_1000=0.08, transmission_2000=0.05, transmission_4000=0.02,
            density=2500, thickness=0.006,
            description="Window glass, 6mm",
            source="ASTM E90"
        ))
        
        self.add(Material(
            name="glass_heavy",
            absorption_125=0.02, absorption_250=0.02, absorption_500=0.02,
            absorption_1000=0.02, absorption_2000=0.02, absorption_4000=0.02,
            scattering_500=0.05, scattering_1000=0.05, scattering_2000=0.05,
            transmission_125=0.25, transmission_250=0.2, transmission_500=0.15,
            transmission_1000=0.1, transmission_2000=0.08, transmission_4000=0.05,
            density=2700, thickness=0.01,
            description="Plate glass, 10mm",
            source="ASTM E90"
        ))
        
        # Wood
        self.add(Material(
            name="hardwood_floor",
            absorption_125=0.10, absorption_250=0.11, absorption_500=0.10,
            absorption_1000=0.08, absorption_2000=0.06, absorption_4000=0.06,
            scattering_500=0.1, scattering_1000=0.1, scattering_2000=0.1,
            density=700, thickness=0.02,
            description="Hardwood flooring on joists",
            source="ISO 11654"
        ))
        
        self.add(Material(
            name="plywood_panel",
            absorption_125=0.20, absorption_250=0.22, absorption_500=0.15,
            absorption_1000=0.10, absorption_2000=0.08, absorption_4000=0.06,
            scattering_500=0.08, scattering_1000=0.08, scattering_2000=0.1,
            density=540, thickness=0.018,
            description="Plywood panel, 18mm",
            source="ISO 11654"
        ))
        
        self.add(Material(
            name="chipboard",
            absorption_125=0.10, absorption_250=0.15, absorption_500=0.15,
            absorption_1000=0.10, absorption_2000=0.08, absorption_4000=0.06,
            scattering_500=0.1, scattering_1000=0.1, scattering_2000=0.12,
            density=650, thickness=0.018,
            description="Chipboard particle board",
            source="generic"
        ))
        
        # Plaster and gypsum
        self.add(Material(
            name="plaster_smooth",
            absorption_125=0.02, absorption_250=0.03, absorption_500=0.03,
            absorption_1000=0.04, absorption_2000=0.04, absorption_4000=0.05,
            scattering_500=0.05, scattering_1000=0.05, scattering_2000=0.05,
            density=1200, thickness=0.013,
            description="Smooth plaster or gypsum board",
            source="ISO 11654"
        ))
        
        self.add(Material(
            name="plaster_perforated",
            absorption_125=0.20, absorption_250=0.40, absorption_500=0.60,
            absorption_1000=0.50, absorption_2000=0.40, absorption_4000=0.30,
            scattering_500=0.3, scattering_1000=0.3, scattering_2000=0.3,
            density=800, thickness=0.013,
            description="Perforated acoustic plaster",
            source="generic"
        ))
        
        # Floor coverings
        self.add(Material(
            name="carpet_heavy",
            absorption_125=0.02, absorption_250=0.06, absorption_500=0.15,
            absorption_1000=0.35, absorption_2000=0.60, absorption_4000=0.65,
            scattering_500=0.2, scattering_1000=0.2, scattering_2000=0.2,
            density=30, thickness=0.012,
            description="Heavy carpet on concrete",
            source="ISO 11654"
        ))
        
        self.add(Material(
            name="carpet_light",
            absorption_125=0.01, absorption_250=0.03, absorption_500=0.10,
            absorption_1000=0.25, absorption_2000=0.40, absorption_4000=0.45,
            scattering_500=0.15, scattering_1000=0.15, scattering_2000=0.15,
            density=15, thickness=0.006,
            description="Light carpet on concrete",
            source="ISO 11654"
        ))
        
        self.add(Material(
            name="linoleum",
            absorption_125=0.02, absorption_250=0.03, absorption_500=0.03,
            absorption_1000=0.03, absorption_2000=0.03, absorption_4000=0.03,
            scattering_500=0.05, scattering_1000=0.05, scattering_2000=0.05,
            density=1100, thickness=0.003,
            description="Linoleum on concrete",
            source="generic"
        ))
        
        # Ceiling treatments
        self.add(Material(
            name="acoustic_tile",
            absorption_125=0.05, absorption_250=0.15, absorption_500=0.40,
            absorption_1000=0.65, absorption_2000=0.75, absorption_4000=0.80,
            scattering_500=0.4, scattering_1000=0.4, scattering_2000=0.4,
            density=50, thickness=0.025,
            description="Suspended acoustic ceiling tile",
            source="ISO 11654"
        ))
        
        self.add(Material(
            name="acoustic_panel",
            absorption_125=0.10, absorption_250=0.25, absorption_500=0.55,
            absorption_1000=0.75, absorption_2000=0.70, absorption_4000=0.65,
            scattering_500=0.35, scattering_1000=0.35, scattering_2000=0.4,
            density=80, thickness=0.05,
            description="Wall-mounted acoustic panel, 50mm",
            source="generic"
        ))
        
        self.add(Material(
            name="acoustic_foam",
            absorption_125=0.03, absorption_250=0.10, absorption_500=0.35,
            absorption_1000=0.60, absorption_2000=0.80, absorption_4000=0.85,
            scattering_500=0.3, scattering_1000=0.3, scattering_2000=0.3,
            porosity=0.9, flow_resistivity=5000,
            density=25, thickness=0.03,
            description="Open-cell acoustic foam",
            source="generic"
        ))
        
        # Upholstery and furniture
        self.add(Material(
            name="curtain_heavy",
            absorption_125=0.10, absorption_250=0.25, absorption_500=0.45,
            absorption_1000=0.65, absorption_2000=0.75, absorption_4000=0.80,
            scattering_500=0.5, scattering_1000=0.5, scattering_2000=0.5,
            density=50, thickness=0.003,
            description="Heavy velour curtain, draped",
            source="ISO 11654"
        ))
        
        self.add(Material(
            name="curtain_light",
            absorption_125=0.05, absorption_250=0.08, absorption_500=0.15,
            absorption_1000=0.30, absorption_2000=0.45, absorption_4000=0.55,
            scattering_500=0.4, scattering_1000=0.4, scattering_2000=0.4,
            density=25, thickness=0.002,
            description="Light curtain, loosely draped",
            source="ISO 11654"
        ))
        
        self.add(Material(
            name="upholstered_furniture",
            absorption_125=0.10, absorption_250=0.20, absorption_500=0.35,
            absorption_1000=0.50, absorption_2000=0.55, absorption_4000=0.60,
            scattering_500=0.4, scattering_1000=0.4, scattering_2000=0.4,
            density=100, thickness=0.1,
            description="Fully upholstered seating",
            source="ISO 11654"
        ))
        
        self.add(Material(
            name="office_chair",
            absorption_125=0.05, absorption_250=0.10, absorption_500=0.20,
            absorption_1000=0.35, absorption_2000=0.45, absorption_4000=0.50,
            scattering_500=0.35, scattering_1000=0.35, scattering_2000=0.35,
            density=60, thickness=0.08,
            description="Typical office task chair",
            source="generic"
        ))
        
        # Special materials
        self.add(Material(
            name="water_surface",
            absorption_125=0.01, absorption_250=0.01, absorption_500=0.01,
            absorption_1000=0.02, absorption_2000=0.02, absorption_4000=0.03,
            scattering_500=0.02, scattering_1000=0.02, scattering_2000=0.02,
            density=1000,
            description="Water surface (pool, fountain)",
            source="generic"
        ))
        
        self.add(Material(
            name="audience",
            absorption_125=0.25, absorption_250=0.40, absorption_500=0.60,
            absorption_1000=0.75, absorption_2000=0.80, absorption_4000=0.85,
            scattering_500=0.6, scattering_1000=0.6, scattering_2000=0.6,
            density=80, thickness=0.5,
            description="Seated audience (theater type seating)",
            source="ISO 11654"
        ))
        
        self.add(Material(
            name="open_window",
            absorption_125=0.0, absorption_250=0.0, absorption_500=0.0,
            absorption_1000=0.0, absorption_2000=0.0, absorption_4000=0.0,
            scattering_500=0.0, scattering_1000=0.0, scattering_2000=0.0,
            transmission_125=1.0, transmission_250=1.0, transmission_500=1.0,
            transmission_1000=1.0, transmission_2000=1.0, transmission_4000=1.0,
            description="Fully open window",
            source="ideal"
        ))
        
        # Metal (for industrial environments)
        self.add(Material(
            name="metal_surface",
            absorption_125=0.02, absorption_250=0.02, absorption_500=0.02,
            absorption_1000=0.03, absorption_2000=0.03, absorption_4000=0.04,
            scattering_500=0.08, scattering_1000=0.08, scattering_2000=0.1,
            density=2700,
            description="Smooth metal surface",
            source="generic"
        ))
        
        # Sport surfaces
        self.add(Material(
            name="gymnasium_floor",
            absorption_125=0.05, absorption_250=0.10, absorption_500=0.15,
            absorption_1000=0.20, absorption_2000=0.20, absorption_4000=0.20,
            scattering_500=0.15, scattering_1000=0.15, scattering_2000=0.15,
            density=500, thickness=0.03,
            description="Wood gymnasium floor",
            source="generic"
        ))
        
    def add(self, material: Material) -> None:
        """Add a material to the database."""
        self._materials[material.name.lower()] = material
        
    def get(self, name: str) -> Optional[Material]:
        """Get a material by name (case-insensitive)."""
        return self._materials.get(name.lower())
    
    def search(self, query: str) -> List[Material]:
        """Search materials by name substring."""
        query = query.lower()
        return [
            mat for name, mat in self._materials.items()
            if query in name
        ]
    
    def list_all(self) -> List[str]:
        """List all material names."""
        return list(self._materials.keys())
    
    def export(self, path: Union[str, Path]) -> None:
        """Export database to JSON file."""
        data = {
            name: mat.to_dict()
            for name, mat in self._materials.items()
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
            
    @classmethod
    def load(cls, path: Union[str, Path]) -> "MaterialDatabase":
        """Load database from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
            
        materials = {
            name: Material.from_dict(mat_data)
            for name, mat_data in data.items()
        }
        
        return cls(custom_materials=materials)


def compute_absorption_delany_basmey(
    frequency: float,
    flow_resistivity: float,
    porosity: float = 0.9,
    thickness: float = 0.05,
    density: float = 50.0
) -> float:
    """
    Compute absorption coefficient using Delany-Basmey model.
    
    This semi-empirical model is valid for porous absorbers.
    
    Args:
        frequency: Frequency in Hz
        flow_resistivity: Flow resistivity in Rayls/m
        porosity: Material porosity (0-1)
        thickness: Material thickness in meters
        density: Material density in kg/m³
        
    Returns:
        Absorption coefficient
    """
    # Characteristic impedance and wavenumber coefficients
    C1 = 1 + 9.08 * (frequency / flow_resistivity)**0.75
    C2 = -11.9 * (frequency / flow_resistivity)**0.73
    
    Z = density * 343 * (C1 - 1j * C2)
    
    k1 = 11.9 * (frequency / flow_resistivity)**0.70
    k2 = -10.3 * (frequency / flow_resistivity)**0.59
    
    k = frequency / 343 * (k1 - 1j * k2)
    
    # Reflection coefficient
    z_r = Z / (density * 343)
    z_s = np.tanh(1j * 2 * np.pi * frequency / 343 * thickness * k) / k
    
    r = (z_s - z_r) / (z_s + z_r)
    
    alpha = 1 - np.abs(r)**2
    
    return max(0, min(1, alpha))


def compute_diffraction_coefficient(
    frequency: float,
    source_angle: float,
    obstacle_height: float,
    distance: float
) -> float:
    """
    Compute Fresnel diffraction coefficient.
    
    Args:
        frequency: Frequency in Hz
        source_angle: Angle from obstacle edge in radians
        obstacle_height: Height of obstacle in meters
        distance: Distance from source to edge in meters
        
    Returns:
        Diffraction coefficient (0-1)
    """
    wavelength = 343 / frequency
    
    # Fresnel number
    N = 2 * obstacle_height**2 / (wavelength * distance + 1e-10)
    
    if N < 0.1:
        return 1.0  # No diffraction
    elif N > 10:
        return 0.0  # Full shadow
        
    # Fresnel integrals
    from scipy.special import fresnel
    
    # Standard diffraction parameter
    nu = np.sqrt(2 * N) * np.cos(source_angle + np.pi/4)
    
    # Approximate diffraction coefficient
    C, S = fresnel(nu)
    
    D = 0.5 + 0.5 * (C + S)**2 / (C**2 + S**2)
    
    return D
