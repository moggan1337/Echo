"""
Echo - Acoustic Simulation Engine
Setup configuration
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="echo-engine",
    version="1.0.0",
    author="Moggan1337",
    author_email="moggan1337@echo-engine.dev",
    description="Comprehensive acoustic simulation engine for room acoustics and spatial audio",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/moggan1337/Echo",
    packages=find_packages(exclude=["tests", "tests.*", "examples", "docs"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Multimedia :: Sound/Audio",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Acoustics",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
        "gpu": [
            "numba>=0.55.0",
        ],
        "visualization": [
            "matplotlib>=3.5.0",
            "pyvista>=0.34.0",
        ],
    },
    keywords=[
        "acoustics",
        "room acoustics",
        "auralization",
        "spatial audio",
        "HRTF",
        "Ambisonics",
        "ray tracing",
        "FDTD",
        "wave equation",
        "audio simulation",
        "impulse response",
    ],
    project_urls={
        "Bug Reports": "https://github.com/moggan1337/Echo/issues",
        "Source": "https://github.com/moggan1337/Echo",
        "Documentation": "https://github.com/moggan1337/Echo#readme",
    },
)
