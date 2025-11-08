"""Setup script for weather derivatives library."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="weather-derivatives",
    version="1.0.0",
    author="Weather Derivatives Team",
    author_email="",
    description="A comprehensive Python library for calculating and pricing weather derivatives",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/libwd",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Science/Research",
        "Topic :: Office/Business :: Financial",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.19.0",
        "scipy>=1.5.0",
        "python-dateutil>=2.8.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.10.0",
            "black>=20.8b1",
            "flake8>=3.8.0",
        ],
    },
    keywords="weather derivatives, finance, risk management, temperature, precipitation, wind, HDD, CDD",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/libwd/issues",
        "Source": "https://github.com/yourusername/libwd",
    },
)
