"""
Weather Derivatives Library

A comprehensive Python library for calculating and pricing weather derivatives.
Supports various input formats: (time, lon, lat), (time, site), or (time).

Main Components:
- WeatherData: Core weather data handling
- Derivatives: HDD, CDD, CAT, and other weather derivative calculations
- Parsers: Input data parsing for different formats
- Pricing: Valuation and pricing methods
"""

from .core.weather_data import WeatherData
from .derivatives.temperature import HDD, CDD, CAT, PAC
from .derivatives.precipitation import PrecipitationDerivative
from .derivatives.wind import WindDerivative
from .parsers.input_parser import WeatherInputParser
from .pricing.valuation import DerivativeValuation

__version__ = "1.0.0"
__author__ = "Weather Derivatives Team"

__all__ = [
    "WeatherData",
    "HDD",
    "CDD",
    "CAT",
    "PAC",
    "PrecipitationDerivative",
    "WindDerivative",
    "WeatherInputParser",
    "DerivativeValuation",
]
