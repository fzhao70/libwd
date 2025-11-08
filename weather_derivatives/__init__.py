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
from .derivatives.snow import SnowDerivative
from .derivatives.frost import FrostDerivative
from .derivatives.humidity import HumidityDerivative
from .derivatives.solar import SolarDerivative
from .parsers.input_parser import WeatherInputParser
from .pricing.valuation import DerivativeValuation

__version__ = "2.0.0"
__author__ = "Weather Derivatives Team"

__all__ = [
    "WeatherData",
    "HDD",
    "CDD",
    "CAT",
    "PAC",
    "PrecipitationDerivative",
    "WindDerivative",
    "SnowDerivative",
    "FrostDerivative",
    "HumidityDerivative",
    "SolarDerivative",
    "WeatherInputParser",
    "DerivativeValuation",
]
