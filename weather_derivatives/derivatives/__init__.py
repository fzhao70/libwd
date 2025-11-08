"""Weather derivatives calculation module."""

from .temperature import HDD, CDD, CAT, PAC
from .precipitation import PrecipitationDerivative
from .wind import WindDerivative
from .snow import SnowDerivative
from .frost import FrostDerivative
from .humidity import HumidityDerivative
from .solar import SolarDerivative

__all__ = [
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
]
