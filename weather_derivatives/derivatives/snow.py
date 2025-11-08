"""
Snow-based weather derivatives.

Implements derivatives based on snowfall and snow depth.
Used in winter sports, transportation, and energy sectors.
"""

from typing import List, Optional, Union
import numpy as np
from ..core.weather_data import WeatherData


class SnowDerivative:
    """
    Snow-based derivative calculator.

    Supports snow derivatives including:
    - Total snowfall accumulation
    - Snow days counting
    - Snow depth derivatives
    - First/last snow date
    """

    def __init__(
        self,
        weather_data: WeatherData,
        snow_day_threshold: float = 1.0
    ):
        """
        Initialize snow derivative.

        Args:
            weather_data: WeatherData object containing snowfall/snow depth data
            snow_day_threshold: Minimum snowfall to count as snow day (cm)
        """
        self.weather_data = weather_data
        self.threshold = snow_day_threshold

        # Validate that snow data exists
        if "snowfall" not in weather_data.data and "snow_depth" not in weather_data.data:
            raise ValueError("Weather data must contain snowfall or snow_depth information")

    def calculate_total_snowfall(self) -> float:
        """
        Calculate total snowfall over the period.

        Returns:
            Total snowfall in cm
        """
        if "snowfall" not in self.weather_data.data:
            raise ValueError("No snowfall data available")

        snowfall = self.weather_data.get_data("snowfall")

        if isinstance(snowfall, (int, float)):
            return snowfall

        return sum(snowfall)

    def calculate_snow_days(self) -> int:
        """
        Calculate number of days with snowfall above threshold.

        Returns:
            Number of snow days
        """
        if "snowfall" not in self.weather_data.data:
            raise ValueError("No snowfall data available")

        snowfall = self.weather_data.get_data("snowfall")

        if isinstance(snowfall, (int, float)):
            snowfall = [snowfall]

        return sum(1 for s in snowfall if s >= self.threshold)

    def calculate_max_snow_depth(self) -> float:
        """
        Calculate maximum snow depth during the period.

        Returns:
            Maximum snow depth in cm
        """
        if "snow_depth" not in self.weather_data.data:
            raise ValueError("No snow_depth data available")

        snow_depth = self.weather_data.get_data("snow_depth")

        if isinstance(snow_depth, (int, float)):
            return snow_depth

        return max(snow_depth)

    def calculate_average_snow_depth(self) -> float:
        """
        Calculate average snow depth during the period.

        Returns:
            Average snow depth in cm
        """
        if "snow_depth" not in self.weather_data.data:
            raise ValueError("No snow_depth data available")

        snow_depth = self.weather_data.get_data("snow_depth")

        if isinstance(snow_depth, (int, float)):
            return snow_depth

        return np.mean(snow_depth)

    def calculate_snow_cover_days(self, min_depth: float = 2.5) -> int:
        """
        Calculate number of days with snow cover above minimum depth.

        Args:
            min_depth: Minimum snow depth for snow cover (cm)

        Returns:
            Number of days with snow cover
        """
        if "snow_depth" not in self.weather_data.data:
            raise ValueError("No snow_depth data available")

        snow_depth = self.weather_data.get_data("snow_depth")

        if isinstance(snow_depth, (int, float)):
            snow_depth = [snow_depth]

        return sum(1 for d in snow_depth if d >= min_depth)

    def payoff_total_snowfall(
        self,
        strike: float,
        tick_value: float = 1.0,
        option_type: str = "call"
    ) -> float:
        """
        Calculate payoff based on total snowfall.

        Args:
            strike: Strike level (cm)
            tick_value: Dollar value per cm
            option_type: "call" or "put"

        Returns:
            Payoff value
        """
        total_snow = self.calculate_total_snowfall()

        if option_type.lower() == "call":
            return max(total_snow - strike, 0) * tick_value
        elif option_type.lower() == "put":
            return max(strike - total_snow, 0) * tick_value
        else:
            raise ValueError(f"Unknown option type: {option_type}")

    def payoff_snow_days(
        self,
        strike: int,
        tick_value: float = 100.0,
        option_type: str = "call"
    ) -> float:
        """
        Calculate payoff based on number of snow days.

        Args:
            strike: Strike level (number of days)
            tick_value: Dollar value per day
            option_type: "call" or "put"

        Returns:
            Payoff value
        """
        snow_days = self.calculate_snow_days()

        if option_type.lower() == "call":
            return max(snow_days - strike, 0) * tick_value
        elif option_type.lower() == "put":
            return max(strike - snow_days, 0) * tick_value
        else:
            raise ValueError(f"Unknown option type: {option_type}")
