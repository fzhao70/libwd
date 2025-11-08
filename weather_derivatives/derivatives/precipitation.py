"""
Precipitation-based weather derivatives.

Implements derivatives based on rainfall, snowfall, and other precipitation metrics.
"""

from typing import List, Optional, Union
import numpy as np
from ..core.weather_data import WeatherData


class PrecipitationDerivative:
    """
    Precipitation-based derivative calculator.

    Supports various precipitation-based contracts including:
    - Total precipitation over a period
    - Number of rain days
    - Extreme precipitation events
    """

    def __init__(
        self,
        weather_data: WeatherData,
        precipitation_threshold: float = 0.1
    ):
        """
        Initialize precipitation derivative.

        Args:
            weather_data: WeatherData object containing precipitation data
            precipitation_threshold: Minimum precipitation to count as a rain day (mm)
        """
        self.weather_data = weather_data
        self.threshold = precipitation_threshold

        # Validate that precipitation data exists
        if "precipitation" not in weather_data.data:
            raise ValueError("Weather data must contain precipitation information")

    def calculate_total_precipitation(self) -> float:
        """
        Calculate total precipitation over the period.

        Returns:
            Total precipitation in mm
        """
        precipitation = self.weather_data.get_precipitation()

        if isinstance(precipitation, (int, float)):
            return precipitation

        return sum(precipitation)

    def calculate_rain_days(self) -> int:
        """
        Calculate number of days with precipitation above threshold.

        Returns:
            Number of rain days
        """
        precipitation = self.weather_data.get_precipitation()

        if isinstance(precipitation, (int, float)):
            precipitation = [precipitation]

        return sum(1 for p in precipitation if p >= self.threshold)

    def calculate_dry_days(self) -> int:
        """
        Calculate number of days with precipitation below threshold.

        Returns:
            Number of dry days
        """
        precipitation = self.weather_data.get_precipitation()

        if isinstance(precipitation, (int, float)):
            precipitation = [precipitation]

        return sum(1 for p in precipitation if p < self.threshold)

    def calculate_extreme_events(self, extreme_threshold: float = 50.0) -> int:
        """
        Count extreme precipitation events.

        Args:
            extreme_threshold: Threshold for extreme event (mm)

        Returns:
            Number of extreme events
        """
        precipitation = self.weather_data.get_precipitation()

        if isinstance(precipitation, (int, float)):
            precipitation = [precipitation]

        return sum(1 for p in precipitation if p >= extreme_threshold)

    def calculate_percentile(self, percentile: float = 90) -> float:
        """
        Calculate precipitation percentile.

        Args:
            percentile: Percentile to calculate (0-100)

        Returns:
            Precipitation value at given percentile
        """
        precipitation = self.weather_data.get_precipitation()

        if isinstance(precipitation, (int, float)):
            return precipitation

        return np.percentile(precipitation, percentile)

    def payoff_total_precipitation(
        self,
        strike: float,
        tick_value: float = 1.0,
        option_type: str = "call"
    ) -> float:
        """
        Calculate payoff based on total precipitation.

        Args:
            strike: Strike level (mm)
            tick_value: Dollar value per mm
            option_type: "call" or "put"

        Returns:
            Payoff value
        """
        total_precip = self.calculate_total_precipitation()

        if option_type.lower() == "call":
            return max(total_precip - strike, 0) * tick_value
        elif option_type.lower() == "put":
            return max(strike - total_precip, 0) * tick_value
        else:
            raise ValueError(f"Unknown option type: {option_type}")

    def payoff_rain_days(
        self,
        strike: int,
        tick_value: float = 100.0,
        option_type: str = "call"
    ) -> float:
        """
        Calculate payoff based on number of rain days.

        Args:
            strike: Strike level (number of days)
            tick_value: Dollar value per day
            option_type: "call" or "put"

        Returns:
            Payoff value
        """
        rain_days = self.calculate_rain_days()

        if option_type.lower() == "call":
            return max(rain_days - strike, 0) * tick_value
        elif option_type.lower() == "put":
            return max(strike - rain_days, 0) * tick_value
        else:
            raise ValueError(f"Unknown option type: {option_type}")

    def calculate_monthly_precipitation(self) -> dict:
        """
        Calculate monthly precipitation totals.

        Returns:
            Dictionary with month as key and precipitation as value
        """
        precipitation = self.weather_data.get_precipitation()
        times = self.weather_data.time

        if not isinstance(precipitation, list):
            precipitation = [precipitation]

        monthly_precip = {}

        for precip, time in zip(precipitation, times):
            month_key = f"{time.year}-{time.month:02d}"

            if month_key not in monthly_precip:
                monthly_precip[month_key] = 0
            monthly_precip[month_key] += precip

        return monthly_precip
