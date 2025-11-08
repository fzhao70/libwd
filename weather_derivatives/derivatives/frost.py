"""
Frost and agricultural temperature derivatives.

Implements derivatives based on frost events and growing degree days.
Primarily used in agriculture and crop insurance.
"""

from typing import List, Optional, Union
import numpy as np
from ..core.weather_data import WeatherData


class FrostDerivative:
    """
    Frost-based derivative calculator.

    Supports agricultural derivatives including:
    - Frost days counting
    - Growing Degree Days (GDD)
    - Freeze events
    - Crop Heat Units
    """

    def __init__(
        self,
        weather_data: WeatherData,
        frost_threshold: float = 0.0,
        base_temperature: float = 10.0
    ):
        """
        Initialize frost derivative.

        Args:
            weather_data: WeatherData object containing temperature data
            frost_threshold: Temperature threshold for frost (Celsius, default 0°C)
            base_temperature: Base temperature for GDD calculation (Celsius, default 10°C)
        """
        self.weather_data = weather_data
        self.frost_threshold = frost_threshold
        self.base_temp = base_temperature

        # Validate that temperature data exists
        if "temperature" not in weather_data.data:
            raise ValueError("Weather data must contain temperature information")

    def calculate_frost_days(self) -> int:
        """
        Calculate number of days with minimum temperature below frost threshold.

        Returns:
            Number of frost days
        """
        temperature = self.weather_data.get_temperature()

        if isinstance(temperature, (int, float)):
            temperature = [temperature]

        # If we have min_temperature data, use it; otherwise use temperature
        if "min_temperature" in self.weather_data.data:
            min_temps = self.weather_data.get_data("min_temperature")
        else:
            min_temps = temperature

        if isinstance(min_temps, (int, float)):
            min_temps = [min_temps]

        return sum(1 for t in min_temps if t <= self.frost_threshold)

    def calculate_growing_degree_days(
        self,
        max_temperature: Optional[float] = 30.0
    ) -> float:
        """
        Calculate Growing Degree Days (GDD).

        GDD = sum of (avg_temp - base_temp) for days where avg_temp > base_temp
        Often capped at a maximum temperature.

        Args:
            max_temperature: Maximum temperature cap for GDD (Celsius)

        Returns:
            Total GDD
        """
        temperature = self.weather_data.get_temperature()

        if isinstance(temperature, (int, float)):
            temperature = [temperature]

        gdd_total = 0.0

        for temp in temperature:
            # Cap temperature if maximum is specified
            if max_temperature is not None:
                temp = min(temp, max_temperature)

            # Calculate GDD for the day
            daily_gdd = max(temp - self.base_temp, 0)
            gdd_total += daily_gdd

        return gdd_total

    def calculate_crop_heat_units(
        self,
        min_night_temp: float = 4.4,
        max_day_temp: float = 30.0
    ) -> float:
        """
        Calculate Crop Heat Units (CHU) for corn/maize.

        CHU is a specialized measure used for corn maturity prediction.

        Args:
            min_night_temp: Minimum night temperature (Celsius)
            max_day_temp: Maximum day temperature (Celsius)

        Returns:
            Total CHU
        """
        # This requires both min and max daily temperatures
        if "min_temperature" not in self.weather_data.data or "max_temperature" not in self.weather_data.data:
            raise ValueError("CHU requires both min_temperature and max_temperature data")

        min_temps = self.weather_data.get_data("min_temperature")
        max_temps = self.weather_data.get_data("max_temperature")

        if isinstance(min_temps, (int, float)):
            min_temps = [min_temps]
            max_temps = [max_temps]

        chu_total = 0.0

        for t_min, t_max in zip(min_temps, max_temps):
            # Night CHU
            y_min = 1.8 * (t_min - min_night_temp)
            night_chu = max(y_min, 0)

            # Day CHU
            y_max = 3.33 * (t_max - 10) - 0.084 * (t_max - 10) ** 2
            day_chu = max(min(y_max, max_day_temp), 0)

            # Daily CHU is average of night and day
            chu_total += (night_chu + day_chu) / 2

        return chu_total

    def calculate_freeze_events(self, severe_threshold: float = -5.0) -> int:
        """
        Calculate number of severe freeze events.

        Args:
            severe_threshold: Temperature threshold for severe freeze (Celsius)

        Returns:
            Number of freeze events
        """
        if "min_temperature" in self.weather_data.data:
            temps = self.weather_data.get_data("min_temperature")
        else:
            temps = self.weather_data.get_temperature()

        if isinstance(temps, (int, float)):
            temps = [temps]

        return sum(1 for t in temps if t <= severe_threshold)

    def calculate_frost_free_days(self) -> int:
        """
        Calculate number of frost-free days.

        Returns:
            Number of days without frost
        """
        return len(self.weather_data) - self.calculate_frost_days()

    def payoff_gdd(
        self,
        strike: float,
        tick_value: float = 1.0,
        option_type: str = "call"
    ) -> float:
        """
        Calculate payoff based on Growing Degree Days.

        Args:
            strike: Strike level (GDD)
            tick_value: Dollar value per GDD unit
            option_type: "call" or "put"

        Returns:
            Payoff value
        """
        gdd = self.calculate_growing_degree_days()

        if option_type.lower() == "call":
            return max(gdd - strike, 0) * tick_value
        elif option_type.lower() == "put":
            return max(strike - gdd, 0) * tick_value
        else:
            raise ValueError(f"Unknown option type: {option_type}")

    def payoff_frost_days(
        self,
        strike: int,
        tick_value: float = 500.0,
        option_type: str = "put"
    ) -> float:
        """
        Calculate payoff based on frost days.

        Typically a put option (pays when there are FEWER frost days than expected).

        Args:
            strike: Strike level (number of days)
            tick_value: Dollar value per day
            option_type: "call" or "put"

        Returns:
            Payoff value
        """
        frost_days = self.calculate_frost_days()

        if option_type.lower() == "call":
            return max(frost_days - strike, 0) * tick_value
        elif option_type.lower() == "put":
            return max(strike - frost_days, 0) * tick_value
        else:
            raise ValueError(f"Unknown option type: {option_type}")
