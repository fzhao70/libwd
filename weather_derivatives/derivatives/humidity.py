"""
Humidity-based weather derivatives.

Implements derivatives based on humidity and related moisture metrics.
Used in agriculture, health sectors, and industrial applications.
"""

from typing import List, Optional, Union
import numpy as np
from ..core.weather_data import WeatherData


class HumidityDerivative:
    """
    Humidity-based derivative calculator.

    Supports humidity derivatives including:
    - Average relative humidity
    - High humidity days
    - Low humidity days
    - Vapor pressure deficit
    """

    def __init__(
        self,
        weather_data: WeatherData,
        high_humidity_threshold: float = 80.0,
        low_humidity_threshold: float = 30.0
    ):
        """
        Initialize humidity derivative.

        Args:
            weather_data: WeatherData object containing humidity data
            high_humidity_threshold: Threshold for high humidity days (%)
            low_humidity_threshold: Threshold for low humidity days (%)
        """
        self.weather_data = weather_data
        self.high_threshold = high_humidity_threshold
        self.low_threshold = low_humidity_threshold

        # Validate that humidity data exists
        if "humidity" not in weather_data.data and "relative_humidity" not in weather_data.data:
            raise ValueError("Weather data must contain humidity or relative_humidity information")

    def _get_humidity(self) -> Union[float, List[float]]:
        """Get humidity data from weather data."""
        if "humidity" in self.weather_data.data:
            return self.weather_data.get_data("humidity")
        elif "relative_humidity" in self.weather_data.data:
            return self.weather_data.get_data("relative_humidity")
        else:
            raise ValueError("No humidity data available")

    def calculate_average_humidity(self) -> float:
        """
        Calculate average humidity over the period.

        Returns:
            Average relative humidity (%)
        """
        humidity = self._get_humidity()

        if isinstance(humidity, (int, float)):
            return humidity

        return np.mean(humidity)

    def calculate_high_humidity_days(self) -> int:
        """
        Calculate number of days with humidity above high threshold.

        Returns:
            Number of high humidity days
        """
        humidity = self._get_humidity()

        if isinstance(humidity, (int, float)):
            humidity = [humidity]

        return sum(1 for h in humidity if h >= self.high_threshold)

    def calculate_low_humidity_days(self) -> int:
        """
        Calculate number of days with humidity below low threshold.

        Returns:
            Number of low humidity days
        """
        humidity = self._get_humidity()

        if isinstance(humidity, (int, float)):
            humidity = [humidity]

        return sum(1 for h in humidity if h <= self.low_threshold)

    def calculate_humidity_stress_index(self) -> float:
        """
        Calculate humidity stress index (days outside comfort zone).

        Returns:
            Number of days with humidity outside 30-70% range
        """
        humidity = self._get_humidity()

        if isinstance(humidity, (int, float)):
            humidity = [humidity]

        stress_days = sum(1 for h in humidity if h < 30 or h > 70)
        return stress_days

    def calculate_vapor_pressure_deficit(
        self,
        temperature_data: Optional[List[float]] = None
    ) -> Union[float, List[float]]:
        """
        Calculate Vapor Pressure Deficit (VPD).

        VPD = (1 - RH/100) * SVP(T)
        where SVP is saturation vapor pressure

        Args:
            temperature_data: Optional temperature data (Celsius)

        Returns:
            VPD value(s) in kPa
        """
        # Get humidity
        humidity = self._get_humidity()
        if isinstance(humidity, (int, float)):
            humidity = [humidity]

        # Get temperature
        if temperature_data is None:
            if "temperature" not in self.weather_data.data:
                raise ValueError("Temperature data required for VPD calculation")
            temperature = self.weather_data.get_temperature()
        else:
            temperature = temperature_data

        if isinstance(temperature, (int, float)):
            temperature = [temperature]

        # Calculate VPD for each day
        vpd_values = []
        for t, rh in zip(temperature, humidity):
            # Saturation vapor pressure (Tetens equation)
            svp = 0.6108 * np.exp((17.27 * t) / (t + 237.3))
            # VPD in kPa
            vpd = (1 - rh / 100) * svp
            vpd_values.append(vpd)

        return vpd_values if len(vpd_values) > 1 else vpd_values[0]

    def calculate_heat_index_days(
        self,
        heat_index_threshold: float = 32.0
    ) -> int:
        """
        Calculate number of days exceeding heat index threshold.

        Requires both temperature and humidity data.

        Args:
            heat_index_threshold: Heat index threshold (Celsius)

        Returns:
            Number of days exceeding threshold
        """
        if "temperature" not in self.weather_data.data:
            raise ValueError("Temperature data required for heat index")

        temperature = self.weather_data.get_temperature()
        humidity = self._get_humidity()

        if isinstance(temperature, (int, float)):
            temperature = [temperature]
            humidity = [humidity]

        days_exceeding = 0
        for t, rh in zip(temperature, humidity):
            # Convert to Fahrenheit for heat index formula
            t_f = (t * 9/5) + 32

            # Simplified heat index formula (Rothfusz regression)
            if t_f >= 80:
                hi = (-42.379 + 2.04901523*t_f + 10.14333127*rh
                      - 0.22475541*t_f*rh - 0.00683783*t_f*t_f
                      - 0.05481717*rh*rh + 0.00122874*t_f*t_f*rh
                      + 0.00085282*t_f*rh*rh - 0.00000199*t_f*t_f*rh*rh)

                # Convert back to Celsius
                hi_c = (hi - 32) * 5/9

                if hi_c >= heat_index_threshold:
                    days_exceeding += 1

        return days_exceeding

    def payoff_humidity_days(
        self,
        strike: int,
        tick_value: float = 100.0,
        option_type: str = "call",
        metric: str = "high"
    ) -> float:
        """
        Calculate payoff based on humidity days.

        Args:
            strike: Strike level (number of days)
            tick_value: Dollar value per day
            option_type: "call" or "put"
            metric: "high" or "low" humidity days

        Returns:
            Payoff value
        """
        if metric == "high":
            days = self.calculate_high_humidity_days()
        elif metric == "low":
            days = self.calculate_low_humidity_days()
        else:
            raise ValueError(f"Unknown metric: {metric}")

        if option_type.lower() == "call":
            return max(days - strike, 0) * tick_value
        elif option_type.lower() == "put":
            return max(strike - days, 0) * tick_value
        else:
            raise ValueError(f"Unknown option type: {option_type}")
