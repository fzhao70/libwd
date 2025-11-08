"""
Wind-based weather derivatives.

Implements derivatives based on wind speed and wind power.
"""

from typing import List, Optional, Union
import numpy as np
from ..core.weather_data import WeatherData


class WindDerivative:
    """
    Wind-based derivative calculator.

    Supports wind derivatives including:
    - Average wind speed
    - Wind power calculations
    - Calm days (low wind)
    - High wind events
    """

    def __init__(
        self,
        weather_data: WeatherData,
        air_density: float = 1.225,
        rotor_diameter: Optional[float] = None
    ):
        """
        Initialize wind derivative.

        Args:
            weather_data: WeatherData object containing wind speed data
            air_density: Air density in kg/m³ (default 1.225 at sea level)
            rotor_diameter: Wind turbine rotor diameter in meters (optional)
        """
        self.weather_data = weather_data
        self.air_density = air_density
        self.rotor_diameter = rotor_diameter

        # Validate that wind speed data exists
        if "wind_speed" not in weather_data.data:
            raise ValueError("Weather data must contain wind speed information")

    def calculate_average_wind_speed(self) -> float:
        """
        Calculate average wind speed over the period.

        Returns:
            Average wind speed in m/s
        """
        wind_speed = self.weather_data.get_wind_speed()

        if isinstance(wind_speed, (int, float)):
            return wind_speed

        return np.mean(wind_speed)

    def calculate_wind_power(self) -> Union[float, List[float]]:
        """
        Calculate wind power based on wind speed.

        Power = 0.5 * air_density * area * velocity³

        Returns:
            Wind power in watts (if rotor_diameter provided) or
            power coefficient (if no rotor_diameter)
        """
        wind_speed = self.weather_data.get_wind_speed()

        if isinstance(wind_speed, (int, float)):
            wind_speed = [wind_speed]

        if self.rotor_diameter is not None:
            area = np.pi * (self.rotor_diameter / 2) ** 2
            power = [
                0.5 * self.air_density * area * (v ** 3)
                for v in wind_speed
            ]
        else:
            # Return power coefficient without area
            power = [0.5 * self.air_density * (v ** 3) for v in wind_speed]

        return power if len(power) > 1 else power[0]

    def calculate_total_wind_energy(self, hours_per_reading: float = 24.0) -> float:
        """
        Calculate total wind energy over the period.

        Args:
            hours_per_reading: Number of hours each reading represents

        Returns:
            Total energy in watt-hours
        """
        power = self.calculate_wind_power()

        if isinstance(power, (int, float)):
            power = [power]

        return sum(power) * hours_per_reading

    def calculate_calm_days(self, calm_threshold: float = 3.0) -> int:
        """
        Calculate number of calm days (low wind).

        Args:
            calm_threshold: Maximum wind speed for calm day (m/s)

        Returns:
            Number of calm days
        """
        wind_speed = self.weather_data.get_wind_speed()

        if isinstance(wind_speed, (int, float)):
            wind_speed = [wind_speed]

        return sum(1 for v in wind_speed if v <= calm_threshold)

    def calculate_high_wind_days(self, high_threshold: float = 15.0) -> int:
        """
        Calculate number of high wind days.

        Args:
            high_threshold: Minimum wind speed for high wind day (m/s)

        Returns:
            Number of high wind days
        """
        wind_speed = self.weather_data.get_wind_speed()

        if isinstance(wind_speed, (int, float)):
            wind_speed = [wind_speed]

        return sum(1 for v in wind_speed if v >= high_threshold)

    def payoff_wind_power(
        self,
        strike: float,
        tick_value: float = 1.0,
        option_type: str = "call"
    ) -> float:
        """
        Calculate payoff based on total wind energy.

        Args:
            strike: Strike level (energy units)
            tick_value: Dollar value per energy unit
            option_type: "call" or "put"

        Returns:
            Payoff value
        """
        total_energy = self.calculate_total_wind_energy()

        if option_type.lower() == "call":
            return max(total_energy - strike, 0) * tick_value
        elif option_type.lower() == "put":
            return max(strike - total_energy, 0) * tick_value
        else:
            raise ValueError(f"Unknown option type: {option_type}")

    def payoff_calm_days(
        self,
        strike: int,
        tick_value: float = 100.0,
        option_type: str = "put"
    ) -> float:
        """
        Calculate payoff based on calm days.

        Typically a put option (pays when there are MORE calm days than strike).

        Args:
            strike: Strike level (number of days)
            tick_value: Dollar value per day
            option_type: "call" or "put"

        Returns:
            Payoff value
        """
        calm_days = self.calculate_calm_days()

        if option_type.lower() == "call":
            return max(calm_days - strike, 0) * tick_value
        elif option_type.lower() == "put":
            return max(strike - calm_days, 0) * tick_value
        else:
            raise ValueError(f"Unknown option type: {option_type}")
