"""
Solar radiation-based weather derivatives.

Implements derivatives based on solar irradiance and sunshine duration.
Used in solar power generation and renewable energy markets.
"""

from typing import List, Optional, Union
import numpy as np
from ..core.weather_data import WeatherData


class SolarDerivative:
    """
    Solar radiation-based derivative calculator.

    Supports solar derivatives including:
    - Total solar irradiance
    - Sunshine hours/duration
    - Cloudy days counting
    - Solar energy production estimates
    """

    def __init__(
        self,
        weather_data: WeatherData,
        panel_efficiency: float = 0.18,
        panel_area: Optional[float] = None
    ):
        """
        Initialize solar derivative.

        Args:
            weather_data: WeatherData object containing solar radiation data
            panel_efficiency: Solar panel efficiency (default 18%)
            panel_area: Solar panel area in m² (optional)
        """
        self.weather_data = weather_data
        self.panel_efficiency = panel_efficiency
        self.panel_area = panel_area

        # Validate that solar data exists
        if ("solar_radiation" not in weather_data.data and
            "sunshine_hours" not in weather_data.data and
            "irradiance" not in weather_data.data):
            raise ValueError(
                "Weather data must contain solar_radiation, sunshine_hours, or irradiance information"
            )

    def calculate_total_irradiance(self) -> float:
        """
        Calculate total solar irradiance over the period.

        Returns:
            Total irradiance in MJ/m² or kWh/m²
        """
        # Try different field names
        for field in ["solar_radiation", "irradiance", "ghi"]:  # ghi = global horizontal irradiance
            if field in self.weather_data.data:
                irradiance = self.weather_data.get_data(field)
                break
        else:
            raise ValueError("No solar irradiance data available")

        if isinstance(irradiance, (int, float)):
            return irradiance

        return sum(irradiance)

    def calculate_average_irradiance(self) -> float:
        """
        Calculate average daily solar irradiance.

        Returns:
            Average irradiance
        """
        total = self.calculate_total_irradiance()
        return total / len(self.weather_data)

    def calculate_sunshine_hours(self) -> float:
        """
        Calculate total sunshine hours over the period.

        Returns:
            Total sunshine hours
        """
        if "sunshine_hours" not in self.weather_data.data:
            raise ValueError("No sunshine hours data available")

        sunshine = self.weather_data.get_data("sunshine_hours")

        if isinstance(sunshine, (int, float)):
            return sunshine

        return sum(sunshine)

    def calculate_cloudy_days(
        self,
        max_sunshine_hours: float = 4.0
    ) -> int:
        """
        Calculate number of cloudy days.

        Args:
            max_sunshine_hours: Maximum sunshine hours to count as cloudy

        Returns:
            Number of cloudy days
        """
        if "sunshine_hours" not in self.weather_data.data:
            # Estimate from irradiance if available
            if "solar_radiation" in self.weather_data.data:
                radiation = self.weather_data.get_data("solar_radiation")
                if isinstance(radiation, (int, float)):
                    radiation = [radiation]
                # Estimate: cloudy if radiation < 3 MJ/m²/day
                return sum(1 for r in radiation if r < 3.0)
            else:
                raise ValueError("No sunshine or radiation data available")

        sunshine = self.weather_data.get_data("sunshine_hours")
        if isinstance(sunshine, (int, float)):
            sunshine = [sunshine]

        return sum(1 for s in sunshine if s <= max_sunshine_hours)

    def calculate_solar_energy_production(
        self,
        hours_per_reading: float = 24.0
    ) -> float:
        """
        Calculate estimated solar energy production.

        Args:
            hours_per_reading: Hours represented by each irradiance reading

        Returns:
            Total energy production in kWh (if panel_area provided) or kWh/m²
        """
        total_irradiance = self.calculate_total_irradiance()

        # Assuming irradiance is in W/m² averaged over the period
        # Energy = Power × Time × Efficiency
        energy_per_m2 = total_irradiance * self.panel_efficiency

        if self.panel_area is not None:
            return energy_per_m2 * self.panel_area

        return energy_per_m2

    def calculate_peak_sun_hours(self) -> float:
        """
        Calculate peak sun hours (PSH).

        PSH = Total daily irradiation / 1000 W/m²

        Returns:
            Total peak sun hours
        """
        total_irradiance = self.calculate_total_irradiance()

        # Convert to peak sun hours (assuming irradiance in kWh/m²)
        # If in MJ/m², convert: 1 kWh = 3.6 MJ
        # Assuming data is already in appropriate units
        return total_irradiance

    def calculate_capacity_factor(
        self,
        rated_capacity: float
    ) -> float:
        """
        Calculate solar plant capacity factor.

        Args:
            rated_capacity: Rated capacity in kW

        Returns:
            Capacity factor (0-1)
        """
        if self.panel_area is None:
            raise ValueError("panel_area required for capacity factor calculation")

        actual_production = self.calculate_solar_energy_production()
        hours = len(self.weather_data) * 24  # Assuming daily data

        theoretical_max = rated_capacity * hours
        capacity_factor = actual_production / theoretical_max

        return min(capacity_factor, 1.0)

    def payoff_irradiance(
        self,
        strike: float,
        tick_value: float = 1.0,
        option_type: str = "put"
    ) -> float:
        """
        Calculate payoff based on total irradiance.

        Typically a put option for solar producers (pays when irradiance is LOW).

        Args:
            strike: Strike level (irradiance units)
            tick_value: Dollar value per unit
            option_type: "call" or "put"

        Returns:
            Payoff value
        """
        total_irradiance = self.calculate_total_irradiance()

        if option_type.lower() == "call":
            return max(total_irradiance - strike, 0) * tick_value
        elif option_type.lower() == "put":
            return max(strike - total_irradiance, 0) * tick_value
        else:
            raise ValueError(f"Unknown option type: {option_type}")

    def payoff_sunshine_hours(
        self,
        strike: float,
        tick_value: float = 10.0,
        option_type: str = "put"
    ) -> float:
        """
        Calculate payoff based on sunshine hours.

        Args:
            strike: Strike level (hours)
            tick_value: Dollar value per hour
            option_type: "call" or "put"

        Returns:
            Payoff value
        """
        sunshine_hours = self.calculate_sunshine_hours()

        if option_type.lower() == "call":
            return max(sunshine_hours - strike, 0) * tick_value
        elif option_type.lower() == "put":
            return max(strike - sunshine_hours, 0) * tick_value
        else:
            raise ValueError(f"Unknown option type: {option_type}")

    def payoff_energy_production(
        self,
        strike: float,
        tick_value: float = 0.1,
        option_type: str = "put"
    ) -> float:
        """
        Calculate payoff based on energy production.

        Args:
            strike: Strike level (kWh)
            tick_value: Dollar value per kWh
            option_type: "call" or "put"

        Returns:
            Payoff value
        """
        energy_production = self.calculate_solar_energy_production()

        if option_type.lower() == "call":
            return max(energy_production - strike, 0) * tick_value
        elif option_type.lower() == "put":
            return max(strike - energy_production, 0) * tick_value
        else:
            raise ValueError(f"Unknown option type: {option_type}")
