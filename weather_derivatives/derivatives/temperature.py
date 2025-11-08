"""
Temperature-based weather derivatives.

Implements common temperature derivatives:
- HDD (Heating Degree Days)
- CDD (Cooling Degree Days)
- CAT (Cumulative Average Temperature)
- PAC (Pacific Temperature)
"""

from typing import List, Optional, Union
import numpy as np
from ..core.weather_data import WeatherData


class BaseTemperatureDerivative:
    """Base class for temperature-based derivatives."""

    def __init__(self, weather_data: WeatherData, reference_temperature: float = 65.0):
        """
        Initialize temperature derivative.

        Args:
            weather_data: WeatherData object containing temperature data
            reference_temperature: Reference/base temperature for calculations (default 65°F)
        """
        self.weather_data = weather_data
        self.reference_temp = reference_temperature

        # Validate that temperature data exists
        if "temperature" not in weather_data.data:
            raise ValueError("Weather data must contain temperature information")

    def _fahrenheit_to_celsius(self, temp_f: float) -> float:
        """Convert Fahrenheit to Celsius."""
        return (temp_f - 32) * 5 / 9

    def _celsius_to_fahrenheit(self, temp_c: float) -> float:
        """Convert Celsius to Fahrenheit."""
        return (temp_c * 9 / 5) + 32


class HDD(BaseTemperatureDerivative):
    """
    Heating Degree Days (HDD) derivative.

    HDD measures how cold the temperature was on a given day.
    Formula: HDD = max(reference_temp - actual_temp, 0)
    """

    def calculate(self, cumulative: bool = True) -> Union[float, List[float]]:
        """
        Calculate Heating Degree Days.

        Args:
            cumulative: If True, return cumulative HDD, else daily HDD

        Returns:
            HDD value(s)
        """
        temperatures = self.weather_data.get_temperature()

        if isinstance(temperatures, (int, float)):
            temperatures = [temperatures]

        # Calculate daily HDD
        daily_hdd = [max(self.reference_temp - temp, 0) for temp in temperatures]

        if cumulative:
            return sum(daily_hdd)

        return daily_hdd if len(daily_hdd) > 1 else daily_hdd[0]

    def calculate_monthly(self) -> dict:
        """
        Calculate monthly HDD aggregations.

        Returns:
            Dictionary with month as key and HDD as value
        """
        temperatures = self.weather_data.get_temperature()
        times = self.weather_data.time

        if not isinstance(temperatures, list):
            temperatures = [temperatures]

        monthly_hdd = {}

        for temp, time in zip(temperatures, times):
            month_key = f"{time.year}-{time.month:02d}"
            hdd_value = max(self.reference_temp - temp, 0)

            if month_key not in monthly_hdd:
                monthly_hdd[month_key] = 0
            monthly_hdd[month_key] += hdd_value

        return monthly_hdd

    def payoff(self, strike: float, tick_value: float = 1.0, option_type: str = "call") -> float:
        """
        Calculate option payoff.

        Args:
            strike: Strike level for the option
            tick_value: Dollar value per HDD unit (default $1)
            option_type: "call" or "put"

        Returns:
            Payoff value
        """
        hdd_value = self.calculate(cumulative=True)

        if option_type.lower() == "call":
            return max(hdd_value - strike, 0) * tick_value
        elif option_type.lower() == "put":
            return max(strike - hdd_value, 0) * tick_value
        else:
            raise ValueError(f"Unknown option type: {option_type}")


class CDD(BaseTemperatureDerivative):
    """
    Cooling Degree Days (CDD) derivative.

    CDD measures how hot the temperature was on a given day.
    Formula: CDD = max(actual_temp - reference_temp, 0)
    """

    def calculate(self, cumulative: bool = True) -> Union[float, List[float]]:
        """
        Calculate Cooling Degree Days.

        Args:
            cumulative: If True, return cumulative CDD, else daily CDD

        Returns:
            CDD value(s)
        """
        temperatures = self.weather_data.get_temperature()

        if isinstance(temperatures, (int, float)):
            temperatures = [temperatures]

        # Calculate daily CDD
        daily_cdd = [max(temp - self.reference_temp, 0) for temp in temperatures]

        if cumulative:
            return sum(daily_cdd)

        return daily_cdd if len(daily_cdd) > 1 else daily_cdd[0]

    def calculate_monthly(self) -> dict:
        """
        Calculate monthly CDD aggregations.

        Returns:
            Dictionary with month as key and CDD as value
        """
        temperatures = self.weather_data.get_temperature()
        times = self.weather_data.time

        if not isinstance(temperatures, list):
            temperatures = [temperatures]

        monthly_cdd = {}

        for temp, time in zip(temperatures, times):
            month_key = f"{time.year}-{time.month:02d}"
            cdd_value = max(temp - self.reference_temp, 0)

            if month_key not in monthly_cdd:
                monthly_cdd[month_key] = 0
            monthly_cdd[month_key] += cdd_value

        return monthly_cdd

    def payoff(self, strike: float, tick_value: float = 1.0, option_type: str = "call") -> float:
        """
        Calculate option payoff.

        Args:
            strike: Strike level for the option
            tick_value: Dollar value per CDD unit (default $1)
            option_type: "call" or "put"

        Returns:
            Payoff value
        """
        cdd_value = self.calculate(cumulative=True)

        if option_type.lower() == "call":
            return max(cdd_value - strike, 0) * tick_value
        elif option_type.lower() == "put":
            return max(strike - cdd_value, 0) * tick_value
        else:
            raise ValueError(f"Unknown option type: {option_type}")


class CAT(BaseTemperatureDerivative):
    """
    Cumulative Average Temperature (CAT) derivative.

    CAT is the sum or average of daily temperatures over a period.
    """

    def __init__(self, weather_data: WeatherData):
        """Initialize CAT derivative (no reference temperature needed)."""
        super().__init__(weather_data, reference_temperature=0.0)

    def calculate(self, method: str = "sum") -> float:
        """
        Calculate Cumulative Average Temperature.

        Args:
            method: "sum" for total, "mean" for average

        Returns:
            CAT value
        """
        temperatures = self.weather_data.get_temperature()

        if isinstance(temperatures, (int, float)):
            temperatures = [temperatures]

        if method == "sum":
            return sum(temperatures)
        elif method == "mean":
            return np.mean(temperatures)
        else:
            raise ValueError(f"Unknown method: {method}")

    def payoff(self, strike: float, tick_value: float = 1.0, option_type: str = "call") -> float:
        """
        Calculate option payoff.

        Args:
            strike: Strike level for the option
            tick_value: Dollar value per degree unit (default $1)
            option_type: "call" or "put"

        Returns:
            Payoff value
        """
        cat_value = self.calculate(method="sum")

        if option_type.lower() == "call":
            return max(cat_value - strike, 0) * tick_value
        elif option_type.lower() == "put":
            return max(strike - cat_value, 0) * tick_value
        else:
            raise ValueError(f"Unknown option type: {option_type}")


class PAC(BaseTemperatureDerivative):
    """
    Pacific (PAC) Temperature derivative.

    PAC measures temperature deviations from a reference over a period.
    Similar to CAT but with a reference baseline.
    """

    def calculate(self, method: str = "cumulative") -> Union[float, List[float]]:
        """
        Calculate Pacific temperature metric.

        Args:
            method: "cumulative" for total deviation, "average" for mean deviation

        Returns:
            PAC value(s)
        """
        temperatures = self.weather_data.get_temperature()

        if isinstance(temperatures, (int, float)):
            temperatures = [temperatures]

        # Calculate deviations
        deviations = [temp - self.reference_temp for temp in temperatures]

        if method == "cumulative":
            return sum(deviations)
        elif method == "average":
            return np.mean(deviations)
        elif method == "daily":
            return deviations if len(deviations) > 1 else deviations[0]
        else:
            raise ValueError(f"Unknown method: {method}")

    def payoff(self, strike: float, tick_value: float = 1.0, option_type: str = "call") -> float:
        """
        Calculate option payoff.

        Args:
            strike: Strike level for the option
            tick_value: Dollar value per unit (default $1)
            option_type: "call" or "put"

        Returns:
            Payoff value
        """
        pac_value = self.calculate(method="cumulative")

        if option_type.lower() == "call":
            return max(pac_value - strike, 0) * tick_value
        elif option_type.lower() == "put":
            return max(strike - pac_value, 0) * tick_value
        else:
            raise ValueError(f"Unknown option type: {option_type}")
