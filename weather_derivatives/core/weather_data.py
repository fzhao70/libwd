"""
Core weather data handling classes.

Supports multiple input formats:
- (time, lon, lat): Geographical coordinates
- (time, site): Named location
- (time): Time-based data only
"""

from datetime import datetime
from typing import Union, Optional, Dict, List, Any
import numpy as np


class WeatherData:
    """
    Core class for handling weather data in various input formats.

    Attributes:
        time: DateTime or list of DateTimes
        location: Dictionary containing location info (lon/lat or site name)
        data: Dictionary containing weather metrics (temperature, precipitation, etc.)
    """

    def __init__(
        self,
        time: Union[datetime, List[datetime]],
        temperature: Optional[Union[float, List[float]]] = None,
        precipitation: Optional[Union[float, List[float]]] = None,
        wind_speed: Optional[Union[float, List[float]]] = None,
        lon: Optional[float] = None,
        lat: Optional[float] = None,
        site: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize weather data.

        Args:
            time: Single datetime or list of datetimes
            temperature: Temperature in Celsius (single value or list)
            precipitation: Precipitation in mm (single value or list)
            wind_speed: Wind speed in m/s (single value or list)
            lon: Longitude (for geographical input)
            lat: Latitude (for geographical input)
            site: Site name (for named location input)
            **kwargs: Additional weather parameters
        """
        # Handle time
        if isinstance(time, (list, np.ndarray)):
            self.time = list(time)
            self.is_timeseries = True
        else:
            self.time = [time]
            self.is_timeseries = False

        # Handle location
        self.location = {}
        if lon is not None and lat is not None:
            self.location = {"type": "coordinates", "lon": lon, "lat": lat}
        elif site is not None:
            self.location = {"type": "site", "name": site}
        else:
            self.location = {"type": "none"}

        # Handle weather data
        self.data = {}

        if temperature is not None:
            self.data["temperature"] = self._ensure_list(temperature)

        if precipitation is not None:
            self.data["precipitation"] = self._ensure_list(precipitation)

        if wind_speed is not None:
            self.data["wind_speed"] = self._ensure_list(wind_speed)

        # Add any additional parameters
        for key, value in kwargs.items():
            self.data[key] = self._ensure_list(value)

        # Validate data length consistency
        self._validate_data_length()

    def _ensure_list(self, value: Union[Any, List[Any]]) -> List[Any]:
        """Ensure value is a list."""
        if isinstance(value, (list, np.ndarray)):
            return list(value)
        return [value]

    def _validate_data_length(self):
        """Validate that all data arrays have consistent length."""
        expected_length = len(self.time)

        for key, values in self.data.items():
            if len(values) != expected_length:
                raise ValueError(
                    f"Length mismatch: {key} has {len(values)} values "
                    f"but time has {expected_length} values"
                )

    def get_temperature(self, index: Optional[int] = None) -> Union[float, List[float]]:
        """Get temperature data."""
        if "temperature" not in self.data:
            raise ValueError("No temperature data available")

        if index is not None:
            return self.data["temperature"][index]
        return self.data["temperature"] if self.is_timeseries else self.data["temperature"][0]

    def get_precipitation(self, index: Optional[int] = None) -> Union[float, List[float]]:
        """Get precipitation data."""
        if "precipitation" not in self.data:
            raise ValueError("No precipitation data available")

        if index is not None:
            return self.data["precipitation"][index]
        return self.data["precipitation"] if self.is_timeseries else self.data["precipitation"][0]

    def get_wind_speed(self, index: Optional[int] = None) -> Union[float, List[float]]:
        """Get wind speed data."""
        if "wind_speed" not in self.data:
            raise ValueError("No wind speed data available")

        if index is not None:
            return self.data["wind_speed"][index]
        return self.data["wind_speed"] if self.is_timeseries else self.data["wind_speed"][0]

    def get_data(self, key: str, index: Optional[int] = None) -> Union[Any, List[Any]]:
        """Get any data by key."""
        if key not in self.data:
            raise ValueError(f"No data available for key: {key}")

        if index is not None:
            return self.data[key][index]
        return self.data[key] if self.is_timeseries else self.data[key][0]

    def __len__(self) -> int:
        """Return number of time points."""
        return len(self.time)

    def __repr__(self) -> str:
        """String representation."""
        loc_str = ""
        if self.location["type"] == "coordinates":
            loc_str = f", location=({self.location['lon']}, {self.location['lat']})"
        elif self.location["type"] == "site":
            loc_str = f", site={self.location['name']}"

        metrics = list(self.data.keys())
        return f"WeatherData(points={len(self)}, metrics={metrics}{loc_str})"
