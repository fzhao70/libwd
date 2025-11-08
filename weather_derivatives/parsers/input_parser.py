"""
Weather data input parser supporting multiple formats.

Supports:
1. (time, lon, lat) - Geographical coordinates
2. (time, site) - Named location
3. (time) - Time-based only
"""

from datetime import datetime
from typing import Union, List, Dict, Any, Optional, Tuple
import numpy as np
from ..core.weather_data import WeatherData


class WeatherInputParser:
    """
    Parser for different weather data input formats.

    Provides convenience methods for creating WeatherData objects
    from various input formats.
    """

    @staticmethod
    def from_coordinates(
        time: Union[datetime, List[datetime]],
        lon: float,
        lat: float,
        temperature: Optional[Union[float, List[float]]] = None,
        precipitation: Optional[Union[float, List[float]]] = None,
        wind_speed: Optional[Union[float, List[float]]] = None,
        **kwargs
    ) -> WeatherData:
        """
        Create WeatherData from geographical coordinates.

        Args:
            time: Single datetime or list of datetimes
            lon: Longitude
            lat: Latitude
            temperature: Temperature data (optional)
            precipitation: Precipitation data (optional)
            wind_speed: Wind speed data (optional)
            **kwargs: Additional weather parameters

        Returns:
            WeatherData object

        Example:
            >>> from datetime import datetime
            >>> parser = WeatherInputParser()
            >>> data = parser.from_coordinates(
            ...     time=datetime(2024, 1, 1),
            ...     lon=-74.006,
            ...     lat=40.7128,
            ...     temperature=15.5
            ... )
        """
        return WeatherData(
            time=time,
            lon=lon,
            lat=lat,
            temperature=temperature,
            precipitation=precipitation,
            wind_speed=wind_speed,
            **kwargs
        )

    @staticmethod
    def from_site(
        time: Union[datetime, List[datetime]],
        site: str,
        temperature: Optional[Union[float, List[float]]] = None,
        precipitation: Optional[Union[float, List[float]]] = None,
        wind_speed: Optional[Union[float, List[float]]] = None,
        **kwargs
    ) -> WeatherData:
        """
        Create WeatherData from named site/location.

        Args:
            time: Single datetime or list of datetimes
            site: Site name (e.g., "New York JFK", "London Heathrow")
            temperature: Temperature data (optional)
            precipitation: Precipitation data (optional)
            wind_speed: Wind speed data (optional)
            **kwargs: Additional weather parameters

        Returns:
            WeatherData object

        Example:
            >>> from datetime import datetime
            >>> parser = WeatherInputParser()
            >>> data = parser.from_site(
            ...     time=datetime(2024, 1, 1),
            ...     site="New York JFK",
            ...     temperature=15.5
            ... )
        """
        return WeatherData(
            time=time,
            site=site,
            temperature=temperature,
            precipitation=precipitation,
            wind_speed=wind_speed,
            **kwargs
        )

    @staticmethod
    def from_time_only(
        time: Union[datetime, List[datetime]],
        temperature: Optional[Union[float, List[float]]] = None,
        precipitation: Optional[Union[float, List[float]]] = None,
        wind_speed: Optional[Union[float, List[float]]] = None,
        **kwargs
    ) -> WeatherData:
        """
        Create WeatherData from time data only (no location).

        Args:
            time: Single datetime or list of datetimes
            temperature: Temperature data (optional)
            precipitation: Precipitation data (optional)
            wind_speed: Wind speed data (optional)
            **kwargs: Additional weather parameters

        Returns:
            WeatherData object

        Example:
            >>> from datetime import datetime
            >>> parser = WeatherInputParser()
            >>> data = parser.from_time_only(
            ...     time=datetime(2024, 1, 1),
            ...     temperature=15.5
            ... )
        """
        return WeatherData(
            time=time,
            temperature=temperature,
            precipitation=precipitation,
            wind_speed=wind_speed,
            **kwargs
        )

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> WeatherData:
        """
        Create WeatherData from dictionary.

        Args:
            data: Dictionary containing weather data fields

        Returns:
            WeatherData object

        Example:
            >>> parser = WeatherInputParser()
            >>> data = parser.from_dict({
            ...     'time': datetime(2024, 1, 1),
            ...     'site': 'New York',
            ...     'temperature': 15.5
            ... })
        """
        # Extract known fields
        time = data.get('time')
        if time is None:
            raise ValueError("'time' field is required")

        lon = data.get('lon')
        lat = data.get('lat')
        site = data.get('site')
        temperature = data.get('temperature')
        precipitation = data.get('precipitation')
        wind_speed = data.get('wind_speed')

        # Get any additional fields
        known_fields = {'time', 'lon', 'lat', 'site', 'temperature', 'precipitation', 'wind_speed'}
        additional_fields = {k: v for k, v in data.items() if k not in known_fields}

        return WeatherData(
            time=time,
            lon=lon,
            lat=lat,
            site=site,
            temperature=temperature,
            precipitation=precipitation,
            wind_speed=wind_speed,
            **additional_fields
        )

    @staticmethod
    def from_csv_dict(records: List[Dict[str, Any]]) -> WeatherData:
        """
        Create WeatherData from list of CSV-like dictionaries.

        Args:
            records: List of dictionaries (like from csv.DictReader)

        Returns:
            WeatherData object

        Example:
            >>> records = [
            ...     {'time': '2024-01-01', 'temperature': 15.5, 'site': 'NYC'},
            ...     {'time': '2024-01-02', 'temperature': 16.0, 'site': 'NYC'}
            ... ]
            >>> parser = WeatherInputParser()
            >>> data = parser.from_csv_dict(records)
        """
        if not records:
            raise ValueError("No records provided")

        # Parse times
        times = []
        for record in records:
            time_val = record.get('time')
            if isinstance(time_val, str):
                # Try to parse string to datetime
                try:
                    times.append(datetime.fromisoformat(time_val))
                except ValueError:
                    # Try other common formats
                    from dateutil import parser as date_parser
                    times.append(date_parser.parse(time_val))
            elif isinstance(time_val, datetime):
                times.append(time_val)
            else:
                raise ValueError(f"Invalid time format: {time_val}")

        # Extract other fields
        first_record = records[0]
        lon = first_record.get('lon')
        lat = first_record.get('lat')
        site = first_record.get('site')

        # Collect time-series data
        data_fields = {}
        for key in first_record.keys():
            if key not in ['time', 'lon', 'lat', 'site']:
                values = []
                for record in records:
                    val = record.get(key)
                    # Try to convert to float if possible
                    try:
                        values.append(float(val))
                    except (ValueError, TypeError):
                        values.append(val)
                data_fields[key] = values

        return WeatherData(
            time=times,
            lon=float(lon) if lon is not None else None,
            lat=float(lat) if lat is not None else None,
            site=site,
            **data_fields
        )

    @staticmethod
    def from_arrays(
        time: Union[List[datetime], np.ndarray],
        temperature: Optional[Union[List[float], np.ndarray]] = None,
        precipitation: Optional[Union[List[float], np.ndarray]] = None,
        wind_speed: Optional[Union[List[float], np.ndarray]] = None,
        lon: Optional[float] = None,
        lat: Optional[float] = None,
        site: Optional[str] = None,
        **kwargs
    ) -> WeatherData:
        """
        Create WeatherData from numpy arrays or lists.

        Args:
            time: Array of datetimes
            temperature: Array of temperatures (optional)
            precipitation: Array of precipitation (optional)
            wind_speed: Array of wind speeds (optional)
            lon: Longitude (optional)
            lat: Latitude (optional)
            site: Site name (optional)
            **kwargs: Additional weather parameter arrays

        Returns:
            WeatherData object

        Example:
            >>> import numpy as np
            >>> from datetime import datetime, timedelta
            >>> times = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(10)]
            >>> temps = np.random.uniform(10, 30, 10)
            >>> parser = WeatherInputParser()
            >>> data = parser.from_arrays(time=times, temperature=temps, site="NYC")
        """
        return WeatherData(
            time=time,
            temperature=temperature,
            precipitation=precipitation,
            wind_speed=wind_speed,
            lon=lon,
            lat=lat,
            site=site,
            **kwargs
        )
