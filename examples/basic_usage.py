"""
Basic usage examples for the weather derivatives library.

Demonstrates the three input formats:
1. (time, lon, lat) - Geographical coordinates
2. (time, site) - Named location
3. (time) - Time-based only
"""

from datetime import datetime, timedelta
from weather_derivatives import (
    WeatherInputParser,
    HDD, CDD, CAT,
    PrecipitationDerivative,
    WindDerivative,
)


def example_1_coordinates():
    """Example 1: Using (time, lon, lat) format."""
    print("=" * 60)
    print("Example 1: Using (time, lon, lat) - Geographical Coordinates")
    print("=" * 60)

    # Create parser
    parser = WeatherInputParser()

    # Single time point with coordinates
    weather_data = parser.from_coordinates(
        time=datetime(2024, 1, 15),
        lon=-74.006,  # New York City
        lat=40.7128,
        temperature=25.0,  # Fahrenheit
        precipitation=0.5,  # mm
        wind_speed=5.2  # m/s
    )

    print(f"Weather data: {weather_data}")

    # Calculate HDD
    hdd = HDD(weather_data, reference_temperature=65.0)
    hdd_value = hdd.calculate()
    print(f"Heating Degree Days: {hdd_value:.2f}")

    # Calculate option payoff
    payoff = hdd.payoff(strike=40, tick_value=250, option_type="call")
    print(f"HDD Call Option Payoff (Strike=40): ${payoff:,.2f}")

    print()


def example_2_site():
    """Example 2: Using (time, site) format."""
    print("=" * 60)
    print("Example 2: Using (time, site) - Named Location")
    print("=" * 60)

    # Create parser
    parser = WeatherInputParser()

    # Time series data for a specific site
    dates = [datetime(2024, 7, 1) + timedelta(days=i) for i in range(30)]
    temperatures = [85 + i % 10 for i in range(30)]  # Varying temps in summer

    weather_data = parser.from_site(
        time=dates,
        site="Chicago O'Hare",
        temperature=temperatures
    )

    print(f"Weather data: {weather_data}")

    # Calculate CDD
    cdd = CDD(weather_data, reference_temperature=65.0)
    cdd_value = cdd.calculate()
    print(f"Cooling Degree Days: {cdd_value:.2f}")

    # Monthly aggregation
    monthly_cdd = cdd.calculate_monthly()
    print(f"Monthly CDD: {monthly_cdd}")

    # Calculate option payoff
    payoff = cdd.payoff(strike=600, tick_value=100, option_type="call")
    print(f"CDD Call Option Payoff (Strike=600): ${payoff:,.2f}")

    print()


def example_3_time_only():
    """Example 3: Using (time) format - time data only."""
    print("=" * 60)
    print("Example 3: Using (time) - Time-Based Only")
    print("=" * 60)

    # Create parser
    parser = WeatherInputParser()

    # Time series without location (generic or aggregated data)
    dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(90)]
    temperatures = [30 + 20 * (i / 90) for i in range(90)]  # Warming trend

    weather_data = parser.from_time_only(
        time=dates,
        temperature=temperatures
    )

    print(f"Weather data: {weather_data}")

    # Calculate CAT (Cumulative Average Temperature)
    cat = CAT(weather_data)
    cat_sum = cat.calculate(method="sum")
    cat_mean = cat.calculate(method="mean")

    print(f"CAT (Sum): {cat_sum:.2f}")
    print(f"CAT (Mean): {cat_mean:.2f}")

    # Calculate option payoff
    payoff = cat.payoff(strike=4000, tick_value=50, option_type="put")
    print(f"CAT Put Option Payoff (Strike=4000): ${payoff:,.2f}")

    print()


def example_4_precipitation():
    """Example 4: Precipitation derivatives."""
    print("=" * 60)
    print("Example 4: Precipitation Derivatives")
    print("=" * 60)

    # Create parser
    parser = WeatherInputParser()

    # Rainfall data for agricultural derivative
    dates = [datetime(2024, 6, 1) + timedelta(days=i) for i in range(30)]
    rainfall = [0, 2.5, 0, 0, 10.5, 0, 0, 0, 3.2, 0] * 3  # Sporadic rainfall

    weather_data = parser.from_site(
        time=dates,
        site="Iowa Corn Belt",
        precipitation=rainfall
    )

    # Calculate precipitation metrics
    precip_deriv = PrecipitationDerivative(weather_data, precipitation_threshold=0.1)

    total_precip = precip_deriv.calculate_total_precipitation()
    rain_days = precip_deriv.calculate_rain_days()
    dry_days = precip_deriv.calculate_dry_days()

    print(f"Total Precipitation: {total_precip:.2f} mm")
    print(f"Rain Days: {rain_days}")
    print(f"Dry Days: {dry_days}")

    # Option payoffs
    payoff_precip = precip_deriv.payoff_total_precipitation(
        strike=50, tick_value=10, option_type="put"
    )
    payoff_days = precip_deriv.payoff_rain_days(
        strike=8, tick_value=500, option_type="call"
    )

    print(f"Precipitation Put Payoff (Strike=50mm): ${payoff_precip:,.2f}")
    print(f"Rain Days Call Payoff (Strike=8 days): ${payoff_days:,.2f}")

    print()


def example_5_wind():
    """Example 5: Wind derivatives."""
    print("=" * 60)
    print("Example 5: Wind Derivatives")
    print("=" * 60)

    # Create parser
    parser = WeatherInputParser()

    # Wind data for wind farm
    dates = [datetime(2024, 3, 1) + timedelta(days=i) for i in range(31)]
    wind_speeds = [5 + 3 * (i % 7) / 7 for i in range(31)]  # Varying wind

    weather_data = parser.from_coordinates(
        time=dates,
        lon=-100.0,  # West Texas
        lat=32.0,
        wind_speed=wind_speeds
    )

    # Calculate wind metrics
    wind_deriv = WindDerivative(weather_data, rotor_diameter=90.0)  # 90m rotor

    avg_wind = wind_deriv.calculate_average_wind_speed()
    total_energy = wind_deriv.calculate_total_wind_energy(hours_per_reading=24)
    calm_days = wind_deriv.calculate_calm_days(calm_threshold=3.0)

    print(f"Average Wind Speed: {avg_wind:.2f} m/s")
    print(f"Total Wind Energy: {total_energy:,.0f} Wh")
    print(f"Calm Days: {calm_days}")

    # Option payoff
    payoff = wind_deriv.payoff_wind_power(
        strike=3000000, tick_value=0.01, option_type="put"
    )
    print(f"Wind Power Put Payoff: ${payoff:,.2f}")

    print()


def example_6_arrays():
    """Example 6: Using numpy arrays."""
    print("=" * 60)
    print("Example 6: Using NumPy Arrays")
    print("=" * 60)

    import numpy as np

    # Create parser
    parser = WeatherInputParser()

    # Generate synthetic data with numpy
    num_days = 60
    dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(num_days)]
    temperatures = np.random.normal(50, 15, num_days)  # Random temps around 50°F
    precipitation = np.random.exponential(2, num_days)  # Exponential rainfall

    weather_data = parser.from_arrays(
        time=dates,
        temperature=temperatures,
        precipitation=precipitation,
        site="Generic Location"
    )

    print(f"Weather data: {weather_data}")

    # Calculate HDD and CDD together
    hdd = HDD(weather_data, reference_temperature=65.0)
    cdd = CDD(weather_data, reference_temperature=65.0)

    print(f"Total HDD: {hdd.calculate():.2f}")
    print(f"Total CDD: {cdd.calculate():.2f}")

    # Precipitation analysis
    precip = PrecipitationDerivative(weather_data)
    print(f"Total Precipitation: {precip.calculate_total_precipitation():.2f} mm")
    print(f"90th Percentile Precipitation: {precip.calculate_percentile(90):.2f} mm")

    print()


if __name__ == "__main__":
    example_1_coordinates()
    example_2_site()
    example_3_time_only()
    example_4_precipitation()
    example_5_wind()
    example_6_arrays()

    print("=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)
