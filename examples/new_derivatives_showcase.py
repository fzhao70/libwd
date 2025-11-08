"""
Showcase of new weather derivatives (v2.0).

Demonstrates:
- Snow derivatives
- Frost and agricultural derivatives (GDD, CHU)
- Humidity derivatives
- Solar radiation derivatives
"""

from datetime import datetime, timedelta
import numpy as np
from weather_derivatives import (
    WeatherInputParser,
    SnowDerivative,
    FrostDerivative,
    HumidityDerivative,
    SolarDerivative,
)


def example_snow_derivatives():
    """Example: Snow derivatives for ski resort."""
    print("=" * 70)
    print("Snow Derivatives - Ski Resort Risk Management")
    print("=" * 70)

    parser = WeatherInputParser()

    # Winter season data for ski resort
    dates = [datetime(2024, 12, 1) + timedelta(days=i) for i in range(90)]
    snowfall_data = np.random.gamma(2, 3, 90)  # Daily snowfall in cm
    snow_depth_data = np.cumsum(snowfall_data) - np.random.uniform(0, 2, 90)  # Melting
    snow_depth_data = np.maximum(snow_depth_data, 0)

    weather_data = parser.from_site(
        time=dates,
        site="Aspen Ski Resort",
        snowfall=snowfall_data.tolist(),
        snow_depth=snow_depth_data.tolist()
    )

    # Calculate snow metrics
    snow = SnowDerivative(weather_data, snow_day_threshold=2.5)

    total_snowfall = snow.calculate_total_snowfall()
    snow_days = snow.calculate_snow_days()
    max_depth = snow.calculate_max_snow_depth()
    avg_depth = snow.calculate_average_snow_depth()
    snow_cover_days = snow.calculate_snow_cover_days(min_depth=10)

    print(f"Season Statistics:")
    print(f"  Total Snowfall: {total_snowfall:.1f} cm")
    print(f"  Snow Days (>2.5cm): {snow_days}")
    print(f"  Max Snow Depth: {max_depth:.1f} cm")
    print(f"  Average Snow Depth: {avg_depth:.1f} cm")
    print(f"  Days with >10cm Snow Cover: {snow_cover_days}")

    # Derivative payoffs
    payoff_snowfall = snow.payoff_total_snowfall(
        strike=200,  # 200cm strike
        tick_value=1000,  # $1000 per cm
        option_type="put"  # Pays if snowfall is low
    )

    payoff_snow_days = snow.payoff_snow_days(
        strike=30,  # 30 snow days
        tick_value=5000,  # $5000 per day
        option_type="put"
    )

    print(f"\nDerivative Payoffs:")
    print(f"  Snowfall Put (Strike=200cm): ${payoff_snowfall:,.2f}")
    print(f"  Snow Days Put (Strike=30): ${payoff_snow_days:,.2f}")

    print()


def example_frost_gdd_derivatives():
    """Example: Agricultural derivatives for crop insurance."""
    print("=" * 70)
    print("Frost & Growing Degree Days - Crop Insurance")
    print("=" * 70)

    parser = WeatherInputParser()

    # Growing season data
    dates = [datetime(2024, 4, 1) + timedelta(days=i) for i in range(150)]

    # Simulate temperatures (warming through growing season)
    base_temps = np.linspace(12, 25, 150)  # Seasonal trend
    daily_variation = np.random.normal(0, 3, 150)
    temperatures = base_temps + daily_variation

    # Min and max temperatures
    min_temps = temperatures - np.random.uniform(3, 7, 150)
    max_temps = temperatures + np.random.uniform(3, 7, 150)

    weather_data = parser.from_site(
        time=dates,
        site="Iowa Corn Belt",
        temperature=temperatures.tolist(),
        min_temperature=min_temps.tolist(),
        max_temperature=max_temps.tolist()
    )

    # Calculate agricultural metrics
    frost = FrostDerivative(
        weather_data,
        frost_threshold=0.0,  # 0°C
        base_temperature=10.0  # 10°C base for corn
    )

    gdd = frost.calculate_growing_degree_days(max_temperature=30.0)
    frost_days = frost.calculate_frost_days()
    frost_free = frost.calculate_frost_free_days()
    freeze_events = frost.calculate_freeze_events(severe_threshold=-2.0)

    # Calculate CHU (for corn)
    chu = frost.calculate_crop_heat_units()

    print(f"Growing Season Metrics:")
    print(f"  Growing Degree Days: {gdd:.0f}")
    print(f"  Crop Heat Units: {chu:.0f}")
    print(f"  Frost Days: {frost_days}")
    print(f"  Frost-Free Days: {frost_free}")
    print(f"  Severe Freeze Events: {freeze_events}")

    # Derivative payoffs
    payoff_gdd = frost.payoff_gdd(
        strike=2500,  # 2500 GDD target
        tick_value=50,  # $50 per GDD unit
        option_type="put"  # Pays if GDD is low
    )

    payoff_frost = frost.payoff_frost_days(
        strike=5,  # More than 5 frost days
        tick_value=2000,  # $2000 per excess frost day
        option_type="call"  # Pays for excess frost
    )

    print(f"\nDerivative Payoffs:")
    print(f"  GDD Put (Strike=2500): ${payoff_gdd:,.2f}")
    print(f"  Frost Days Call (Strike=5): ${payoff_frost:,.2f}")

    print()


def example_humidity_derivatives():
    """Example: Humidity derivatives for health sector."""
    print("=" * 70)
    print("Humidity Derivatives - Health Sector Risk")
    print("=" * 70)

    parser = WeatherInputParser()

    # Summer humidity data
    dates = [datetime(2024, 6, 1) + timedelta(days=i) for i in range(92)]

    # Simulate humidity (varies throughout summer)
    humidity_data = np.random.beta(6, 4, 92) * 100  # Beta distribution, scaled to 0-100%
    temperature_data = np.random.normal(28, 5, 92)  # Summer temperatures in Celsius

    weather_data = parser.from_site(
        time=dates,
        site="Houston Medical Center",
        humidity=humidity_data.tolist(),
        temperature=temperature_data.tolist()
    )

    # Calculate humidity metrics
    humidity = HumidityDerivative(
        weather_data,
        high_humidity_threshold=80.0,
        low_humidity_threshold=30.0
    )

    avg_humidity = humidity.calculate_average_humidity()
    high_days = humidity.calculate_high_humidity_days()
    low_days = humidity.calculate_low_humidity_days()
    stress_days = humidity.calculate_humidity_stress_index()
    heat_index_days = humidity.calculate_heat_index_days(heat_index_threshold=35.0)

    print(f"Humidity Metrics:")
    print(f"  Average Humidity: {avg_humidity:.1f}%")
    print(f"  High Humidity Days (>80%): {high_days}")
    print(f"  Low Humidity Days (<30%): {low_days}")
    print(f"  Humidity Stress Days: {stress_days}")
    print(f"  Heat Index Days (>35°C): {heat_index_days}")

    # Derivative payoffs
    payoff_high = humidity.payoff_humidity_days(
        strike=30,
        tick_value=500,
        option_type="call",
        metric="high"
    )

    print(f"\nDerivative Payoffs:")
    print(f"  High Humidity Call (Strike=30 days): ${payoff_high:,.2f}")

    print()


def example_solar_derivatives():
    """Example: Solar derivatives for renewable energy."""
    print("=" * 70)
    print("Solar Radiation Derivatives - Solar Power Plant")
    print("=" * 70)

    parser = WeatherInputParser()

    # Annual solar data
    dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(365)]

    # Simulate solar radiation (seasonal pattern)
    day_of_year = np.arange(365)
    seasonal_pattern = 5 + 3 * np.sin(2 * np.pi * (day_of_year - 80) / 365)  # Peak in summer
    daily_variation = np.random.gamma(2, 0.5, 365)
    solar_radiation = seasonal_pattern + daily_variation  # MJ/m²/day or kWh/m²/day

    # Sunshine hours
    sunshine_hours = solar_radiation * 1.5 + np.random.normal(0, 1, 365)
    sunshine_hours = np.clip(sunshine_hours, 0, 14)

    weather_data = parser.from_coordinates(
        time=dates,
        lon=-110.0,  # Arizona
        lat=33.0,
        solar_radiation=solar_radiation.tolist(),
        sunshine_hours=sunshine_hours.tolist()
    )

    # Calculate solar metrics
    solar = SolarDerivative(
        weather_data,
        panel_efficiency=0.20,  # 20% efficient panels
        panel_area=10000  # 10,000 m² solar farm
    )

    total_irradiance = solar.calculate_total_irradiance()
    avg_irradiance = solar.calculate_average_irradiance()
    total_sunshine = solar.calculate_sunshine_hours()
    cloudy_days = solar.calculate_cloudy_days(max_sunshine_hours=4.0)
    energy_production = solar.calculate_solar_energy_production()

    print(f"Solar Metrics:")
    print(f"  Total Annual Irradiance: {total_irradiance:.0f} units")
    print(f"  Average Daily Irradiance: {avg_irradiance:.2f} units")
    print(f"  Total Sunshine Hours: {total_sunshine:.0f} hours")
    print(f"  Cloudy Days (<4h sun): {cloudy_days}")
    print(f"  Estimated Energy Production: {energy_production:,.0f} kWh")

    # Derivative payoffs
    payoff_irradiance = solar.payoff_irradiance(
        strike=2500,  # 2500 units annual irradiance
        tick_value=100,  # $100 per unit
        option_type="put"  # Protection against low solar
    )

    payoff_energy = solar.payoff_energy_production(
        strike=15000000,  # 15 million kWh
        tick_value=0.05,  # $0.05 per kWh
        option_type="put"
    )

    print(f"\nDerivative Payoffs:")
    print(f"  Irradiance Put (Strike=2500): ${payoff_irradiance:,.2f}")
    print(f"  Energy Production Put: ${payoff_energy:,.2f}")

    print()


if __name__ == "__main__":
    print("\n")
    print("*" * 70)
    print("   WEATHER DERIVATIVES v2.0 - NEW FEATURES SHOWCASE")
    print("*" * 70)
    print("\n")

    example_snow_derivatives()
    example_frost_gdd_derivatives()
    example_humidity_derivatives()
    example_solar_derivatives()

    print("=" * 70)
    print("All new derivative examples completed successfully!")
    print("=" * 70)
