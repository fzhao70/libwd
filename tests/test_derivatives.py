"""
Comprehensive tests for weather derivatives library.
"""

import pytest
from datetime import datetime, timedelta
import numpy as np

from weather_derivatives import (
    WeatherData,
    WeatherInputParser,
    HDD, CDD, CAT, PAC,
    PrecipitationDerivative,
    WindDerivative,
    SnowDerivative,
    FrostDerivative,
    HumidityDerivative,
    SolarDerivative,
    DerivativeValuation,
)


class TestWeatherData:
    """Test WeatherData class."""

    def test_create_with_coordinates(self):
        """Test creating weather data with coordinates."""
        wd = WeatherData(
            time=datetime(2024, 1, 1),
            lon=-74.006,
            lat=40.7128,
            temperature=25.0
        )

        assert len(wd) == 1
        assert wd.location["type"] == "coordinates"
        assert wd.location["lon"] == -74.006
        assert wd.location["lat"] == 40.7128
        assert wd.get_temperature() == 25.0

    def test_create_with_site(self):
        """Test creating weather data with site name."""
        wd = WeatherData(
            time=datetime(2024, 1, 1),
            site="New York",
            temperature=25.0
        )

        assert wd.location["type"] == "site"
        assert wd.location["name"] == "New York"

    def test_create_time_only(self):
        """Test creating weather data with time only."""
        wd = WeatherData(
            time=datetime(2024, 1, 1),
            temperature=25.0
        )

        assert wd.location["type"] == "none"

    def test_time_series(self):
        """Test time series data."""
        times = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(5)]
        temps = [20.0, 22.0, 24.0, 26.0, 28.0]

        wd = WeatherData(time=times, temperature=temps)

        assert len(wd) == 5
        assert wd.is_timeseries
        assert wd.get_temperature() == temps

    def test_data_length_mismatch(self):
        """Test that data length mismatch raises error."""
        times = [datetime(2024, 1, 1), datetime(2024, 1, 2)]
        temps = [20.0, 22.0, 24.0]  # Wrong length

        with pytest.raises(ValueError, match="Length mismatch"):
            WeatherData(time=times, temperature=temps)


class TestInputParser:
    """Test WeatherInputParser."""

    def test_from_coordinates(self):
        """Test parsing from coordinates."""
        parser = WeatherInputParser()
        wd = parser.from_coordinates(
            time=datetime(2024, 1, 1),
            lon=-74.006,
            lat=40.7128,
            temperature=25.0
        )

        assert wd.location["type"] == "coordinates"
        assert wd.get_temperature() == 25.0

    def test_from_site(self):
        """Test parsing from site name."""
        parser = WeatherInputParser()
        wd = parser.from_site(
            time=datetime(2024, 1, 1),
            site="Chicago",
            temperature=25.0
        )

        assert wd.location["type"] == "site"
        assert wd.location["name"] == "Chicago"

    def test_from_time_only(self):
        """Test parsing from time only."""
        parser = WeatherInputParser()
        wd = parser.from_time_only(
            time=datetime(2024, 1, 1),
            temperature=25.0
        )

        assert wd.location["type"] == "none"

    def test_from_arrays(self):
        """Test parsing from numpy arrays."""
        parser = WeatherInputParser()
        times = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(5)]
        temps = np.array([20.0, 22.0, 24.0, 26.0, 28.0])

        wd = parser.from_arrays(time=times, temperature=temps, site="Test")

        assert len(wd) == 5
        assert wd.location["name"] == "Test"


class TestHDD:
    """Test HDD (Heating Degree Days) derivative."""

    def test_single_day_hdd(self):
        """Test HDD calculation for single day."""
        wd = WeatherData(
            time=datetime(2024, 1, 1),
            temperature=50.0
        )

        hdd = HDD(wd, reference_temperature=65.0)
        result = hdd.calculate()

        assert result == 15.0  # 65 - 50 = 15

    def test_no_hdd_warm_day(self):
        """Test HDD is zero on warm day."""
        wd = WeatherData(
            time=datetime(2024, 1, 1),
            temperature=70.0
        )

        hdd = HDD(wd, reference_temperature=65.0)
        result = hdd.calculate()

        assert result == 0.0

    def test_cumulative_hdd(self):
        """Test cumulative HDD over multiple days."""
        times = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(3)]
        temps = [50.0, 55.0, 60.0]

        wd = WeatherData(time=times, temperature=temps)
        hdd = HDD(wd, reference_temperature=65.0)
        result = hdd.calculate(cumulative=True)

        # (65-50) + (65-55) + (65-60) = 15 + 10 + 5 = 30
        assert result == 30.0

    def test_daily_hdd(self):
        """Test daily HDD values."""
        times = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(3)]
        temps = [50.0, 55.0, 60.0]

        wd = WeatherData(time=times, temperature=temps)
        hdd = HDD(wd, reference_temperature=65.0)
        result = hdd.calculate(cumulative=False)

        assert result == [15.0, 10.0, 5.0]

    def test_hdd_call_payoff(self):
        """Test HDD call option payoff."""
        wd = WeatherData(time=datetime(2024, 1, 1), temperature=50.0)
        hdd = HDD(wd, reference_temperature=65.0)

        # HDD = 15, strike = 10, tick = 100
        payoff = hdd.payoff(strike=10, tick_value=100, option_type="call")
        assert payoff == 500.0  # (15 - 10) * 100

    def test_hdd_put_payoff(self):
        """Test HDD put option payoff."""
        wd = WeatherData(time=datetime(2024, 1, 1), temperature=60.0)
        hdd = HDD(wd, reference_temperature=65.0)

        # HDD = 5, strike = 10, tick = 100
        payoff = hdd.payoff(strike=10, tick_value=100, option_type="put")
        assert payoff == 500.0  # (10 - 5) * 100


class TestCDD:
    """Test CDD (Cooling Degree Days) derivative."""

    def test_single_day_cdd(self):
        """Test CDD calculation for single day."""
        wd = WeatherData(time=datetime(2024, 7, 1), temperature=85.0)
        cdd = CDD(wd, reference_temperature=65.0)
        result = cdd.calculate()

        assert result == 20.0  # 85 - 65 = 20

    def test_no_cdd_cool_day(self):
        """Test CDD is zero on cool day."""
        wd = WeatherData(time=datetime(2024, 7, 1), temperature=60.0)
        cdd = CDD(wd, reference_temperature=65.0)
        result = cdd.calculate()

        assert result == 0.0

    def test_cumulative_cdd(self):
        """Test cumulative CDD over multiple days."""
        times = [datetime(2024, 7, 1) + timedelta(days=i) for i in range(3)]
        temps = [85.0, 90.0, 95.0]

        wd = WeatherData(time=times, temperature=temps)
        cdd = CDD(wd, reference_temperature=65.0)
        result = cdd.calculate(cumulative=True)

        # (85-65) + (90-65) + (95-65) = 20 + 25 + 30 = 75
        assert result == 75.0


class TestCAT:
    """Test CAT (Cumulative Average Temperature) derivative."""

    def test_cat_sum(self):
        """Test CAT sum calculation."""
        times = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(3)]
        temps = [50.0, 55.0, 60.0]

        wd = WeatherData(time=times, temperature=temps)
        cat = CAT(wd)
        result = cat.calculate(method="sum")

        assert result == 165.0

    def test_cat_mean(self):
        """Test CAT mean calculation."""
        times = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(3)]
        temps = [50.0, 55.0, 60.0]

        wd = WeatherData(time=times, temperature=temps)
        cat = CAT(wd)
        result = cat.calculate(method="mean")

        assert result == 55.0


class TestPrecipitationDerivative:
    """Test precipitation derivatives."""

    def test_total_precipitation(self):
        """Test total precipitation calculation."""
        times = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(3)]
        precip = [10.0, 5.0, 15.0]

        wd = WeatherData(time=times, precipitation=precip)
        pd = PrecipitationDerivative(wd)
        result = pd.calculate_total_precipitation()

        assert result == 30.0

    def test_rain_days(self):
        """Test rain days counting."""
        times = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(5)]
        precip = [10.0, 0.0, 5.0, 0.05, 15.0]

        wd = WeatherData(time=times, precipitation=precip)
        pd = PrecipitationDerivative(wd, precipitation_threshold=0.1)
        result = pd.calculate_rain_days()

        assert result == 3  # Days with > 0.1mm

    def test_dry_days(self):
        """Test dry days counting."""
        times = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(5)]
        precip = [10.0, 0.0, 5.0, 0.05, 15.0]

        wd = WeatherData(time=times, precipitation=precip)
        pd = PrecipitationDerivative(wd, precipitation_threshold=0.1)
        result = pd.calculate_dry_days()

        assert result == 2


class TestWindDerivative:
    """Test wind derivatives."""

    def test_average_wind_speed(self):
        """Test average wind speed calculation."""
        times = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(3)]
        wind = [5.0, 7.0, 6.0]

        wd = WeatherData(time=times, wind_speed=wind)
        wd_deriv = WindDerivative(wd)
        result = wd_deriv.calculate_average_wind_speed()

        assert result == 6.0

    def test_calm_days(self):
        """Test calm days counting."""
        times = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(5)]
        wind = [2.0, 5.0, 1.5, 8.0, 2.5]

        wd = WeatherData(time=times, wind_speed=wind)
        wd_deriv = WindDerivative(wd)
        result = wd_deriv.calculate_calm_days(calm_threshold=3.0)

        assert result == 3


class TestDerivativeValuation:
    """Test valuation methods."""

    def test_historical_burn_analysis(self):
        """Test historical burn analysis."""
        historical_data = [1000, 1100, 1200, 1300, 1400]
        valuation = DerivativeValuation()

        result = valuation.historical_burn_analysis(
            historical_data=historical_data,
            strike=1200,
            tick_value=100,
            option_type="call"
        )

        assert "mean_payoff" in result
        assert "median_payoff" in result
        assert "probability_itm" in result
        assert result["num_scenarios"] == 5

    def test_monte_carlo_valuation(self):
        """Test Monte Carlo valuation."""
        valuation = DerivativeValuation()

        result = valuation.monte_carlo_valuation(
            mean=1000,
            std=100,
            strike=1100,
            tick_value=50,
            option_type="call",
            num_simulations=1000
        )

        assert "expected_payoff" in result
        assert "probability_itm" in result
        assert result["num_simulations"] == 1000
        assert result["expected_payoff"] >= 0

    def test_present_value(self):
        """Test present value calculation."""
        valuation = DerivativeValuation(discount_rate=0.05)

        pv = valuation.calculate_present_value(
            expected_payoff=1000,
            time_to_maturity_days=365
        )

        # Should be less than 1000 due to discounting
        assert pv < 1000
        assert pv > 950  # Roughly 1000 / 1.05

    def test_risk_metrics(self):
        """Test risk metrics calculation."""
        payoffs = np.random.normal(1000, 200, 1000)
        valuation = DerivativeValuation()

        metrics = valuation.calculate_risk_metrics(payoffs, confidence_level=0.95)

        assert "var" in metrics
        assert "cvar" in metrics
        assert "expected_value" in metrics
        assert "volatility" in metrics

    def test_black_scholes_approximation(self):
        """Test Black-Scholes approximation."""
        valuation = DerivativeValuation(discount_rate=0.05)

        call_value = valuation.black_scholes_approximation(
            current_index=1000,
            strike=1000,
            time_to_maturity_years=0.5,
            volatility=0.20,
            option_type="call"
        )

        # ATM call should have positive value
        assert call_value > 0

    def test_greeks_calculation(self):
        """Test Greeks calculation."""
        valuation = DerivativeValuation(discount_rate=0.05)

        greeks = valuation.calculate_greeks(
            current_index=1000,
            strike=1000,
            time_to_maturity_years=0.5,
            volatility=0.20,
            option_type="call"
        )

        assert "delta" in greeks
        assert "gamma" in greeks
        assert "vega" in greeks
        assert "theta" in greeks
        assert "price" in greeks

        # Delta for ATM call should be around 0.5
        assert 0.3 < greeks["delta"] < 0.7


class TestIntegration:
    """Integration tests for complete workflows."""

    def test_complete_hdd_workflow(self):
        """Test complete HDD workflow from data to valuation."""
        # Create data
        parser = WeatherInputParser()
        times = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(30)]
        temps = [40.0 + i % 20 for i in range(30)]

        wd = parser.from_site(time=times, site="Chicago", temperature=temps)

        # Calculate HDD
        hdd = HDD(wd, reference_temperature=65.0)
        hdd_value = hdd.calculate()

        assert hdd_value > 0

        # Calculate payoff
        payoff = hdd.payoff(strike=400, tick_value=1000, option_type="call")

        assert payoff >= 0

    def test_multiple_derivatives_same_data(self):
        """Test using multiple derivatives with same data."""
        times = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(10)]
        temps = [60.0, 65.0, 70.0, 65.0, 60.0, 55.0, 60.0, 65.0, 70.0, 75.0]

        wd = WeatherData(time=times, temperature=temps)

        # Calculate both HDD and CDD
        hdd = HDD(wd, reference_temperature=65.0)
        cdd = CDD(wd, reference_temperature=65.0)
        cat = CAT(wd)

        hdd_val = hdd.calculate()
        cdd_val = cdd.calculate()
        cat_val = cat.calculate(method="sum")

        assert hdd_val >= 0
        assert cdd_val >= 0
        assert cat_val == sum(temps)


class TestSnowDerivative:
    """Test snow derivatives."""

    def test_total_snowfall(self):
        """Test total snowfall calculation."""
        times = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(5)]
        snowfall = [10.0, 5.0, 15.0, 0.0, 8.0]

        wd = WeatherData(time=times, snowfall=snowfall)
        snow = SnowDerivative(wd)
        result = snow.calculate_total_snowfall()

        assert result == 38.0

    def test_snow_days(self):
        """Test snow days counting."""
        times = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(5)]
        snowfall = [10.0, 0.5, 15.0, 0.0, 3.0]

        wd = WeatherData(time=times, snowfall=snowfall)
        snow = SnowDerivative(wd, snow_day_threshold=1.0)
        result = snow.calculate_snow_days()

        assert result == 3  # Days with >= 1.0cm


class TestFrostDerivative:
    """Test frost and agricultural derivatives."""

    def test_frost_days(self):
        """Test frost days counting."""
        times = [datetime(2024, 4, 1) + timedelta(days=i) for i in range(5)]
        min_temps = [2.0, -1.0, 5.0, -0.5, 3.0]

        wd = WeatherData(time=times, min_temperature=min_temps, temperature=[15, 10, 18, 12, 16])
        frost = FrostDerivative(wd, frost_threshold=0.0)
        result = frost.calculate_frost_days()

        assert result == 2  # Days with temp <= 0°C

    def test_growing_degree_days(self):
        """Test GDD calculation."""
        times = [datetime(2024, 5, 1) + timedelta(days=i) for i in range(3)]
        temps = [20.0, 25.0, 15.0]

        wd = WeatherData(time=times, temperature=temps)
        frost = FrostDerivative(wd, base_temperature=10.0)
        gdd = frost.calculate_growing_degree_days()

        # (20-10) + (25-10) + (15-10) = 10 + 15 + 5 = 30
        assert gdd == 30.0


class TestHumidityDerivative:
    """Test humidity derivatives."""

    def test_average_humidity(self):
        """Test average humidity calculation."""
        times = [datetime(2024, 6, 1) + timedelta(days=i) for i in range(3)]
        humidity = [60.0, 80.0, 70.0]

        wd = WeatherData(time=times, humidity=humidity)
        humid = HumidityDerivative(wd)
        result = humid.calculate_average_humidity()

        assert result == 70.0

    def test_high_humidity_days(self):
        """Test high humidity days counting."""
        times = [datetime(2024, 6, 1) + timedelta(days=i) for i in range(5)]
        humidity = [60.0, 85.0, 90.0, 70.0, 95.0]

        wd = WeatherData(time=times, humidity=humidity)
        humid = HumidityDerivative(wd, high_humidity_threshold=80.0)
        result = humid.calculate_high_humidity_days()

        assert result == 3  # Days with >= 80%


class TestSolarDerivative:
    """Test solar radiation derivatives."""

    def test_total_irradiance(self):
        """Test total irradiance calculation."""
        times = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(3)]
        irradiance = [5.0, 6.0, 4.5]

        wd = WeatherData(time=times, solar_radiation=irradiance)
        solar = SolarDerivative(wd)
        result = solar.calculate_total_irradiance()

        assert result == 15.5

    def test_sunshine_hours(self):
        """Test sunshine hours calculation."""
        times = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(3)]
        sunshine = [8.0, 10.0, 6.0]

        wd = WeatherData(time=times, sunshine_hours=sunshine)
        solar = SolarDerivative(wd)
        result = solar.calculate_sunshine_hours()

        assert result == 24.0


class TestAdvancedValuation:
    """Test advanced valuation methods."""

    def test_burn_rate_analysis(self):
        """Test burn rate analysis."""
        valuation = DerivativeValuation()
        historical_data = [1000, 1100, 1200, 1300, 1400]

        result = valuation.burn_rate_analysis(
            historical_data=historical_data,
            strike=1200,
            tick_value=100,
            distribution_fit="empirical"
        )

        assert "burn_rate" in result
        assert "mean_payoff" in result
        assert 0 <= result["burn_rate"] <= 1

    def test_time_series_forecast(self):
        """Test time series forecasting."""
        valuation = DerivativeValuation()
        historical_data = np.random.normal(1000, 100, 30)

        result = valuation.time_series_forecast_valuation(
            historical_data=historical_data,
            strike=1100,
            tick_value=50,
            forecast_periods=10,
            num_simulations=100
        )

        assert "expected_payoff" in result
        assert "ar_coefficient" in result
        assert result["expected_payoff"] >= 0

    def test_weather_index_insurance(self):
        """Test weather index insurance pricing."""
        valuation = DerivativeValuation()
        historical_data = np.random.normal(1000, 150, 20)

        result = valuation.weather_index_insurance_pricing(
            historical_data=historical_data,
            trigger=900,
            exit=1200,
            limit=50000
        )

        assert "total_premium" in result
        assert "pure_premium" in result
        assert "loss_ratio" in result
        assert result["total_premium"] > result["pure_premium"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
