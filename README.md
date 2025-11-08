# Weather Derivatives Library

A comprehensive Python library for calculating and pricing weather derivatives. Supports multiple input formats and a wide range of weather derivative types.

## Features

- **Flexible Input Formats**: Supports three input formats:
  - `(time, lon, lat)` - Geographical coordinates
  - `(time, site)` - Named location
  - `(time)` - Time-based data only

- **Temperature Derivatives**:
  - HDD (Heating Degree Days)
  - CDD (Cooling Degree Days)
  - CAT (Cumulative Average Temperature)
  - PAC (Pacific Temperature)

- **Precipitation Derivatives**:
  - Total precipitation
  - Rain days counting
  - Extreme event analysis
  - Dry days counting

- **Wind Derivatives**:
  - Wind power calculations
  - Calm/high wind day counting
  - Wind energy estimates
  - Turbine production modeling

- **Snow Derivatives**:
  - Total snowfall accumulation
  - Snow days counting
  - Snow depth analysis
  - Snow cover duration

- **Frost & Agricultural Derivatives**:
  - Frost days counting
  - Growing Degree Days (GDD)
  - Crop Heat Units (CHU)
  - Freeze events tracking

- **Humidity Derivatives**:
  - Average humidity calculations
  - High/low humidity days
  - Vapor Pressure Deficit (VPD)
  - Heat index analysis

- **Solar Radiation Derivatives**:
  - Solar irradiance totals
  - Sunshine hours
  - Solar energy production estimates
  - Cloudy days counting
  - Capacity factor analysis

- **Advanced Pricing & Valuation** (20+ state-of-the-art methods):
  - Historical burn analysis with distribution fitting
  - Monte Carlo simulation (standard, importance sampling, control variates)
  - Black-Scholes approximation
  - Greeks calculation (delta, gamma, vega, theta)
  - Risk metrics (VaR, CVaR, skewness, kurtosis)
  - Bootstrap resampling (block bootstrap)
  - Extreme Value Theory (EVT) pricing
  - Quantile regression
  - Regime-switching models
  - Time series forecasting (AR models)
  - Weather index insurance pricing
  - Asian option pricing
  - Spread option pricing
  - Portfolio optimization
  - Stochastic volatility models
  - Copula-based multi-index valuation
  - Sensitivity analysis

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/libwd.git
cd libwd

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

## Quick Start

### Example 1: Temperature Derivatives with Coordinates

```python
from datetime import datetime
from weather_derivatives import WeatherInputParser, HDD, CDD

# Create weather data with geographical coordinates
parser = WeatherInputParser()
weather_data = parser.from_coordinates(
    time=datetime(2024, 1, 15),
    lon=-74.006,  # New York City
    lat=40.7128,
    temperature=25.0  # Fahrenheit
)

# Calculate Heating Degree Days
hdd = HDD(weather_data, reference_temperature=65.0)
hdd_value = hdd.calculate()
print(f"HDD: {hdd_value}")

# Calculate option payoff
payoff = hdd.payoff(strike=40, tick_value=250, option_type="call")
print(f"Payoff: ${payoff:,.2f}")
```

### Example 2: Using Named Locations

```python
from datetime import datetime, timedelta
from weather_derivatives import WeatherInputParser, CDD

# Create time series for a specific site
parser = WeatherInputParser()
dates = [datetime(2024, 7, 1) + timedelta(days=i) for i in range(30)]
temperatures = [85, 87, 90, 88, 92, 89, 91] * 4 + [86, 88]

weather_data = parser.from_site(
    time=dates,
    site="Chicago O'Hare",
    temperature=temperatures
)

# Calculate Cooling Degree Days
cdd = CDD(weather_data, reference_temperature=65.0)
print(f"Total CDD: {cdd.calculate()}")
print(f"Monthly CDD: {cdd.calculate_monthly()}")
```

### Example 3: Time-Based Data Only

```python
from datetime import datetime, timedelta
from weather_derivatives import WeatherInputParser, CAT

# Create weather data without location
parser = WeatherInputParser()
dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(90)]
temperatures = [30 + i * 0.5 for i in range(90)]  # Warming trend

weather_data = parser.from_time_only(
    time=dates,
    temperature=temperatures
)

# Calculate Cumulative Average Temperature
cat = CAT(weather_data)
print(f"CAT Sum: {cat.calculate(method='sum')}")
print(f"CAT Mean: {cat.calculate(method='mean')}")
```

### Example 4: Precipitation Derivatives

```python
from weather_derivatives import WeatherInputParser, PrecipitationDerivative

parser = WeatherInputParser()
weather_data = parser.from_site(
    time=dates,
    site="Iowa",
    precipitation=[0, 2.5, 0, 10.5, 3.2, 0, 5.1]  # mm
)

precip = PrecipitationDerivative(weather_data, precipitation_threshold=0.1)
print(f"Total Precipitation: {precip.calculate_total_precipitation()} mm")
print(f"Rain Days: {precip.calculate_rain_days()}")

# Calculate payoff
payoff = precip.payoff_total_precipitation(strike=50, tick_value=10, option_type="put")
print(f"Payoff: ${payoff:,.2f}")
```

### Example 5: Snow Derivatives

```python
from weather_derivatives import WeatherInputParser, SnowDerivative

parser = WeatherInputParser()
data = parser.from_site(
    time=dates,
    site="Aspen Ski Resort",
    snowfall=[10, 15, 5, 20, 8],  # cm per day
    snow_depth=[30, 45, 50, 70, 78]  # cm
)

snow = SnowDerivative(data, snow_day_threshold=2.5)
print(f"Total Snowfall: {snow.calculate_total_snowfall()} cm")
print(f"Snow Days: {snow.calculate_snow_days()}")
print(f"Max Snow Depth: {snow.calculate_max_snow_depth()} cm")
```

### Example 6: Agricultural (Frost/GDD) Derivatives

```python
from weather_derivatives import FrostDerivative

frost = FrostDerivative(weather_data, frost_threshold=0.0, base_temperature=10.0)
gdd = frost.calculate_growing_degree_days()
frost_days = frost.calculate_frost_days()

print(f"Growing Degree Days: {gdd}")
print(f"Frost Days: {frost_days}")
```

### Example 7: Advanced Valuation Methods

```python
from weather_derivatives import DerivativeValuation, AdvancedValuation
import numpy as np

# Standard methods
valuation = DerivativeValuation(discount_rate=0.05)
historical_hdd = np.random.normal(1200, 200, 20)

# Burn rate analysis with distribution fitting
burn_analysis = valuation.burn_rate_analysis(
    historical_data=historical_hdd,
    strike=1300,
    tick_value=5000,
    distribution_fit="gev"  # Generalized Extreme Value
)

# Time series forecasting
ts_valuation = valuation.time_series_forecast_valuation(
    historical_data=historical_hdd,
    strike=1300,
    tick_value=5000,
    forecast_periods=30
)

# Weather index insurance pricing
insurance = valuation.weather_index_insurance_pricing(
    historical_data=historical_hdd,
    trigger=1000,
    exit=1500,
    limit=100000
)
print(f"Insurance Premium: ${insurance['total_premium']:,.2f}")

# Advanced methods
adv_val = AdvancedValuation()

# Bootstrap resampling
bootstrap = adv_val.bootstrap_valuation(
    historical_data=historical_hdd,
    strike=1300,
    tick_value=5000,
    num_bootstrap_samples=10000
)

# Extreme Value Theory
evt = adv_val.extreme_value_theory_pricing(
    historical_data=historical_hdd,
    strike=1300,
    tick_value=5000
)

# Regime-switching model
regime = adv_val.regime_switching_valuation(
    historical_data=historical_hdd,
    strike=1300,
    tick_value=5000,
    num_regimes=2
)

print(f"Bootstrap Mean: ${bootstrap['mean_payoff']:,.2f}")
print(f"EVT Expected Payoff: ${evt['expected_payoff']:,.2f}")
print(f"Regime-Switching: ${regime['expected_payoff']:,.2f}")
```

## Documentation

### Input Formats

The library supports three ways to specify weather data:

1. **Geographical Coordinates**: `(time, lon, lat)`
   ```python
   parser.from_coordinates(time=datetime, lon=-74.006, lat=40.7128, temperature=25.0)
   ```

2. **Named Site**: `(time, site)`
   ```python
   parser.from_site(time=datetime, site="Chicago O'Hare", temperature=25.0)
   ```

3. **Time Only**: `(time)`
   ```python
   parser.from_time_only(time=datetime, temperature=25.0)
   ```

### Derivative Types

#### Temperature Derivatives

- **HDD (Heating Degree Days)**: Measures how cold temperatures are
  - Formula: `max(reference_temp - actual_temp, 0)`
  - Used by energy companies for heating demand

- **CDD (Cooling Degree Days)**: Measures how hot temperatures are
  - Formula: `max(actual_temp - reference_temp, 0)`
  - Used for cooling demand forecasting

- **CAT (Cumulative Average Temperature)**: Sum or average of temperatures
  - Used for seasonal temperature exposure

- **PAC (Pacific Temperature)**: Temperature deviations from reference
  - Used for temperature swing exposure

#### Precipitation Derivatives

- Total precipitation over a period
- Rain day counting
- Dry day counting
- Extreme precipitation events
- Percentile analysis

#### Wind Derivatives

- Average wind speed
- Wind power calculation
- Total wind energy
- Calm days (low wind)
- High wind events

### Pricing & Valuation

The library includes sophisticated pricing tools:

- **Historical Burn Analysis**: Analyze historical payoffs
- **Monte Carlo Simulation**: Simulate future scenarios
- **Black-Scholes Approximation**: Option pricing
- **Greeks Calculation**: Delta, gamma, vega, theta
- **Risk Metrics**: VaR, CVaR, skewness, kurtosis

## Examples

See the `examples/` directory for comprehensive examples:

- `basic_usage.py` - Basic usage with all input formats
- `advanced_pricing.py` - Advanced pricing and valuation techniques

Run examples:
```bash
python examples/basic_usage.py
python examples/advanced_pricing.py
```

## Requirements

- Python 3.7+
- NumPy
- SciPy
- python-dateutil

## Project Structure

```
libwd/
├── weather_derivatives/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   └── weather_data.py
│   ├── derivatives/
│   │   ├── __init__.py
│   │   ├── temperature.py      # HDD, CDD, CAT, PAC
│   │   ├── precipitation.py    # Rainfall derivatives
│   │   ├── wind.py             # Wind power derivatives
│   │   ├── snow.py             # Snowfall derivatives
│   │   ├── frost.py            # GDD, frost days
│   │   ├── humidity.py         # Humidity derivatives
│   │   └── solar.py            # Solar radiation derivatives
│   ├── parsers/
│   │   ├── __init__.py
│   │   └── input_parser.py
│   └── pricing/
│       ├── __init__.py
│       ├── valuation.py         # 15+ pricing methods
│       └── advanced_valuation.py # Advanced techniques
├── examples/
│   ├── basic_usage.py
│   └── advanced_pricing.py
├── tests/
│   └── test_derivatives.py
├── README.md
├── setup.py
└── requirements.txt
```

## Testing

```bash
# Run tests
python -m pytest tests/

# Run specific test
python -m pytest tests/test_derivatives.py
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## References

- Weather derivatives markets and trading
- CME Weather Derivative Products
- Energy and commodity risk management

## Support

For questions and support, please open an issue on GitHub.
