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

- **Wind Derivatives**:
  - Wind power calculations
  - Calm/high wind day counting
  - Wind energy estimates

- **Advanced Pricing & Valuation**:
  - Historical burn analysis
  - Monte Carlo simulation
  - Black-Scholes approximation
  - Greeks calculation
  - Risk metrics (VaR, CVaR, etc.)

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

### Example 5: Advanced Valuation

```python
from weather_derivatives import DerivativeValuation
import numpy as np

# Historical burn analysis
valuation = DerivativeValuation(discount_rate=0.05)
historical_hdd = np.random.normal(1200, 200, 20)  # 20 years

analysis = valuation.historical_burn_analysis(
    historical_data=historical_hdd,
    strike=1300,
    tick_value=5000,
    option_type="call"
)

print(f"Mean Payoff: ${analysis['mean_payoff']:,.2f}")
print(f"Probability ITM: {analysis['probability_itm']:.1%}")

# Monte Carlo simulation
mc_results = valuation.monte_carlo_valuation(
    mean=800,
    std=150,
    strike=900,
    tick_value=2500,
    option_type="call",
    num_simulations=10000
)

print(f"Expected Payoff: ${mc_results['expected_payoff']:,.2f}")
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
│   │   ├── temperature.py
│   │   ├── precipitation.py
│   │   └── wind.py
│   ├── parsers/
│   │   ├── __init__.py
│   │   └── input_parser.py
│   └── pricing/
│       ├── __init__.py
│       └── valuation.py
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

## Citation

If you use this library in your research, please cite:

```
@software{weather_derivatives,
  title = {Weather Derivatives Library},
  author = {Weather Derivatives Team},
  year = {2024},
  url = {https://github.com/yourusername/libwd}
}
```

## References

- Weather derivatives markets and trading
- CME Weather Derivative Products
- Energy and commodity risk management

## Support

For questions and support, please open an issue on GitHub.
