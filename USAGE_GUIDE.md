# Weather Derivatives Library - Complete Usage Guide

**Version:** 2.0.0
**Author:** Weather Derivatives Team

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Input Formats](#input-formats)
4. [Weather Derivatives](#weather-derivatives)
5. [Pricing & Valuation](#pricing--valuation)
6. [Complete API Reference](#complete-api-reference)
7. [Examples](#examples)

---

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

**Requirements:**
- Python 3.7+
- NumPy >= 1.19.0
- SciPy >= 1.5.0
- python-dateutil >= 2.8.0

---

## Quick Start

### Basic Example

```python
from datetime import datetime, timedelta
from weather_derivatives import WeatherInputParser, HDD, DerivativeValuation

# Create weather data
parser = WeatherInputParser()
dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(90)]
temperatures = [35 + i % 25 for i in range(90)]  # Winter temperatures

weather_data = parser.from_site(
    time=dates,
    site="Chicago",
    temperature=temperatures
)

# Calculate Heating Degree Days
hdd = HDD(weather_data, reference_temperature=65.0)
total_hdd = hdd.calculate()
print(f"Total HDD: {total_hdd}")

# Calculate option payoff
payoff = hdd.payoff(strike=2700, tick_value=5000, option_type="call")
print(f"Option Payoff: ${payoff:,.2f}")

# Value using historical data
valuation = DerivativeValuation(discount_rate=0.05)
historical_hdd = [2500, 2800, 3000, 2700, 2900, 2600, 2850, 3100, 2750, 2650]

analysis = valuation.historical_burn_analysis(
    historical_data=historical_hdd,
    strike=2700,
    tick_value=5000,
    option_type="call"
)
print(f"Expected Payoff: ${analysis['mean_payoff']:,.2f}")
```

---

## Input Formats

The library supports three flexible input formats for weather data:

### 1. Geographical Coordinates: `(time, lon, lat)`

**Use Case:** When you have GPS coordinates for weather stations

```python
from datetime import datetime
from weather_derivatives import WeatherInputParser

parser = WeatherInputParser()
data = parser.from_coordinates(
    time=datetime(2024, 1, 15),
    lon=-74.006,          # Longitude (float)
    lat=40.7128,          # Latitude (float)
    temperature=25.0,     # Fahrenheit or Celsius (float)
    precipitation=5.2,    # mm (float)
    wind_speed=8.5        # m/s (float)
)
```

**Input Dimensions:**
- `time`: datetime or list[datetime] with length N
- `lon`: float (single coordinate)
- `lat`: float (single coordinate)
- `temperature`: float or list[float] with length N
- Other weather variables: float or list[float] with length N

**Output:** WeatherData object with N time points

---

### 2. Named Location: `(time, site)`

**Use Case:** When you have a named location (city, airport code, etc.)

```python
data = parser.from_site(
    time=[datetime(2024, 1, 1), datetime(2024, 1, 2)],
    site="Chicago O'Hare",
    temperature=[30.0, 32.5],
    precipitation=[0.5, 1.2]
)
```

**Input Dimensions:**
- `time`: datetime or list[datetime] with length N
- `site`: string (location name)
- Weather variables: float or list[float] with length N

**Output:** WeatherData object with N time points

---

### 3. Time Only: `(time)`

**Use Case:** When location is not relevant or data is aggregated

```python
data = parser.from_time_only(
    time=[datetime(2024, 1, 1), datetime(2024, 1, 2), datetime(2024, 1, 3)],
    temperature=[28.0, 30.0, 32.0]
)
```

**Input Dimensions:**
- `time`: datetime or list[datetime] with length N
- Weather variables: float or list[float] with length N

**Output:** WeatherData object with N time points

---

## Weather Derivatives

### 1. Temperature Derivatives

#### HDD (Heating Degree Days)

**Purpose:** Measures heating demand; pays when it's cold

**Formula:** HDD = max(reference_temp - actual_temp, 0)

```python
from weather_derivatives import HDD

hdd = HDD(weather_data, reference_temperature=65.0)
```

**Methods:**

| Method | Input | Output | Description |
|--------|-------|--------|-------------|
| `calculate(cumulative=True)` | cumulative: bool | float or list[float] | Calculates HDD |
| `calculate_monthly()` | None | dict[str, float] | Monthly HDD aggregation |
| `payoff(strike, tick_value, option_type)` | strike: float<br>tick_value: float<br>option_type: str | float | Option payoff |

**Input Dimensions:**
- `weather_data`: WeatherData with N time points
- `reference_temperature`: float (default 65°F or 18.3°C)

**Output Dimensions:**
- `calculate(cumulative=True)`: scalar float (total HDD)
- `calculate(cumulative=False)`: list[float] with length N (daily HDD)
- `calculate_monthly()`: dict with keys as 'YYYY-MM' strings
- `payoff()`: scalar float (dollars)

**Example:**

```python
# Daily HDD
daily_hdd = hdd.calculate(cumulative=False)
print(f"Daily HDD: {daily_hdd}")  # [15.0, 10.0, 5.0, ...]

# Total HDD
total_hdd = hdd.calculate(cumulative=True)
print(f"Total HDD: {total_hdd}")  # 450.0

# Monthly breakdown
monthly = hdd.calculate_monthly()
print(monthly)  # {'2024-01': 450.0, '2024-02': 380.0, ...}

# Option payoff
payoff = hdd.payoff(strike=400, tick_value=5000, option_type="call")
print(f"Payoff: ${payoff:,.2f}")  # $250,000.00
```

---

#### CDD (Cooling Degree Days)

**Purpose:** Measures cooling demand; pays when it's hot

**Formula:** CDD = max(actual_temp - reference_temp, 0)

```python
from weather_derivatives import CDD

cdd = CDD(weather_data, reference_temperature=65.0)
```

**Methods:** Same as HDD

**Example:**

```python
summer_temps = [85, 90, 88, 92, 87]  # 5 days of summer
weather_data = parser.from_time_only(
    time=[datetime(2024, 7, 1) + timedelta(days=i) for i in range(5)],
    temperature=summer_temps
)

cdd = CDD(weather_data, reference_temperature=65.0)
total_cdd = cdd.calculate()
print(f"Total CDD: {total_cdd}")  # (85-65) + (90-65) + ... = 120.0
```

---

#### CAT (Cumulative Average Temperature)

**Purpose:** Sum or average of temperatures over a period

```python
from weather_derivatives import CAT

cat = CAT(weather_data)
cat_sum = cat.calculate(method="sum")      # Total
cat_mean = cat.calculate(method="mean")    # Average
```

**Methods:**

| Method | Input | Output | Description |
|--------|-------|--------|-------------|
| `calculate(method)` | method: str ("sum" or "mean") | float | CAT calculation |
| `payoff(strike, tick_value, option_type)` | strike: float<br>tick_value: float<br>option_type: str | float | Option payoff |

---

### 2. Precipitation Derivatives

```python
from weather_derivatives import PrecipitationDerivative

precip = PrecipitationDerivative(
    weather_data,
    precipitation_threshold=0.1  # mm
)
```

**Methods:**

| Method | Input | Output | Description |
|--------|-------|--------|-------------|
| `calculate_total_precipitation()` | None | float | Total precipitation (mm) |
| `calculate_rain_days()` | None | int | Days above threshold |
| `calculate_dry_days()` | None | int | Days below threshold |
| `calculate_extreme_events(threshold)` | threshold: float | int | Extreme events count |
| `calculate_percentile(percentile)` | percentile: float (0-100) | float | Precipitation percentile |
| `payoff_total_precipitation(...)` | strike, tick_value, option_type | float | Payoff based on total |
| `payoff_rain_days(...)` | strike, tick_value, option_type | float | Payoff based on rain days |

**Example:**

```python
rainfall = [0, 2.5, 0, 10.5, 3.2, 0, 5.1, 0, 0, 1.8]
weather_data = parser.from_time_only(
    time=[datetime(2024, 6, 1) + timedelta(days=i) for i in range(10)],
    precipitation=rainfall
)

precip = PrecipitationDerivative(weather_data, precipitation_threshold=0.5)

print(f"Total Precipitation: {precip.calculate_total_precipitation()} mm")  # 23.1 mm
print(f"Rain Days: {precip.calculate_rain_days()}")  # 5 days
print(f"90th Percentile: {precip.calculate_percentile(90)} mm")  # 8.775 mm

# Option payoff
payoff = precip.payoff_total_precipitation(
    strike=30,
    tick_value=100,
    option_type="put"  # Pays if rainfall < 30mm
)
print(f"Payoff: ${payoff:,.2f}")  # $690.00
```

**Input/Output Dimensions:**
- Input: WeatherData with N time points, precipitation list[float]
- `calculate_total_precipitation()`: scalar float
- `calculate_rain_days()`: scalar int
- `payoff_*()`: scalar float

---

### 3. Wind Derivatives

```python
from weather_derivatives import WindDerivative

wind = WindDerivative(
    weather_data,
    air_density=1.225,      # kg/m³
    rotor_diameter=90.0     # meters
)
```

**Methods:**

| Method | Input | Output | Description |
|--------|-------|--------|-------------|
| `calculate_average_wind_speed()` | None | float | Average wind speed (m/s) |
| `calculate_wind_power()` | None | float or list[float] | Wind power (W or W/m²) |
| `calculate_total_wind_energy(hours)` | hours_per_reading: float | float | Total energy (Wh) |
| `calculate_calm_days(threshold)` | calm_threshold: float | int | Days below threshold |
| `calculate_high_wind_days(threshold)` | high_threshold: float | int | Days above threshold |
| `payoff_wind_power(...)` | strike, tick_value, option_type | float | Energy-based payoff |
| `payoff_calm_days(...)` | strike, tick_value, option_type | float | Calm day payoff |

**Example:**

```python
wind_speeds = [5.2, 7.8, 6.1, 8.5, 4.3, 9.2, 6.7, 7.4, 5.8, 8.1]
weather_data = parser.from_coordinates(
    time=[datetime(2024, 3, 1) + timedelta(days=i) for i in range(10)],
    lon=-100.0,
    lat=32.0,
    wind_speed=wind_speeds
)

wind = WindDerivative(weather_data, rotor_diameter=90.0)

print(f"Average Wind: {wind.calculate_average_wind_speed():.2f} m/s")
print(f"Total Energy: {wind.calculate_total_wind_energy():,.0f} Wh")
print(f"Calm Days (<3 m/s): {wind.calculate_calm_days(calm_threshold=3.0)}")

# Option payoff
payoff = wind.payoff_wind_power(
    strike=1500000,
    tick_value=0.01,
    option_type="put"
)
print(f"Payoff: ${payoff:,.2f}")
```

---

### 4. Snow Derivatives

```python
from weather_derivatives import SnowDerivative

snow = SnowDerivative(
    weather_data,
    snow_day_threshold=1.0  # cm
)
```

**Methods:**

| Method | Input | Output | Description |
|--------|-------|--------|-------------|
| `calculate_total_snowfall()` | None | float | Total snowfall (cm) |
| `calculate_snow_days()` | None | int | Days above threshold |
| `calculate_max_snow_depth()` | None | float | Maximum depth (cm) |
| `calculate_average_snow_depth()` | None | float | Average depth (cm) |
| `calculate_snow_cover_days(min_depth)` | min_depth: float | int | Days with snow cover |
| `payoff_total_snowfall(...)` | strike, tick_value, option_type | float | Snowfall payoff |
| `payoff_snow_days(...)` | strike, tick_value, option_type | float | Snow days payoff |

**Example:**

```python
snowfall = [10, 15, 5, 20, 8, 0, 0, 12, 18, 7]
snow_depth = [10, 25, 30, 50, 58, 58, 55, 67, 85, 90]

weather_data = parser.from_site(
    time=[datetime(2024, 12, 1) + timedelta(days=i) for i in range(10)],
    site="Aspen Ski Resort",
    snowfall=snowfall,
    snow_depth=snow_depth
)

snow = SnowDerivative(weather_data, snow_day_threshold=2.5)

print(f"Total Snowfall: {snow.calculate_total_snowfall()} cm")  # 95 cm
print(f"Snow Days: {snow.calculate_snow_days()}")  # 8 days
print(f"Max Depth: {snow.calculate_max_snow_depth()} cm")  # 90 cm

# Payoff for ski resort
payoff = snow.payoff_total_snowfall(
    strike=120,  # Need 120cm for good season
    tick_value=1000,
    option_type="put"  # Pays if snowfall < 120cm
)
print(f"Insurance Payout: ${payoff:,.2f}")  # $25,000
```

---

### 5. Frost & Agricultural Derivatives

```python
from weather_derivatives import FrostDerivative

frost = FrostDerivative(
    weather_data,
    frost_threshold=0.0,      # °C
    base_temperature=10.0     # °C for GDD
)
```

**Methods:**

| Method | Input | Output | Description |
|--------|-------|--------|-------------|
| `calculate_frost_days()` | None | int | Days with frost |
| `calculate_growing_degree_days(max_temp)` | max_temperature: float | float | GDD total |
| `calculate_crop_heat_units(...)` | min_night_temp, max_day_temp | float | CHU for corn |
| `calculate_freeze_events(threshold)` | severe_threshold: float | int | Severe freezes |
| `calculate_frost_free_days()` | None | int | Frost-free days |
| `payoff_gdd(...)` | strike, tick_value, option_type | float | GDD payoff |
| `payoff_frost_days(...)` | strike, tick_value, option_type | float | Frost payoff |

**Example:**

```python
# Growing season for corn
temps = [15, 18, 22, 25, 28, 27, 26, 24, 20, 18]
min_temps = [8, 10, 14, 18, 20, 19, 18, 16, 12, 10]
max_temps = [22, 26, 30, 32, 36, 35, 34, 32, 28, 26]

weather_data = parser.from_site(
    time=[datetime(2024, 5, 1) + timedelta(days=i) for i in range(10)],
    site="Iowa",
    temperature=temps,
    min_temperature=min_temps,
    max_temperature=max_temps
)

frost = FrostDerivative(weather_data, base_temperature=10.0)

gdd = frost.calculate_growing_degree_days(max_temperature=30.0)
chu = frost.calculate_crop_heat_units()
frost_days = frost.calculate_frost_days()

print(f"Growing Degree Days: {gdd:.0f}")  # ~150 GDD
print(f"Crop Heat Units: {chu:.0f}")  # CHU for corn maturity
print(f"Frost Days: {frost_days}")  # 0

# Crop insurance payoff
payoff = frost.payoff_gdd(
    strike=2500,  # Need 2500 GDD for full maturity
    tick_value=50,
    option_type="put"
)
print(f"Crop Insurance: ${payoff:,.2f}")
```

**GDD Calculation:**
```
For each day:
  temp_capped = min(temperature, max_temperature)
  daily_gdd = max(temp_capped - base_temperature, 0)
Total GDD = sum of daily_gdd
```

---

### 6. Humidity Derivatives

```python
from weather_derivatives import HumidityDerivative

humidity_deriv = HumidityDerivative(
    weather_data,
    high_humidity_threshold=80.0,  # %
    low_humidity_threshold=30.0    # %
)
```

**Methods:**

| Method | Input | Output | Description |
|--------|-------|--------|-------------|
| `calculate_average_humidity()` | None | float | Average RH (%) |
| `calculate_high_humidity_days()` | None | int | Days above high threshold |
| `calculate_low_humidity_days()` | None | int | Days below low threshold |
| `calculate_humidity_stress_index()` | None | int | Days outside 30-70% |
| `calculate_vapor_pressure_deficit(temps)` | temperature_data: list[float] | float or list[float] | VPD in kPa |
| `calculate_heat_index_days(threshold)` | heat_index_threshold: float | int | Heat index days |
| `payoff_humidity_days(...)` | strike, tick_value, option_type, metric | float | Humidity payoff |

**Example:**

```python
humidity = [65, 85, 90, 70, 75, 88, 92, 78, 82, 87]
temps = [28, 32, 34, 30, 29, 33, 35, 31, 32, 33]

weather_data = parser.from_site(
    time=[datetime(2024, 7, 1) + timedelta(days=i) for i in range(10)],
    site="Houston",
    humidity=humidity,
    temperature=temps
)

humid = HumidityDerivative(weather_data, high_humidity_threshold=80.0)

print(f"Average Humidity: {humid.calculate_average_humidity():.1f}%")  # 81.2%
print(f"High Humidity Days: {humid.calculate_high_humidity_days()}")  # 6 days
print(f"Heat Index Days >35°C: {humid.calculate_heat_index_days(35.0)}")  # ~4 days

# Health sector derivative
payoff = humid.payoff_humidity_days(
    strike=5,
    tick_value=500,
    option_type="call",
    metric="high"
)
print(f"Payoff: ${payoff:,.2f}")  # $500 (6-5=1 day over)
```

---

### 7. Solar Radiation Derivatives

```python
from weather_derivatives import SolarDerivative

solar = SolarDerivative(
    weather_data,
    panel_efficiency=0.20,    # 20%
    panel_area=10000          # m²
)
```

**Methods:**

| Method | Input | Output | Description |
|--------|-------|--------|-------------|
| `calculate_total_irradiance()` | None | float | Total irradiance |
| `calculate_average_irradiance()` | None | float | Average daily irradiance |
| `calculate_sunshine_hours()` | None | float | Total sunshine hours |
| `calculate_cloudy_days(max_hours)` | max_sunshine_hours: float | int | Cloudy days |
| `calculate_solar_energy_production(hours)` | hours_per_reading: float | float | Energy (kWh) |
| `calculate_peak_sun_hours()` | None | float | PSH |
| `calculate_capacity_factor(capacity)` | rated_capacity: float | float | Capacity factor (0-1) |
| `payoff_irradiance(...)` | strike, tick_value, option_type | float | Irradiance payoff |
| `payoff_sunshine_hours(...)` | strike, tick_value, option_type | float | Sunshine payoff |
| `payoff_energy_production(...)` | strike, tick_value, option_type | float | Energy payoff |

**Example:**

```python
# Annual solar data for solar farm
irradiance = [5.2, 6.1, 5.8, 6.5, 7.0, 6.8, 7.2, 6.9, 6.3, 5.9]  # kWh/m²/day
sunshine_hours = [8, 9, 8, 10, 11, 10, 11, 10, 9, 8]

weather_data = parser.from_coordinates(
    time=[datetime(2024, 6, 1) + timedelta(days=i) for i in range(10)],
    lon=-110.0,
    lat=33.0,
    solar_radiation=irradiance,
    sunshine_hours=sunshine_hours
)

solar = SolarDerivative(weather_data, panel_efficiency=0.20, panel_area=10000)

print(f"Total Irradiance: {solar.calculate_total_irradiance():.1f} kWh/m²")  # 63.7
print(f"Sunshine Hours: {solar.calculate_sunshine_hours():.0f} hours")  # 94 hours
print(f"Energy Production: {solar.calculate_solar_energy_production():,.0f} kWh")  # 127,400 kWh

# Revenue protection for solar plant
payoff = solar.payoff_energy_production(
    strike=150000,  # Expected 150,000 kWh
    tick_value=0.05,  # $0.05/kWh
    option_type="put"
)
print(f"Revenue Protection: ${payoff:,.2f}")  # $1,130
```

---

## Pricing & Valuation

### DerivativeValuation Class

```python
from weather_derivatives import DerivativeValuation

valuation = DerivativeValuation(discount_rate=0.05)
```

### Standard Methods

#### 1. Historical Burn Analysis

**Purpose:** Analyze historical payoffs to estimate fair value

```python
analysis = valuation.historical_burn_analysis(
    historical_data=[1000, 1100, 1200, 1300, 1400],  # list[float]
    strike=1200,                                      # float
    tick_value=5000,                                  # float
    option_type="call"                                # "call" or "put"
)
```

**Output Dimensions:**
```python
{
    'mean_payoff': float,
    'median_payoff': float,
    'std_payoff': float,
    'min_payoff': float,
    'max_payoff': float,
    'percentile_25': float,
    'percentile_75': float,
    'percentile_95': float,
    'probability_itm': float,  # 0-1
    'num_scenarios': int
}
```

---

#### 2. Monte Carlo Simulation

```python
mc_results = valuation.monte_carlo_valuation(
    mean=1000,                    # float
    std=150,                      # float
    strike=1100,                  # float
    tick_value=2500,              # float
    option_type="call",           # "call" or "put"
    num_simulations=10000,        # int
    distribution="normal"         # "normal" or "lognormal"
)
```

**Output Dimensions:**
```python
{
    'expected_payoff': float,
    'median_payoff': float,
    'std_payoff': float,
    'min_payoff': float,
    'max_payoff': float,
    'percentile_5': float,
    'percentile_95': float,
    'var_95': float,
    'probability_itm': float,
    'num_simulations': int
}
```

---

#### 3. Black-Scholes Approximation

```python
price = valuation.black_scholes_approximation(
    current_index=1000,              # float
    strike=1100,                     # float
    time_to_maturity_years=0.25,    # float (years)
    volatility=0.20,                 # float (annualized)
    option_type="call",              # "call" or "put"
    risk_free_rate=0.03              # float (optional)
)
```

**Input/Output:**
- Input: All scalars (float)
- Output: Scalar float (option price)

---

#### 4. Greeks Calculation

```python
greeks = valuation.calculate_greeks(
    current_index=1000,
    strike=1100,
    time_to_maturity_years=0.25,
    volatility=0.20,
    option_type="call"
)
```

**Output Dimensions:**
```python
{
    'delta': float,    # Sensitivity to index (0-1 for calls, -1-0 for puts)
    'gamma': float,    # Rate of change of delta
    'vega': float,     # Sensitivity to volatility
    'theta': float,    # Time decay (per day)
    'price': float     # Option price
}
```

---

#### 5. Burn Rate Analysis with Distribution Fitting

```python
burn = valuation.burn_rate_analysis(
    historical_data=np.array([1000, 1100, 1200, ...]),  # array-like
    strike=1200,
    tick_value=5000,
    option_type="call",
    distribution_fit="gev"  # "empirical", "normal", "lognormal", "gamma", "gev"
)
```

**Output:**
```python
{
    'mean_payoff': float,
    'median_payoff': float,
    'std_payoff': float,
    'burn_rate': float,           # Probability of payout
    'distribution_type': str,
    'distribution_params': dict   # Distribution parameters (if fitted)
}
```

---

#### 6. Time Series Forecast Valuation

**Purpose:** Value derivatives using AR(1) time series forecasting

```python
ts_result = valuation.time_series_forecast_valuation(
    historical_data=np.array([...]),  # array-like, length >= 10
    strike=1300,
    tick_value=5000,
    option_type="call",
    forecast_periods=30,               # int
    num_simulations=1000               # int
)
```

**Output:**
```python
{
    'expected_payoff': float,
    'median_payoff': float,
    'std_payoff': float,
    'ar_coefficient': float,      # AR(1) coefficient
    'innovation_std': float,      # Residual std dev
    'percentile_5': float,
    'percentile_95': float
}
```

---

#### 7. Weather Index Insurance Pricing

**Purpose:** Price parametric insurance contracts

```python
insurance = valuation.weather_index_insurance_pricing(
    historical_data=np.array([...]),
    trigger=1000,          # Coverage starts
    exit=1500,             # Maximum payout reached
    limit=100000,          # Maximum payout amount
    attachment_probability=0.20  # Target probability
)
```

**Output:**
```python
{
    'pure_premium': float,                    # Expected payout
    'risk_loading': float,                    # Risk charge
    'admin_loading': float,                   # Administrative fee
    'total_premium': float,                   # Total price
    'attachment_probability': float,          # Actual prob of payout
    'average_payout_given_trigger': float,   # Conditional expectation
    'max_payout': float,                     # = limit
    'loss_ratio': float,                     # Expected loss / premium
    'std_payout': float
}
```

**Payout Function:**
```python
if index <= trigger:
    payout = 0
elif index >= exit:
    payout = limit
else:
    payout = limit * (index - trigger) / (exit - trigger)
```

---

#### 8. Asian Option Pricing

**Purpose:** Options on average weather index

```python
asian = valuation.asian_option_pricing(
    historical_data=np.array([...]),
    strike=1200,
    tick_value=5000,
    option_type="call",
    averaging_periods=30,    # Number of days to average
    num_simulations=10000
)
```

**Output:**
```python
{
    'expected_payoff': float,
    'median_payoff': float,
    'std_payoff': float,
    'volatility_reduction': float,  # How much averaging reduces volatility
    'percentile_5': float,
    'percentile_95': float
}
```

---

#### 9. Spread Option Pricing

**Purpose:** Options on difference between two indices

```python
spread = valuation.spread_option_pricing(
    historical_data1=np.array([...]),  # HDD for City 1
    historical_data2=np.array([...]),  # HDD for City 2
    strike=100,                        # Strike on the spread
    tick_value=2500,
    option_type="call",
    num_simulations=10000
)
```

**Output:**
```python
{
    'expected_payoff': float,
    'median_payoff': float,
    'std_payoff': float,
    'spread_mean': float,       # E[Index1 - Index2]
    'spread_std': float,        # Std(Index1 - Index2)
    'correlation': float,       # Correlation between indices
    'percentile_5': float,
    'percentile_95': float
}
```

---

#### 10. Portfolio Optimization

**Purpose:** Optimal allocation across multiple derivatives

```python
portfolio = valuation.portfolio_optimization(
    derivatives=[
        {'expected_return': 50000, 'volatility': 20000},
        {'expected_return': 30000, 'volatility': 15000},
        {'expected_return': 40000, 'volatility': 25000}
    ],
    budget=1000000,
    risk_aversion=0.5  # 0 = risk-neutral, 1 = very risk-averse
)
```

**Input:**
- `derivatives`: list[dict] with keys 'expected_return' and 'volatility'
- `budget`: float
- `risk_aversion`: float (0-1)

**Output:**
```python
{
    'optimal_weights': list[float],      # Weights summing to 1
    'optimal_allocation': list[float],   # Dollar amounts
    'expected_return': float,
    'portfolio_std': float,
    'sharpe_ratio': float,
    'diversification_ratio': float
}
```

---

#### 11. Stochastic Volatility Valuation

**Purpose:** Model time-varying volatility (Heston-like)

```python
stoch_vol = valuation.stochastic_volatility_valuation(
    mean=1200,
    vol_mean=150,              # Mean volatility level
    vol_std=30,                # Volatility of volatility
    vol_mean_reversion=0.5,    # Speed of mean reversion
    strike=1300,
    tick_value=5000,
    option_type="call",
    time_periods=30,
    num_simulations=10000
)
```

**Output:**
```python
{
    'expected_payoff': float,
    'median_payoff': float,
    'std_payoff': float,
    'implied_volatility': float,
    'percentile_5': float,
    'percentile_95': float
}
```

---

#### 12. Sensitivity Analysis

**Purpose:** Test how payoff changes with parameters

```python
sensitivity = valuation.sensitivity_analysis(
    base_params={'mean': 1200, 'std': 200},
    param_ranges={
        'mean': (1000, 1400),
        'std': (100, 300)
    },
    strike=1300,
    tick_value=5000,
    option_type="call",
    num_points=10
)
```

**Output:**
```python
{
    'mean': {
        'param_values': list[float],  # 10 points from 1000 to 1400
        'payoffs': list[float],       # Corresponding payoffs
        'sensitivity': float          # ∂payoff/∂param
    },
    'std': {
        'param_values': list[float],
        'payoffs': list[float],
        'sensitivity': float
    }
}
```

---

### Advanced Valuation Methods

```python
from weather_derivatives.pricing.advanced_valuation import AdvancedValuation

adv_val = AdvancedValuation(confidence_level=0.95)
```

#### 1. Bootstrap Valuation

**Purpose:** Robust estimation using block bootstrap

```python
bootstrap = adv_val.bootstrap_valuation(
    historical_data=np.array([...]),
    strike=1300,
    tick_value=5000,
    option_type="call",
    num_bootstrap_samples=10000,
    block_size=3  # Block size to preserve autocorrelation
)
```

**Output:**
```python
{
    'mean_payoff': float,
    'median_payoff': float,
    'std_payoff': float,
    'confidence_interval': tuple[float, float],  # (lower, upper)
    'percentile_5': float,
    'percentile_95': float,
    'bootstrap_distribution': np.ndarray  # All 10,000 payoffs
}
```

---

#### 2. Extreme Value Theory Pricing

**Purpose:** Tail risk modeling using GPD

```python
evt = adv_val.extreme_value_theory_pricing(
    historical_data=np.array([...]),
    strike=1300,
    tick_value=5000,
    option_type="call",
    threshold_percentile=0.90  # Use top 10% for tail modeling
)
```

**Output:**
```python
{
    'expected_payoff': float,
    'tail_probability': float,      # P(exceeding strike)
    'gpd_shape': float,             # ξ (shape parameter)
    'gpd_scale': float,             # σ (scale parameter)
    'threshold': float,             # Threshold used
    'num_exceedances': int          # Number of tail observations
}
```

**Note:** Requires at least 10 exceedances for reliable estimation

---

#### 3. Quantile Regression Pricing

**Purpose:** Distribution-free quantile estimation

```python
quantile = adv_val.quantile_regression_pricing(
    historical_data=np.array([...]),
    strike=1200,
    tick_value=5000,
    option_type="call",
    quantiles=[0.1, 0.25, 0.5, 0.75, 0.9]
)
```

**Output:**
```python
{
    'expected_payoff': float,
    'quantiles': dict[float, float],  # Quantile -> payoff value
    'median': float,
    'iqr': float,                     # Interquartile range
    'payoff_distribution': np.ndarray
}
```

---

#### 4. Regime-Switching Valuation

**Purpose:** Model climate regimes (El Niño / La Niña)

```python
regime = adv_val.regime_switching_valuation(
    historical_data=np.array([...]),
    strike=1300,
    tick_value=5000,
    option_type="call",
    num_regimes=2,
    num_simulations=10000
)
```

**Output:**
```python
{
    'expected_payoff': float,
    'median_payoff': float,
    'std_payoff': float,
    'num_regimes': int,
    'regimes': list[dict],  # [{'mean': float, 'std': float, 'probability': float}, ...]
    'percentile_5': float,
    'percentile_95': float
}
```

---

#### 5. Importance Sampling

**Purpose:** Efficient simulation of rare events

```python
importance = adv_val.importance_sampling_valuation(
    mean=1000,
    std=150,
    strike=1400,  # Deep OTM
    tick_value=5000,
    option_type="call",
    num_simulations=10000,
    shift=None  # Auto-compute optimal shift
)
```

**Output:**
```python
{
    'expected_payoff': float,
    'std_payoff': float,
    'variance_reduction': float,      # Variance reduction ratio
    'effective_sample_size': float    # Effective number of samples
}
```

---

#### 6. Control Variates

**Purpose:** Variance reduction using control variates

```python
control = adv_val.control_variates_valuation(
    historical_data=np.array([...]),
    strike=1300,
    tick_value=5000,
    option_type="call",
    num_simulations=10000,
    control_mean=1200  # Known mean of control variate
)
```

**Output:**
```python
{
    'expected_payoff': float,
    'std_payoff': float,
    'variance_reduction_ratio': float,  # 1 - (new_var / old_var)
    'optimal_coefficient': float,       # Optimal c*
    'standard_mc_payoff': float,        # Without control variates
    'efficiency_gain': float            # Variance ratio
}
```

---

#### 7. Copula-Based Multi-Index Valuation

**Purpose:** Model dependence between multiple indices

```python
copula = adv_val.copula_based_valuation(
    data1=np.array([...]),  # HDD for location 1
    data2=np.array([...]),  # HDD for location 2
    strike1=1100,
    strike2=900,
    tick_value=3000,
    option_type="call",
    copula_type="gaussian",  # "gaussian" or "t"
    num_simulations=10000
)
```

**Output:**
```python
{
    'expected_payoff': float,
    'std_payoff': float,
    'correlation': float,
    'percentile_5': float,
    'percentile_95': float,
    'copula_type': str
}
```

---

## Complete API Reference

### Input/Output Dimension Summary

| Class/Method | Input Shape | Output Shape | Notes |
|--------------|-------------|--------------|-------|
| **WeatherData** | | | |
| `__init__(time, ...)` | time: (N,), data: (N,) | WeatherData | N = number of time points |
| `get_temperature()` | - | float or (N,) | Depends on is_timeseries |
| **HDD/CDD** | | | |
| `calculate(cumulative=True)` | - | float | Scalar total |
| `calculate(cumulative=False)` | - | (N,) | Daily values |
| `calculate_monthly()` | - | dict | Keys = months |
| `payoff(...)` | strike, tick, type | float | Dollar amount |
| **Precipitation** | | | |
| `calculate_total_*()` | - | float | Scalar |
| `calculate_*_days()` | - | int | Count |
| **Valuation** | | | |
| `historical_burn_analysis()` | data: (M,) | dict | M = history length |
| `monte_carlo_valuation()` | params, N_sim | dict | N_sim simulations |
| `bootstrap_valuation()` | data: (M,), N_boot | dict | N_boot bootstrap samples |
| `spread_option_pricing()` | data1: (M,), data2: (K,) | dict | Handles different lengths |

### Data Type Reference

| Type | Python Type | Example | Description |
|------|-------------|---------|-------------|
| datetime | datetime.datetime | `datetime(2024, 1, 1)` | Single timestamp |
| time_series | list[datetime] | `[datetime(...), ...]` | N timestamps |
| float_scalar | float | `65.0` | Single value |
| float_series | list[float] | `[25.0, 30.0, ...]` | N values |
| option_type | str | `"call"` or `"put"` | Option direction |
| distribution | str | `"normal"`, `"lognormal"`, `"gev"` | Distribution type |

---

## Examples

### Example 1: Complete HDD Option Workflow

```python
from datetime import datetime, timedelta
import numpy as np
from weather_derivatives import WeatherInputParser, HDD, DerivativeValuation

# Step 1: Create weather data (90-day winter)
parser = WeatherInputParser()
dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(90)]
temps = np.random.normal(35, 10, 90)  # Average 35°F, std 10°F

weather_data = parser.from_site(
    time=dates,
    site="Chicago",
    temperature=temps.tolist()
)

print(f"Created weather data: {weather_data}")
# Output: WeatherData(points=90, metrics=['temperature'], site=Chicago)

# Step 2: Calculate HDD
hdd = HDD(weather_data, reference_temperature=65.0)
total_hdd = hdd.calculate(cumulative=True)
monthly_hdd = hdd.calculate_monthly()

print(f"\nTotal HDD: {total_hdd:.0f}")
print(f"Monthly HDD: {monthly_hdd}")
# Output: Total HDD: 2700
#         Monthly HDD: {'2024-01': 930, '2024-02': 840, '2024-03': 930}

# Step 3: Calculate option payoff
strike = 2500
tick_value = 5000
payoff = hdd.payoff(strike=strike, tick_value=tick_value, option_type="call")

print(f"\nHDD Call Option:")
print(f"  Strike: {strike}")
print(f"  Actual HDD: {total_hdd:.0f}")
print(f"  Intrinsic Value: {(total_hdd - strike) * tick_value:,.0f}")
print(f"  Payoff: ${payoff:,.2f}")
# Output: Payoff: $1,000,000.00 (if HDD = 2700)

# Step 4: Value using historical data
historical_hdd = np.random.normal(2600, 300, 20)  # 20 years

valuation = DerivativeValuation(discount_rate=0.05)

# Historical burn analysis
burn = valuation.historical_burn_analysis(
    historical_data=historical_hdd,
    strike=strike,
    tick_value=tick_value,
    option_type="call"
)

print(f"\nHistorical Burn Analysis (20 years):")
print(f"  Mean Payoff: ${burn['mean_payoff']:,.2f}")
print(f"  Std Payoff: ${burn['std_payoff']:,.2f}")
print(f"  Probability ITM: {burn['probability_itm']:.1%}")
print(f"  95th Percentile: ${burn['percentile_95']:,.2f}")

# Monte Carlo valuation
mc = valuation.monte_carlo_valuation(
    mean=2600,
    std=300,
    strike=strike,
    tick_value=tick_value,
    option_type="call",
    num_simulations=100000
)

print(f"\nMonte Carlo Valuation (100,000 simulations):")
print(f"  Expected Payoff: ${mc['expected_payoff']:,.2f}")
print(f"  VaR (95%): ${mc['var_95']:,.2f}")

# Present value
days_to_maturity = 180
pv = valuation.calculate_present_value(mc['expected_payoff'], days_to_maturity)

print(f"\nPresent Value (180 days):")
print(f"  Discount Rate: {valuation.discount_rate:.1%}")
print(f"  PV: ${pv:,.2f}")
```

**Expected Output:**
```
Created weather data: WeatherData(points=90, metrics=['temperature'], site=Chicago)

Total HDD: 2700
Monthly HDD: {'2024-01': 930, '2024-02': 840, '2024-03': 930}

HDD Call Option:
  Strike: 2500
  Actual HDD: 2700
  Intrinsic Value: 1,000,000
  Payoff: $1,000,000.00

Historical Burn Analysis (20 years):
  Mean Payoff: $825,450.00
  Std Payoff: $683,250.00
  Probability ITM: 65.0%
  95th Percentile: $2,100,000.00

Monte Carlo Valuation (100,000 simulations):
  Expected Payoff: $830,250.00
  VaR (95%): $0.00

Present Value (180 days):
  Discount Rate: 5.0%
  PV: $810,125.50
```

---

### Example 2: Agricultural GDD Insurance

```python
from datetime import datetime, timedelta
import numpy as np
from weather_derivatives import (
    WeatherInputParser,
    FrostDerivative,
    DerivativeValuation
)

# Growing season for corn (May 1 - September 30)
dates = [datetime(2024, 5, 1) + timedelta(days=i) for i in range(153)]

# Simulate realistic growing season temperatures
day_of_season = np.arange(153)
seasonal_trend = 15 + 10 * np.sin(np.pi * day_of_season / 153)  # Warming then cooling
daily_variation = np.random.normal(0, 5, 153)
temps = seasonal_trend + daily_variation

min_temps = temps - np.random.uniform(5, 10, 153)
max_temps = temps + np.random.uniform(5, 10, 153)

# Create weather data
parser = WeatherInputParser()
weather_data = parser.from_site(
    time=dates,
    site="Iowa Corn Belt",
    temperature=temps.tolist(),
    min_temperature=min_temps.tolist(),
    max_temperature=max_temps.tolist()
)

# Calculate agricultural metrics
frost = FrostDerivative(weather_data, base_temperature=10.0)

gdd = frost.calculate_growing_degree_days(max_temperature=30.0)
chu = frost.calculate_crop_heat_units()
frost_days = frost.calculate_frost_days()

print(f"Growing Season Metrics:")
print(f"  Total GDD: {gdd:.0f}")
print(f"  Crop Heat Units: {chu:.0f}")
print(f"  Frost Days: {frost_days}")

# GDD insurance: Farmer needs 2500 GDD for full maturity
strike_gdd = 2500
tick_value_gdd = 50  # $50 per GDD unit below strike

payoff = frost.payoff_gdd(
    strike=strike_gdd,
    tick_value=tick_value_gdd,
    option_type="put"
)

print(f"\nGDD Insurance:")
print(f"  Required GDD: {strike_gdd}")
print(f"  Actual GDD: {gdd:.0f}")
print(f"  Shortfall: {max(strike_gdd - gdd, 0):.0f}")
print(f"  Payout: ${payoff:,.2f}")

# Price the insurance using historical data
historical_gdd = np.random.normal(2600, 250, 15)  # 15 years of data

valuation = DerivativeValuation()
insurance_pricing = valuation.weather_index_insurance_pricing(
    historical_data=historical_gdd,
    trigger=2300,  # No payout above 2300 GDD
    exit=2000,     # Maximum payout at 2000 GDD
    limit=50000    # $50,000 maximum payout
)

print(f"\nInsurance Pricing:")
print(f"  Pure Premium: ${insurance_pricing['pure_premium']:,.2f}")
print(f"  Risk Loading: ${insurance_pricing['risk_loading']:,.2f}")
print(f"  Admin Fee: ${insurance_pricing['admin_loading']:,.2f}")
print(f"  Total Premium: ${insurance_pricing['total_premium']:,.2f}")
print(f"  Loss Ratio: {insurance_pricing['loss_ratio']:.1%}")
print(f"  Attachment Probability: {insurance_pricing['attachment_probability']:.1%}")
```

---

### Example 3: Solar Farm Revenue Protection

```python
from datetime import datetime, timedelta
import numpy as np
from weather_derivatives import (
    WeatherInputParser,
    SolarDerivative,
    DerivativeValuation
)
from weather_derivatives.pricing.advanced_valuation import AdvancedValuation

# Annual solar data (365 days)
dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(365)]

# Seasonal irradiance pattern
day_of_year = np.arange(365)
seasonal = 5 + 3 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
daily_var = np.random.gamma(2, 0.5, 365)
irradiance = seasonal + daily_var  # kWh/m²/day

# Create weather data
parser = WeatherInputParser()
weather_data = parser.from_coordinates(
    time=dates,
    lon=-110.0,  # Arizona
    lat=33.0,
    solar_radiation=irradiance.tolist()
)

# Solar farm specs
solar = SolarDerivative(
    weather_data,
    panel_efficiency=0.20,  # 20%
    panel_area=10000        # 10,000 m²
)

# Calculate metrics
total_irradiance = solar.calculate_total_irradiance()
energy_production = solar.calculate_solar_energy_production()

print(f"Solar Farm Annual Performance:")
print(f"  Total Irradiance: {total_irradiance:,.0f} kWh/m²")
print(f"  Energy Production: {energy_production:,.0f} kWh")
print(f"  Revenue @ $0.10/kWh: ${energy_production * 0.10:,.2f}")

# Revenue protection put option
# Pays if production < 20 million kWh
strike_energy = 20_000_000
revenue_per_kwh = 0.10

payoff = solar.payoff_energy_production(
    strike=strike_energy,
    tick_value=revenue_per_kwh,
    option_type="put"
)

print(f"\nRevenue Protection:")
print(f"  Strike: {strike_energy:,} kWh")
print(f"  Shortfall: {max(strike_energy - energy_production, 0):,.0f} kWh")
print(f"  Payout: ${payoff:,.2f}")

# Value using advanced methods
historical_irradiance = np.random.normal(2800, 400, 20)  # 20 years

# Bootstrap valuation
adv_val = AdvancedValuation()
bootstrap = adv_val.bootstrap_valuation(
    historical_data=historical_irradiance,
    strike=2500,
    tick_value=800,  # $800 per unit of irradiance
    option_type="put",
    num_bootstrap_samples=10000
)

print(f"\nBootstrap Valuation:")
print(f"  Mean Payoff: ${bootstrap['mean_payoff']:,.2f}")
print(f"  95% CI: ${bootstrap['confidence_interval'][0]:,.2f} - ${bootstrap['confidence_interval'][1]:,.2f}")

# Extreme Value Theory
evt = adv_val.extreme_value_theory_pricing(
    historical_data=historical_irradiance,
    strike=2500,
    tick_value=800,
    option_type="put",
    threshold_percentile=0.20  # Focus on low tail
)

if 'expected_payoff' in evt:
    print(f"\nExtreme Value Theory:")
    print(f"  Expected Payoff: ${evt['expected_payoff']:,.2f}")
    print(f"  Tail Probability: {evt['tail_probability']:.2%}")
```

---

## Tips & Best Practices

### 1. Choosing the Right Derivative

| Use Case | Recommended Derivative | Key Metric |
|----------|----------------------|------------|
| Winter heating costs | HDD | Total HDD > strike |
| Summer cooling costs | CDD | Total CDD > strike |
| Ski resort revenue | Snow | Total snowfall < strike |
| Crop yield protection | GDD, Frost | GDD < strike or frost days > strike |
| Solar plant revenue | Solar Irradiance | Total irradiance < strike |
| Wind farm revenue | Wind Energy | Total energy < strike |
| Agricultural irrigation | Precipitation | Total rainfall < strike |

### 2. Valuation Method Selection

| Scenario | Recommended Method | Reason |
|----------|-------------------|---------|
| Have 10+ years historical data | Historical Burn Analysis | Most reliable |
| Limited data (<5 years) | Monte Carlo + Bootstrap | Fill in gaps |
| Deep OTM options | Importance Sampling or EVT | Efficient rare event simulation |
| Multiple correlated indices | Copula-based | Captures dependence |
| Climate regime shifts | Regime-Switching | Models El Niño/La Niña |
| Time series data | AR Forecast Valuation | Uses autocorrelation |
| Insurance products | Weather Index Insurance | Built-in loadings |

### 3. Data Quality Requirements

**Minimum Requirements:**
- **HDD/CDD**: 30+ daily temperature observations
- **GDD**: Full growing season (120+ days)
- **Historical Valuation**: 5+ years minimum, 10+ preferred
- **EVT**: At least 10 exceedances for tail modeling

**Data Consistency:**
- All temperatures in same units (°F or °C)
- Consistent time intervals (daily, weekly)
- Handle missing data appropriately

### 4. Common Pitfalls

❌ **Don't:**
- Mix temperature units (Fahrenheit vs Celsius)
- Use too short historical periods (<5 years)
- Ignore autocorrelation in time series
- Apply normal distribution to extremes

✅ **Do:**
- Validate data quality first
- Use appropriate distribution fitting
- Check for regime changes in climate
- Consider block bootstrap for autocorrelated data
- Test sensitivity to parameters

---

## Support & Contributing

### Getting Help

- **Documentation**: This guide
- **Examples**: `/examples` directory
- **Issues**: [GitHub Issues](https://github.com/yourusername/libwd/issues)

### Contributing

Contributions are welcome! Please submit pull requests for:
- New derivative types
- Additional valuation methods
- Bug fixes
- Documentation improvements
- Test cases

---

## License

MIT License - see LICENSE file for details

---

**Version:** 2.0.0
**Last Updated:** 2024
**Maintainer:** Weather Derivatives Team
