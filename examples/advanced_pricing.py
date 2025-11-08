"""
Advanced pricing and valuation examples.

Demonstrates:
- Historical burn analysis
- Monte Carlo simulation
- Risk metrics calculation
- Greeks calculation
"""

from datetime import datetime, timedelta
import numpy as np
from weather_derivatives import (
    WeatherInputParser,
    HDD, CDD,
    DerivativeValuation,
)


def example_historical_burn():
    """Example: Historical burn analysis for HDD derivatives."""
    print("=" * 60)
    print("Historical Burn Analysis Example")
    print("=" * 60)

    # Simulate 20 years of historical HDD data for winter months
    np.random.seed(42)
    historical_hdd = np.random.normal(1200, 200, 20)  # 20 years of data

    print("Historical HDD values:")
    for i, hdd in enumerate(historical_hdd, 1):
        print(f"  Year {i}: {hdd:.0f}")

    # Create valuation engine
    valuation = DerivativeValuation(discount_rate=0.05)

    # Analyze HDD call option
    strike = 1300
    tick_value = 5000  # $5,000 per HDD point

    analysis = valuation.historical_burn_analysis(
        historical_data=historical_hdd,
        strike=strike,
        tick_value=tick_value,
        option_type="call"
    )

    print(f"\nHDD Call Option Analysis (Strike={strike}, Tick=${tick_value:,}):")
    print(f"  Mean Payoff: ${analysis['mean_payoff']:,.2f}")
    print(f"  Median Payoff: ${analysis['median_payoff']:,.2f}")
    print(f"  Std Payoff: ${analysis['std_payoff']:,.2f}")
    print(f"  Max Payoff: ${analysis['max_payoff']:,.2f}")
    print(f"  95th Percentile: ${analysis['percentile_95']:,.2f}")
    print(f"  Probability ITM: {analysis['probability_itm']:.1%}")

    # Present value calculation
    days_to_maturity = 180
    pv = valuation.calculate_present_value(analysis['mean_payoff'], days_to_maturity)
    print(f"\nPresent Value (180 days to maturity): ${pv:,.2f}")

    print()


def example_monte_carlo():
    """Example: Monte Carlo simulation for CDD derivatives."""
    print("=" * 60)
    print("Monte Carlo Simulation Example")
    print("=" * 60)

    # Create valuation engine
    valuation = DerivativeValuation(discount_rate=0.04)

    # CDD parameters based on historical data
    mean_cdd = 800
    std_cdd = 150
    strike = 900
    tick_value = 2500

    # Run Monte Carlo simulation
    mc_results = valuation.monte_carlo_valuation(
        mean=mean_cdd,
        std=std_cdd,
        strike=strike,
        tick_value=tick_value,
        option_type="call",
        num_simulations=50000,
        distribution="normal"
    )

    print(f"CDD Call Option - Monte Carlo Results ({mc_results['num_simulations']:,} simulations):")
    print(f"  Expected Payoff: ${mc_results['expected_payoff']:,.2f}")
    print(f"  Median Payoff: ${mc_results['median_payoff']:,.2f}")
    print(f"  Std Deviation: ${mc_results['std_payoff']:,.2f}")
    print(f"  5th Percentile: ${mc_results['percentile_5']:,.2f}")
    print(f"  95th Percentile: ${mc_results['percentile_95']:,.2f}")
    print(f"  Value at Risk (95%): ${mc_results['var_95']:,.2f}")
    print(f"  Probability ITM: {mc_results['probability_itm']:.1%}")

    print()


def example_risk_metrics():
    """Example: Calculate risk metrics for a portfolio."""
    print("=" * 60)
    print("Risk Metrics Calculation Example")
    print("=" * 60)

    # Simulate portfolio payoffs
    np.random.seed(123)
    num_scenarios = 10000
    payoffs = np.random.normal(50000, 30000, num_scenarios)
    payoffs = np.maximum(payoffs, 0)  # Floor at zero

    # Create valuation engine
    valuation = DerivativeValuation()

    # Calculate risk metrics
    risk_metrics = valuation.calculate_risk_metrics(payoffs, confidence_level=0.95)

    print("Portfolio Risk Metrics:")
    print(f"  Expected Value: ${risk_metrics['expected_value']:,.2f}")
    print(f"  Volatility: ${risk_metrics['volatility']:,.2f}")
    print(f"  Value at Risk (95%): ${risk_metrics['var']:,.2f}")
    print(f"  Conditional VaR: ${risk_metrics['cvar']:,.2f}")
    print(f"  Downside Deviation: ${risk_metrics['downside_deviation']:,.2f}")
    print(f"  Skewness: {risk_metrics['skewness']:.3f}")
    print(f"  Kurtosis: {risk_metrics['kurtosis']:.3f}")

    print()


def example_greeks():
    """Example: Calculate option Greeks."""
    print("=" * 60)
    print("Option Greeks Calculation Example")
    print("=" * 60)

    # Create valuation engine
    valuation = DerivativeValuation(discount_rate=0.05)

    # Option parameters
    current_index = 1100  # Current HDD index level
    strike = 1200
    time_to_maturity = 90 / 365.25  # 90 days
    volatility = 0.20  # 20% annualized volatility

    # Calculate Greeks for call option
    greeks_call = valuation.calculate_greeks(
        current_index=current_index,
        strike=strike,
        time_to_maturity_years=time_to_maturity,
        volatility=volatility,
        option_type="call"
    )

    print(f"HDD Call Option Greeks (Index={current_index}, Strike={strike}):")
    print(f"  Price: ${greeks_call['price']:,.2f}")
    print(f"  Delta: {greeks_call['delta']:.4f}")
    print(f"  Gamma: {greeks_call['gamma']:.6f}")
    print(f"  Vega: {greeks_call['vega']:.4f}")
    print(f"  Theta: ${greeks_call['theta']:,.2f} per day")

    # Calculate Greeks for put option
    greeks_put = valuation.calculate_greeks(
        current_index=current_index,
        strike=strike,
        time_to_maturity_years=time_to_maturity,
        volatility=volatility,
        option_type="put"
    )

    print(f"\nHDD Put Option Greeks (Index={current_index}, Strike={strike}):")
    print(f"  Price: ${greeks_put['price']:,.2f}")
    print(f"  Delta: {greeks_put['delta']:.4f}")
    print(f"  Gamma: {greeks_put['gamma']:.6f}")
    print(f"  Vega: {greeks_put['vega']:.4f}")
    print(f"  Theta: ${greeks_put['theta']:,.2f} per day")

    print()


def example_black_scholes():
    """Example: Black-Scholes approximation."""
    print("=" * 60)
    print("Black-Scholes Approximation Example")
    print("=" * 60)

    # Create valuation engine
    valuation = DerivativeValuation(discount_rate=0.05)

    # Parameters
    current_index = 1000
    volatility = 0.25

    # Calculate prices for different strikes and maturities
    strikes = [900, 1000, 1100, 1200]
    maturities = [30, 60, 90, 180]  # days

    print("Call Option Prices:")
    print(f"{'Strike':<10} {'30d':<10} {'60d':<10} {'90d':<10} {'180d':<10}")
    print("-" * 50)

    for strike in strikes:
        prices = []
        for maturity in maturities:
            price = valuation.black_scholes_approximation(
                current_index=current_index,
                strike=strike,
                time_to_maturity_years=maturity / 365.25,
                volatility=volatility,
                option_type="call"
            )
            prices.append(f"${price:.2f}")

        print(f"{strike:<10} {prices[0]:<10} {prices[1]:<10} {prices[2]:<10} {prices[3]:<10}")

    print()


def example_complete_workflow():
    """Complete workflow: Data -> Derivatives -> Valuation."""
    print("=" * 60)
    print("Complete Workflow Example")
    print("=" * 60)

    # Step 1: Create weather data
    parser = WeatherInputParser()
    dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(90)]
    temperatures = np.random.normal(35, 10, 90)  # Cold winter

    weather_data = parser.from_site(
        time=dates,
        site="Boston Logan",
        temperature=temperatures
    )

    print(f"Step 1: Created weather data with {len(weather_data)} observations")

    # Step 2: Calculate HDD
    hdd = HDD(weather_data, reference_temperature=65.0)
    hdd_value = hdd.calculate()

    print(f"Step 2: Calculated HDD = {hdd_value:.2f}")

    # Step 3: Calculate option payoff
    strike = 2700
    tick_value = 5000
    payoff = hdd.payoff(strike=strike, tick_value=tick_value, option_type="call")

    print(f"Step 3: Call option payoff (Strike={strike}) = ${payoff:,.2f}")

    # Step 4: Historical analysis for pricing
    historical_hdd_values = np.random.normal(2800, 300, 15)
    valuation = DerivativeValuation()

    burn_analysis = valuation.historical_burn_analysis(
        historical_data=historical_hdd_values,
        strike=strike,
        tick_value=tick_value,
        option_type="call"
    )

    print(f"Step 4: Historical burn analysis")
    print(f"  Fair value estimate: ${burn_analysis['mean_payoff']:,.2f}")
    print(f"  Probability ITM: {burn_analysis['probability_itm']:.1%}")

    # Step 5: Risk metrics
    simulated_payoffs = [
        max((hdd_val - strike) * tick_value, 0)
        for hdd_val in historical_hdd_values
    ]

    risk = valuation.calculate_risk_metrics(simulated_payoffs)
    print(f"Step 5: Risk metrics")
    print(f"  Expected value: ${risk['expected_value']:,.2f}")
    print(f"  Volatility: ${risk['volatility']:,.2f}")
    print(f"  VaR (95%): ${risk['var']:,.2f}")

    print()


if __name__ == "__main__":
    example_historical_burn()
    example_monte_carlo()
    example_risk_metrics()
    example_greeks()
    example_black_scholes()
    example_complete_workflow()

    print("=" * 60)
    print("All advanced examples completed successfully!")
    print("=" * 60)
