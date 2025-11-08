"""
State-of-the-art valuation methods showcase (v2.0).

Demonstrates advanced pricing techniques:
- Bootstrap resampling
- Extreme Value Theory
- Quantile regression
- Regime-switching models
- Time series forecasting
- Weather index insurance
- Portfolio optimization
- And more...
"""

from datetime import datetime, timedelta
import numpy as np
from weather_derivatives import (
    WeatherInputParser,
    HDD,
    DerivativeValuation,
)
from weather_derivatives.pricing.advanced_valuation import AdvancedValuation


def example_bootstrap_valuation():
    """Example: Bootstrap resampling for robust valuation."""
    print("=" * 70)
    print("Bootstrap Resampling Valuation")
    print("=" * 70)

    # Historical HDD data (20 years)
    np.random.seed(42)
    historical_hdd = np.random.normal(1200, 250, 20)

    adv_val = AdvancedValuation()

    bootstrap_result = adv_val.bootstrap_valuation(
        historical_data=historical_hdd,
        strike=1300,
        tick_value=5000,
        option_type="call",
        num_bootstrap_samples=10000,
        block_size=3  # 3-year blocks to preserve autocorrelation
    )

    print(f"Bootstrap Results (10,000 samples):")
    print(f"  Mean Payoff: ${bootstrap_result['mean_payoff']:,.2f}")
    print(f"  Median Payoff: ${bootstrap_result['median_payoff']:,.2f}")
    print(f"  Std Dev: ${bootstrap_result['std_payoff']:,.2f}")
    print(f"  95% CI: ${bootstrap_result['confidence_interval'][0]:,.2f} - ${bootstrap_result['confidence_interval'][1]:,.2f}")
    print(f"  5th Percentile: ${bootstrap_result['percentile_5']:,.2f}")
    print(f"  95th Percentile: ${bootstrap_result['percentile_95']:,.2f}")

    print()


def example_extreme_value_theory():
    """Example: EVT for tail risk assessment."""
    print("=" * 70)
    print("Extreme Value Theory (EVT) Pricing")
    print("=" * 70)

    # Historical CDD data with some extreme values
    np.random.seed(123)
    historical_cdd = np.concatenate([
        np.random.normal(800, 150, 18),
        np.array([1200, 1350])  # Two extreme years
    ])

    adv_val = AdvancedValuation()

    evt_result = adv_val.extreme_value_theory_pricing(
        historical_data=historical_cdd,
        strike=1000,
        tick_value=2500,
        option_type="call",
        threshold_percentile=0.80
    )

    if "error" not in evt_result:
        print(f"EVT Analysis:")
        print(f"  Expected Payoff: ${evt_result['expected_payoff']:,.2f}")
        print(f"  Tail Probability: {evt_result['tail_probability']:.2%}")
        print(f"  GPD Shape (ξ): {evt_result['gpd_shape']:.4f}")
        print(f"  GPD Scale (σ): {evt_result['gpd_scale']:.2f}")
        print(f"  Threshold: {evt_result['threshold']:.2f}")
        print(f"  Number of Exceedances: {evt_result['num_exceedances']}")

    print()


def example_quantile_regression():
    """Example: Quantile-based pricing."""
    print("=" * 70)
    print("Quantile Regression Pricing")
    print("=" * 70)

    historical_data = np.random.normal(1000, 200, 50)

    adv_val = AdvancedValuation()

    quantile_result = adv_val.quantile_regression_pricing(
        historical_data=historical_data,
        strike=1100,
        tick_value=1000,
        option_type="call",
        quantiles=[0.05, 0.25, 0.5, 0.75, 0.95]
    )

    print(f"Quantile Analysis:")
    print(f"  Expected Payoff: ${quantile_result['expected_payoff']:,.2f}")
    print(f"  Median Payoff: ${quantile_result['median']:,.2f}")
    print(f"  Interquartile Range: ${quantile_result['iqr']:,.2f}")
    print(f"\n  Quantile Values:")
    for q, val in quantile_result['quantiles'].items():
        print(f"    {q:.0%}: ${val:,.2f}")

    print()


def example_regime_switching():
    """Example: Regime-switching model for climate variability."""
    print("=" * 70)
    print("Regime-Switching Model (El Niño / La Niña)")
    print("=" * 70)

    # Simulate data with two regimes
    np.random.seed(789)
    regime1_data = np.random.normal(900, 100, 12)  # La Niña (cooler)
    regime2_data = np.random.normal(1100, 120, 8)  # El Niño (warmer)
    historical_data = np.concatenate([regime1_data, regime2_data])
    np.random.shuffle(historical_data)

    adv_val = AdvancedValuation()

    regime_result = adv_val.regime_switching_valuation(
        historical_data=historical_data,
        strike=1050,
        tick_value=3000,
        option_type="call",
        num_regimes=2,
        num_simulations=10000
    )

    print(f"Regime-Switching Results:")
    print(f"  Expected Payoff: ${regime_result['expected_payoff']:,.2f}")
    print(f"  Median Payoff: ${regime_result['median_payoff']:,.2f}")
    print(f"  Std Dev: ${regime_result['std_payoff']:,.2f}")
    print(f"\n  Identified Regimes:")
    for i, regime in enumerate(regime_result['regimes'], 1):
        print(f"    Regime {i}: Mean={regime['mean']:.1f}, Std={regime['std']:.1f}, Prob={regime['probability']:.1%}")

    print()


def example_importance_sampling():
    """Example: Importance sampling for deep OTM options."""
    print("=" * 70)
    print("Importance Sampling Monte Carlo")
    print("=" * 70)

    adv_val = AdvancedValuation()

    # Deep OTM call option
    is_result = adv_val.importance_sampling_valuation(
        mean=1000,
        std=150,
        strike=1400,  # Deep OTM
        tick_value=5000,
        option_type="call",
        num_simulations=10000
    )

    print(f"Importance Sampling Results:")
    print(f"  Expected Payoff: ${is_result['expected_payoff']:,.2f}")
    print(f"  Std Dev: ${is_result['std_payoff']:,.2f}")
    print(f"  Variance Reduction: {is_result['variance_reduction']:.2%}")
    print(f"  Effective Sample Size: {is_result['effective_sample_size']:.0f}")

    print()


def example_advanced_valuation_methods():
    """Example: Using new advanced methods in DerivativeValuation."""
    print("=" * 70)
    print("Advanced Valuation Methods in DerivativeValuation")
    print("=" * 70)

    historical_hdd = np.random.normal(1200, 200, 30)

    valuation = DerivativeValuation(discount_rate=0.05)

    # 1. Burn Rate Analysis with distribution fitting
    print("1. Burn Rate Analysis with GEV Distribution:")
    burn_result = valuation.burn_rate_analysis(
        historical_data=historical_hdd,
        strike=1300,
        tick_value=5000,
        distribution_fit="gev"  # Generalized Extreme Value
    )
    print(f"   Mean Payoff: ${burn_result['mean_payoff']:,.2f}")
    print(f"   Burn Rate: {burn_result['burn_rate']:.1%}")
    if 'distribution_params' in burn_result:
        print(f"   GEV Parameters: {burn_result['distribution_params']}")

    # 2. Time Series Forecasting
    print("\n2. Time Series Forecast Valuation:")
    ts_result = valuation.time_series_forecast_valuation(
        historical_data=historical_hdd,
        strike=1300,
        tick_value=5000,
        forecast_periods=30,
        num_simulations=1000
    )
    print(f"   Expected Payoff: ${ts_result['expected_payoff']:,.2f}")
    print(f"   AR Coefficient: {ts_result['ar_coefficient']:.3f}")
    print(f"   Innovation Std: {ts_result['innovation_std']:.2f}")

    # 3. Weather Index Insurance
    print("\n3. Weather Index Insurance Pricing:")
    insurance = valuation.weather_index_insurance_pricing(
        historical_data=historical_hdd,
        trigger=1000,
        exit=1500,
        limit=100000
    )
    print(f"   Pure Premium: ${insurance['pure_premium']:,.2f}")
    print(f"   Risk Loading: ${insurance['risk_loading']:,.2f}")
    print(f"   Total Premium: ${insurance['total_premium']:,.2f}")
    print(f"   Attachment Probability: {insurance['attachment_probability']:.1%}")
    print(f"   Loss Ratio: {insurance['loss_ratio']:.2%}")

    # 4. Asian Option Pricing
    print("\n4. Asian Option (Average) Pricing:")
    asian_result = valuation.asian_option_pricing(
        historical_data=historical_hdd,
        strike=1200,
        tick_value=5000,
        averaging_periods=30
    )
    print(f"   Expected Payoff: ${asian_result['expected_payoff']:,.2f}")
    print(f"   Volatility Reduction: {asian_result['volatility_reduction']:.2f}x")

    # 5. Spread Option
    print("\n5. Spread Option (HDD_City1 - HDD_City2):")
    historical_hdd2 = np.random.normal(1100, 180, 30)
    spread_result = valuation.spread_option_pricing(
        historical_data1=historical_hdd,
        historical_data2=historical_hdd2,
        strike=100,  # Strike on the spread
        tick_value=2500
    )
    print(f"   Expected Payoff: ${spread_result['expected_payoff']:,.2f}")
    print(f"   Spread Mean: {spread_result['spread_mean']:.2f}")
    print(f"   Correlation: {spread_result['correlation']:.3f}")

    # 6. Portfolio Optimization
    print("\n6. Portfolio Optimization:")
    derivatives = [
        {"expected_return": 50000, "volatility": 20000},
        {"expected_return": 30000, "volatility": 15000},
        {"expected_return": 40000, "volatility": 25000},
    ]
    portfolio = valuation.portfolio_optimization(
        derivatives=derivatives,
        budget=1000000,
        risk_aversion=0.5
    )
    print(f"   Optimal Weights: {[f'{w:.1%}' for w in portfolio['optimal_weights']]}")
    print(f"   Expected Return: ${portfolio['expected_return']:,.2f}")
    print(f"   Portfolio Std: ${portfolio['portfolio_std']:,.2f}")
    print(f"   Sharpe Ratio: {portfolio['sharpe_ratio']:.3f}")

    # 7. Stochastic Volatility
    print("\n7. Stochastic Volatility Model:")
    stoch_vol = valuation.stochastic_volatility_valuation(
        mean=1200,
        vol_mean=150,
        vol_std=30,
        vol_mean_reversion=0.5,
        strike=1300,
        tick_value=5000,
        time_periods=30
    )
    print(f"   Expected Payoff: ${stoch_vol['expected_payoff']:,.2f}")
    print(f"   Implied Volatility: {stoch_vol['implied_volatility']:.2f}")

    print()


def example_copula_multi_index():
    """Example: Copula for multi-index derivatives."""
    print("=" * 70)
    print("Copula-Based Multi-Index Valuation")
    print("=" * 70)

    # Two correlated weather indices
    np.random.seed(456)
    data1 = np.random.normal(1000, 150, 30)
    data2 = data1 * 0.7 + np.random.normal(300, 100, 30)  # Correlated

    adv_val = AdvancedValuation()

    # Gaussian copula
    copula_gaussian = adv_val.copula_based_valuation(
        data1=data1,
        data2=data2,
        strike1=1100,
        strike2=900,
        tick_value=3000,
        copula_type="gaussian"
    )

    print(f"Gaussian Copula Results:")
    print(f"  Expected Payoff: ${copula_gaussian['expected_payoff']:,.2f}")
    print(f"  Correlation: {copula_gaussian['correlation']:.3f}")
    print(f"  Std Dev: ${copula_gaussian['std_payoff']:,.2f}")

    # Student-t copula (heavier tails)
    copula_t = adv_val.copula_based_valuation(
        data1=data1,
        data2=data2,
        strike1=1100,
        strike2=900,
        tick_value=3000,
        copula_type="t"
    )

    print(f"\nStudent-t Copula Results:")
    print(f"  Expected Payoff: ${copula_t['expected_payoff']:,.2f}")
    print(f"  Correlation: {copula_t['correlation']:.3f}")

    print()


def example_sensitivity_analysis():
    """Example: Comprehensive sensitivity analysis."""
    print("=" * 70)
    print("Sensitivity Analysis")
    print("=" * 70)

    valuation = DerivativeValuation()

    base_params = {"mean": 1200, "std": 200}
    param_ranges = {
        "mean": (1000, 1400),
        "std": (100, 300)
    }

    sensitivity = valuation.sensitivity_analysis(
        base_params=base_params,
        param_ranges=param_ranges,
        strike=1300,
        tick_value=5000,
        num_points=5
    )

    print(f"Sensitivity Results:")
    for param, result in sensitivity.items():
        print(f"\n  {param}:")
        print(f"    Sensitivity: ${result['sensitivity']:,.2f} per unit change")
        print(f"    Range: {result['param_values'][0]:.0f} - {result['param_values'][-1]:.0f}")
        print(f"    Payoff Range: ${result['payoffs'][0]:,.2f} - ${result['payoffs'][-1]:,.2f}")

    print()


if __name__ == "__main__":
    print("\n")
    print("*" * 70)
    print("   STATE-OF-THE-ART VALUATION METHODS SHOWCASE")
    print("   Weather Derivatives Library v2.0")
    print("*" * 70)
    print("\n")

    example_bootstrap_valuation()
    example_extreme_value_theory()
    example_quantile_regression()
    example_regime_switching()
    example_importance_sampling()
    example_copula_multi_index()
    example_advanced_valuation_methods()
    example_sensitivity_analysis()

    print("=" * 70)
    print("All state-of-the-art valuation examples completed successfully!")
    print("=" * 70)
