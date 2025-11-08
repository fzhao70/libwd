"""
Valuation and pricing methods for weather derivatives.

Includes:
- Historical burn analysis
- Monte Carlo simulation
- Index-based pricing
"""

from typing import List, Optional, Dict, Any, Union
import numpy as np
from datetime import datetime
from ..core.weather_data import WeatherData


class DerivativeValuation:
    """
    Valuation toolkit for weather derivatives.

    Provides methods for pricing and risk analysis.
    """

    def __init__(self, discount_rate: float = 0.05):
        """
        Initialize valuation engine.

        Args:
            discount_rate: Annual discount rate for present value calculations
        """
        self.discount_rate = discount_rate

    def historical_burn_analysis(
        self,
        historical_data: List[float],
        strike: float,
        tick_value: float,
        option_type: str = "call"
    ) -> Dict[str, float]:
        """
        Perform historical burn analysis.

        Analyzes historical payoffs to estimate fair value.

        Args:
            historical_data: List of historical index values
            strike: Strike level
            tick_value: Dollar value per unit
            option_type: "call" or "put"

        Returns:
            Dictionary with statistics (mean, std, percentiles, etc.)
        """
        payoffs = []

        for value in historical_data:
            if option_type.lower() == "call":
                payoff = max(value - strike, 0) * tick_value
            elif option_type.lower() == "put":
                payoff = max(strike - value, 0) * tick_value
            else:
                raise ValueError(f"Unknown option type: {option_type}")

            payoffs.append(payoff)

        payoffs = np.array(payoffs)

        return {
            "mean_payoff": np.mean(payoffs),
            "median_payoff": np.median(payoffs),
            "std_payoff": np.std(payoffs),
            "min_payoff": np.min(payoffs),
            "max_payoff": np.max(payoffs),
            "percentile_25": np.percentile(payoffs, 25),
            "percentile_75": np.percentile(payoffs, 75),
            "percentile_95": np.percentile(payoffs, 95),
            "probability_itm": np.mean(payoffs > 0),
            "num_scenarios": len(payoffs),
        }

    def monte_carlo_valuation(
        self,
        mean: float,
        std: float,
        strike: float,
        tick_value: float,
        option_type: str = "call",
        num_simulations: int = 10000,
        distribution: str = "normal"
    ) -> Dict[str, float]:
        """
        Monte Carlo simulation for derivative pricing.

        Args:
            mean: Mean of the index distribution
            std: Standard deviation of the index
            strike: Strike level
            tick_value: Dollar value per unit
            option_type: "call" or "put"
            num_simulations: Number of Monte Carlo simulations
            distribution: "normal" or "lognormal"

        Returns:
            Dictionary with valuation statistics
        """
        # Generate random scenarios
        if distribution == "normal":
            scenarios = np.random.normal(mean, std, num_simulations)
        elif distribution == "lognormal":
            scenarios = np.random.lognormal(
                np.log(mean) - 0.5 * (std / mean) ** 2,
                std / mean,
                num_simulations
            )
        else:
            raise ValueError(f"Unknown distribution: {distribution}")

        # Calculate payoffs
        payoffs = []
        for value in scenarios:
            if option_type.lower() == "call":
                payoff = max(value - strike, 0) * tick_value
            elif option_type.lower() == "put":
                payoff = max(strike - value, 0) * tick_value
            else:
                raise ValueError(f"Unknown option type: {option_type}")

            payoffs.append(payoff)

        payoffs = np.array(payoffs)

        return {
            "expected_payoff": np.mean(payoffs),
            "median_payoff": np.median(payoffs),
            "std_payoff": np.std(payoffs),
            "min_payoff": np.min(payoffs),
            "max_payoff": np.max(payoffs),
            "percentile_5": np.percentile(payoffs, 5),
            "percentile_95": np.percentile(payoffs, 95),
            "var_95": np.percentile(payoffs, 5),  # Value at Risk
            "probability_itm": np.mean(payoffs > 0),
            "num_simulations": num_simulations,
        }

    def calculate_present_value(
        self,
        expected_payoff: float,
        time_to_maturity_days: int
    ) -> float:
        """
        Calculate present value of expected payoff.

        Args:
            expected_payoff: Expected payoff at maturity
            time_to_maturity_days: Days until maturity

        Returns:
            Present value
        """
        years = time_to_maturity_days / 365.25
        discount_factor = 1 / (1 + self.discount_rate) ** years
        return expected_payoff * discount_factor

    def calculate_risk_metrics(
        self,
        payoffs: Union[List[float], np.ndarray],
        confidence_level: float = 0.95
    ) -> Dict[str, float]:
        """
        Calculate risk metrics for a distribution of payoffs.

        Args:
            payoffs: Array of payoff values
            confidence_level: Confidence level for VaR and CVaR

        Returns:
            Dictionary of risk metrics
        """
        payoffs = np.array(payoffs)
        sorted_payoffs = np.sort(payoffs)

        var_index = int((1 - confidence_level) * len(sorted_payoffs))
        var = sorted_payoffs[var_index]

        # Conditional Value at Risk (CVaR) - expected loss beyond VaR
        cvar = np.mean(sorted_payoffs[:var_index])

        return {
            "var": var,
            "cvar": cvar,
            "expected_value": np.mean(payoffs),
            "volatility": np.std(payoffs),
            "skewness": float(np.mean(((payoffs - np.mean(payoffs)) / np.std(payoffs)) ** 3)),
            "kurtosis": float(np.mean(((payoffs - np.mean(payoffs)) / np.std(payoffs)) ** 4)),
            "downside_deviation": np.std(payoffs[payoffs < np.mean(payoffs)]),
        }

    def black_scholes_approximation(
        self,
        current_index: float,
        strike: float,
        time_to_maturity_years: float,
        volatility: float,
        option_type: str = "call",
        risk_free_rate: Optional[float] = None
    ) -> float:
        """
        Black-Scholes-like approximation for weather derivatives.

        Note: This is an approximation as weather indices may not follow
        geometric Brownian motion, but can provide a useful benchmark.

        Args:
            current_index: Current index level
            strike: Strike level
            time_to_maturity_years: Time to maturity in years
            volatility: Implied volatility (annualized)
            option_type: "call" or "put"
            risk_free_rate: Risk-free rate (uses discount_rate if not provided)

        Returns:
            Option value
        """
        from scipy.stats import norm

        if risk_free_rate is None:
            risk_free_rate = self.discount_rate

        if time_to_maturity_years <= 0:
            # At maturity
            if option_type.lower() == "call":
                return max(current_index - strike, 0)
            else:
                return max(strike - current_index, 0)

        # Black-Scholes formula
        d1 = (
            np.log(current_index / strike)
            + (risk_free_rate + 0.5 * volatility ** 2) * time_to_maturity_years
        ) / (volatility * np.sqrt(time_to_maturity_years))

        d2 = d1 - volatility * np.sqrt(time_to_maturity_years)

        if option_type.lower() == "call":
            value = (
                current_index * norm.cdf(d1)
                - strike * np.exp(-risk_free_rate * time_to_maturity_years) * norm.cdf(d2)
            )
        elif option_type.lower() == "put":
            value = (
                strike * np.exp(-risk_free_rate * time_to_maturity_years) * norm.cdf(-d2)
                - current_index * norm.cdf(-d1)
            )
        else:
            raise ValueError(f"Unknown option type: {option_type}")

        return value

    def calculate_greeks(
        self,
        current_index: float,
        strike: float,
        time_to_maturity_years: float,
        volatility: float,
        option_type: str = "call",
        risk_free_rate: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Calculate option Greeks using finite differences.

        Args:
            current_index: Current index level
            strike: Strike level
            time_to_maturity_years: Time to maturity in years
            volatility: Implied volatility
            option_type: "call" or "put"
            risk_free_rate: Risk-free rate

        Returns:
            Dictionary of Greeks (delta, gamma, vega, theta)
        """
        if risk_free_rate is None:
            risk_free_rate = self.discount_rate

        # Base price
        price = self.black_scholes_approximation(
            current_index, strike, time_to_maturity_years, volatility, option_type, risk_free_rate
        )

        # Delta (sensitivity to index change)
        ds = 0.01 * current_index
        price_up = self.black_scholes_approximation(
            current_index + ds, strike, time_to_maturity_years, volatility, option_type, risk_free_rate
        )
        price_down = self.black_scholes_approximation(
            current_index - ds, strike, time_to_maturity_years, volatility, option_type, risk_free_rate
        )
        delta = (price_up - price_down) / (2 * ds)

        # Gamma (second derivative wrt index)
        gamma = (price_up - 2 * price + price_down) / (ds ** 2)

        # Vega (sensitivity to volatility)
        dvol = 0.01
        price_vega = self.black_scholes_approximation(
            current_index, strike, time_to_maturity_years, volatility + dvol, option_type, risk_free_rate
        )
        vega = (price_vega - price) / dvol

        # Theta (time decay)
        dt = 1 / 365.25  # One day
        if time_to_maturity_years > dt:
            price_theta = self.black_scholes_approximation(
                current_index, strike, time_to_maturity_years - dt, volatility, option_type, risk_free_rate
            )
            theta = (price_theta - price)  # Per day
        else:
            theta = -price  # Approximate for very short time

        return {
            "delta": delta,
            "gamma": gamma,
            "vega": vega,
            "theta": theta,
            "price": price,
        }

    def burn_rate_analysis(
        self,
        historical_data: Union[List[float], np.ndarray],
        strike: float,
        tick_value: float,
        option_type: str = "call",
        distribution_fit: str = "empirical"
    ) -> Dict[str, Any]:
        """
        Enhanced burn rate analysis with distribution fitting.

        Args:
            historical_data: Historical index values
            strike: Strike level
            tick_value: Dollar value per unit
            option_type: "call" or "put"
            distribution_fit: "empirical", "normal", "lognormal", "gamma", or "gev"

        Returns:
            Dictionary with burn analysis and fitted distribution parameters
        """
        from scipy import stats as scipy_stats

        historical_data = np.array(historical_data)

        # Calculate historical payoffs
        if option_type.lower() == "call":
            payoffs = np.maximum(historical_data - strike, 0) * tick_value
        else:
            payoffs = np.maximum(strike - historical_data, 0) * tick_value

        results = {
            "mean_payoff": np.mean(payoffs),
            "median_payoff": np.median(payoffs),
            "std_payoff": np.std(payoffs),
            "burn_rate": np.mean(payoffs > 0),
        }

        # Fit distribution
        if distribution_fit == "normal":
            mu, sigma = scipy_stats.norm.fit(historical_data)
            results["distribution_params"] = {"mu": mu, "sigma": sigma}
            results["distribution_type"] = "normal"

        elif distribution_fit == "lognormal":
            # Fit to positive values only
            positive_data = historical_data[historical_data > 0]
            if len(positive_data) > 0:
                shape, loc, scale = scipy_stats.lognorm.fit(positive_data, floc=0)
                results["distribution_params"] = {"shape": shape, "loc": loc, "scale": scale}
                results["distribution_type"] = "lognormal"

        elif distribution_fit == "gamma":
            shape, loc, scale = scipy_stats.gamma.fit(historical_data)
            results["distribution_params"] = {"shape": shape, "loc": loc, "scale": scale}
            results["distribution_type"] = "gamma"

        elif distribution_fit == "gev":
            # Generalized Extreme Value
            shape, loc, scale = scipy_stats.genextreme.fit(historical_data)
            results["distribution_params"] = {"shape": shape, "loc": loc, "scale": scale}
            results["distribution_type"] = "gev"

        else:
            results["distribution_type"] = "empirical"

        return results

    def time_series_forecast_valuation(
        self,
        historical_data: Union[List[float], np.ndarray],
        strike: float,
        tick_value: float,
        option_type: str = "call",
        forecast_periods: int = 30,
        num_simulations: int = 1000
    ) -> Dict[str, Any]:
        """
        Valuation using time series forecasting with uncertainty.

        Uses autoregressive model for forecasting.

        Args:
            historical_data: Historical index values (time series)
            strike: Strike level
            tick_value: Dollar value per unit
            option_type: "call" or "put"
            forecast_periods: Number of periods to forecast
            num_simulations: Number of simulation paths

        Returns:
            Dictionary with forecasted valuation
        """
        historical_data = np.array(historical_data)

        # Simple AR(1) model estimation
        # y_t = phi * y_{t-1} + epsilon
        y_t = historical_data[1:]
        y_t_1 = historical_data[:-1]

        # OLS estimation
        phi = np.sum(y_t * y_t_1) / np.sum(y_t_1 ** 2)
        residuals = y_t - phi * y_t_1
        sigma_epsilon = np.std(residuals)

        # Forecast with simulation
        simulated_paths = []
        last_value = historical_data[-1]

        for _ in range(num_simulations):
            path = [last_value]
            for _ in range(forecast_periods):
                next_value = phi * path[-1] + np.random.normal(0, sigma_epsilon)
                path.append(next_value)

            # Calculate cumulative index
            cumulative_index = np.mean(path[1:])  # Average over forecast period
            simulated_paths.append(cumulative_index)

        simulated_paths = np.array(simulated_paths)

        # Calculate payoffs
        if option_type.lower() == "call":
            payoffs = np.maximum(simulated_paths - strike, 0) * tick_value
        else:
            payoffs = np.maximum(strike - simulated_paths, 0) * tick_value

        return {
            "expected_payoff": np.mean(payoffs),
            "median_payoff": np.median(payoffs),
            "std_payoff": np.std(payoffs),
            "ar_coefficient": phi,
            "innovation_std": sigma_epsilon,
            "percentile_5": np.percentile(payoffs, 5),
            "percentile_95": np.percentile(payoffs, 95),
        }

    def weather_index_insurance_pricing(
        self,
        historical_data: Union[List[float], np.ndarray],
        trigger: float,
        exit: float,
        limit: float,
        attachment_probability: float = 0.20
    ) -> Dict[str, Any]:
        """
        Weather index insurance pricing.

        Prices parametric insurance based on weather indices.

        Args:
            historical_data: Historical index values
            trigger: Trigger point (coverage starts)
            exit: Exit point (maximum payout reached)
            limit: Maximum payout amount
            attachment_probability: Target probability of payout

        Returns:
            Dictionary with insurance pricing
        """
        historical_data = np.array(historical_data)

        # Calculate payouts under different scenarios
        payouts = []
        for value in historical_data:
            if value <= trigger:
                payout = 0
            elif value >= exit:
                payout = limit
            else:
                # Linear payout between trigger and exit
                payout = limit * (value - trigger) / (exit - trigger)

            payouts.append(max(0, min(payout, limit)))

        payouts = np.array(payouts)

        # Calculate premium components
        pure_premium = np.mean(payouts)
        std_premium = np.std(payouts)

        # Risk loading (using standard deviation principle)
        risk_loading = 0.3 * std_premium

        # Administrative loading (percentage)
        admin_loading = 0.15 * pure_premium

        # Total premium
        total_premium = pure_premium + risk_loading + admin_loading

        # Attachment probability (probability of payout > 0)
        attachment_prob = np.mean(payouts > 0)

        # Loss ratio (expected payout / premium)
        loss_ratio = pure_premium / total_premium if total_premium > 0 else 0

        return {
            "pure_premium": pure_premium,
            "risk_loading": risk_loading,
            "admin_loading": admin_loading,
            "total_premium": total_premium,
            "attachment_probability": attachment_prob,
            "average_payout_given_trigger": np.mean(payouts[payouts > 0]) if np.any(payouts > 0) else 0,
            "max_payout": limit,
            "loss_ratio": loss_ratio,
            "std_payout": std_premium,
        }

    def asian_option_pricing(
        self,
        historical_data: Union[List[float], np.ndarray],
        strike: float,
        tick_value: float,
        option_type: str = "call",
        averaging_periods: int = 30,
        num_simulations: int = 10000
    ) -> Dict[str, Any]:
        """
        Asian (average rate) option pricing for weather derivatives.

        Values options based on average weather index over a period.

        Args:
            historical_data: Historical data for parameter estimation
            strike: Strike level
            tick_value: Dollar value per unit
            option_type: "call" or "put"
            averaging_periods: Number of periods for averaging
            num_simulations: Number of Monte Carlo paths

        Returns:
            Dictionary with Asian option valuation
        """
        historical_data = np.array(historical_data)
        mean = np.mean(historical_data)
        std = np.std(historical_data)

        # Simulate paths and calculate average
        averaged_values = []

        for _ in range(num_simulations):
            # Generate path
            path = np.random.normal(mean, std, averaging_periods)
            # Calculate average
            avg_value = np.mean(path)
            averaged_values.append(avg_value)

        averaged_values = np.array(averaged_values)

        # Calculate payoffs based on average
        if option_type.lower() == "call":
            payoffs = np.maximum(averaged_values - strike, 0) * tick_value
        else:
            payoffs = np.maximum(strike - averaged_values, 0) * tick_value

        # Asian option typically has lower volatility than European
        european_std = std / np.sqrt(averaging_periods)

        return {
            "expected_payoff": np.mean(payoffs),
            "median_payoff": np.median(payoffs),
            "std_payoff": np.std(payoffs),
            "volatility_reduction": std / np.std(averaged_values) if np.std(averaged_values) > 0 else 1.0,
            "percentile_5": np.percentile(payoffs, 5),
            "percentile_95": np.percentile(payoffs, 95),
        }

    def spread_option_pricing(
        self,
        historical_data1: Union[List[float], np.ndarray],
        historical_data2: Union[List[float], np.ndarray],
        strike: float,
        tick_value: float,
        option_type: str = "call",
        num_simulations: int = 10000
    ) -> Dict[str, Any]:
        """
        Spread option pricing for weather derivatives.

        Values options on the difference between two weather indices.

        Args:
            historical_data1: Historical data for first index
            historical_data2: Historical data for second index
            strike: Strike on the spread
            tick_value: Dollar value per unit
            option_type: "call" or "put"
            num_simulations: Number of simulations

        Returns:
            Dictionary with spread option valuation
        """
        data1 = np.array(historical_data1)
        data2 = np.array(historical_data2)

        # Estimate parameters
        mean1, std1 = np.mean(data1), np.std(data1)
        mean2, std2 = np.mean(data2), np.std(data2)

        # Estimate correlation
        min_len = min(len(data1), len(data2))
        correlation = np.corrcoef(data1[:min_len], data2[:min_len])[0, 1]

        # Generate correlated samples
        mean_vec = [mean1, mean2]
        cov_matrix = [[std1**2, correlation * std1 * std2],
                     [correlation * std1 * std2, std2**2]]

        samples = np.random.multivariate_normal(mean_vec, cov_matrix, num_simulations)

        # Calculate spread
        spreads = samples[:, 0] - samples[:, 1]

        # Calculate payoffs
        if option_type.lower() == "call":
            payoffs = np.maximum(spreads - strike, 0) * tick_value
        else:
            payoffs = np.maximum(strike - spreads, 0) * tick_value

        return {
            "expected_payoff": np.mean(payoffs),
            "median_payoff": np.median(payoffs),
            "std_payoff": np.std(payoffs),
            "spread_mean": np.mean(spreads),
            "spread_std": np.std(spreads),
            "correlation": correlation,
            "percentile_5": np.percentile(payoffs, 5),
            "percentile_95": np.percentile(payoffs, 95),
        }

    def portfolio_optimization(
        self,
        derivatives: List[Dict[str, Any]],
        budget: float,
        risk_aversion: float = 0.5
    ) -> Dict[str, Any]:
        """
        Portfolio optimization for weather derivatives.

        Optimizes allocation across multiple derivatives using mean-variance framework.

        Args:
            derivatives: List of derivative specifications with expected returns and risks
            budget: Total budget for allocation
            risk_aversion: Risk aversion parameter (0 = risk-neutral, 1 = very risk-averse)

        Returns:
            Dictionary with optimal allocation
        """
        from scipy.optimize import minimize as scipy_minimize

        n = len(derivatives)

        # Extract expected returns and volatilities
        expected_returns = np.array([d.get("expected_return", 0) for d in derivatives])
        volatilities = np.array([d.get("volatility", 0) for d in derivatives])

        # Correlation matrix (simplified: use provided or assume independence)
        if "correlation_matrix" in derivatives[0]:
            corr_matrix = derivatives[0]["correlation_matrix"]
        else:
            corr_matrix = np.eye(n)

        # Covariance matrix
        cov_matrix = np.outer(volatilities, volatilities) * corr_matrix

        def portfolio_objective(weights):
            # Mean-variance utility: return - risk_aversion * variance
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
            return -(portfolio_return - risk_aversion * portfolio_variance)

        # Constraints
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},  # Weights sum to 1
        ]

        # Bounds (no short selling, no more than 100% in any asset)
        bounds = [(0, 1) for _ in range(n)]

        # Initial guess
        initial_weights = np.ones(n) / n

        # Optimize
        result = scipy_minimize(
            portfolio_objective,
            initial_weights,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints
        )

        optimal_weights = result.x
        optimal_allocation = optimal_weights * budget

        # Calculate portfolio metrics
        portfolio_return = np.dot(optimal_weights, expected_returns)
        portfolio_variance = np.dot(optimal_weights, np.dot(cov_matrix, optimal_weights))
        portfolio_std = np.sqrt(portfolio_variance)

        # Sharpe ratio (assuming zero risk-free rate)
        sharpe_ratio = portfolio_return / portfolio_std if portfolio_std > 0 else 0

        return {
            "optimal_weights": optimal_weights.tolist(),
            "optimal_allocation": optimal_allocation.tolist(),
            "expected_return": portfolio_return,
            "portfolio_std": portfolio_std,
            "sharpe_ratio": sharpe_ratio,
            "diversification_ratio": np.sum(optimal_weights * volatilities) / portfolio_std if portfolio_std > 0 else 1,
        }

    def stochastic_volatility_valuation(
        self,
        mean: float,
        vol_mean: float,
        vol_std: float,
        vol_mean_reversion: float,
        strike: float,
        tick_value: float,
        option_type: str = "call",
        time_periods: int = 30,
        num_simulations: int = 10000
    ) -> Dict[str, Any]:
        """
        Stochastic volatility model valuation (Heston-like).

        Models weather index with time-varying volatility.

        Args:
            mean: Mean of the index
            vol_mean: Mean volatility level
            vol_std: Volatility of volatility
            vol_mean_reversion: Mean reversion speed
            strike: Strike level
            tick_value: Dollar value per unit
            option_type: "call" or "put"
            time_periods: Number of time periods
            num_simulations: Number of simulation paths

        Returns:
            Dictionary with stochastic volatility valuation
        """
        dt = 1.0  # Time step

        simulated_finals = []

        for _ in range(num_simulations):
            value = mean
            vol = vol_mean

            for _ in range(time_periods):
                # Update volatility (CIR process)
                dvol = vol_mean_reversion * (vol_mean - vol) * dt + vol_std * np.sqrt(max(vol, 0)) * np.random.normal() * np.sqrt(dt)
                vol = max(vol + dvol, 0.01)  # Floor to avoid negative volatility

                # Update index value
                dvalue = 0 * dt + np.sqrt(max(vol, 0)) * np.random.normal() * np.sqrt(dt)
                value += dvalue

            simulated_finals.append(value)

        simulated_finals = np.array(simulated_finals)

        # Calculate payoffs
        if option_type.lower() == "call":
            payoffs = np.maximum(simulated_finals - strike, 0) * tick_value
        else:
            payoffs = np.maximum(strike - simulated_finals, 0) * tick_value

        return {
            "expected_payoff": np.mean(payoffs),
            "median_payoff": np.median(payoffs),
            "std_payoff": np.std(payoffs),
            "implied_volatility": np.std(simulated_finals),
            "percentile_5": np.percentile(payoffs, 5),
            "percentile_95": np.percentile(payoffs, 95),
        }

    def sensitivity_analysis(
        self,
        base_params: Dict[str, float],
        param_ranges: Dict[str, Tuple[float, float]],
        strike: float,
        tick_value: float,
        option_type: str = "call",
        num_points: int = 10
    ) -> Dict[str, Any]:
        """
        Comprehensive sensitivity analysis for derivative pricing.

        Args:
            base_params: Base parameter values (e.g., {"mean": 1000, "std": 100})
            param_ranges: Ranges for each parameter
            strike: Strike level
            tick_value: Dollar value per unit
            option_type: "call" or "put"
            num_points: Number of points to evaluate for each parameter

        Returns:
            Dictionary with sensitivity results
        """
        sensitivity_results = {}

        for param_name, (min_val, max_val) in param_ranges.items():
            param_values = np.linspace(min_val, max_val, num_points)
            payoffs = []

            for param_val in param_values:
                # Update parameter
                params = base_params.copy()
                params[param_name] = param_val

                # Run Monte Carlo with these parameters
                mc_result = self.monte_carlo_valuation(
                    mean=params.get("mean", 1000),
                    std=params.get("std", 100),
                    strike=strike,
                    tick_value=tick_value,
                    option_type=option_type,
                    num_simulations=1000
                )

                payoffs.append(mc_result["expected_payoff"])

            sensitivity_results[param_name] = {
                "param_values": param_values.tolist(),
                "payoffs": payoffs,
                "sensitivity": (payoffs[-1] - payoffs[0]) / (param_values[-1] - param_values[0])
            }

        return sensitivity_results
