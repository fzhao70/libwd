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
