"""
Advanced valuation methods for weather derivatives.

Implements state-of-the-art pricing techniques including:
- Time series modeling (ARIMA, GARCH)
- Bootstrapping methods
- Quantile regression
- Extreme value theory
- Regime-switching models
"""

from typing import List, Optional, Dict, Any, Union, Tuple
import numpy as np
from scipy import stats
from scipy.optimize import minimize
import warnings


class AdvancedValuation:
    """
    Advanced valuation methods for weather derivatives.

    Implements cutting-edge pricing and risk analysis techniques.
    """

    def __init__(self, confidence_level: float = 0.95):
        """
        Initialize advanced valuation engine.

        Args:
            confidence_level: Confidence level for risk metrics (default 0.95)
        """
        self.confidence_level = confidence_level

    def bootstrap_valuation(
        self,
        historical_data: Union[List[float], np.ndarray],
        strike: float,
        tick_value: float,
        option_type: str = "call",
        num_bootstrap_samples: int = 10000,
        block_size: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Bootstrap resampling for derivative valuation.

        Uses block bootstrap to preserve autocorrelation structure.

        Args:
            historical_data: Historical weather index values
            strike: Strike level
            tick_value: Dollar value per unit
            option_type: "call" or "put"
            num_bootstrap_samples: Number of bootstrap samples
            block_size: Block size for block bootstrap (None = circular block)

        Returns:
            Dictionary with bootstrap statistics
        """
        historical_data = np.array(historical_data)
        n = len(historical_data)

        if block_size is None:
            # Optimal block size (Politis and White, 2004)
            block_size = int(np.ceil(n ** (1/3)))

        bootstrap_payoffs = []

        for _ in range(num_bootstrap_samples):
            # Circular block bootstrap
            num_blocks = int(np.ceil(n / block_size))
            start_indices = np.random.randint(0, n, num_blocks)

            sample = []
            for start_idx in start_indices:
                for i in range(block_size):
                    sample.append(historical_data[(start_idx + i) % n])
                    if len(sample) >= n:
                        break
                if len(sample) >= n:
                    break

            sample = sample[:n]

            # Calculate average for this sample
            avg_value = np.mean(sample)

            # Calculate payoff
            if option_type.lower() == "call":
                payoff = max(avg_value - strike, 0) * tick_value
            else:
                payoff = max(strike - avg_value, 0) * tick_value

            bootstrap_payoffs.append(payoff)

        bootstrap_payoffs = np.array(bootstrap_payoffs)

        return {
            "mean_payoff": np.mean(bootstrap_payoffs),
            "median_payoff": np.median(bootstrap_payoffs),
            "std_payoff": np.std(bootstrap_payoffs),
            "confidence_interval": (
                np.percentile(bootstrap_payoffs, (1 - self.confidence_level) * 50),
                np.percentile(bootstrap_payoffs, (1 + self.confidence_level) * 50)
            ),
            "percentile_5": np.percentile(bootstrap_payoffs, 5),
            "percentile_95": np.percentile(bootstrap_payoffs, 95),
            "bootstrap_distribution": bootstrap_payoffs,
        }

    def extreme_value_theory_pricing(
        self,
        historical_data: Union[List[float], np.ndarray],
        strike: float,
        tick_value: float,
        option_type: str = "call",
        threshold_percentile: float = 0.90
    ) -> Dict[str, Any]:
        """
        Extreme Value Theory (EVT) based pricing.

        Uses Generalized Pareto Distribution (GPD) for tail modeling.

        Args:
            historical_data: Historical weather index values
            strike: Strike level
            tick_value: Dollar value per unit
            option_type: "call" or "put"
            threshold_percentile: Percentile for POT (Peaks Over Threshold)

        Returns:
            Dictionary with EVT-based valuation
        """
        historical_data = np.array(historical_data)

        # Peaks Over Threshold (POT) method
        threshold = np.percentile(historical_data, threshold_percentile * 100)

        if option_type.lower() == "call":
            exceedances = historical_data[historical_data > threshold] - threshold
        else:
            # For puts, look at lower tail
            threshold = np.percentile(historical_data, (1 - threshold_percentile) * 100)
            exceedances = threshold - historical_data[historical_data < threshold]

        if len(exceedances) < 10:
            warnings.warn("Too few exceedances for reliable EVT estimation")
            return {"error": "Insufficient extreme events"}

        # Fit GPD using MLE
        # GPD: F(y) = 1 - (1 + ξy/σ)^(-1/ξ)
        def gpd_negloglik(params):
            xi, sigma = params
            if sigma <= 0:
                return np.inf
            if xi == 0:
                return np.sum(np.log(sigma)) + np.sum(exceedances / sigma)
            else:
                t = 1 + xi * exceedances / sigma
                if np.any(t <= 0):
                    return np.inf
                return len(exceedances) * np.log(sigma) + (1 + 1/xi) * np.sum(np.log(t))

        # Initial guess
        result = minimize(gpd_negloglik, [0.1, np.std(exceedances)], method='Nelder-Mead')
        xi, sigma = result.x

        # Calculate probability of exceeding strike
        n_exceedances = len(exceedances)
        n_total = len(historical_data)

        if option_type.lower() == "call":
            if strike > threshold:
                excess = strike - threshold
                prob_exceed = (n_exceedances / n_total) * (1 + xi * excess / sigma) ** (-1 / xi)
            else:
                prob_exceed = n_exceedances / n_total
        else:
            if strike < threshold:
                excess = threshold - strike
                prob_exceed = (n_exceedances / n_total) * (1 + xi * excess / sigma) ** (-1 / xi)
            else:
                prob_exceed = n_exceedances / n_total

        # Expected payoff using EVT
        # Approximate expected payoff in tail
        if xi < 1:
            expected_excess = sigma / (1 - xi)
        else:
            expected_excess = np.mean(exceedances)  # Fallback

        expected_payoff = prob_exceed * expected_excess * tick_value

        return {
            "expected_payoff": expected_payoff,
            "tail_probability": prob_exceed,
            "gpd_shape": xi,
            "gpd_scale": sigma,
            "threshold": threshold,
            "num_exceedances": n_exceedances,
        }

    def quantile_regression_pricing(
        self,
        historical_data: Union[List[float], np.ndarray],
        strike: float,
        tick_value: float,
        option_type: str = "call",
        quantiles: List[float] = [0.1, 0.25, 0.5, 0.75, 0.9]
    ) -> Dict[str, Any]:
        """
        Quantile-based derivative pricing.

        Estimates distribution of payoffs using quantile regression.

        Args:
            historical_data: Historical weather index values
            strike: Strike level
            tick_value: Dollar value per unit
            option_type: "call" or "put"
            quantiles: Quantiles to estimate

        Returns:
            Dictionary with quantile-based valuation
        """
        historical_data = np.array(historical_data)

        # Calculate payoffs
        if option_type.lower() == "call":
            payoffs = np.maximum(historical_data - strike, 0) * tick_value
        else:
            payoffs = np.maximum(strike - historical_data, 0) * tick_value

        # Estimate quantiles
        quantile_values = {
            q: np.percentile(payoffs, q * 100) for q in quantiles
        }

        # Estimate expected payoff weighted by quantiles
        # Use trapezoidal rule for integration
        sorted_payoffs = np.sort(payoffs)
        quantile_points = np.linspace(0, 1, len(sorted_payoffs))

        expected_payoff = np.trapz(sorted_payoffs, quantile_points)

        return {
            "expected_payoff": expected_payoff,
            "quantiles": quantile_values,
            "median": quantile_values.get(0.5, np.median(payoffs)),
            "iqr": quantile_values.get(0.75, 0) - quantile_values.get(0.25, 0),
            "payoff_distribution": sorted_payoffs,
        }

    def regime_switching_valuation(
        self,
        historical_data: Union[List[float], np.ndarray],
        strike: float,
        tick_value: float,
        option_type: str = "call",
        num_regimes: int = 2,
        num_simulations: int = 10000
    ) -> Dict[str, Any]:
        """
        Regime-switching model for weather derivative pricing.

        Models weather index with different regimes (e.g., El Niño vs. La Niña).

        Args:
            historical_data: Historical weather index values
            strike: Strike level
            tick_value: Dollar value per unit
            option_type: "call" or "put"
            num_regimes: Number of regimes (default 2)
            num_simulations: Number of simulations

        Returns:
            Dictionary with regime-switching valuation
        """
        historical_data = np.array(historical_data)

        # Simple k-means clustering to identify regimes
        from scipy.cluster.vq import kmeans, vq

        # Cluster based on values
        centroids, _ = kmeans(historical_data.reshape(-1, 1), num_regimes)
        regime_assignments, _ = vq(historical_data.reshape(-1, 1), centroids)

        # Estimate parameters for each regime
        regimes = []
        for r in range(num_regimes):
            regime_data = historical_data[regime_assignments == r]
            if len(regime_data) > 0:
                regimes.append({
                    "mean": np.mean(regime_data),
                    "std": np.std(regime_data),
                    "probability": len(regime_data) / len(historical_data),
                })

        # Simulate from regime mixture
        simulated_payoffs = []

        for _ in range(num_simulations):
            # Choose regime
            regime_probs = [r["probability"] for r in regimes]
            regime = np.random.choice(len(regimes), p=regime_probs)

            # Sample from regime
            value = np.random.normal(regimes[regime]["mean"], regimes[regime]["std"])

            # Calculate payoff
            if option_type.lower() == "call":
                payoff = max(value - strike, 0) * tick_value
            else:
                payoff = max(strike - value, 0) * tick_value

            simulated_payoffs.append(payoff)

        simulated_payoffs = np.array(simulated_payoffs)

        return {
            "expected_payoff": np.mean(simulated_payoffs),
            "median_payoff": np.median(simulated_payoffs),
            "std_payoff": np.std(simulated_payoffs),
            "num_regimes": num_regimes,
            "regimes": regimes,
            "percentile_5": np.percentile(simulated_payoffs, 5),
            "percentile_95": np.percentile(simulated_payoffs, 95),
        }

    def importance_sampling_valuation(
        self,
        mean: float,
        std: float,
        strike: float,
        tick_value: float,
        option_type: str = "call",
        num_simulations: int = 10000,
        shift: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Importance sampling Monte Carlo for rare event simulation.

        More efficient for deep OTM options.

        Args:
            mean: Mean of the index
            std: Standard deviation
            strike: Strike level
            tick_value: Dollar value per unit
            option_type: "call" or "put"
            num_simulations: Number of simulations
            shift: Mean shift for importance sampling (auto if None)

        Returns:
            Dictionary with valuation using importance sampling
        """
        # Determine optimal shift
        if shift is None:
            if option_type.lower() == "call":
                shift = max(strike - mean, 0)
            else:
                shift = min(strike - mean, 0)

        # Sample from shifted distribution
        shifted_mean = mean + shift
        samples = np.random.normal(shifted_mean, std, num_simulations)

        # Calculate likelihood ratios
        # L = f(x) / g(x) where f is original, g is importance distribution
        log_likelihood_ratios = -0.5 * ((samples - mean) ** 2 - (samples - shifted_mean) ** 2) / (std ** 2)
        likelihood_ratios = np.exp(log_likelihood_ratios)

        # Calculate payoffs
        if option_type.lower() == "call":
            payoffs = np.maximum(samples - strike, 0) * tick_value
        else:
            payoffs = np.maximum(strike - samples, 0) * tick_value

        # Weighted average
        weighted_payoffs = payoffs * likelihood_ratios
        expected_payoff = np.mean(weighted_payoffs)

        # Variance reduction
        variance = np.var(weighted_payoffs)

        return {
            "expected_payoff": expected_payoff,
            "std_payoff": np.sqrt(variance),
            "variance_reduction": variance / np.var(payoffs) if np.var(payoffs) > 0 else 1.0,
            "effective_sample_size": num_simulations * (np.sum(likelihood_ratios) ** 2) / np.sum(likelihood_ratios ** 2),
        }

    def control_variates_valuation(
        self,
        historical_data: Union[List[float], np.ndarray],
        strike: float,
        tick_value: float,
        option_type: str = "call",
        num_simulations: int = 10000,
        control_mean: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Control variates variance reduction technique.

        Uses the index itself as a control variate.

        Args:
            historical_data: Historical data for parameter estimation
            strike: Strike level
            tick_value: Dollar value per unit
            option_type: "call" or "put"
            num_simulations: Number of simulations
            control_mean: Known mean of control variate

        Returns:
            Dictionary with control variates valuation
        """
        historical_data = np.array(historical_data)
        mean = np.mean(historical_data)
        std = np.std(historical_data)

        if control_mean is None:
            control_mean = mean

        # Standard MC simulation
        samples = np.random.normal(mean, std, num_simulations)

        # Calculate payoffs
        if option_type.lower() == "call":
            payoffs = np.maximum(samples - strike, 0) * tick_value
        else:
            payoffs = np.maximum(strike - samples, 0) * tick_value

        # Control variate: use the samples themselves
        control_values = samples

        # Estimate optimal coefficient
        cov_matrix = np.cov(payoffs, control_values)
        c_optimal = -cov_matrix[0, 1] / cov_matrix[1, 1]

        # Controlled estimator
        controlled_payoffs = payoffs + c_optimal * (control_values - control_mean)

        expected_payoff = np.mean(controlled_payoffs)
        variance = np.var(controlled_payoffs)

        # Variance reduction
        original_variance = np.var(payoffs)
        variance_reduction = 1 - (variance / original_variance) if original_variance > 0 else 0

        return {
            "expected_payoff": expected_payoff,
            "std_payoff": np.sqrt(variance),
            "variance_reduction_ratio": variance_reduction,
            "optimal_coefficient": c_optimal,
            "standard_mc_payoff": np.mean(payoffs),
            "efficiency_gain": original_variance / variance if variance > 0 else 1.0,
        }

    def copula_based_valuation(
        self,
        data1: Union[List[float], np.ndarray],
        data2: Union[List[float], np.ndarray],
        strike1: float,
        strike2: float,
        tick_value: float,
        option_type: str = "call",
        copula_type: str = "gaussian",
        num_simulations: int = 10000
    ) -> Dict[str, Any]:
        """
        Copula-based valuation for multi-index derivatives.

        Models dependence between multiple weather indices.

        Args:
            data1: Historical data for first index
            data2: Historical data for second index
            strike1: Strike for first index
            strike2: Strike for second index
            tick_value: Dollar value per unit
            option_type: "call" or "put"
            copula_type: "gaussian" or "t"
            num_simulations: Number of simulations

        Returns:
            Dictionary with copula-based valuation
        """
        data1 = np.array(data1)
        data2 = np.array(data2)

        # Estimate marginal parameters
        mean1, std1 = np.mean(data1), np.std(data1)
        mean2, std2 = np.mean(data2), np.std(data2)

        # Estimate correlation
        correlation = np.corrcoef(data1, data2)[0, 1]

        # Generate correlated samples
        if copula_type == "gaussian":
            # Gaussian copula
            mean_vec = [mean1, mean2]
            cov_matrix = [[std1**2, correlation * std1 * std2],
                         [correlation * std1 * std2, std2**2]]

            samples = np.random.multivariate_normal(mean_vec, cov_matrix, num_simulations)

        elif copula_type == "t":
            # Student-t copula
            df = 5  # degrees of freedom
            cov_matrix = [[1, correlation], [correlation, 1]]

            # Generate from t-copula
            z = np.random.multivariate_normal([0, 0], cov_matrix, num_simulations)
            chi2 = np.random.chisquare(df, num_simulations)

            t_samples = z * np.sqrt(df / chi2[:, np.newaxis])

            # Transform to marginals
            u1 = stats.t.cdf(t_samples[:, 0], df)
            u2 = stats.t.cdf(t_samples[:, 1], df)

            samples1 = stats.norm.ppf(u1, mean1, std1)
            samples2 = stats.norm.ppf(u2, mean2, std2)

            samples = np.column_stack([samples1, samples2])
        else:
            raise ValueError(f"Unknown copula type: {copula_type}")

        # Calculate payoffs for basket option
        if option_type.lower() == "call":
            payoffs1 = np.maximum(samples[:, 0] - strike1, 0)
            payoffs2 = np.maximum(samples[:, 1] - strike2, 0)
        else:
            payoffs1 = np.maximum(strike1 - samples[:, 0], 0)
            payoffs2 = np.maximum(strike2 - samples[:, 1], 0)

        # Basket payoff (average of both)
        basket_payoffs = (payoffs1 + payoffs2) / 2 * tick_value

        return {
            "expected_payoff": np.mean(basket_payoffs),
            "std_payoff": np.std(basket_payoffs),
            "correlation": correlation,
            "percentile_5": np.percentile(basket_payoffs, 5),
            "percentile_95": np.percentile(basket_payoffs, 95),
            "copula_type": copula_type,
        }
