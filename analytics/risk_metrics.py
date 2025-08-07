"""
Advanced risk metrics calculations including VaR, drawdown, and Sharpe ratio
"""
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import logging
import sys
from config import Config

class RiskMetricsCalculator:
    """
    Comprehensive risk metrics calculation engine
    Features: VaR, CVaR, drawdown analysis, volatility modeling, correlation analysis
    """
    
    def __init__(self, risk_free_rate: float = None):
        self.risk_free_rate = risk_free_rate or Config.DEFAULT_RISK_FREE_RATE
        self.logger = logging.getLogger(__name__)
    
    def calculate_comprehensive_risk_metrics(
        self, 
        portfolio_data: Dict[str, Any],
        benchmark_data: Optional[pd.Series] = None,
        confidence_levels: List[float] = None
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive risk metrics for portfolio
        
        Args:
            portfolio_data: Portfolio data with equity history
            benchmark_data: Benchmark returns for comparison
            confidence_levels: VaR confidence levels (default: [0.95, 0.99])
            
        Returns:
            Comprehensive risk metrics dictionary
        """
        
        try:
            if confidence_levels is None:
                confidence_levels = [0.95, 0.99]
            
            # Extract equity history and calculate returns
            equity_history = portfolio_data.get('equity_history', [])
            
            if len(equity_history) < 2:
                return self._get_default_risk_metrics()
            
            # Convert to DataFrame and calculate returns
            df = pd.DataFrame(equity_history)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            returns = df['total_value'].pct_change().dropna()
            
            if len(returns) < 2:
                return self._get_default_risk_metrics()
            
            # Basic risk metrics
            basic_metrics = self._calculate_basic_risk_metrics(returns)
            
            # VaR and CVaR calculations
            var_metrics = self._calculate_var_metrics(returns, confidence_levels)
            
            # Drawdown analysis
            drawdown_metrics = self._calculate_drawdown_metrics(df['total_value'])
            
            # Volatility analysis
            volatility_metrics = self._calculate_volatility_metrics(returns)
            
            # Performance ratios
            performance_ratios = self._calculate_performance_ratios(returns, df)
            
            # Benchmark comparison (if available)
            benchmark_metrics = {}
            if benchmark_data is not None:
                benchmark_metrics = self._calculate_benchmark_metrics(returns, benchmark_data)
            
            # Combine all metrics
            return {
                **basic_metrics,
                **var_metrics,
                **drawdown_metrics,
                **volatility_metrics,
                **performance_ratios,
                **benchmark_metrics,
                'calculation_timestamp': datetime.utcnow().isoformat(),
                'sample_size': len(returns),
                'time_period_days': (df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]).days
            }
            
        except Exception as e:
            self.logger.error(f"Risk metrics calculation failed: {e}")
            return self._get_default_risk_metrics()
    
    def _calculate_basic_risk_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate basic risk statistics"""
        
        # Annualization factor (assuming daily returns)
        annualization_factor = np.sqrt(252)
        
        return {
            'volatility_daily': returns.std(),
            'volatility_annualized': returns.std() * annualization_factor,
            'skewness': returns.skew(),
            'kurtosis': returns.kurtosis(),
            'mean_return_daily': returns.mean(),
            'mean_return_annualized': returns.mean() * 252,
            'min_return': returns.min(),
            'max_return': returns.max(),
            'positive_days_pct': (returns > 0).mean() * 100,
            'negative_days_pct': (returns < 0).mean() * 100
        }
    
    def _calculate_var_metrics(self, returns: pd.Series, confidence_levels: List[float]) -> Dict[str, float]:
        """Calculate Value at Risk and Conditional VaR metrics"""
        
        var_metrics = {}
        
        for confidence in confidence_levels:
            alpha = 1 - confidence
            conf_str = f"{int(confidence*100)}"
            
            # Historical VaR
            historical_var = returns.quantile(alpha)
            var_metrics[f'var_{conf_str}_historical'] = historical_var
            
            # Parametric VaR (assuming normal distribution)
            parametric_var = norm.ppf(alpha, returns.mean(), returns.std())
            var_metrics[f'var_{conf_str}_parametric'] = parametric_var
            
            # Conditional VaR (Expected Shortfall)
            conditional_returns = returns[returns <= historical_var]
            if len(conditional_returns) > 0:
                cvar = conditional_returns.mean()
            else:
                cvar = historical_var
            
            var_metrics[f'cvar_{conf_str}'] = cvar
            var_metrics[f'expected_shortfall_{conf_str}'] = cvar
        
        return var_metrics
    
    def _calculate_drawdown_metrics(self, equity_curve: pd.Series) -> Dict[str, float]:
        """Calculate comprehensive drawdown analysis"""
        
        # Calculate drawdowns
        peak = equity_curve.expanding().max()
        drawdown = (equity_curve - peak) / peak
        
        # Maximum drawdown
        max_drawdown = drawdown.min()
        
        # Find drawdown periods
        is_drawdown = drawdown < 0
        drawdown_periods = []
        
        if is_drawdown.any():
            # Find start and end of drawdown periods
            drawdown_starts = is_drawdown & ~is_drawdown.shift(1, fill_value=False)
            drawdown_ends = ~is_drawdown & is_drawdown.shift(1, fill_value=False)
            
            starts = drawdown_starts[drawdown_starts].index.tolist()
            ends = drawdown_ends[drawdown_ends].index.tolist()
            
            # Handle case where portfolio ends in drawdown
            if len(starts) > len(ends):
                ends.append(len(equity_curve) - 1)
            
            for start, end in zip(starts, ends):
                duration = end - start + 1
                depth = drawdown.iloc[start:end+1].min()
                drawdown_periods.append({'duration': duration, 'depth': depth})
        
        # Calculate statistics
        avg_drawdown_duration = np.mean([dd['duration'] for dd in drawdown_periods]) if drawdown_periods else 0
        max_drawdown_duration = max([dd['duration'] for dd in drawdown_periods]) if drawdown_periods else 0
        
        # Recovery analysis
        recovery_factor = abs(max_drawdown) / equity_curve.iloc[-1] if max_drawdown < 0 else 0
        
        return {
            'max_drawdown': max_drawdown,
            'avg_drawdown_duration': avg_drawdown_duration,
            'max_drawdown_duration': max_drawdown_duration,
            'drawdown_periods_count': len(drawdown_periods),
            'recovery_factor': recovery_factor,
            'current_drawdown': drawdown.iloc[-1],
            'time_underwater_pct': (drawdown < 0).mean() * 100
        }
    
    def _calculate_volatility_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate advanced volatility metrics"""
        
        # Downside volatility
        downside_returns = returns[returns < 0]
        downside_volatility = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        
        # Upside volatility
        upside_returns = returns[returns > 0]
        upside_volatility = upside_returns.std() * np.sqrt(252) if len(upside_returns) > 0 else 0
        
        # Rolling volatility analysis
        if len(returns) >= 30:
            rolling_vol_30 = returns.rolling(window=30).std()
            vol_of_vol = rolling_vol_30.std() * np.sqrt(252)
            max_30d_vol = rolling_vol_30.max() * np.sqrt(252)
            min_30d_vol = rolling_vol_30.min() * np.sqrt(252)
        else:
            vol_of_vol = max_30d_vol = min_30d_vol = 0
        
        # Semi-variance
        mean_return = returns.mean()
        semi_variance = ((returns[returns < mean_return] - mean_return) ** 2).mean()
        semi_deviation = np.sqrt(semi_variance) * np.sqrt(252) if semi_variance > 0 else 0
        
        return {
            'downside_volatility': downside_volatility,
            'upside_volatility': upside_volatility,
            'volatility_of_volatility': vol_of_vol,
            'max_30d_volatility': max_30d_vol,
            'min_30d_volatility': min_30d_vol,
            'semi_deviation': semi_deviation,
            'volatility_ratio': upside_volatility / downside_volatility if downside_volatility > 0 else 0
        }
    
    def _calculate_performance_ratios(self, returns: pd.Series, equity_df: pd.DataFrame) -> Dict[str, float]:
        """Calculate risk-adjusted performance ratios"""
        
        # Annualized metrics
        annualized_return = returns.mean() * 252
        annualized_volatility = returns.std() * np.sqrt(252)
        
        # Sharpe ratio
        sharpe_ratio = (annualized_return - self.risk_free_rate) / annualized_volatility if annualized_volatility > 0 else 0
        
        # Sortino ratio (using downside deviation)
        downside_returns = returns[returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else annualized_volatility
        sortino_ratio = (annualized_return - self.risk_free_rate) / downside_deviation if downside_deviation > 0 else 0
        
        # Calmar ratio (return vs max drawdown)
        max_drawdown = self._calculate_drawdown_metrics(equity_df['total_value'])['max_drawdown']
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown < 0 else 0
        
        # Information ratio (excess return vs tracking error)
        # This would require benchmark data, using volatility as proxy
        information_ratio = annualized_return / annualized_volatility if annualized_volatility > 0 else 0
        
        # Omega ratio (simplified version)
        threshold = 0  # Risk-free rate threshold
        gains = returns[returns > threshold].sum()
        losses = abs(returns[returns <= threshold].sum())
        omega_ratio = gains / losses if losses > 0 else float('inf') if gains > 0 else 0
        
        # Gain-to-Pain ratio
        positive_returns = returns[returns > 0].sum()
        negative_returns = abs(returns[returns < 0].sum())
        gain_to_pain = positive_returns / negative_returns if negative_returns > 0 else float('inf') if positive_returns > 0 else 0
        
        # Sterling ratio
        average_drawdown = returns[returns < 0].mean() if len(returns[returns < 0]) > 0 else -0.01
        sterling_ratio = annualized_return / abs(average_drawdown) if average_drawdown < 0 else 0
        
        return {
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'information_ratio': information_ratio,
            'omega_ratio': min(omega_ratio, 10),  # Cap for display purposes
            'gain_to_pain_ratio': min(gain_to_pain, 10),
            'sterling_ratio': sterling_ratio,
            'treynor_ratio': 0,  # Would need beta calculation
            'modigliani_ratio': 0  # Would need benchmark comparison
        }
    
    def _calculate_benchmark_metrics(self, portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> Dict[str, float]:
        """Calculate benchmark comparison metrics"""
        
        try:
            # Align returns
            min_length = min(len(portfolio_returns), len(benchmark_returns))
            port_returns = portfolio_returns.iloc[-min_length:]
            bench_returns = benchmark_returns.iloc[-min_length:]
            
            if len(port_returns) < 2:
                return {}
            
            # Beta calculation
            covariance = np.cov(port_returns, bench_returns)[0, 1]
            benchmark_variance = np.var(bench_returns)
            beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
            
            # Alpha calculation
            benchmark_return = bench_returns.mean() * 252
            portfolio_return = port_returns.mean() * 252
            expected_return = self.risk_free_rate + beta * (benchmark_return - self.risk_free_rate)
            alpha = portfolio_return - expected_return
            
            # Tracking error
            excess_returns = port_returns - bench_returns
            tracking_error = excess_returns.std() * np.sqrt(252)
            
            # Information ratio
            information_ratio = excess_returns.mean() * 252 / tracking_error if tracking_error > 0 else 0
            
            # Correlation
            correlation = np.corrcoef(port_returns, bench_returns)[0, 1]
            
            # Up/Down capture ratios
            up_market = bench_returns > 0
            down_market = bench_returns < 0
            
            if up_market.any():
                up_capture = (port_returns[up_market].mean() / bench_returns[up_market].mean()) if bench_returns[up_market].mean() != 0 else 0
            else:
                up_capture = 0
            
            if down_market.any():
                down_capture = (port_returns[down_market].mean() / bench_returns[down_market].mean()) if bench_returns[down_market].mean() != 0 else 0
            else:
                down_capture = 0
            
            # Treynor ratio
            treynor_ratio = (portfolio_return - self.risk_free_rate) / beta if beta != 0 else 0
            
            return {
                'beta': beta,
                'alpha': alpha,
                'tracking_error': tracking_error,
                'information_ratio': information_ratio,
                'correlation': correlation,
                'up_capture_ratio': up_capture,
                'down_capture_ratio': down_capture,
                'treynor_ratio': treynor_ratio,
                'active_return': portfolio_return - benchmark_return
            }
            
        except Exception as e:
            self.logger.error(f"Benchmark metrics calculation failed: {e}")
            return {}
    
    def calculate_portfolio_var(
        self, 
        positions: Dict[str, Any], 
        correlation_matrix: Optional[pd.DataFrame] = None,
        confidence_level: float = 0.95,
        time_horizon: int = 1
    ) -> Dict[str, float]:
        """
        Calculate portfolio Value at Risk using position data
        
        Args:
            positions: Portfolio positions
            correlation_matrix: Asset correlation matrix
            confidence_level: VaR confidence level
            time_horizon: Time horizon in days
            
        Returns:
            VaR metrics
        """
        
        try:
            if not positions:
                return {'portfolio_var': 0.0, 'marginal_var': {}, 'component_var': {}}
            
            # Extract position values and weights
            total_value = sum(pos.get('market_value', 0) for pos in positions.values())
            
            if total_value <= 0:
                return {'portfolio_var': 0.0, 'marginal_var': {}, 'component_var': {}}
            
            weights = np.array([pos.get('market_value', 0) / total_value for pos in positions.values()])
            tickers = list(positions.keys())
            
            # Simplified volatility assumptions (would use historical data in practice)
            volatilities = np.array([0.02] * len(tickers))  # 2% daily volatility assumption
            
            # Create correlation matrix if not provided
            if correlation_matrix is None:
                # Use simplified correlation structure
                n_assets = len(tickers)
                correlation_matrix = np.eye(n_assets) * 0.6 + np.ones((n_assets, n_assets)) * 0.4
            
            # Calculate portfolio volatility
            cov_matrix = np.outer(volatilities, volatilities) * correlation_matrix
            portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
            portfolio_volatility = np.sqrt(portfolio_variance)
            
            # Adjust for time horizon
            portfolio_volatility_adjusted = portfolio_volatility * np.sqrt(time_horizon)
            
            # Calculate VaR
            alpha = 1 - confidence_level
            portfolio_var = norm.ppf(alpha) * portfolio_volatility_adjusted * total_value
            
            # Calculate marginal VaR
            marginal_var = {}
            component_var = {}
            
            for i, ticker in enumerate(tickers):
                # Marginal contribution to risk
                marginal_contribution = np.dot(cov_matrix[i], weights) / portfolio_volatility
                marginal_var[ticker] = marginal_contribution * norm.ppf(alpha) * np.sqrt(time_horizon)
                
                # Component VaR
                component_var[ticker] = weights[i] * marginal_var[ticker] * total_value
            
            return {
                'portfolio_var': abs(portfolio_var),
                'portfolio_volatility': portfolio_volatility_adjusted,
                'marginal_var': marginal_var,
                'component_var': component_var,
                'diversification_benefit': sum(component_var.values()) - abs(portfolio_var)
            }
            
        except Exception as e:
            self.logger.error(f"Portfolio VaR calculation failed: {e}")
            return {'portfolio_var': 0.0, 'error': str(e)}
    
    def calculate_stress_test_scenarios(self, portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate portfolio performance under stress scenarios"""
        
        try:
            positions = portfolio_data.get('positions', {})
            total_value = sum(pos.get('market_value', 0) for pos in positions.values())
            
            if total_value <= 0:
                return {}
            
            # Define stress scenarios
            scenarios = {
                'market_crash_2008': {'equity': -0.40, 'volatility': 3.0},
                'covid_crash_2020': {'equity': -0.35, 'volatility': 2.5},
                'interest_rate_shock': {'equity': -0.15, 'volatility': 1.5},
                'sector_rotation': {'equity': -0.10, 'volatility': 1.2},
                'currency_devaluation': {'equity': -0.08, 'volatility': 1.3}
            }
            
            scenario_results = {}
            
            for scenario_name, scenario_params in scenarios.items():
                equity_shock = scenario_params['equity']
                vol_multiplier = scenario_params['volatility']
                
                # Calculate scenario impact
                scenario_loss = total_value * equity_shock
                scenario_volatility = 0.02 * vol_multiplier  # Base 2% daily vol
                
                # VaR under stress
                stress_var_95 = norm.ppf(0.05) * scenario_volatility * total_value
                stress_var_99 = norm.ppf(0.01) * scenario_volatility * total_value
                
                scenario_results[scenario_name] = {
                    'immediate_loss': scenario_loss,
                    'loss_percentage': equity_shock * 100,
                    'stress_var_95': stress_var_95,
                    'stress_var_99': stress_var_99,
                    'volatility_multiplier': vol_multiplier
                }
            
            return scenario_results
            
        except Exception as e:
            self.logger.error(f"Stress test calculation failed: {e}")
            return {}
    
    def _get_default_risk_metrics(self) -> Dict[str, float]:
        """Return default risk metrics when calculation fails"""
        
        return {
            'volatility_daily': 0.0,
            'volatility_annualized': 0.0,
            'var_95_historical': 0.0,
            'var_99_historical': 0.0,
            'cvar_95': 0.0,
            'cvar_99': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'calmar_ratio': 0.0,
            'beta': 0.0,
            'alpha': 0.0,
            'tracking_error': 0.0,
            'correlation': 0.0,
            'skewness': 0.0,
            'kurtosis': 0.0
        }
