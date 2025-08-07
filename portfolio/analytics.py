"""
Portfolio analytics and Modern Portfolio Theory optimization
"""
import numpy as np
import pandas as pd
from scipy import optimize
from scipy.stats import norm
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import logging
import sys

class RiskProfile(Enum):
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
    SPECULATIVE = "speculative"

@dataclass
class AssetAllocation:
    """Asset allocation configuration"""
    equity_pct: float = 0.60
    bond_pct: float = 0.30
    commodity_pct: float = 0.05
    cash_pct: float = 0.05
    alternatives_pct: float = 0.00
    
    def validate(self) -> bool:
        total = sum([
            self.equity_pct, self.bond_pct, self.commodity_pct, 
            self.cash_pct, self.alternatives_pct
        ])
        return abs(total - 1.0) <= 0.001

@dataclass 
class RiskConstraints:
    """Portfolio risk constraints"""
    max_position_weight: float = 0.10
    max_sector_weight: float = 0.25
    target_volatility: Optional[float] = None
    max_drawdown: Optional[float] = None
    var_limit: Optional[float] = None
    tracking_error_limit: Optional[float] = None

class PortfolioAnalytics:
    """Advanced portfolio analytics and risk calculations"""
    
    def __init__(self, risk_free_rate: float = 0.07):
        self.risk_free_rate = risk_free_rate
        self.logger = logging.getLogger(__name__)
    
    def calculate_comprehensive_metrics(self, 
        equity_history: List[Dict], 
        positions: Dict[str, Any],
        balances: Dict[str, float]
    ) -> Dict[str, Any]:
        """Calculate comprehensive portfolio performance and risk metrics"""
        
        try:
            if len(equity_history) < 2:
                return self._get_default_metrics()
            
            # Convert to DataFrame for analysis
            df = pd.DataFrame(equity_history)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Calculate returns
            returns = df['total_value'].pct_change().dropna()
            
            if len(returns) < 2:
                return self._get_default_metrics()
            
            # Basic performance metrics
            performance_metrics = self._calculate_performance_metrics(df, returns)
            
            # Risk metrics
            risk_metrics = self._calculate_risk_metrics(returns)
            
            # Position analytics
            position_analytics = self._calculate_position_analytics(positions, balances)
            
            # Factor exposure
            factor_exposure = self._calculate_factor_exposure(positions)
            
            result = {
                'risk_metrics': risk_metrics,
                'performance_metrics': performance_metrics,
                'analytics': {
                    'position_analytics': position_analytics,
                    'factor_exposure': factor_exposure,
                    'calculation_timestamp': datetime.utcnow().isoformat()
                }
            }
            
        except Exception as e:
            self.logger.error(f"Portfolio metrics calculation failed: {e}")
            self.logger.info("PortfolioAnalytics.calculate_comprehensive_metrics EXIT (error)")
            return self._get_default_metrics()
    
    def _calculate_performance_metrics(self, df: pd.DataFrame, returns: pd.Series) -> Dict[str, float]:
        """Calculate performance metrics"""
        
        # Total return
        initial_value = df['total_value'].iloc[0]
        final_value = df['total_value'].iloc[-1]
        total_return = (final_value / initial_value) - 1
        
        # Annualized metrics
        days = (df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]).days
        years = max(days / 365.25, 1/252)  # At least 1 trading day
        
        annualized_return = (1 + total_return) ** (1 / years) - 1
        
        # Volatility (annualized)
        volatility = returns.std() * np.sqrt(252)
        
        # Sharpe ratio
        sharpe_ratio = (annualized_return - self.risk_free_rate) / volatility if volatility != 0 else 0
        
        # Downside metrics
        downside_returns = returns[returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = (annualized_return - self.risk_free_rate) / downside_deviation if downside_deviation != 0 else 0
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'best_day': returns.max() if len(returns) > 0 else 0,
            'worst_day': returns.min() if len(returns) > 0 else 0,
            'positive_days_pct': (returns > 0).mean() * 100 if len(returns) > 0 else 0
        }
    
    def _calculate_risk_metrics(self, returns: pd.Series) -> Dict[str, float]:
        """Calculate comprehensive risk metrics"""
        
        # VaR calculations
        var_95 = returns.quantile(0.05) if len(returns) > 0 else 0
        var_99 = returns.quantile(0.01) if len(returns) > 0 else 0
        
        # Expected Shortfall (Conditional VaR)
        es_95 = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else 0
        es_99 = returns[returns <= var_99].mean() if len(returns[returns <= var_99]) > 0 else 0
        
        # Maximum Drawdown
        cumulative = (1 + returns).cumprod()
        peak = cumulative.expanding().max()
        drawdown = (cumulative - peak) / peak
        max_drawdown = drawdown.min() if len(drawdown) > 0 else 0
        
        # Additional risk metrics
        skewness = returns.skew() if len(returns) > 2 else 0
        kurtosis = returns.kurtosis() if len(returns) > 3 else 0
        
        return {
            'var_95': var_95,
            'var_99': var_99,
            'expected_shortfall_95': es_95,
            'expected_shortfall_99': es_99,
            'max_drawdown': max_drawdown,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'beta': 0.0,  # Would need benchmark data
            'alpha': 0.0  # Would need benchmark data
        }
    
    def _calculate_position_analytics(self, positions: Dict[str, Any], balances: Dict[str, float]) -> Dict[str, Any]:
        """Calculate position-level analytics"""
        
        if not positions:
            return {
                'concentration': 0.0,
                'diversification_ratio': 1.0,
                'largest_position_pct': 0.0,
                'number_of_positions': 0,
                'cash_allocation_pct': 100.0
            }
        
        total_value = balances.get('total_value', 0)
        
        if total_value <= 0:
            return {
                'concentration': 0.0,
                'diversification_ratio': 1.0,
                'largest_position_pct': 0.0,
                'number_of_positions': len(positions),
                'cash_allocation_pct': 100.0
            }
        
        # Position weights
        weights = []
        for position in positions.values():
            market_value = position.get('market_value', 0)
            weight = market_value / total_value
            weights.append(weight)
        
        weights = np.array(weights)
        
        # Concentration measures
        herfindahl_index = np.sum(weights ** 2) if len(weights) > 0 else 0
        effective_positions = 1 / herfindahl_index if herfindahl_index > 0 else 1
        
        # Largest position
        largest_position_pct = np.max(weights) * 100 if len(weights) > 0 else 0
        
        # Cash allocation
        cash_pct = (balances.get('cash', 0) / total_value) * 100
        
        return {
            'concentration': herfindahl_index,
            'diversification_ratio': effective_positions,
            'largest_position_pct': largest_position_pct,
            'number_of_positions': len(positions),
            'cash_allocation_pct': cash_pct
        }
    
    def _calculate_factor_exposure(self, positions: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate sector and factor exposures"""
        
        # Sample sector mapping - in practice, this would come from a data provider
        sector_mapping = {
            'RELIANCE': 'Energy',
            'TCS': 'IT',
            'HDFCBANK': 'Banking',
            'INFY': 'IT',
            'HINDUNILVR': 'FMCG',
            'ITC': 'FMCG',
            'ICICIBANK': 'Banking',
            'MARUTI': 'Auto',
            'SBIN': 'Banking'
        }
        
        sector_exposure = {}
        total_value = sum(pos.get('market_value', 0) for pos in positions.values())
        
        if total_value <= 0:
            return {'sector_exposure': {}, 'style_exposure': {}}
        
        for ticker, position in positions.items():
            clean_ticker = ticker.replace('.NS', '').replace('.BO', '')
            sector = sector_mapping.get(clean_ticker, 'Other')
            weight = position.get('market_value', 0) / total_value
            
            sector_exposure[sector] = sector_exposure.get(sector, 0) + weight
        
        return {
            'sector_exposure': sector_exposure,
            'style_exposure': {
                'large_cap': 0.7,  # Sample data
                'mid_cap': 0.2,
                'small_cap': 0.1
            }
        }
    
    def _get_default_metrics(self) -> Dict[str, Any]:
        """Return default metrics when calculation is not possible"""
        
        return {
            'risk_metrics': {
                'var_95': 0.0, 'var_99': 0.0, 'expected_shortfall_95': 0.0,
                'expected_shortfall_99': 0.0, 'max_drawdown': 0.0,
                'beta': 0.0, 'alpha': 0.0, 'sharpe_ratio': 0.0,
                'sortino_ratio': 0.0, 'skewness': 0.0, 'kurtosis': 0.0
            },
            'performance_metrics': {
                'total_return': 0.0, 'annualized_return': 0.0, 'volatility': 0.0,
                'sharpe_ratio': 0.0, 'sortino_ratio': 0.0, 'best_day': 0.0,
                'worst_day': 0.0, 'positive_days_pct': 0.0
            },
            'analytics': {
                'position_analytics': {
                    'concentration': 0.0, 'diversification_ratio': 1.0,
                    'largest_position_pct': 0.0, 'number_of_positions': 0,
                    'cash_allocation_pct': 100.0
                },
                'factor_exposure': {
                    'sector_exposure': {}, 'style_exposure': {}
                }
            }
        }

class ModernPortfolioTheory:
    """Modern Portfolio Theory optimization and analysis"""
    
    def __init__(self, risk_free_rate: float = 0.07):
        self.risk_free_rate = risk_free_rate
        self.logger = logging.getLogger(__name__)
        
    def optimize_portfolio(self,
        expected_returns: pd.Series,
        covariance_matrix: pd.DataFrame,
        constraints: RiskConstraints,
        method: str = "max_sharpe"
    ) -> Dict[str, Any]:
        """
        Optimize portfolio using Modern Portfolio Theory
        
        Args:
            expected_returns: Expected returns for each asset
            covariance_matrix: Covariance matrix of returns
            constraints: Risk and position constraints
            method: 'max_sharpe', 'min_variance', 'risk_parity', 'max_return'
            
        Returns:
            Optimization results with weights and metrics
        """
        
        try:
            if method == "max_sharpe":
                self.logger.info("ModernPortfolioTheory.optimize_portfolio method: max_sharpe")
                return self._maximize_sharpe_ratio(expected_returns, covariance_matrix, constraints)
            elif method == "min_variance":
                self.logger.info("ModernPortfolioTheory.optimize_portfolio method: min_variance")
                return self._minimize_variance(covariance_matrix, constraints)
            elif method == "risk_parity":
                self.logger.info("ModernPortfolioTheory.optimize_portfolio method: risk_parity")
                return self._risk_parity_optimization(covariance_matrix, constraints)
            elif method == "max_return":
                self.logger.info("ModernPortfolioTheory.optimize_portfolio method: max_return")
                return self._maximize_return(expected_returns, covariance_matrix, constraints)
            else:
                raise ValueError(f"Unknown optimization method: {method}")
                
        except Exception as e:
            self.logger.error(f"Portfolio optimization failed: {e}")
            self.logger.info("ModernPortfolioTheory.optimize_portfolio EXIT (error)")
            return self._get_equal_weight_fallback(expected_returns.index)
    
    def _maximize_sharpe_ratio(
        self, 
        expected_returns: pd.Series, 
        covariance_matrix: pd.DataFrame,
        constraints: RiskConstraints
    ) -> Dict[str, Any]:
        """Maximize Sharpe ratio using optimization"""
        
        n_assets = len(expected_returns)
        
        return self._scipy_max_sharpe(expected_returns, covariance_matrix, constraints)
    
    def _scipy_max_sharpe(self, expected_returns: pd.Series, cov_matrix: pd.DataFrame, constraints: RiskConstraints):
        """Maximize Sharpe ratio using scipy optimization"""
        
        n_assets = len(expected_returns)
        
        def negative_sharpe(weights):
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
            
            if portfolio_vol == 0:
                return -np.inf
            
            sharpe = (portfolio_return - self.risk_free_rate) / portfolio_vol
            return -sharpe  # Minimize negative Sharpe
        
        # Constraints
        constraints_list = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}  # Weights sum to 1
        ]
        
        # Bounds
        bounds = [(0, constraints.max_position_weight or 1.0) for _ in range(n_assets)]
        
        # Initial guess
        x0 = np.ones(n_assets) / n_assets
        
        try:
            result = optimize.minimize(
                negative_sharpe, x0, method='SLSQP',
                bounds=bounds, constraints=constraints_list,
                options={'maxiter': 1000, 'ftol': 1e-9}
            )
            
            if result.success:
                weights = pd.Series(result.x, index=expected_returns.index)
                return self._create_optimization_result(weights, expected_returns, cov_matrix, "max_sharpe")
            else:
                self.logger.warning(f"Sharpe optimization failed: {result.message}")
                return self._get_equal_weight_fallback(expected_returns.index)
                
        except Exception as e:
            self.logger.error(f"Scipy Sharpe optimization error: {e}")
            return self._get_equal_weight_fallback(expected_returns.index)
    
    def _minimize_variance(self, cov_matrix: pd.DataFrame, constraints: RiskConstraints):
        """Global Minimum Variance Portfolio"""
        
        n_assets = len(cov_matrix)
        
        def portfolio_variance(weights):
            return np.dot(weights, np.dot(cov_matrix, weights))
        
        # Constraints
        constraints_list = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}
        ]
        
        # Bounds
        bounds = [(0, constraints.max_position_weight or 1.0) for _ in range(n_assets)]
        
        # Initial guess
        x0 = np.ones(n_assets) / n_assets
        
        try:
            result = optimize.minimize(
                portfolio_variance, x0, method='SLSQP',
                bounds=bounds, constraints=constraints_list
            )
            
            if result.success:
                weights = pd.Series(result.x, index=cov_matrix.index)
                # Use zero expected returns for min variance calculation
                zero_returns = pd.Series(np.zeros(n_assets), index=cov_matrix.index)
                return self._create_optimization_result(weights, zero_returns, cov_matrix, "min_variance")
            else:
                return self._get_equal_weight_fallback(cov_matrix.index)
                
        except Exception as e:
            self.logger.error(f"Min variance optimization error: {e}")
            return self._get_equal_weight_fallback(cov_matrix.index)
    
    def _risk_parity_optimization(self, cov_matrix: pd.DataFrame, constraints: RiskConstraints):
        """Risk Parity Portfolio - equal risk contribution"""
        
        def risk_parity_objective(weights):
            weights = np.array(weights)
            portfolio_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
            
            if portfolio_vol == 0:
                return 1e6
            
            # Risk contributions
            marginal_contrib = np.dot(cov_matrix, weights)
            risk_contrib = weights * marginal_contrib / (portfolio_vol ** 2)
            
            # Target equal risk contribution
            target = 1.0 / len(weights)
            
            return np.sum((risk_contrib - target) ** 2)
        
        n_assets = len(cov_matrix)
        
        # Constraints
        constraints_list = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}
        ]
        
        # Bounds
        bounds = [(0.001, constraints.max_position_weight or 1.0) for _ in range(n_assets)]
        
        # Initial guess
        x0 = np.ones(n_assets) / n_assets
        
        try:
            result = optimize.minimize(
                risk_parity_objective, x0, method='SLSQP',
                bounds=bounds, constraints=constraints_list,
                options={'maxiter': 1000}
            )
            
            if result.success:
                weights = pd.Series(result.x, index=cov_matrix.index)
                zero_returns = pd.Series(np.zeros(n_assets), index=cov_matrix.index)
                return self._create_optimization_result(weights, zero_returns, cov_matrix, "risk_parity")
            else:
                return self._get_equal_weight_fallback(cov_matrix.index)
                
        except Exception as e:
            self.logger.error(f"Risk parity optimization error: {e}")
            return self._get_equal_weight_fallback(cov_matrix.index)
    
    def _create_optimization_result(
        self, 
        weights: pd.Series, 
        expected_returns: pd.Series,
        cov_matrix: pd.DataFrame, 
        method: str
    ) -> Dict[str, Any]:
        """Create standardized optimization result"""
        
        # Calculate portfolio metrics
        portfolio_return = np.dot(weights, expected_returns) if not expected_returns.isna().all() else 0
        portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
        portfolio_vol = np.sqrt(portfolio_variance * 252)  # Annualized
        
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_vol if portfolio_vol != 0 else 0
        
        # Risk contributions
        if portfolio_variance > 0:
            marginal_contrib = np.dot(cov_matrix, weights)
            risk_contributions = weights * marginal_contrib / portfolio_variance
        else:
            risk_contributions = weights * 0
        
        return {
            'weights': weights.to_dict(),
            'expected_return': portfolio_return,
            'volatility': portfolio_vol,
            'sharpe_ratio': sharpe_ratio,
            'risk_contributions': risk_contributions.to_dict(),
            'largest_weight': weights.max(),
            'effective_positions': 1 / np.sum(weights ** 2),
            'method': method,
            'success': True
        }
    
    def _get_equal_weight_fallback(self, asset_names) -> Dict[str, Any]:
        """Equal weight fallback when optimization fails"""
        
        n_assets = len(asset_names)
        equal_weight = 1.0 / n_assets
        weights = pd.Series([equal_weight] * n_assets, index=asset_names)
        
        return {
            'weights': weights.to_dict(),
            'expected_return': 0.0,
            'volatility': 0.0,
            'sharpe_ratio': 0.0,
            'risk_contributions': weights.to_dict(),
            'largest_weight': equal_weight,
            'effective_positions': n_assets,
            'method': 'equal_weight_fallback',
            'success': False
        }
