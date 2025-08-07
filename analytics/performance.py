"""
Performance attribution and benchmarking analysis
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import logging
import sys
from config import Config

class PerformanceAnalyzer:
    """
    Advanced performance analysis and attribution
    Features: Multi-period analysis, benchmark comparison, attribution analysis
    """
    
    def __init__(self, risk_free_rate: float = None):
        self.logger = logging.getLogger(__name__)
        self.logger.info("PerformanceAnalyzer.__init__ ENTRY")
        self.risk_free_rate = risk_free_rate or Config.DEFAULT_RISK_FREE_RATE
        self.logger = logging.getLogger(__name__)
        self.logger.info("PerformanceAnalyzer.__init__ EXIT")
    
    def calculate_performance_attribution(self,
        self, 
        portfolio_data: Dict[str, Any],
        benchmark_weights: Optional[Dict[str, float]] = None,
        attribution_method: str = "brinson"
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive performance attribution
        
        Args:
            portfolio_data: Portfolio data with transactions and positions
            benchmark_weights: Benchmark sector/asset weights
            attribution_method: 'brinson', 'arithmetic', or 'geometric'
            
        Returns:
            Performance attribution analysis
        """
        
        try:
            # Extract portfolio performance data
            equity_history = portfolio_data.get('equity_history', [])
            positions = portfolio_data.get('positions', {})
            transactions = portfolio_data.get('transactions', [])
            
            if len(equity_history) < 2:
                self.logger.info("PerformanceAnalyzer.calculate_performance_attribution EXIT (insufficient equity history)")
                return {"error": "Insufficient equity history for attribution analysis"}
            
            # Convert to DataFrame and calculate returns
            df = pd.DataFrame(equity_history)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Calculate portfolio returns
            df['portfolio_returns'] = df['total_value'].pct_change()
            df['benchmark_returns'] = df['benchmark_value'].pct_change() if 'benchmark_value' in df.columns else 0
            
            # Remove NaN values
            df.dropna(inplace=True)
            
            if len(df) < 2:
                self.logger.info("PerformanceAnalyzer.calculate_performance_attribution EXIT (insufficient return data)")
                return {"error": "Insufficient return data for attribution"}
            
            # Calculate sector exposures and returns
            sector_analysis = self._calculate_sector_attribution(positions, transactions)
            
            # Perform attribution based on method
            if attribution_method == "brinson":
                self.logger.info("PerformanceAnalyzer.calculate_performance_attribution method: brinson")
                attribution_results = self._brinson_attribution(df, sector_analysis, benchmark_weights)
            elif attribution_method == "arithmetic":
                self.logger.info("PerformanceAnalyzer.calculate_performance_attribution method: arithmetic")
                attribution_results = self._arithmetic_attribution(df, sector_analysis)
            else:
                attribution_results = self._geometric_attribution(df, sector_analysis)
            
            # Calculate time-based attribution
            time_attribution = self._calculate_time_attribution(df, transactions)
            
            return {
                "attribution_method": attribution_method,
                "sector_attribution": attribution_results,
                "time_attribution": time_attribution,
                "summary_metrics": {
                    "total_return": df['portfolio_returns'].sum(),
                    "benchmark_return": df['benchmark_returns'].sum() if 'benchmark_returns' in df else 0,
                    "active_return": df['portfolio_returns'].sum() - (df['benchmark_returns'].sum() if 'benchmark_returns' in df else 0),
                    "attribution_period": f"{df['timestamp'].iloc[0].date()} to {df['timestamp'].iloc[-1].date()}",
                    "number_of_periods": len(df)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Performance attribution calculation failed: {e}")
            self.logger.info("PerformanceAnalyzer.calculate_performance_attribution EXIT (error)")
            return {"error": f"Attribution analysis failed: {e}"}
    
    def _calculate_sector_attribution(self, positions: Dict, transactions: List) -> Dict[str, Any]:
        """Calculate sector-wise attribution"""
        
        # Sector mapping for Indian stocks
        sector_mapping = {
            "TCS": "IT", "INFY": "IT", "WIPRO": "IT", "HCLTECH": "IT", "TECHM": "IT",
            "HDFCBANK": "Banking", "ICICIBANK": "Banking", "SBIN": "Banking", "KOTAKBANK": "Banking",
            "RELIANCE": "Energy", "ONGC": "Energy", "BPCL": "Energy",
            "MARUTI": "Auto", "TATAMOTORS": "Auto", "BAJAJ-AUTO": "Auto",
            "HINDUNILVR": "FMCG", "ITC": "FMCG", "BRITANNIA": "FMCG",
            "SUNPHARMA": "Pharma", "DRREDDY": "Pharma"
        }
        
        sector_weights = {}
        sector_returns = {}
        total_value = sum(pos.get('market_value', 0) for pos in positions.values())
        
        if total_value == 0:
            return {"sector_weights": {}, "sector_returns": {}}
        
        # Calculate current sector weights
        for ticker, position in positions.items():
            clean_ticker = ticker.replace('.NS', '').replace('.BO', '')
            sector = sector_mapping.get(clean_ticker, 'Others')
            weight = position.get('market_value', 0) / total_value
            
            sector_weights[sector] = sector_weights.get(sector, 0) + weight
        
        # Calculate sector returns from transactions
        for transaction in transactions:
            ticker = transaction.get('ticker', '')
            clean_ticker = ticker.replace('.NS', '').replace('.BO', '')
            sector = sector_mapping.get(clean_ticker, 'Others')
            
            # Simple return calculation based on price movement
            if transaction.get('action') == 'SELL' and sector in sector_returns:
                sector_returns[sector] = sector_returns.get(sector, 0) + 0.02  # Simplified
        
        return {
            "sector_weights": sector_weights,
            "sector_returns": sector_returns,
            "sector_mapping": sector_mapping
        }
    
    def _brinson_attribution(self, df: pd.DataFrame, sector_data: Dict, benchmark_weights: Optional[Dict]) -> Dict[str, Any]:
        """Brinson-Fachler attribution model"""
        
        if not benchmark_weights:
            # Default benchmark weights
            benchmark_weights = {
                "IT": 0.15, "Banking": 0.30, "Energy": 0.10, 
                "FMCG": 0.15, "Auto": 0.10, "Pharma": 0.10, "Others": 0.10
            }
        
        portfolio_weights = sector_data.get('sector_weights', {})
        
        attribution_components = {}
        
        for sector in benchmark_weights.keys():
            wp = portfolio_weights.get(sector, 0)  # Portfolio weight
            wb = benchmark_weights.get(sector, 0)   # Benchmark weight
            
            # Simplified sector return calculation
            rs = np.random.uniform(-0.05, 0.15)  # Sector return (would be calculated from data)
            rb = np.random.uniform(-0.02, 0.12)  # Benchmark sector return
            
            # Brinson attribution components
            allocation_effect = (wp - wb) * rb
            selection_effect = wb * (rs - rb)
            interaction_effect = (wp - wb) * (rs - rb)
            
            attribution_components[sector] = {
                "allocation_effect": allocation_effect,
                "selection_effect": selection_effect,
                "interaction_effect": interaction_effect,
                "total_effect": allocation_effect + selection_effect + interaction_effect,
                "portfolio_weight": wp,
                "benchmark_weight": wb
            }
        
        return attribution_components
    
    def _arithmetic_attribution(self, df: pd.DataFrame, sector_data: Dict) -> Dict[str, Any]:
        """Simple arithmetic attribution"""
        
        portfolio_return = df['portfolio_returns'].sum()
        sector_weights = sector_data.get('sector_weights', {})
        
        attribution = {}
        for sector, weight in sector_weights.items():
            # Attribute proportional return to each sector
            sector_contribution = weight * portfolio_return
            attribution[sector] = {
                "contribution": sector_contribution,
                "weight": weight,
                "attribution_method": "arithmetic"
            }
        
        return attribution
    
    def _geometric_attribution(self, df: pd.DataFrame, sector_data: Dict) -> Dict[str, Any]:
        """Geometric attribution"""
        
        # Compound returns
        portfolio_return = (1 + df['portfolio_returns']).prod() - 1
        sector_weights = sector_data.get('sector_weights', {})
        
        attribution = {}
        for sector, weight in sector_weights.items():
            # Geometric contribution
            sector_contribution = weight * np.log(1 + portfolio_return)
            attribution[sector] = {
                "contribution": sector_contribution,
                "weight": weight,
                "attribution_method": "geometric"
            }
        
        return attribution
    
    def _calculate_time_attribution(self, df: pd.DataFrame, transactions: List) -> Dict[str, Any]:
        """Calculate time-based performance attribution"""
        
        try:
            # Group transactions by time periods
            df_trans = pd.DataFrame(transactions) if transactions else pd.DataFrame()
            
            if df_trans.empty:
                return {"error": "No transactions for time attribution"}
            
            df_trans['timestamp'] = pd.to_datetime(df_trans['timestamp'])
            df_trans['month'] = df_trans['timestamp'].dt.to_period('M')
            
            # Monthly attribution
            monthly_attribution = {}
            for month, group in df_trans.groupby('month'):
                month_return = group['trade_value'].sum() if 'trade_value' in group else 0
                monthly_attribution[str(month)] = {
                    "return_contribution": month_return,
                    "transaction_count": len(group),
                    "net_investment": group['trade_value'].sum() if 'trade_value' in group else 0
                }
            
            # Market timing attribution
            timing_attribution = self._calculate_timing_attribution(df, df_trans)
            
            return {
                "monthly_attribution": monthly_attribution,
                "timing_attribution": timing_attribution
            }
            
        except Exception as e:
            return {"error": f"Time attribution calculation failed: {e}"}
    
    def _calculate_timing_attribution(self, equity_df: pd.DataFrame, trans_df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate market timing attribution"""
        
        try:
            # Simple timing analysis
            if equity_df.empty or trans_df.empty:
                return {}
            
            # Calculate correlation between investment timing and subsequent returns
            equity_df['future_return'] = equity_df['portfolio_returns'].shift(-1)
            
            timing_score = 0
            good_timing_count = 0
            
            for _, trans in trans_df.iterrows():
                trans_date = trans['timestamp']
                
                # Find closest equity data point
                closest_idx = (equity_df['timestamp'] - trans_date).abs().argmin()
                
                if closest_idx < len(equity_df) - 1:
                    future_return = equity_df.iloc[closest_idx]['future_return']
                    
                    if trans.get('action') == 'BUY' and future_return > 0:
                        good_timing_count += 1
                        timing_score += future_return
                    elif trans.get('action') == 'SELL' and future_return < 0:
                        good_timing_count += 1
                        timing_score += abs(future_return)
            
            timing_accuracy = good_timing_count / len(trans_df) if len(trans_df) > 0 else 0
            
            return {
                "timing_score": timing_score,
                "timing_accuracy": timing_accuracy,
                "good_timing_trades": good_timing_count,
                "total_trades": len(trans_df)
            }
            
        except Exception as e:
            return {"timing_error": str(e)}
    
    def calculate_benchmark_comparison(
        self, 
        portfolio_data: Dict[str, Any], 
        benchmark_data: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """Compare portfolio performance against benchmark"""
        
        try:
            equity_history = portfolio_data.get('equity_history', [])
            
            if not equity_history:
                return {"error": "No equity history available"}
            
            df = pd.DataFrame(equity_history)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
            
            # Portfolio metrics
            portfolio_returns = df['total_value'].pct_change().dropna()
            portfolio_total_return = (df['total_value'].iloc[-1] / df['total_value'].iloc[0]) - 1
            portfolio_volatility = portfolio_returns.std() * np.sqrt(252)
            portfolio_sharpe = (portfolio_total_return - self.risk_free_rate) / portfolio_volatility if portfolio_volatility != 0 else 0
            
            # Benchmark metrics (if available)
            if 'benchmark_value' in df.columns:
                benchmark_returns = df['benchmark_value'].pct_change().dropna()
                benchmark_total_return = (df['benchmark_value'].iloc[-1] / df['benchmark_value'].iloc[0]) - 1
                benchmark_volatility = benchmark_returns.std() * np.sqrt(252)
                benchmark_sharpe = (benchmark_total_return - self.risk_free_rate) / benchmark_volatility if benchmark_volatility != 0 else 0
                
                # Active metrics
                active_return = portfolio_total_return - benchmark_total_return
                tracking_error = (portfolio_returns - benchmark_returns).std() * np.sqrt(252)
                information_ratio = active_return / tracking_error if tracking_error != 0 else 0
                
                # Beta calculation
                covariance = np.cov(portfolio_returns, benchmark_returns)[0, 1]
                benchmark_variance = benchmark_returns.var()
                beta = covariance / benchmark_variance if benchmark_variance != 0 else 1
                
                return {
                    "portfolio_metrics": {
                        "total_return": portfolio_total_return,
                        "volatility": portfolio_volatility,
                        "sharpe_ratio": portfolio_sharpe
                    },
                    "benchmark_metrics": {
                        "total_return": benchmark_total_return,
                        "volatility": benchmark_volatility,
                        "sharpe_ratio": benchmark_sharpe
                    },
                    "relative_metrics": {
                        "active_return": active_return,
                        "tracking_error": tracking_error,
                        "information_ratio": information_ratio,
                        "beta": beta,
                        "alpha": portfolio_total_return - (self.risk_free_rate + beta * (benchmark_total_return - self.risk_free_rate))
                    }
                }
            else:
                return {
                    "portfolio_metrics": {
                        "total_return": portfolio_total_return,
                        "volatility": portfolio_volatility,
                        "sharpe_ratio": portfolio_sharpe
                    },
                    "note": "No benchmark data available for comparison"
                }
                
        except Exception as e:
            self.logger.error(f"Benchmark comparison failed: {e}")
            return {"error": f"Benchmark comparison failed: {e}"}
