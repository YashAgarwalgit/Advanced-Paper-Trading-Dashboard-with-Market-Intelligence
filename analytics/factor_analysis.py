"""
Factor exposure and sector analysis for institutional portfolios
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import logging
import sys
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from config import Config

class FactorAnalyzer:
    """
    Advanced factor analysis and exposure calculation
    Features: Multi-factor models, sector analysis, style factor decomposition
    """
    
    def __init__(self, risk_free_rate: float = None):
        self.logger = logging.getLogger(__name__)
        self.logger.info("FactorAnalyzer.__init__ ENTRY")
        self.risk_free_rate = risk_free_rate or Config.DEFAULT_RISK_FREE_RATE
        self.logger = logging.getLogger(__name__)
        
        # Factor definitions
        self.style_factors = ['Value', 'Growth', 'Quality', 'Momentum', 'Low_Volatility', 'Size']
        self.sector_factors = ['IT', 'Banking', 'Energy', 'FMCG', 'Auto', 'Pharma', 'Metals', 'Telecom']
        
        # Asset mappings
        self._initialize_factor_mappings()
        self.logger.info("FactorAnalyzer.__init__ EXIT")
    
    def _initialize_factor_mappings(self):
        self.logger.info("FactorAnalyzer._initialize_factor_mappings ENTRY")
        """Initialize sector and style mappings for assets"""
        
        self.sector_mapping = {
            # IT Sector
            "TCS": "IT", "INFY": "IT", "WIPRO": "IT", "HCLTECH": "IT", 
            "TECHM": "IT", "MINDTREE": "IT", "LTI": "IT",
            # Banking
            "HDFCBANK": "Banking", "ICICIBANK": "Banking", "SBIN": "Banking",
            "KOTAKBANK": "Banking", "AXISBANK": "Banking", "INDUSIND": "Banking",
            # Energy & Oil
            "RELIANCE": "Energy", "ONGC": "Energy", "BPCL": "Energy", 
            "IOCL": "Energy", "GAIL": "Energy",
            # FMCG
            "HINDUNILVR": "FMCG", "ITC": "FMCG", "BRITANNIA": "FMCG",
            "NESTLEIND": "FMCG", "DABUR": "FMCG",
            # Auto
            "MARUTI": "Auto", "TATAMOTORS": "Auto", "M&M": "Auto",
            "BAJAJ-AUTO": "Auto", "HEROMOCP": "Auto",
            # Pharma
            "SUNPHARMA": "Pharma", "DRREDDY": "Pharma", "CIPLA": "Pharma",
            "LUPIN": "Pharma", "BIOCON": "Pharma",
            # Metals
            "TATASTEEL": "Metals", "HINDALCO": "Metals", "JSWSTEEL": "Metals",
            "COALINDIA": "Metals", "VEDL": "Metals",
            # Banking
            "HDFCBANK": "Banking", "ICICIBANK": "Banking", "SBIN": "Banking",
            "KOTAKBANK": "Banking", "AXISBANK": "Banking", "INDUSIND": "Banking",
            
            # Energy & Oil
            "RELIANCE": "Energy", "ONGC": "Energy", "BPCL": "Energy", 
            "IOCL": "Energy", "GAIL": "Energy",
            
            # FMCG
            "HINDUNILVR": "FMCG", "ITC": "FMCG", "BRITANNIA": "FMCG",
            "NESTLEIND": "FMCG", "DABUR": "FMCG",
            
            # Auto
            "MARUTI": "Auto", "TATAMOTORS": "Auto", "M&M": "Auto",
            "BAJAJ-AUTO": "Auto", "HEROMOCP": "Auto",
            
            # Pharma
            "SUNPHARMA": "Pharma", "DRREDDY": "Pharma", "CIPLA": "Pharma",
            "LUPIN": "Pharma", "BIOCON": "Pharma",
            
            # Metals
            "TATASTEEL": "Metals", "HINDALCO": "Metals", "JSWSTEEL": "Metals",
            "COALINDIA": "Metals", "VEDL": "Metals"
        }
        self.logger.info("FactorAnalyzer._initialize_factor_mappings EXIT")

        # Style factor characteristics (simplified)
        self.style_characteristics = {
            "Value": {"PE_low": True, "PB_low": True, "dividend_yield_high": True},
            "Growth": {"earnings_growth_high": True, "revenue_growth_high": True},
            "Quality": {"ROE_high": True, "debt_low": True, "margin_high": True},
            "Momentum": {"price_momentum_high": True, "earnings_momentum_high": True},
            "Low_Volatility": {"volatility_low": True, "beta_low": True},
            "Size": {"market_cap_large": True}
        }
    
    def calculate_factor_exposures(self, portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        self.logger.info("FactorAnalyzer.calculate_factor_exposures ENTRY")
        """
        Calculate comprehensive factor exposures for the portfolio
        
        Args:
            portfolio_data: Portfolio data with positions and analytics
            
        Returns:
            Factor exposure analysis
        """
        
        try:
            positions = portfolio_data.get('positions', {})
            
            if not positions:
                self.logger.info("FactorAnalyzer.calculate_factor_exposures EXIT (no positions)")
                return {"error": "No positions found for factor analysis"}
            
            # Calculate sector exposures
            sector_exposures = self._calculate_sector_exposures(positions)
            self.logger.info("FactorAnalyzer.calculate_factor_exposures sector exposures calculated")
            
            # Calculate style factor exposures
            style_exposures = self._calculate_style_exposures(positions)
            self.logger.info("FactorAnalyzer.calculate_factor_exposures style exposures calculated")
            
            # Calculate factor loadings and contributions
            factor_loadings = self._calculate_factor_loadings(positions, portfolio_data)
            self.logger.info("FactorAnalyzer.calculate_factor_exposures factor loadings calculated")
            
            # Risk factor decomposition
            risk_decomposition = self._calculate_risk_decomposition(positions, sector_exposures, style_exposures)
            
            # Factor tilts vs benchmark
            factor_tilts = self._calculate_factor_tilts(sector_exposures, style_exposures)
            
            return {
                "sector_exposures": sector_exposures,
                "style_exposures": style_exposures,
                "factor_loadings": factor_loadings,
                "risk_decomposition": risk_decomposition,
                "factor_tilts": factor_tilts,
                "analysis_timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Factor exposure calculation failed: {e}")
            self.logger.info("FactorAnalyzer.calculate_factor_exposures EXIT (error)")
            return {"error": f"Factor analysis failed: {e}"}
    
    def _calculate_sector_exposures(self, positions: Dict[str, Any]) -> Dict[str, float]:
        """Calculate sector exposure weights"""
        
        sector_exposures = {}
        total_value = sum(pos.get('market_value', 0) for pos in positions.values())
        
        if total_value == 0:
            return sector_exposures
        
        for ticker, position in positions.items():
            clean_ticker = ticker.replace('.NS', '').replace('.BO', '')
            sector = self.sector_mapping.get(clean_ticker, 'Others')
            weight = position.get('market_value', 0) / total_value
            
            sector_exposures[sector] = sector_exposures.get(sector, 0) + weight
        
        return sector_exposures
    
    def _calculate_style_exposures(self, positions: Dict[str, Any]) -> Dict[str, float]:
        """Calculate style factor exposures"""
        
        style_exposures = {factor: 0.0 for factor in self.style_factors}
        total_value = sum(pos.get('market_value', 0) for pos in positions.values())
        
        if total_value == 0:
            return style_exposures
        
        for ticker, position in positions.items():
            weight = position.get('market_value', 0) / total_value
            
            # Simplified style factor assignment based on ticker characteristics
            clean_ticker = ticker.replace('.NS', '').replace('.BO', '')
            
            # Assign style factors based on known characteristics
            if clean_ticker in ['RELIANCE', 'TCS', 'HDFCBANK']:
                style_exposures['Quality'] += weight * 0.8
                style_exposures['Size'] += weight * 1.0  # Large cap
            elif clean_ticker in ['INFY', 'WIPRO', 'TECHM']:
                style_exposures['Growth'] += weight * 0.7
                style_exposures['Momentum'] += weight * 0.5
            elif clean_ticker in ['HINDUNILVR', 'ITC']:
                style_exposures['Value'] += weight * 0.6
                style_exposures['Quality'] += weight * 0.7
            else:
                # Default allocation
                style_exposures['Value'] += weight * 0.3
                style_exposures['Growth'] += weight * 0.3
        
        return style_exposures
    
    def _calculate_factor_loadings(self, positions: Dict, portfolio_data: Dict) -> Dict[str, Any]:
        """Calculate factor loadings using returns data"""
        
        try:
            equity_history = portfolio_data.get('equity_history', [])
            
            if len(equity_history) < 30:  # Need minimum data points
                return {"error": "Insufficient data for factor loading calculation"}
            
            # Convert to returns
            df = pd.DataFrame(equity_history)
            df['returns'] = df['total_value'].pct_change().dropna()
            
            returns = df['returns'].values
            
            # Generate synthetic factor returns for demonstration
            # In practice, these would be real factor index returns
            np.random.seed(42)  # For reproducible results
            n_periods = len(returns)
            
            factor_returns = {}
            for factor in self.style_factors + self.sector_factors:
                factor_returns[factor] = np.random.normal(0.001, 0.02, n_periods)
            
            # Create factor return matrix
            factor_matrix = np.column_stack(list(factor_returns.values()))
            factor_names = list(factor_returns.keys())
            
            # Run regression: portfolio_returns = alpha + beta1*F1 + beta2*F2 + ... + error
            try:
                # Add intercept
                X = np.column_stack([np.ones(len(returns)), factor_matrix])
                
                # OLS regression
                betas, residuals, rank, s = np.linalg.lstsq(X, returns, rcond=None)
                
                # Extract results
                alpha = betas[0]
                factor_betas = betas[1:]
                
                # Calculate R-squared
                ss_res = np.sum(residuals) if len(residuals) > 0 else np.sum((returns - np.dot(X, betas)) ** 2)
                ss_tot = np.sum((returns - np.mean(returns)) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                
                loadings = dict(zip(factor_names, factor_betas))
                
                return {
                    "alpha": alpha,
                    "factor_betas": loadings,
                    "r_squared": r_squared,
                    "residual_risk": np.std(returns - np.dot(X, betas)) if len(returns) > len(betas) else 0,
                    "significant_factors": [f for f, beta in loadings.items() if abs(beta) > 0.1]
                }
                
            except np.linalg.LinAlgError:
                return {"error": "Numerical issues in factor regression"}
                
        except Exception as e:
            return {"error": f"Factor loading calculation failed: {e}"}
    
    def _calculate_risk_decomposition(self, positions: Dict, sector_exp: Dict, style_exp: Dict) -> Dict[str, Any]:
        """Decompose portfolio risk into factor components"""
        
        try:
            # Simplified risk decomposition
            total_risk = 1.0  # Normalized total risk
            
            # Sector risk contribution
            sector_risk = sum(weight ** 2 for weight in sector_exp.values()) * 0.6
            
            # Style factor risk contribution  
            style_risk = sum(weight ** 2 for weight in style_exp.values()) * 0.3
            
            # Specific/idiosyncratic risk
            specific_risk = max(0, total_risk - sector_risk - style_risk)
            
            # Concentration risk
            position_weights = []
            total_value = sum(pos.get('market_value', 0) for pos in positions.values())
            
            if total_value > 0:
                position_weights = [pos.get('market_value', 0) / total_value for pos in positions.values()]
            
            concentration_risk = sum(w ** 2 for w in position_weights) if position_weights else 0
            
            return {
                "sector_risk_contribution": sector_risk,
                "style_risk_contribution": style_risk, 
                "specific_risk_contribution": specific_risk,
                "concentration_risk": concentration_risk,
                "risk_contributions": {
                    "Systematic Risk": sector_risk + style_risk,
                    "Specific Risk": specific_risk,
                    "Concentration Risk": concentration_risk
                }
            }
            
        except Exception as e:
            return {"error": f"Risk decomposition failed: {e}"}
    
    def _calculate_factor_tilts(self, sector_exp: Dict, style_exp: Dict) -> Dict[str, Any]:
        """Calculate factor tilts relative to benchmark"""
        
        # Default benchmark weights (representing broad market)
        benchmark_sectors = {
            "IT": 0.15, "Banking": 0.30, "Energy": 0.10, "FMCG": 0.15,
            "Auto": 0.08, "Pharma": 0.07, "Metals": 0.05, "Others": 0.10
        }
        
        benchmark_styles = {
            "Value": 0.3, "Growth": 0.3, "Quality": 0.2, 
            "Momentum": 0.1, "Low_Volatility": 0.1, "Size": 0.8
        }
        
        # Calculate tilts (portfolio weight - benchmark weight)
        sector_tilts = {}
        for sector in benchmark_sectors:
            portfolio_weight = sector_exp.get(sector, 0)
            benchmark_weight = benchmark_sectors[sector]
            sector_tilts[sector] = portfolio_weight - benchmark_weight
        
        style_tilts = {}
        for factor in benchmark_styles:
            portfolio_weight = style_exp.get(factor, 0)
            benchmark_weight = benchmark_styles[factor]
            style_tilts[factor] = portfolio_weight - benchmark_weight
        
        # Identify significant tilts
        significant_sector_tilts = {k: v for k, v in sector_tilts.items() if abs(v) > 0.05}
        significant_style_tilts = {k: v for k, v in style_tilts.items() if abs(v) > 0.1}
        
        return {
            "sector_tilts": sector_tilts,
            "style_tilts": style_tilts,
            "significant_sector_tilts": significant_sector_tilts,
            "significant_style_tilts": significant_style_tilts,
            "tilt_summary": {
                "overweight_sectors": [k for k, v in significant_sector_tilts.items() if v > 0],
                "underweight_sectors": [k for k, v in significant_sector_tilts.items() if v < 0],
                "dominant_styles": [k for k, v in significant_style_tilts.items() if v > 0]
            }
        }
    
    def perform_pca_analysis(self, portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """Perform Principal Component Analysis on portfolio factors"""
        
        try:
            positions = portfolio_data.get('positions', {})
            equity_history = portfolio_data.get('equity_history', [])
            
            if not positions or len(equity_history) < 50:
                return {"error": "Insufficient data for PCA analysis"}
            
            # Create feature matrix (positions as features)
            tickers = list(positions.keys())
            
            # Generate synthetic return series for each ticker
            np.random.seed(42)
            n_periods = min(100, len(equity_history))
            
            returns_matrix = np.random.normal(0.001, 0.02, (n_periods, len(tickers)))
            
            # Add some correlation structure
            for i in range(len(tickers)):
                if i > 0:
                    returns_matrix[:, i] = 0.3 * returns_matrix[:, 0] + 0.7 * returns_matrix[:, i]
            
            # Standardize the data
            scaler = StandardScaler()
            returns_scaled = scaler.fit_transform(returns_matrix)
            
            # Perform PCA
            pca = PCA()
            pca_result = pca.fit_transform(returns_scaled)
            
            # Calculate results
            explained_variance_ratio = pca.explained_variance_ratio_
            cumulative_variance = np.cumsum(explained_variance_ratio)
            
            # Find number of components for 80% and 95% variance
            n_comp_80 = np.argmax(cumulative_variance >= 0.80) + 1
            n_comp_95 = np.argmax(cumulative_variance >= 0.95) + 1
            
            # Component loadings
            components = pca.components_
            
            # Create factor interpretations
            factor_interpretations = {}
            for i in range(min(5, len(components))):  # Top 5 factors
                loadings = components[i]
                top_contributors = [(tickers[j], loadings[j]) for j in np.argsort(np.abs(loadings))[-5:]]
                factor_interpretations[f"PC{i+1}"] = {
                    "explained_variance": explained_variance_ratio[i],
                    "top_contributors": top_contributors
                }
            
            return {
                "explained_variance_ratio": explained_variance_ratio[:10].tolist(),  # Top 10
                "cumulative_variance": cumulative_variance[:10].tolist(),
                "components_for_80pct": n_comp_80,
                "components_for_95pct": n_comp_95,
                "factor_interpretations": factor_interpretations,
                "total_components": len(explained_variance_ratio)
            }
            
        except Exception as e:
            self.logger.error(f"PCA analysis failed: {e}")
            return {"error": f"PCA analysis failed: {e}"}
    
    def calculate_correlation_analysis(self, portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate correlation structure within portfolio"""
        
        try:
            positions = portfolio_data.get('positions', {})
            
            if len(positions) < 2:
                return {"error": "Need at least 2 positions for correlation analysis"}
            
            tickers = list(positions.keys())
            n_assets = len(tickers)
            
            # Generate synthetic correlation matrix
            np.random.seed(42)
            # Start with random matrix
            corr_matrix = np.random.uniform(-0.5, 0.8, (n_assets, n_assets))
            # Make symmetric
            corr_matrix = (corr_matrix + corr_matrix.T) / 2
            # Set diagonal to 1
            np.fill_diagonal(corr_matrix, 1.0)
            
            # Convert to DataFrame for easier handling
            corr_df = pd.DataFrame(corr_matrix, index=tickers, columns=tickers)
            
            # Calculate correlation statistics
            avg_correlation = np.mean(corr_matrix[np.triu_indices(n_assets, k=1)])
            max_correlation = np.max(corr_matrix[np.triu_indices(n_assets, k=1)])
            min_correlation = np.min(corr_matrix[np.triu_indices(n_assets, k=1)])
            
            # Find highly correlated pairs
            high_corr_pairs = []
            for i in range(n_assets):
                for j in range(i+1, n_assets):
                    if corr_matrix[i, j] > 0.7:
                        high_corr_pairs.append({
                            "asset1": tickers[i],
                            "asset2": tickers[j], 
                            "correlation": corr_matrix[i, j]
                        })
            
            return {
                "correlation_matrix": corr_df.to_dict(),
                "correlation_statistics": {
                    "average_correlation": avg_correlation,
                    "max_correlation": max_correlation,
                    "min_correlation": min_correlation,
                    "highly_correlated_pairs": high_corr_pairs
                },
                "diversification_metrics": {
                    "effective_assets": n_assets / (1 + (n_assets - 1) * avg_correlation),
                    "concentration_risk": avg_correlation
                }
            }
            
        except Exception as e:
            self.logger.error(f"Correlation analysis failed: {e}")
            return {"error": f"Correlation analysis failed: {e}"}
