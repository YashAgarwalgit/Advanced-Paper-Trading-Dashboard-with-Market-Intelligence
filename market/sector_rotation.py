"""
Sector rotation analysis for market intelligence
"""
import numpy as np
import pandas as pd
import yfinance as yf
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import logging
from config import Config
from utils.helpers import safe_float, format_percentage, get_timestamp_iso
from utils.decorators import measure_performance

@dataclass
class SectorPerformance:
    """Sector performance data structure"""
    sector: str
    performance_1d: float
    performance_1w: float
    performance_1m: float
    performance_3m: float
    performance_ytd: float
    volatility: float
    momentum_score: float
    relative_strength: float

class SectorRotationAnalyzer:
    """
    Sector rotation analysis and momentum detection
    Features: Relative strength analysis, momentum scoring, rotation signals
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Indian sector indices mapping
        self.sector_indices = {
            "IT": "^CNXIT",
            "Banking": "^NSEBANK", 
            "Auto": "^CNXAUTO",
            "Pharma": "^CNXPHARMA",
            "FMCG": "^CNXFMCG",
            "Energy": "^CNXENERGY",
            "Metals": "^CNXMETAL",
            "Realty": "^CNXREALTY",
            "PSU": "^CNXPSE",
            "Media": "^CNXMEDIA"
        }
        
        # Sector characteristics
        self.sector_characteristics = {
            "IT": {"cyclical": False, "defensive": False, "growth": True},
            "Banking": {"cyclical": True, "defensive": False, "growth": False},
            "Auto": {"cyclical": True, "defensive": False, "growth": False},
            "Pharma": {"cyclical": False, "defensive": True, "growth": True},
            "FMCG": {"cyclical": False, "defensive": True, "growth": False},
            "Energy": {"cyclical": True, "defensive": False, "growth": False},
            "Metals": {"cyclical": True, "defensive": False, "growth": False},
            "Realty": {"cyclical": True, "defensive": False, "growth": False}
        }
    
    @measure_performance
    def analyze_sector_rotation(
        self, 
        lookback_days: int = 90,
        benchmark: str = "^NSEI"
    ) -> Dict[str, Any]:
        """
        Comprehensive sector rotation analysis
        
        Args:
            lookback_days: Analysis period in days
            benchmark: Benchmark index for relative performance
            
        Returns:
            Sector rotation analysis results
        """
        
        try:
            self.logger.info("Starting sector rotation analysis")
            
            # Get sector performance data
            sector_data = self._get_sector_performance_data(lookback_days)
            
            if not sector_data:
                return {"error": "Failed to fetch sector data"}
            
            # Calculate relative performance vs benchmark
            benchmark_data = self._get_benchmark_data(benchmark, lookback_days)
            
            # Analyze momentum and trends
            momentum_analysis = self._analyze_sector_momentum(sector_data)
            
            # Detect rotation patterns
            rotation_signals = self._detect_rotation_signals(sector_data, momentum_analysis)
            
            # Calculate sector scores
            sector_scores = self._calculate_sector_scores(sector_data, momentum_analysis)
            
            # Generate recommendations
            recommendations = self._generate_rotation_recommendations(
                sector_scores, rotation_signals, momentum_analysis
            )
            
            return {
                "timestamp": get_timestamp_iso(),
                "sector_performance": sector_data,
                "momentum_analysis": momentum_analysis,
                "rotation_signals": rotation_signals,
                "sector_scores": sector_scores,
                "recommendations": recommendations,
                "market_regime": self._determine_market_regime(sector_data),
                "defensive_vs_cyclical": self._analyze_defensive_cyclical(sector_data)
            }
            
        except Exception as e:
            self.logger.error(f"Sector rotation analysis failed: {e}")
            return {"error": f"Analysis failed: {e}"}
    
    def _get_sector_performance_data(self, lookback_days: int) -> Dict[str, SectorPerformance]:
        """Fetch and calculate sector performance data"""
        
        sector_data = {}
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)
        
        for sector, ticker in self.sector_indices.items():
            try:
                # Download sector data
                data = yf.download(ticker, start=start_date, end=end_date, progress=False)
                
                if data.empty or len(data) < 5:
                    continue
                
                # Calculate returns
                returns = data['Close'].pct_change().dropna()
                
                # Performance calculations
                current_price = data['Close'].iloc[-1]
                
                # 1D performance
                perf_1d = 0
                if len(data) >= 2:
                    perf_1d = (current_price / data['Close'].iloc[-2] - 1) * 100
                
                # 1W performance
                perf_1w = 0
                if len(data) >= 5:
                    perf_1w = (current_price / data['Close'].iloc[-6] - 1) * 100
                
                # 1M performance
                perf_1m = 0
                if len(data) >= 21:
                    perf_1m = (current_price / data['Close'].iloc[-22] - 1) * 100
                
                # 3M performance
                perf_3m = 0
                if len(data) >= 63:
                    perf_3m = (current_price / data['Close'].iloc[-64] - 1) * 100
                
                # YTD performance
                perf_ytd = 0
                ytd_start = datetime(end_date.year, 1, 1)
                ytd_data = data[data.index >= ytd_start]
                if not ytd_data.empty:
                    perf_ytd = (current_price / ytd_data['Close'].iloc[0] - 1) * 100
                
                # Volatility (annualized)
                volatility = returns.std() * np.sqrt(252) * 100
                
                # Momentum score (combination of different periods)
                momentum_score = (perf_1w * 0.3 + perf_1m * 0.5 + perf_3m * 0.2)
                
                # Relative strength (vs average of all sectors calculated later)
                relative_strength = momentum_score  # Will be adjusted later
                
                sector_data[sector] = SectorPerformance(
                    sector=sector,
                    performance_1d=perf_1d,
                    performance_1w=perf_1w,
                    performance_1m=perf_1m,
                    performance_3m=perf_3m,
                    performance_ytd=perf_ytd,
                    volatility=volatility,
                    momentum_score=momentum_score,
                    relative_strength=relative_strength
                )
                
            except Exception as e:
                self.logger.warning(f"Failed to get data for {sector}: {e}")
                continue
        
        # Calculate relative strength vs sector average
        if sector_data:
            avg_momentum = np.mean([s.momentum_score for s in sector_data.values()])
            for sector_perf in sector_data.values():
                sector_perf.relative_strength = sector_perf.momentum_score - avg_momentum
        
        return sector_data
    
    def _get_benchmark_data(self, benchmark: str, lookback_days: int) -> Optional[pd.Series]:
        """Get benchmark data for relative performance calculation"""
        
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_days)
            
            data = yf.download(benchmark, start=start_date, end=end_date, progress=False)
            if not data.empty:
                return data['Close'].pct_change().dropna()
            
        except Exception as e:
            self.logger.warning(f"Failed to get benchmark data: {e}")
        
        return None
    
    def _analyze_sector_momentum(self, sector_data: Dict[str, SectorPerformance]) -> Dict[str, Any]:
        """Analyze momentum characteristics across sectors"""
        
        if not sector_data:
            return {}
        
        # Calculate momentum statistics
        momentum_scores = [s.momentum_score for s in sector_data.values()]
        
        # Identify leaders and laggards
        sorted_sectors = sorted(
            sector_data.items(), 
            key=lambda x: x[1].momentum_score, 
            reverse=True
        )
        
        leaders = sorted_sectors[:3]
        laggards = sorted_sectors[-3:]
        
        # Momentum distribution
        momentum_std = np.std(momentum_scores)
        momentum_mean = np.mean(momentum_scores)
        
        # Trend consistency (sectors with consistent direction)
        consistent_up = []
        consistent_down = []
        
        for sector, perf in sector_data.items():
            if (perf.performance_1w > 0 and perf.performance_1m > 0 and perf.performance_3m > 0):
                consistent_up.append(sector)
            elif (perf.performance_1w < 0 and perf.performance_1m < 0 and perf.performance_3m < 0):
                consistent_down.append(sector)
        
        return {
            "leaders": [(sector, perf.momentum_score) for sector, perf in leaders],
            "laggards": [(sector, perf.momentum_score) for sector, perf in laggards],
            "momentum_mean": momentum_mean,
            "momentum_std": momentum_std,
            "momentum_dispersion": momentum_std / abs(momentum_mean) if momentum_mean != 0 else 0,
            "consistent_uptrend": consistent_up,
            "consistent_downtrend": consistent_down,
            "sector_count": len(sector_data)
        }
    
    def _detect_rotation_signals(
        self, 
        sector_data: Dict[str, SectorPerformance], 
        momentum_analysis: Dict[str, Any]
    ) -> List[str]:
        """Detect sector rotation patterns and signals"""
        
        signals = []
        
        try:
            if not sector_data:
                return signals
            
            # Growth vs Value rotation
            growth_sectors = ['IT', 'Pharma']
            value_sectors = ['Banking', 'Energy', 'Metals']
            
            growth_performance = np.mean([
                sector_data[sector].momentum_score 
                for sector in growth_sectors 
                if sector in sector_data
            ])
            
            value_performance = np.mean([
                sector_data[sector].momentum_score 
                for sector in value_sectors 
                if sector in sector_data
            ])
            
            if growth_performance > value_performance + 3:
                signals.append("Growth sectors outperforming - Growth rotation in progress")
            elif value_performance > growth_performance + 3:
                signals.append("Value sectors outperforming - Value rotation detected")
            
            # Defensive vs Cyclical rotation
            defensive_sectors = ['FMCG', 'Pharma']
            cyclical_sectors = ['Auto', 'Banking', 'Metals']
            
            defensive_perf = np.mean([
                sector_data[sector].momentum_score 
                for sector in defensive_sectors 
                if sector in sector_data
            ])
            
            cyclical_perf = np.mean([
                sector_data[sector].momentum_score 
                for sector in cyclical_sectors 
                if sector in sector_data
            ])
            
            if defensive_perf > cyclical_perf + 2:
                signals.append("Defensive sectors leading - Risk-off rotation")
            elif cyclical_perf > defensive_perf + 2:
                signals.append("Cyclical sectors leading - Risk-on rotation")
            
            # High momentum dispersion
            momentum_dispersion = momentum_analysis.get('momentum_dispersion', 0)
            if momentum_dispersion > 0.5:
                signals.append("High momentum dispersion - Active rotation environment")
            elif momentum_dispersion < 0.2:
                signals.append("Low momentum dispersion - Sector convergence")
            
            # Specific sector breakouts
            for sector, perf in sector_data.items():
                if perf.momentum_score > 10 and perf.performance_1w > 3:
                    signals.append(f"{sector} sector showing strong momentum breakout")
                elif perf.momentum_score < -10 and perf.performance_1w < -3:
                    signals.append(f"{sector} sector showing significant weakness")
            
        except Exception as e:
            self.logger.error(f"Rotation signal detection failed: {e}")
            signals.append("Signal detection temporarily unavailable")
        
        return signals
    
    def _calculate_sector_scores(
        self, 
        sector_data: Dict[str, SectorPerformance], 
        momentum_analysis: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate comprehensive sector attractiveness scores"""
        
        scores = {}
        
        for sector, perf in sector_data.items():
            # Score components (0-100 scale)
            momentum_component = min(max((perf.momentum_score + 20) / 40 * 100, 0), 100)
            
            consistency_component = 50  # Base score
            if perf.performance_1w > 0 and perf.performance_1m > 0:
                consistency_component += 25
            elif perf.performance_1w < 0 and perf.performance_1m < 0:
                consistency_component -= 25
            
            relative_strength_component = min(max((perf.relative_strength + 10) / 20 * 100, 0), 100)
            
            volatility_component = max(100 - perf.volatility, 0)  # Lower volatility is better
            
            # Weighted score
            total_score = (
                momentum_component * 0.4 +
                consistency_component * 0.3 +
                relative_strength_component * 0.2 +
                volatility_component * 0.1
            )
            
            scores[sector] = min(max(total_score, 0), 100)
        
        return scores
    
    def _generate_rotation_recommendations(
        self, 
        sector_scores: Dict[str, float], 
        rotation_signals: List[str], 
        momentum_analysis: Dict[str, Any]
    ) -> List[str]:
        """Generate actionable sector rotation recommendations"""
        
        recommendations = []
        
        try:
            if not sector_scores:
                return ["Insufficient data for recommendations"]
            
            # Top rated sectors
            sorted_scores = sorted(sector_scores.items(), key=lambda x: x[1], reverse=True)
            top_sectors = sorted_scores[:3]
            bottom_sectors = sorted_scores[-2:]
            
            recommendations.append(f"🟢 Overweight: {', '.join([s[0] for s in top_sectors])}")
            recommendations.append(f"🔴 Underweight: {', '.join([s[0] for s in bottom_sectors])}")
            
            # Momentum-based recommendations
            leaders = momentum_analysis.get('leaders', [])
            if leaders:
                top_momentum = leaders[0][0]
                recommendations.append(f"📈 Momentum play: {top_momentum} sector showing strongest momentum")
            
            # Risk-based recommendations
            if any("Risk-off" in signal for signal in rotation_signals):
                recommendations.append("🛡️ Consider defensive positioning in FMCG and Pharma")
            elif any("Risk-on" in signal for signal in rotation_signals):
                recommendations.append("⚡ Cyclical sectors favored - Banking, Auto, Metals attractive")
            
            # Contrarian opportunities
            laggards = momentum_analysis.get('laggards', [])
            if laggards and laggards[-1][1] < -5:  # Significant underperformance
                laggard_sector = laggards[-1][0]
                recommendations.append(f"🔄 Contrarian opportunity: {laggard_sector} may be oversold")
            
        except Exception as e:
            recommendations.append("Recommendation engine temporarily unavailable")
        
        return recommendations
    
    def _determine_market_regime(self, sector_data: Dict[str, SectorPerformance]) -> str:
        """Determine overall market regime from sector performance"""
        
        if not sector_data:
            return "Unknown"
        
        # Calculate overall momentum
        avg_momentum = np.mean([s.momentum_score for s in sector_data.values()])
        momentum_dispersion = np.std([s.momentum_score for s in sector_data.values()])
        
        positive_sectors = sum(1 for s in sector_data.values() if s.momentum_score > 0)
        total_sectors = len(sector_data)
        positive_ratio = positive_sectors / total_sectors
        
        if avg_momentum > 5 and positive_ratio > 0.7:
            return "Broad-based Rally"
        elif avg_momentum < -5 and positive_ratio < 0.3:
            return "Broad-based Decline"
        elif momentum_dispersion > 5:
            return "Active Rotation"
        elif abs(avg_momentum) < 2:
            return "Sideways/Consolidation"
        else:
            return "Mixed Signals"
    
    def _analyze_defensive_cyclical(self, sector_data: Dict[str, SectorPerformance]) -> Dict[str, Any]:
        """Analyze defensive vs cyclical sector performance"""
        
        defensive_sectors = ['FMCG', 'Pharma']
        cyclical_sectors = ['Banking', 'Auto', 'Energy', 'Metals']
        
        defensive_performance = []
        cyclical_performance = []
        
        for sector, perf in sector_data.items():
            if sector in defensive_sectors:
                defensive_performance.append(perf.momentum_score)
            elif sector in cyclical_sectors:
                cyclical_performance.append(perf.momentum_score)
        
        def_avg = np.mean(defensive_performance) if defensive_performance else 0
        cyc_avg = np.mean(cyclical_performance) if cyclical_performance else 0
        
        if def_avg > cyc_avg + 2:
            bias = "Defensive"
            signal = "Risk-off environment"
        elif cyc_avg > def_avg + 2:
            bias = "Cyclical" 
            signal = "Risk-on environment"
        else:
            bias = "Neutral"
            signal = "Balanced market"
        
        return {
            "defensive_avg": def_avg,
            "cyclical_avg": cyc_avg,
            "bias": bias,
            "signal": signal,
            "spread": abs(def_avg - cyc_avg)
        }
    
    def get_sector_heatmap_data(self, lookback_days: int = 90) -> pd.DataFrame:
        """Get data formatted for sector heatmap visualization"""
        
        try:
            analysis = self.analyze_sector_rotation(lookback_days)
            sector_data = analysis.get('sector_performance', {})
            
            if not sector_data:
                return pd.DataFrame()
            
            # Create DataFrame for heatmap
            heatmap_data = []
            for sector, perf in sector_data.items():
                heatmap_data.append({
                    'Sector': sector,
                    '1D': perf.performance_1d,
                    '1W': perf.performance_1w,
                    '1M': perf.performance_1m,
                    '3M': perf.performance_3m,
                    'YTD': perf.performance_ytd
                })
            
            return pd.DataFrame(heatmap_data).set_index('Sector')
            
        except Exception as e:
            self.logger.error(f"Heatmap data generation failed: {e}")
            return pd.DataFrame()
