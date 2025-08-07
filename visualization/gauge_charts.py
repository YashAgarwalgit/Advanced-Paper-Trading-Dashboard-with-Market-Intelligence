"""
Gauge and indicator chart components
"""
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging
import sys
from utils.helpers import safe_float

class GaugeCharts:
    """
    Gauge and indicator chart components
    Features: Risk gauges, performance indicators, market sentiment gauges
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def create_risk_gauge(self, risk_score: float, title: str = "Risk Level") -> go.Figure:
        """Create risk level gauge"""
        
        try:
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=risk_score,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': title},
                delta={'reference': 50},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': self._get_risk_color(risk_score)},
                    'steps': [
                        {'range': [0, 30], 'color': "rgba(0, 255, 0, 0.3)"},
                        {'range': [30, 70], 'color': "rgba(255, 255, 0, 0.3)"},
                        {'range': [70, 100], 'color': "rgba(255, 0, 0, 0.3)"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            
            fig.update_layout(
                height=300,
                template="plotly_dark",
                margin=dict(l=20, r=20, t=50, b=20)
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Risk gauge creation failed: {e}")
            return self._create_simple_gauge(risk_score, title)
    
    def create_performance_gauge(self, performance: float, benchmark: float = 0, title: str = "Performance") -> go.Figure:
        """Create performance gauge with benchmark comparison"""
        
        try:
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=performance,
                number={'suffix': '%'},
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': title},
                delta={'reference': benchmark, 'suffix': '%'},
                gauge={
                    'axis': {'range': [-50, 50]},
                    'bar': {'color': "green" if performance > benchmark else "red"},
                    'steps': [
                        {'range': [-50, -10], 'color': "rgba(255, 0, 0, 0.3)"},
                        {'range': [-10, 10], 'color': "rgba(255, 255, 0, 0.3)"},
                        {'range': [10, 50], 'color': "rgba(0, 255, 0, 0.3)"}
                    ],
                    'threshold': {
                        'line': {'color': "blue", 'width': 4},
                        'thickness': 0.75,
                        'value': benchmark
                    }
                }
            ))
            
            fig.update_layout(
                height=300,
                template="plotly_dark",
                margin=dict(l=20, r=20, t=50, b=20)
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Performance gauge creation failed: {e}")
            return self._create_simple_gauge(performance, title)
    
    def create_vix_gauge(self, vix_value: float) -> go.Figure:
        """Create VIX (volatility) gauge"""
        
        try:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=vix_value,
                title={'text': "India VIX (Fear Index)"},
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [10, 35], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': "rgba(0,0,0,0)"},  # Invisible bar
                    'steps': [
                        {'range': [10, 18], 'color': 'rgba(0, 255, 0, 0.3)'},
                        {'range': [18, 25], 'color': 'rgba(255, 255, 0, 0.3)'},
                        {'range': [25, 35], 'color': 'rgba(255, 0, 0, 0.3)'}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': vix_value
                    }
                }
            ))
            
            fig.update_layout(
                height=250,
                template="plotly_dark",
                margin=dict(l=20, r=20, t=50, b=20)
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"VIX gauge creation failed: {e}")
            return self._create_simple_gauge(vix_value, "India VIX")
    
    def create_sharpe_ratio_gauge(self, sharpe_ratio: float) -> go.Figure:
        """Create Sharpe ratio gauge"""
        
        try:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=sharpe_ratio,
                title={'text': "Sharpe Ratio"},
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [-2, 3]},
                    'bar': {'color': self._get_sharpe_color(sharpe_ratio)},
                    'steps': [
                        {'range': [-2, 0], 'color': "rgba(255, 0, 0, 0.3)"},
                        {'range': [0, 1], 'color': "rgba(255, 255, 0, 0.3)"},
                        {'range': [1, 2], 'color': "rgba(144, 238, 144, 0.3)"},
                        {'range': [2, 3], 'color': "rgba(0, 255, 0, 0.3)"}
                    ],
                    'threshold': {
                        'line': {'color': "blue", 'width': 4},
                        'thickness': 0.75,
                        'value': 1.0
                    }
                }
            ))
            
            fig.update_layout(
                height=250,
                template="plotly_dark",
                margin=dict(l=20, r=20, t=50, b=20)
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Sharpe ratio gauge creation failed: {e}")
            return self._create_simple_gauge(sharpe_ratio, "Sharpe Ratio")
    
    def create_portfolio_health_dashboard(self, portfolio_metrics: Dict[str, float]) -> go.Figure:
        """Create comprehensive portfolio health dashboard"""
        
        try:
            fig = make_subplots(
                rows=2, cols=3,
                specs=[
                    [{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}],
                    [{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}]
                ],
                subplot_titles=[
                    'Total Return', 'Volatility', 'Sharpe Ratio',
                    'Max Drawdown', 'Win Rate', 'Risk Score'
                ],
                vertical_spacing=0.15
            )
            
            # Total Return
            total_return = portfolio_metrics.get('total_return', 0) * 100
            fig.add_trace(
                go.Indicator(
                    mode="number+delta",
                    value=total_return,
                    number={'suffix': '%'},
                    title={"text": "Total Return"},
                    delta={'reference': 0, 'position': "bottom"},
                    domain={'x': [0, 1], 'y': [0, 1]}
                ),
                row=1, col=1
            )
            
            # Volatility
            volatility = portfolio_metrics.get('volatility', 0) * 100
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=volatility,
                    number={'suffix': '%'},
                    title={"text": "Volatility"},
                    gauge={
                        'axis': {'range': [0, 50]},
                        'bar': {'color': self._get_volatility_color(volatility)},
                        'steps': [
                            {'range': [0, 15], 'color': "rgba(0, 255, 0, 0.2)"},
                            {'range': [15, 30], 'color': "rgba(255, 255, 0, 0.2)"},
                            {'range': [30, 50], 'color': "rgba(255, 0, 0, 0.2)"}
                        ]
                    },
                    domain={'x': [0, 1], 'y': [0, 1]}
                ),
                row=1, col=2
            )
            
            # Sharpe Ratio
            sharpe = portfolio_metrics.get('sharpe_ratio', 0)
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=sharpe,
                    title={"text": "Sharpe Ratio"},
                    gauge={
                        'axis': {'range': [-2, 3]},
                        'bar': {'color': self._get_sharpe_color(sharpe)},
                        'steps': [
                            {'range': [-2, 0], 'color': "rgba(255, 0, 0, 0.2)"},
                            {'range': [0, 1], 'color': "rgba(255, 255, 0, 0.2)"},
                            {'range': [1, 3], 'color': "rgba(0, 255, 0, 0.2)"}
                        ]
                    },
                    domain={'x': [0, 1], 'y': [0, 1]}
                ),
                row=1, col=3
            )
            
            # Max Drawdown
            max_drawdown = abs(portfolio_metrics.get('max_drawdown', 0)) * 100
            fig.add_trace(
                go.Indicator(
                    mode="number",
                    value=max_drawdown,
                    number={'suffix': '%'},
                    title={"text": "Max Drawdown"},
                    domain={'x': [0, 1], 'y': [0, 1]}
                ),
                row=2, col=1
            )
            
            # Win Rate
            win_rate = portfolio_metrics.get('win_rate', 0.5) * 100
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=win_rate,
                    number={'suffix': '%'},
                    title={"text": "Win Rate"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': self._get_win_rate_color(win_rate)},
                        'steps': [
                            {'range': [0, 40], 'color': "rgba(255, 0, 0, 0.2)"},
                            {'range': [40, 60], 'color': "rgba(255, 255, 0, 0.2)"},
                            {'range': [60, 100], 'color': "rgba(0, 255, 0, 0.2)"}
                        ]
                    },
                    domain={'x': [0, 1], 'y': [0, 1]}
                ),
                row=2, col=2
            )
            
            # Overall Risk Score
            risk_score = self._calculate_overall_risk_score(portfolio_metrics)
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=risk_score,
                    title={"text": "Risk Score"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': self._get_risk_color(risk_score)},
                        'steps': [
                            {'range': [0, 30], 'color': "rgba(0, 255, 0, 0.2)"},
                            {'range': [30, 70], 'color': "rgba(255, 255, 0, 0.2)"},
                            {'range': [70, 100], 'color': "rgba(255, 0, 0, 0.2)"}
                        ]
                    },
                    domain={'x': [0, 1], 'y': [0, 1]}
                ),
                row=2, col=3
            )
            
            fig.update_layout(
                height=500,
                template="plotly_dark",
                title_text="Portfolio Health Dashboard",
                title_x=0.5,
                showlegend=False
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Portfolio health dashboard creation failed: {e}")
            return self._create_error_chart("Health Dashboard Unavailable")
    
    def create_market_sentiment_gauge(self, sentiment_score: float) -> go.Figure:
        """Create market sentiment gauge"""
        
        try:
            sentiment_label = self._get_sentiment_label(sentiment_score)
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=sentiment_score,
                title={'text': f"Market Sentiment<br><span style='font-size:0.8em'>{sentiment_label}</span>"},
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': self._get_sentiment_color(sentiment_score)},
                    'steps': [
                        {'range': [0, 25], 'color': "rgba(255, 0, 0, 0.3)"},
                        {'range': [25, 40], 'color': "rgba(255, 165, 0, 0.3)"},
                        {'range': [40, 60], 'color': "rgba(255, 255, 0, 0.3)"},
                        {'range': [60, 75], 'color': "rgba(144, 238, 144, 0.3)"},
                        {'range': [75, 100], 'color': "rgba(0, 255, 0, 0.3)"}
                    ],
                    'threshold': {
                        'line': {'color': "blue", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            
            fig.update_layout(
                height=300,
                template="plotly_dark",
                margin=dict(l=20, r=20, t=50, b=20)
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Market sentiment gauge creation failed: {e}")
            return self._create_simple_gauge(sentiment_score, "Market Sentiment")
    
    def create_leverage_gauge(self, leverage_ratio: float, max_leverage: float = 3.0) -> go.Figure:
        """Create leverage utilization gauge"""
        
        try:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=leverage_ratio,
                number={'suffix': 'x'},
                title={'text': "Leverage Ratio"},
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [0, max_leverage]},
                    'bar': {'color': self._get_leverage_color(leverage_ratio, max_leverage)},
                    'steps': [
                        {'range': [0, max_leverage * 0.5], 'color': "rgba(0, 255, 0, 0.3)"},
                        {'range': [max_leverage * 0.5, max_leverage * 0.8], 'color': "rgba(255, 255, 0, 0.3)"},
                        {'range': [max_leverage * 0.8, max_leverage], 'color': "rgba(255, 0, 0, 0.3)"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': max_leverage * 0.9
                    }
                }
            ))
            
            fig.update_layout(
                height=250,
                template="plotly_dark",
                margin=dict(l=20, r=20, t=50, b=20)
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Leverage gauge creation failed: {e}")
            return self._create_simple_gauge(leverage_ratio, "Leverage")
    
    def create_var_gauge(self, var_value: float, confidence_level: int = 95) -> go.Figure:
        """Create Value at Risk gauge"""
        
        try:
            # Convert to positive percentage for display
            var_display = abs(var_value) * 100
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=var_display,
                number={'suffix': '%'},
                title={'text': f"VaR {confidence_level}%<br>Daily Risk"},
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [0, 10]},  # 0-10% range
                    'bar': {'color': self._get_var_color(var_display)},
                    'steps': [
                        {'range': [0, 2], 'color': "rgba(0, 255, 0, 0.3)"},
                        {'range': [2, 5], 'color': "rgba(255, 255, 0, 0.3)"},
                        {'range': [5, 10], 'color': "rgba(255, 0, 0, 0.3)"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 5.0
                    }
                }
            ))
            
            fig.update_layout(
                height=250,
                template="plotly_dark",
                margin=dict(l=20, r=20, t=50, b=20)
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"VaR gauge creation failed: {e}")
            return self._create_simple_gauge(var_display, "VaR")
    
    def create_concentration_gauge(self, concentration_score: float) -> go.Figure:
        """Create portfolio concentration gauge"""
        
        try:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=concentration_score * 100,
                number={'suffix': '%'},
                title={'text': "Position Concentration<br>Largest Holding"},
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [0, 50]},
                    'bar': {'color': self._get_concentration_color(concentration_score)},
                    'steps': [
                        {'range': [0, 10], 'color': "rgba(0, 255, 0, 0.3)"},
                        {'range': [10, 25], 'color': "rgba(255, 255, 0, 0.3)"},
                        {'range': [25, 50], 'color': "rgba(255, 0, 0, 0.3)"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 20
                    }
                }
            ))
            
            fig.update_layout(
                height=250,
                template="plotly_dark",
                margin=dict(l=20, r=20, t=50, b=20)
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Concentration gauge creation failed: {e}")
            return self._create_simple_gauge(concentration_score * 100, "Concentration")
    
    def create_multi_gauge_dashboard(self, metrics: Dict[str, float], title: str = "Portfolio Metrics") -> go.Figure:
        """Create multi-gauge dashboard with custom metrics"""
        
        try:
            # Dynamically create subplots based on number of metrics
            n_metrics = len(metrics)
            if n_metrics <= 3:
                rows, cols = 1, n_metrics
            elif n_metrics <= 6:
                rows, cols = 2, 3
            else:
                rows, cols = 3, 3
            
            # Create subplot specs
            specs = [[{"type": "indicator"} for _ in range(cols)] for _ in range(rows)]
            
            fig = make_subplots(
                rows=rows, cols=cols,
                specs=specs,
                subplot_titles=list(metrics.keys()),
                vertical_spacing=0.15,
                horizontal_spacing=0.1
            )
            
            # Add gauges
            for i, (metric_name, value) in enumerate(metrics.items()):
                row = (i // cols) + 1
                col = (i % cols) + 1
                
                # Determine gauge configuration based on metric name
                gauge_config = self._get_gauge_config_for_metric(metric_name, value)
                
                fig.add_trace(
                    go.Indicator(
                        mode="gauge+number",
                        value=value,
                        title={"text": metric_name},
                        **gauge_config,
                        domain={'x': [0, 1], 'y': [0, 1]}
                    ),
                    row=row, col=col
                )
            
            fig.update_layout(
                height=200 * rows,
                template="plotly_dark",
                title_text=title,
                title_x=0.5,
                showlegend=False
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Multi-gauge dashboard creation failed: {e}")
            return self._create_error_chart("Multi-Gauge Dashboard Unavailable")
    
    # Helper Methods
    def _get_risk_color(self, risk_score: float) -> str:
        """Get color based on risk score"""
        if risk_score <= 30:
            return "green"
        elif risk_score <= 70:
            return "orange"
        else:
            return "red"
    
    def _get_sharpe_color(self, sharpe_ratio: float) -> str:
        """Get color based on Sharpe ratio"""
        if sharpe_ratio >= 1.5:
            return "green"
        elif sharpe_ratio >= 0.5:
            return "orange"
        else:
            return "red"
    
    def _get_volatility_color(self, volatility: float) -> str:
        """Get color based on volatility percentage"""
        if volatility <= 15:
            return "green"
        elif volatility <= 30:
            return "orange"
        else:
            return "red"
    
    def _get_win_rate_color(self, win_rate: float) -> str:
        """Get color based on win rate percentage"""
        if win_rate >= 60:
            return "green"
        elif win_rate >= 40:
            return "orange"
        else:
            return "red"
    
    def _get_sentiment_color(self, sentiment_score: float) -> str:
        """Get color based on sentiment score"""
        if sentiment_score >= 75:
            return "lightgreen"
        elif sentiment_score >= 60:
            return "green"
        elif sentiment_score >= 40:
            return "orange"
        elif sentiment_score >= 25:
            return "red"
        else:
            return "darkred"
    
    def _get_leverage_color(self, leverage_ratio: float, max_leverage: float) -> str:
        """Get color based on leverage ratio"""
        utilization = leverage_ratio / max_leverage
        if utilization <= 0.5:
            return "green"
        elif utilization <= 0.8:
            return "orange"
        else:
            return "red"
    
    def _get_var_color(self, var_percentage: float) -> str:
        """Get color based on VaR percentage"""
        if var_percentage <= 2:
            return "green"
        elif var_percentage <= 5:
            return "orange"
        else:
            return "red"
    
    def _get_concentration_color(self, concentration: float) -> str:
        """Get color based on concentration ratio"""
        concentration_pct = concentration * 100
        if concentration_pct <= 10:
            return "green"
        elif concentration_pct <= 25:
            return "orange"
        else:
            return "red"
    
    def _get_sentiment_label(self, sentiment_score: float) -> str:
        """Get sentiment label from score"""
        if sentiment_score >= 80:
            return "Extreme Greed"
        elif sentiment_score >= 60:
            return "Greed"
        elif sentiment_score >= 40:
            return "Neutral"
        elif sentiment_score >= 20:
            return "Fear"
        else:
            return "Extreme Fear"
    
    def _calculate_overall_risk_score(self, portfolio_metrics: Dict[str, float]) -> float:
        """Calculate overall risk score from portfolio metrics"""
        
        try:
            risk_factors = []
            
            # Volatility component
            volatility = portfolio_metrics.get('volatility', 0) * 100
            vol_score = min(volatility / 0.5, 100)  # Normalize to 0-100
            risk_factors.append(vol_score)
            
            # Drawdown component
            max_drawdown = abs(portfolio_metrics.get('max_drawdown', 0)) * 100
            dd_score = min(max_drawdown / 0.3, 100)  # 30% max drawdown = 100 risk
            risk_factors.append(dd_score)
            
            # Sharpe ratio component (inverted - lower Sharpe = higher risk)
            sharpe = portfolio_metrics.get('sharpe_ratio', 0)
            sharpe_score = max(0, 100 - (sharpe + 2) / 5 * 100)  # Normalize -2 to 3 range
            risk_factors.append(sharpe_score)
            
            # Average risk score
            overall_risk = sum(risk_factors) / len(risk_factors) if risk_factors else 50
            return min(100, max(0, overall_risk))
            
        except Exception as e:
            self.logger.error(f"Risk score calculation failed: {e}")
            return 50.0  # Default neutral risk
    
    def _get_gauge_config_for_metric(self, metric_name: str, value: float) -> Dict[str, Any]:
        """Get gauge configuration based on metric type"""
        
        metric_configs = {
            'sharpe_ratio': {
                'gauge': {
                    'axis': {'range': [-2, 3]},
                    'bar': {'color': self._get_sharpe_color(value)},
                    'steps': [
                        {'range': [-2, 0], 'color': "rgba(255, 0, 0, 0.2)"},
                        {'range': [0, 1], 'color': "rgba(255, 255, 0, 0.2)"},
                        {'range': [1, 3], 'color': "rgba(0, 255, 0, 0.2)"}
                    ]
                }
            },
            'volatility': {
                'number': {'suffix': '%'},
                'gauge': {
                    'axis': {'range': [0, 50]},
                    'bar': {'color': self._get_volatility_color(value)},
                    'steps': [
                        {'range': [0, 15], 'color': "rgba(0, 255, 0, 0.2)"},
                        {'range': [15, 30], 'color': "rgba(255, 255, 0, 0.2)"},
                        {'range': [30, 50], 'color': "rgba(255, 0, 0, 0.2)"}
                    ]
                }
            },
            'return': {
                'number': {'suffix': '%'},
                'gauge': {
                    'axis': {'range': [-50, 50]},
                    'bar': {'color': "green" if value > 0 else "red"},
                    'steps': [
                        {'range': [-50, -10], 'color': "rgba(255, 0, 0, 0.2)"},
                        {'range': [-10, 10], 'color': "rgba(255, 255, 0, 0.2)"},
                        {'range': [10, 50], 'color': "rgba(0, 255, 0, 0.2)"}
                    ]
                }
            }
        }
        
        # Default configuration
        default_config = {
            'gauge': {
                'axis': {'range': [0, 100]},
                'bar': {'color': "blue"},
                'steps': [
                    {'range': [0, 33], 'color': "rgba(255, 0, 0, 0.2)"},
                    {'range': [33, 66], 'color': "rgba(255, 255, 0, 0.2)"},
                    {'range': [66, 100], 'color': "rgba(0, 255, 0, 0.2)"}
                ]
            }
        }
        
        # Find best match for metric name
        for key, config in metric_configs.items():
            if key.lower() in metric_name.lower():
                return config
        
        return default_config
    
    def _create_simple_gauge(self, value: float, title: str) -> go.Figure:
        """Create simple fallback gauge"""
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=value,
            title={'text': title},
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "blue"},
                'steps': [
                    {'range': [0, 50], 'color': "rgba(255, 255, 255, 0.2)"},
                    {'range': [50, 100], 'color': "rgba(0, 0, 255, 0.2)"}
                ]
            }
        ))
        
        fig.update_layout(
            height=300,
            template="plotly_dark",
            margin=dict(l=20, r=20, t=50, b=20)
        )
        
        return fig
    
    def _create_error_chart(self, message: str) -> go.Figure:
        """Create error chart with message"""
        
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color="orange")
        )
        fig.update_layout(
            template="plotly_dark",
            height=400,
            showlegend=False
        )
        return fig
