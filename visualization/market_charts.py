"""
Market intelligence chart components
"""
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import logging
import sys
from market.sector_rotation import SectorRotationAnalyzer
from utils.helpers import safe_float, format_percentage

class MarketCharts:
    """
    Market intelligence visualization components
    Features: Sector heatmaps, regime charts, correlation analysis
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.sector_analyzer = SectorRotationAnalyzer()
        self.color_scales = {
            'performance': 'RdYlGn',
            'correlation': 'RdYlBu', 
            'volatility': 'Reds'
        }
    
    def create_sector_rotation_heatmap(self, lookback_days: int = 90) -> go.Figure:
        """Create sector rotation performance heatmap"""
        
        try:
            # Get sector heatmap data
            heatmap_data = self.sector_analyzer.get_sector_heatmap_data(lookback_days)
            
            if heatmap_data.empty:
                return self._create_error_chart("Sector Data Unavailable")
            
            fig = go.Figure(data=go.Heatmap(
                z=heatmap_data.values,
                x=heatmap_data.columns,
                y=heatmap_data.index,
                colorscale='RdYlGn',
                zmid=0,
                text=heatmap_data.round(2).values,
                texttemplate="%{text}%",
                textfont={"size": 10},
                hovertemplate='<b>%{y}</b><br>%{x}: %{z:.2f}%<extra></extra>',
                colorbar=dict(title="Performance %")
            ))
            
            fig.update_layout(
                title="Sector Performance Heatmap",
                xaxis_title="Time Period",
                yaxis_title="Sectors",
                template="plotly_dark",
                height=500,
                width=800
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Sector heatmap creation failed: {e}")
            return self._create_error_chart("Sector Heatmap Unavailable")
    
    def create_market_regime_chart(self, regime_data: Dict[str, Any]) -> go.Figure:
        """Create market regime visualization"""
        
        try:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=[
                    'Market Regime Score',
                    'Regime Components',
                    'Volatility Clustering',
                    'Market Stress Level'
                ],
                specs=[
                    [{"type": "indicator"}, {"type": "domain"}],
                    [{"type": "xy"}, {"type": "indicator"}]
                ]
            )
            
            # Regime score gauge
            regime_score = regime_data.get('regime_score', 5.0)
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=regime_score,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Regime Score"},
                    delta={'reference': 5.0},
                    gauge={
                        'axis': {'range': [0, 10]},
                        'bar': {'color': self._get_regime_color(regime_score)},
                        'steps': [
                            {'range': [0, 3], 'color': "rgba(255, 0, 0, 0.3)"},
                            {'range': [3, 7], 'color': "rgba(255, 255, 0, 0.3)"},
                            {'range': [7, 10], 'color': "rgba(0, 255, 0, 0.3)"}
                        ],
                        'threshold': {
                            'line': {'color': "black", 'width': 4},
                            'thickness': 0.75,
                            'value': regime_score
                        }
                    }
                ),
                row=1, col=1
            )
            
            # Regime components pie chart
            components = regime_data.get('components', {})
            if components:
                fig.add_trace(
                    go.Pie(
                        labels=list(components.keys()),
                        values=list(components.values()),
                        name="Components",
                        hole=0.3
                    ),
                    row=1, col=2
                )
            
            # Mock volatility clustering
            x_data = list(range(30))
            volatility_data = np.random.normal(0.02, 0.01, 30)  # Mock data
            volatility_data = np.abs(volatility_data)  # Ensure positive
            
            fig.add_trace(
                go.Scatter(
                    x=x_data,
                    y=volatility_data * 100,
                    name='Daily Volatility',
                    line=dict(color='orange', width=2),
                    hovertemplate='Day: %{x}<br>Volatility: %{y:.2f}%<extra></extra>'
                ),
                row=2, col=1
            )
            
            # Market stress indicator
            stress_level = regime_data.get('market_stress_level', 0.3)
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=stress_level * 100,
                    title={'text': "Market Stress"},
                    number={'suffix': '%'},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "red" if stress_level > 0.7 else "orange" if stress_level > 0.4 else "green"},
                        'steps': [
                            {'range': [0, 40], 'color': "rgba(0, 255, 0, 0.3)"},
                            {'range': [40, 70], 'color': "rgba(255, 255, 0, 0.3)"},
                            {'range': [70, 100], 'color': "rgba(255, 0, 0, 0.3)"}
                        ]
                    },
                    domain={'x': [0, 1], 'y': [0, 1]}
                ),
                row=2, col=2
            )
            
            fig.update_layout(
                height=700,
                showlegend=False,
                template="plotly_dark",
                title_text="Market Regime Analysis",
                title_x=0.5
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Market regime chart creation failed: {e}")
            return self._create_error_chart("Market Regime Chart Unavailable")
    
    def create_cross_asset_correlation(self, correlation_data: Dict[str, Any]) -> go.Figure:
        """Create cross-asset correlation analysis"""
        
        try:
            correlation_matrix = correlation_data.get('correlation_matrix', {})
            
            if not correlation_matrix:
                # Create mock correlation matrix
                assets = ['Equities', 'Bonds', 'Gold', 'USD/INR', 'Oil']
                np.random.seed(42)
                corr_values = np.random.uniform(-0.5, 0.8, (len(assets), len(assets)))
                corr_values = (corr_values + corr_values.T) / 2
                np.fill_diagonal(corr_values, 1.0)
                
                correlation_matrix = pd.DataFrame(corr_values, index=assets, columns=assets)
            elif isinstance(correlation_matrix, dict):
                # Convert dict to DataFrame
                df_data = []
                for asset1, correlations in correlation_matrix.items():
                    if isinstance(correlations, dict):
                        df_data.append(correlations)
                
                if df_data:
                    correlation_matrix = pd.DataFrame(df_data, index=correlation_matrix.keys())
                else:
                    return self._create_error_chart("Invalid Correlation Data")
            
            fig = go.Figure(data=go.Heatmap(
                z=correlation_matrix.values,
                x=correlation_matrix.columns,
                y=correlation_matrix.index,
                colorscale='RdYlBu',
                zmid=0,
                text=correlation_matrix.round(3).values,
                texttemplate="%{text}",
                textfont={"size": 10},
                hovertemplate='<b>%{y} vs %{x}</b><br>Correlation: %{z:.3f}<extra></extra>',
                colorbar=dict(title="Correlation")
            ))
            
            fig.update_layout(
                title="Cross-Asset Correlation Matrix",
                template="plotly_dark",
                height=500,
                width=600
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Correlation chart creation failed: {e}")
            return self._create_error_chart("Correlation Chart Unavailable")
    
    def create_sentiment_analysis_chart(self, sentiment_data: Dict[str, Any]) -> go.Figure:
        """Create market sentiment analysis visualization"""
        
        try:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=[
                    'Fear & Greed Index',
                    'Market Breadth',
                    'Sentiment Components',
                    'Sentiment Trend'
                ],
                specs=[
                    [{"type": "indicator"}, {"type": "indicator"}],
                    [{"type": "domain"}, {"type": "xy"}]
                ]
            )
            
            # Fear & Greed Index
            sentiment_score = sentiment_data.get('sentiment_score', 50)
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=sentiment_score,
                    title={'text': "Fear & Greed"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': self._get_sentiment_color(sentiment_score)},
                        'steps': [
                            {'range': [0, 25], 'color': "rgba(255, 0, 0, 0.3)"},
                            {'range': [25, 75], 'color': "rgba(255, 255, 0, 0.3)"},
                            {'range': [75, 100], 'color': "rgba(0, 255, 0, 0.3)"}
                        ]
                    },
                    domain={'x': [0, 1], 'y': [0, 1]}
                ),
                row=1, col=1
            )
            
            # Market Breadth
            breadth_ratio = sentiment_data.get('components', {}).get('market_breadth', {}).get('breadth_ratio', 0.5)
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=breadth_ratio * 100,
                    title={'text': "Market Breadth"},
                    number={'suffix': '%'},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "green" if breadth_ratio > 0.6 else "red" if breadth_ratio < 0.4 else "orange"}
                    },
                    domain={'x': [0, 1], 'y': [0, 1]}
                ),
                row=1, col=2
            )
            
            # Sentiment Components
            components = sentiment_data.get('components', {})
            if components:
                component_names = []
                component_values = []
                
                for comp_name, comp_data in components.items():
                    if isinstance(comp_data, dict):
                        score_key = next((k for k in comp_data.keys() if 'score' in k.lower()), None)
                        if score_key:
                            component_names.append(comp_name.replace('_', ' ').title())
                            component_values.append(comp_data[score_key])
                
                if component_names and component_values:
                    fig.add_trace(
                        go.Pie(
                            labels=component_names,
                            values=component_values,
                            name="Components",
                            hole=0.3
                        ),
                        row=2, col=1
                    )
            
            # Sentiment Trend (mock data)
            x_data = list(range(30))
            sentiment_trend = np.random.normal(sentiment_score, 10, 30)  # Mock trend around current sentiment
            sentiment_trend = np.clip(sentiment_trend, 0, 100)  # Ensure 0-100 range
            
            fig.add_trace(
                go.Scatter(
                    x=x_data,
                    y=sentiment_trend,
                    name='Sentiment Trend',
                    line=dict(color='blue', width=2),
                    hovertemplate='Day: %{x}<br>Sentiment: %{y:.1f}<extra></extra>'
                ),
                row=2, col=2
            )
            
            fig.update_layout(
                height=600,
                showlegend=False,
                template="plotly_dark",
                title_text="Market Sentiment Analysis",
                title_x=0.5
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Sentiment chart creation failed: {e}")
            return self._create_error_chart("Sentiment Analysis Unavailable")
    
    def create_factor_analysis_chart(self, factor_data: Dict[str, Any]) -> go.Figure:
        """Create factor analysis visualization"""
        
        try:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=[
                    'Sector Exposure',
                    'Style Factors',
                    'Factor Performance',
                    'Risk Attribution'
                ],
                specs=[
                    [{"type": "domain"}, {"type": "domain"}],
                    [{"type": "xy"}, {"type": "domain"}]
                ]
            )
            
            # Sector Exposure
            sector_exposure = factor_data.get('sector_exposure', {})
            if sector_exposure:
                fig.add_trace(
                    go.Pie(
                        labels=list(sector_exposure.keys()),
                        values=list(sector_exposure.values()),
                        name="Sectors",
                        hole=0.3
                    ),
                    row=1, col=1
                )
            
            # Style Factors
            style_exposure = factor_data.get('style_exposure', {})
            if not style_exposure:
                # Mock style factors
                style_exposure = {
                    'Value': 0.3, 'Growth': 0.4, 'Quality': 0.2, 
                    'Momentum': 0.1, 'Low Vol': 0.15, 'Size': 0.8
                }
            
            fig.add_trace(
                go.Pie(
                    labels=list(style_exposure.keys()),
                    values=list(style_exposure.values()),
                    name="Style Factors",
                    hole=0.3
                ),
                row=1, col=2
            )
            
            # Factor Performance (mock data)
            factors = list(style_exposure.keys())
            performance = np.random.normal(0.05, 0.15, len(factors))  # Mock performance data
            
            fig.add_trace(
                go.Bar(
                    x=factors,
                    y=performance * 100,
                    name='Factor Performance',
                    marker_color=['green' if p > 0 else 'red' for p in performance],
                    hovertemplate='<b>%{x}</b><br>Performance: %{y:.2f}%<extra></extra>'
                ),
                row=2, col=1
            )
            
            # Risk Attribution
            risk_sources = {'Sector Risk': 40, 'Style Risk': 30, 'Specific Risk': 30}
            fig.add_trace(
                go.Pie(
                    labels=list(risk_sources.keys()),
                    values=list(risk_sources.values()),
                    name="Risk Sources",
                    hole=0.3
                ),
                row=2, col=2
            )
            
            fig.update_layout(
                height=600,
                showlegend=False,
                template="plotly_dark",
                title_text="Factor Analysis Dashboard",
                title_x=0.5
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Factor analysis chart creation failed: {e}")
            return self._create_error_chart("Factor Analysis Unavailable")
    
    def _get_regime_color(self, score: float) -> str:
        """Get color based on regime score"""
        if score >= 7:
            return "green"
        elif score >= 4:
            return "orange"
        else:
            return "red"
    
    def _get_sentiment_color(self, score: float) -> str:
        """Get color based on sentiment score"""
        if score >= 75:
            return "lightgreen"
        elif score >= 60:
            return "yellow"
        elif score >= 40:
            return "orange"
        else:
            return "red"
    
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
        fig.update_layout(template="plotly_dark", height=400)
        return fig
