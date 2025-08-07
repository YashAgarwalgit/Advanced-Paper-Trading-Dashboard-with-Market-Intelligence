"""
Risk dashboard visualization components
"""
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging
import sys
from scipy.stats import norm
from utils.helpers import safe_float

class RiskDashboard:
    """
    Risk visualization dashboard components
    Features: VaR charts, drawdown analysis, correlation heatmaps
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def create_risk_overview(self, portfolio_data: Dict[str, Any]) -> go.Figure:
        """Create comprehensive risk dashboard"""
        
        try:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=[
                    'Value at Risk Distribution',
                    'Risk Decomposition',
                    'Drawdown Analysis',
                    'Correlation Matrix'
                ],
                specs=[
                    [{"type": "xy"}, {"type": "domain"}],
                    [{"type": "xy"}, {"type": "xy"}]
                ],
                vertical_spacing=0.1
            )
            
            # VaR Distribution
            equity_history = portfolio_data.get('equity_history', [])
            if len(equity_history) > 10:
                returns = self._calculate_returns(equity_history)
                
                # Create return distribution
                fig.add_trace(
                    go.Histogram(
                        x=returns * 100,
                        nbinsx=30,
                        name="Return Distribution",
                        opacity=0.7,
                        marker_color='lightblue'
                    ),
                    row=1, col=1
                )
                
                # Add VaR lines
                var_95 = np.percentile(returns, 5) * 100
                var_99 = np.percentile(returns, 1) * 100
                
                fig.add_vline(x=var_95, line_dash="dash", line_color="orange", 
                            annotation_text="VaR 95%", row=1, col=1)
                fig.add_vline(x=var_99, line_dash="dash", line_color="red", 
                            annotation_text="VaR 99%", row=1, col=1)
            
            # Risk Decomposition
            positions = portfolio_data.get('positions', {})
            if positions:
                risk_data = self._calculate_risk_decomposition(positions)
                
                fig.add_trace(
                    go.Pie(
                        labels=list(risk_data.keys()),
                        values=list(risk_data.values()),
                        name="Risk Sources",
                        hole=0.3
                    ),
                    row=1, col=2
                )
            
            # Drawdown Analysis
            if len(equity_history) > 1:
                drawdown_data = self._calculate_drawdown(equity_history)
                
                fig.add_trace(
                    go.Scatter(
                        x=range(len(drawdown_data)),
                        y=drawdown_data * 100,
                        name='Drawdown %',
                        fill='tonexty',
                        line=dict(color='red'),
                        hovertemplate='Drawdown: %{y:.2f}%<extra></extra>'
                    ),
                    row=2, col=1
                )
            
            # Correlation Matrix (simplified)
            if len(positions) > 1:
                corr_matrix = self._create_correlation_matrix(positions)
                
                fig.add_trace(
                    go.Heatmap(
                        z=corr_matrix.values,
                        x=corr_matrix.columns,
                        y=corr_matrix.index,
                        colorscale='RdYlBu',
                        zmid=0,
                        showscale=True
                    ),
                    row=2, col=2
                )
            
            fig.update_layout(
                height=700,
                showlegend=False,
                template="plotly_dark",
                title_text="Risk Analysis Dashboard",
                title_x=0.5
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Risk overview creation failed: {e}")
            return self._create_error_chart("Risk Dashboard Unavailable")
    
    def create_var_analysis(self, portfolio_data: Dict[str, Any], confidence_levels: List[float] = None) -> go.Figure:
        """Create detailed VaR analysis"""
        
        try:
            if confidence_levels is None:
                confidence_levels = [0.95, 0.99]
            
            equity_history = portfolio_data.get('equity_history', [])
            if len(equity_history) < 30:
                return self._create_error_chart("Insufficient Data for VaR Analysis")
            
            returns = self._calculate_returns(equity_history)
            
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=['Return Distribution with VaR', 'VaR Time Series'],
                vertical_spacing=0.15
            )
            
            # Distribution with VaR
            fig.add_trace(
                go.Histogram(
                    x=returns * 100,
                    nbinsx=50,
                    name="Daily Returns",
                    opacity=0.7,
                    marker_color='lightblue',
                    hovertemplate='Return: %{x:.2f}%<br>Frequency: %{y}<extra></extra>'
                ),
                row=1, col=1
            )
            
            # Add VaR lines and statistics
            for confidence in confidence_levels:
                alpha = 1 - confidence
                historical_var = np.percentile(returns, alpha * 100) * 100
                parametric_var = norm.ppf(alpha, returns.mean(), returns.std()) * 100
                
                fig.add_vline(x=historical_var, line_dash="dash", 
                            line_color="red" if confidence == 0.99 else "orange",
                            annotation_text=f"VaR {int(confidence*100)}%: {historical_var:.2f}%",
                            row=1, col=1)
            
            # Rolling VaR
            if len(returns) > 30:
                rolling_var = []
                window = 30
                
                for i in range(window, len(returns)):
                    window_returns = returns[i-window:i]
                    var_95 = np.percentile(window_returns, 5) * 100
                    rolling_var.append(var_95)
                
                fig.add_trace(
                    go.Scatter(
                        x=list(range(window, len(returns))),
                        y=rolling_var,
                        name='Rolling VaR 95%',
                        line=dict(color='red', width=2),
                        hovertemplate='Day: %{x}<br>VaR: %{y:.2f}%<extra></extra>'
                    ),
                    row=2, col=1
                )
            
            fig.update_layout(
                height=600,
                template="plotly_dark",
                title_text="Value at Risk Analysis"
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"VaR analysis creation failed: {e}")
            return self._create_error_chart("VaR Analysis Unavailable")
    
    def create_drawdown_analysis(self, portfolio_data: Dict[str, Any]) -> go.Figure:
        """Create detailed drawdown analysis"""
        
        try:
            equity_history = portfolio_data.get('equity_history', [])
            if len(equity_history) < 2:
                return self._create_error_chart("Insufficient Data for Drawdown Analysis")
            
            df = pd.DataFrame(equity_history)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
            
            # Calculate drawdown
            values = df['total_value'].values
            peak = np.maximum.accumulate(values)
            drawdown = (values - peak) / peak
            
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=['Portfolio Value with Peaks', 'Drawdown Over Time'],
                shared_xaxes=True,
                vertical_spacing=0.1
            )
            
            # Portfolio value with peaks
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=values,
                    name='Portfolio Value',
                    line=dict(color='#00d4aa', width=2),
                    hovertemplate='<b>%{x}</b><br>Value: ₹%{y:,.0f}<extra></extra>'
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=peak,
                    name='Peak Value',
                    line=dict(color='orange', width=1, dash='dot'),
                    hovertemplate='<b>%{x}</b><br>Peak: ₹%{y:,.0f}<extra></extra>'
                ),
                row=1, col=1
            )
            
            # Drawdown
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=drawdown * 100,
                    name='Drawdown %',
                    fill='tonexty',
                    line=dict(color='red'),
                    hovertemplate='<b>%{x}</b><br>Drawdown: %{y:.2f}%<extra></extra>'
                ),
                row=2, col=1
            )
            
            # Add max drawdown line
            max_dd = drawdown.min() * 100
            fig.add_hline(y=max_dd, line_dash="dash", line_color="darkred",
                         annotation_text=f"Max DD: {max_dd:.2f}%", row=2, col=1)
            
            fig.update_layout(
                height=500,
                template="plotly_dark",
                title_text="Drawdown Analysis",
                hovermode='x unified'
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Drawdown analysis creation failed: {e}")
            return self._create_error_chart("Drawdown Analysis Unavailable")
    
    def create_risk_metrics_summary(self, portfolio_data: Dict[str, Any]) -> go.Figure:
        """Create risk metrics summary dashboard"""
        
        try:
            equity_history = portfolio_data.get('equity_history', [])
            if len(equity_history) < 10:
                return self._create_error_chart("Insufficient Data for Risk Metrics")
            
            # Calculate risk metrics
            returns = self._calculate_returns(equity_history)
            risk_metrics = self._calculate_comprehensive_risk_metrics(returns, equity_history)
            
            # Create indicators
            fig = make_subplots(
                rows=2, cols=3,
                specs=[
                    [{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}],
                    [{"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"}]
                ],
                subplot_titles=[
                    'Volatility (Ann.)', 'VaR 95%', 'Max Drawdown',
                    'Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio'
                ]
            )
            
            # Volatility
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=risk_metrics['volatility'],
                    number={'suffix': '%'},
                    title={"text": "Volatility"},
                    gauge={'axis': {'range': [0, 50]}, 'bar': {'color': "orange"}},
                    domain={'x': [0, 1], 'y': [0, 1]}
                ),
                row=1, col=1
            )
            
            # VaR
            fig.add_trace(
                go.Indicator(
                    mode="number",
                    value=risk_metrics['var_95'],
                    number={'suffix': '%'},
                    title={"text": "VaR 95%"},
                    domain={'x': [0, 1], 'y': [0, 1]}
                ),
                row=1, col=2
            )
            
            # Max Drawdown
            fig.add_trace(
                go.Indicator(
                    mode="number",
                    value=risk_metrics['max_drawdown'],
                    number={'suffix': '%'},
                    title={"text": "Max DD"},
                    domain={'x': [0, 1], 'y': [0, 1]}
                ),
                row=1, col=3
            )
            
            # Sharpe Ratio
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=risk_metrics['sharpe_ratio'],
                    title={"text": "Sharpe"},
                    gauge={
                        'axis': {'range': [-2, 3]},
                        'bar': {'color': "green" if risk_metrics['sharpe_ratio'] > 1 else "orange"},
                        'steps': [
                            {'range': [0, 1], 'color': "lightgray"},
                            {'range': [1, 2], 'color': "yellow"},
                            {'range': [2, 3], 'color': "lightgreen"}
                        ]
                    },
                    domain={'x': [0, 1], 'y': [0, 1]}
                ),
                row=2, col=1
            )
            
            # Sortino Ratio
            fig.add_trace(
                go.Indicator(
                    mode="number",
                    value=risk_metrics['sortino_ratio'],
                    title={"text": "Sortino"},
                    domain={'x': [0, 1], 'y': [0, 1]}
                ),
                row=2, col=2
            )
            
            # Calmar Ratio
            fig.add_trace(
                go.Indicator(
                    mode="number",
                    value=risk_metrics['calmar_ratio'],
                    title={"text": "Calmar"},
                    domain={'x': [0, 1], 'y': [0, 1]}
                ),
                row=2, col=3
            )
            
            fig.update_layout(
                height=500,
                template="plotly_dark",
                title_text="Risk Metrics Summary"
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Risk metrics summary creation failed: {e}")
            return self._create_error_chart("Risk Metrics Unavailable")
    
    def _calculate_returns(self, equity_history: List[Dict]) -> np.ndarray:
        """Calculate returns from equity history"""
        df = pd.DataFrame(equity_history)
        returns = df['total_value'].pct_change().dropna()
        return returns.values
    
    def _calculate_drawdown(self, equity_history: List[Dict]) -> np.ndarray:
        """Calculate drawdown series"""
        df = pd.DataFrame(equity_history)
        values = df['total_value'].values
        peak = np.maximum.accumulate(values)
        drawdown = (values - peak) / peak
        return drawdown
    
    def _calculate_risk_decomposition(self, positions: Dict[str, Any]) -> Dict[str, float]:
        """Calculate simplified risk decomposition"""
        # Simplified risk attribution
        total_value = sum(pos.get('market_value', 0) for pos in positions.values())
        
        if total_value == 0:
            return {"No Risk": 100}
        
        risk_sources = {}
        for ticker, position in positions.items():
            weight = position.get('market_value', 0) / total_value * 100
            if weight > 5:  # Only show significant positions
                risk_sources[ticker] = weight
        
        # Group small positions
        small_positions = sum(w for w in risk_sources.values() if w < 5)
        if small_positions > 0:
            risk_sources = {k: v for k, v in risk_sources.items() if v >= 5}
            risk_sources['Others'] = small_positions
        
        return risk_sources if risk_sources else {"No Positions": 100}
    
    def _create_correlation_matrix(self, positions: Dict[str, Any]) -> pd.DataFrame:
        """Create mock correlation matrix for positions"""
        tickers = list(positions.keys())[:5]  # Limit to 5 for display
        
        # Generate mock correlation matrix
        np.random.seed(42)  # For reproducibility
        corr_matrix = np.random.rand(len(tickers), len(tickers))
        corr_matrix = (corr_matrix + corr_matrix.T) / 2  # Make symmetric
        np.fill_diagonal(corr_matrix, 1.0)  # Set diagonal to 1
        
        return pd.DataFrame(corr_matrix, index=tickers, columns=tickers)
    
    def _calculate_comprehensive_risk_metrics(self, returns: np.ndarray, equity_history: List[Dict]) -> Dict[str, float]:
        """Calculate comprehensive risk metrics"""
        
        # Annualized metrics
        volatility = returns.std() * np.sqrt(252) * 100
        
        # VaR
        var_95 = np.percentile(returns, 5) * 100
        
        # Max drawdown
        drawdown = self._calculate_drawdown(equity_history)
        max_drawdown = drawdown.min() * 100
        
        # Performance metrics
        annual_return = returns.mean() * 252 * 100
        risk_free_rate = 7.0  # 7% risk-free rate
        
        # Ratios
        sharpe_ratio = (annual_return - risk_free_rate) / volatility if volatility != 0 else 0
        
        # Downside metrics
        downside_returns = returns[returns < 0]
        downside_volatility = downside_returns.std() * np.sqrt(252) * 100 if len(downside_returns) > 0 else volatility
        sortino_ratio = (annual_return - risk_free_rate) / downside_volatility if downside_volatility != 0 else 0
        
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        return {
            'volatility': volatility,
            'var_95': var_95,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio
        }
    
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
