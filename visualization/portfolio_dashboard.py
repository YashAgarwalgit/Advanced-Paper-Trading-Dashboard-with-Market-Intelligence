"""
Portfolio dashboard visualization components
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
from utils.helpers import safe_float, format_percentage, format_currency

class PortfolioDashboard:
    """
    Portfolio visualization dashboard components
    Features: Equity curves, allocation charts, performance metrics
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.color_palette = [
            '#00d4aa', '#ff6b6b', '#4ecdc4', '#45b7d1', 
            '#f39c12', '#9b59b6', '#1abc9c', '#e74c3c'
        ]
    
    def create_portfolio_overview(self, portfolio_data: Dict[str, Any]) -> go.Figure:
        """Create comprehensive portfolio overview dashboard"""
        
        try:
            fig = make_subplots(
                rows=2, cols=3,
                subplot_titles=[
                    'Portfolio Equity Curve',
                    'Asset Allocation',
                    'Top Holdings',
                    'Performance Metrics',
                    'Monthly Returns',
                    'Risk Metrics'
                ],
                specs=[
                    [{"colspan": 2}, None, {"type": "pie"}],
                    [{"type": "indicator"}, {"type": "xy"}, {"type": "indicator"}]
                ],
                vertical_spacing=0.08
            )
            
            # Portfolio equity curve
            equity_history = portfolio_data.get('equity_history', [])
            if equity_history:
                df = pd.DataFrame(equity_history)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
                fig.add_trace(
                    go.Scatter(
                        x=df['timestamp'],
                        y=df['total_value'],
                        name='Portfolio Value',
                        line=dict(color='#00d4aa', width=3),
                        hovertemplate='<b>Date:</b> %{x}<br><b>Value:</b> ₹%{y:,.0f}<extra></extra>'
                    ),
                    row=1, col=1
                )
                
                # Add benchmark if available
                if 'benchmark_value' in df.columns and df['benchmark_value'].notna().any():
                    initial_portfolio = df['total_value'].iloc[0]
                    initial_benchmark = df['benchmark_value'].iloc[0]
                    scale_factor = initial_portfolio / initial_benchmark if initial_benchmark != 0 else 1
                    
                    fig.add_trace(
                        go.Scatter(
                            x=df['timestamp'],
                            y=df['benchmark_value'] * scale_factor,
                            name='Benchmark (Scaled)',
                            line=dict(color='#ff6b6b', width=2, dash='dash'),
                            hovertemplate='<b>Date:</b> %{x}<br><b>Benchmark:</b> ₹%{y:,.0f}<extra></extra>'
                        ),
                        row=1, col=1
                    )
            
            # Asset allocation pie chart
            positions = portfolio_data.get('positions', {})
            balances = portfolio_data.get('balances', {})
            
            if positions:
                allocation_data = []
                total_value = balances.get('total_value', 0)
                
                for ticker, position in positions.items():
                    market_value = position.get('market_value', 0)
                    if market_value > 0:
                        allocation_data.append({
                            'Asset': ticker,
                            'Value': market_value,
                            'Weight': (market_value / total_value) * 100 if total_value > 0 else 0
                        })
                
                # Add cash
                cash = balances.get('cash', 0)
                if cash > 0:
                    allocation_data.append({
                        'Asset': 'Cash',
                        'Value': cash,
                        'Weight': (cash / total_value) * 100 if total_value > 0 else 0
                    })
                
                if allocation_data:
                    df_alloc = pd.DataFrame(allocation_data)
                    fig.add_trace(
                        go.Pie(
                            labels=df_alloc['Asset'],
                            values=df_alloc['Weight'],
                            name="Allocation",
                            hovertemplate='<b>%{label}</b><br>Weight: %{value:.1f}%<br>Value: ₹%{customdata:,.0f}<extra></extra>',
                            customdata=df_alloc['Value'],
                            marker=dict(colors=self.color_palette)
                        ),
                        row=1, col=3
                    )
            
            # Performance indicator
            total_value = balances.get('total_value', 0)
            initial_capital = balances.get('initial_capital', 1)
            total_return = ((total_value / initial_capital) - 1) * 100 if initial_capital > 0 else 0
            
            fig.add_trace(
                go.Indicator(
                    mode="number+delta",
                    value=total_return,
                    number={'suffix': '%', 'font': {'size': 40}},
                    title={"text": "Total Return"},
                    delta={'reference': 0, 'position': "bottom"},
                    domain={'x': [0, 1], 'y': [0, 1]}
                ),
                row=2, col=1
            )
            
            # Monthly returns heatmap (mock data)
            if len(equity_history) > 30:
                df = pd.DataFrame(equity_history)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df['month'] = df['timestamp'].dt.to_period('M')
                df['returns'] = df['total_value'].pct_change()
                
                monthly_returns = df.groupby('month')['returns'].apply(lambda x: (1 + x).prod() - 1)
                
                if len(monthly_returns) > 1:
                    fig.add_trace(
                        go.Bar(
                            x=[str(m) for m in monthly_returns.index[-12:]],  # Last 12 months
                            y=monthly_returns.values[-12:] * 100,
                            name='Monthly Returns',
                            marker_color=['green' if r > 0 else 'red' for r in monthly_returns.values[-12:]],
                            hovertemplate='<b>%{x}</b><br>Return: %{y:.2f}%<extra></extra>'
                        ),
                        row=2, col=2
                    )
            
            # Risk indicator
            volatility = self._calculate_portfolio_volatility(equity_history)
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=volatility,
                    number={'suffix': '%', 'font': {'size': 30}},
                    title={"text": "Volatility (Ann.)"},
                    gauge={
                        'axis': {'range': [0, 50]},
                        'bar': {'color': "lightgreen" if volatility < 15 else "orange" if volatility < 25 else "red"},
                        'steps': [
                            {'range': [0, 15], 'color': "rgba(0, 255, 0, 0.2)"},
                            {'range': [15, 25], 'color': "rgba(255, 255, 0, 0.2)"},
                            {'range': [25, 50], 'color': "rgba(255, 0, 0, 0.2)"}
                        ]
                    },
                    domain={'x': [0, 1], 'y': [0, 1]}
                ),
                row=2, col=3
            )
            
            fig.update_layout(
                height=700,
                showlegend=True,
                template="plotly_dark",
                title_text="Portfolio Performance Dashboard",
                title_x=0.5,
                title_font_size=20
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Portfolio overview creation failed: {e}")
            return self._create_error_chart("Portfolio Overview Unavailable")
    
    def create_equity_curve(self, portfolio_data: Dict[str, Any], show_benchmark: bool = True) -> go.Figure:
        """Create detailed equity curve chart"""
        
        try:
            fig = go.Figure()
            
            equity_history = portfolio_data.get('equity_history', [])
            if not equity_history:
                return self._create_error_chart("No Equity History Available")
            
            df = pd.DataFrame(equity_history)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
            
            # Portfolio equity
            fig.add_trace(
                go.Scatter(
                    x=df['timestamp'],
                    y=df['total_value'],
                    name='Portfolio Value',
                    line=dict(color='#00d4aa', width=3),
                    hovertemplate='<b>%{x}</b><br>Portfolio: ₹%{y:,.0f}<extra></extra>'
                )
            )
            
            # Cash component
            if 'cash' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df['timestamp'],
                        y=df['cash'],
                        name='Cash',
                        line=dict(color='#45b7d1', width=1, dash='dot'),
                        hovertemplate='<b>%{x}</b><br>Cash: ₹%{y:,.0f}<extra></extra>'
                    )
                )
            
            # Market value component
            if 'market_value' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=df['timestamp'],
                        y=df['market_value'],
                        name='Investments',
                        line=dict(color='#f39c12', width=1, dash='dot'),
                        hovertemplate='<b>%{x}</b><br>Investments: ₹%{y:,.0f}<extra></extra>'
                    )
                )
            
            # Benchmark comparison
            if show_benchmark and 'benchmark_value' in df.columns and df['benchmark_value'].notna().any():
                initial_portfolio = df['total_value'].iloc[0]
                initial_benchmark = df['benchmark_value'].iloc[0]
                scale_factor = initial_portfolio / initial_benchmark if initial_benchmark != 0 else 1
                
                fig.add_trace(
                    go.Scatter(
                        x=df['timestamp'],
                        y=df['benchmark_value'] * scale_factor,
                        name='Benchmark (Scaled)',
                        line=dict(color='#ff6b6b', width=2, dash='dash'),
                        hovertemplate='<b>%{x}</b><br>Benchmark: ₹%{y:,.0f}<extra></extra>'
                    )
                )
            
            fig.update_layout(
                title="Portfolio Equity Curve",
                xaxis_title="Date",
                yaxis_title="Value (₹)",
                template="plotly_dark",
                height=500,
                hovermode='x unified'
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Equity curve creation failed: {e}")
            return self._create_error_chart("Equity Curve Unavailable")
    
    def create_allocation_breakdown(self, portfolio_data: Dict[str, Any]) -> go.Figure:
        """Create detailed allocation breakdown charts"""
        
        try:
            positions = portfolio_data.get('positions', {})
            balances = portfolio_data.get('balances', {})
            
            if not positions:
                return self._create_error_chart("No Positions to Display")
            
            # Create subplot for multiple views
            fig = make_subplots(
                rows=1, cols=2,
                specs=[
                    [{"type": "pie"}, {"type": "xy"}]
                ],
                subplot_titles=['Allocation by Asset', 'Position Sizes']
            )
            
            # Prepare allocation data
            allocation_data = []
            total_value = balances.get('total_value', 0)
            
            for ticker, position in positions.items():
                market_value = position.get('market_value', 0)
                if market_value > 0:
                    allocation_data.append({
                        'Asset': ticker,
                        'Value': market_value,
                        'Weight': (market_value / total_value) * 100 if total_value > 0 else 0
                    })
            
            # Add cash
            cash = balances.get('cash', 0)
            if cash > 0:
                allocation_data.append({
                    'Asset': 'Cash',
                    'Value': cash,
                    'Weight': (cash / total_value) * 100 if total_value > 0 else 0
                })
            
            if allocation_data:
                df_alloc = pd.DataFrame(allocation_data)
                df_alloc = df_alloc.sort_values('Weight', ascending=False)
                
                # Pie chart
                fig.add_trace(
                    go.Pie(
                        labels=df_alloc['Asset'],
                        values=df_alloc['Weight'],
                        name="Allocation",
                        hovertemplate='<b>%{label}</b><br>%{value:.1f}%<br>₹%{customdata:,.0f}<extra></extra>',
                        customdata=df_alloc['Value'],
                        marker=dict(colors=self.color_palette),
                        textinfo='label+percent'
                    ),
                    row=1, col=1
                )
                
                # Bar chart
                fig.add_trace(
                    go.Bar(
                        x=df_alloc['Weight'],
                        y=df_alloc['Asset'],
                        orientation='h',
                        name='Position Size',
                        marker_color=self.color_palette[:len(df_alloc)],
                        hovertemplate='<b>%{y}</b><br>Weight: %{x:.1f}%<br>Value: ₹%{customdata:,.0f}<extra></extra>',
                        customdata=df_alloc['Value']
                    ),
                    row=1, col=2
                )
            
            fig.update_layout(
                height=400,
                showlegend=False,
                template="plotly_dark",
                title_text="Portfolio Allocation Analysis"
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Allocation breakdown creation failed: {e}")
            return self._create_error_chart("Allocation Analysis Unavailable")
    
    def create_performance_attribution(self, portfolio_data: Dict[str, Any]) -> go.Figure:
        """Create performance attribution analysis"""
        
        try:
            transactions = portfolio_data.get('transactions', [])
            positions = portfolio_data.get('positions', {})
            
            if not transactions:
                return self._create_error_chart("No Transaction History Available")
            
            # Calculate position contributions
            attribution_data = []
            
            for ticker, position in positions.items():
                unrealized_pnl = position.get('unrealized_pnl', 0)
                market_value = position.get('market_value', 0)
                
                attribution_data.append({
                    'Asset': ticker,
                    'P&L': unrealized_pnl,
                    'Weight': market_value,
                    'Contribution': unrealized_pnl  # Simplified
                })
            
            if attribution_data:
                df_attr = pd.DataFrame(attribution_data)
                df_attr = df_attr.sort_values('P&L')
                
                fig = go.Figure()
                
                # Waterfall chart for attribution
                fig.add_trace(
                    go.Bar(
                        x=df_attr['Asset'],
                        y=df_attr['P&L'],
                        name='P&L Contribution',
                        marker_color=['green' if pnl > 0 else 'red' for pnl in df_attr['P&L']],
                        hovertemplate='<b>%{x}</b><br>P&L: ₹%{y:,.0f}<extra></extra>'
                    )
                )
                
                fig.update_layout(
                    title="Performance Attribution by Position",
                    xaxis_title="Assets",
                    yaxis_title="P&L Contribution (₹)",
                    template="plotly_dark",
                    height=400
                )
                
                return fig
            else:
                return self._create_error_chart("No Attribution Data Available")
                
        except Exception as e:
            self.logger.error(f"Performance attribution creation failed: {e}")
            return self._create_error_chart("Attribution Analysis Unavailable")
    
    def _calculate_portfolio_volatility(self, equity_history: List[Dict]) -> float:
        """Calculate portfolio volatility from equity history"""
        
        try:
            if len(equity_history) < 2:
                return 0.0
            
            df = pd.DataFrame(equity_history)
            returns = df['total_value'].pct_change().dropna()
            
            if len(returns) < 2:
                return 0.0
            
            # Annualized volatility
            return returns.std() * np.sqrt(252) * 100
            
        except Exception:
            return 0.0
    
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
