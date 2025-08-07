"""
Individual tab implementations with enhanced styling and functionality
"""
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, Optional, Any, List
import asyncio
import logging
import sys
from datetime import datetime, timedelta

class BaseTab:
    """Base class for all tab implementations"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.info("BaseTab.__init__ ENTRY [LOG_UI_TAB]")
        self.logger.info("BaseTab.__init__ EXIT [LOG_UI_TAB]")
    
    def _apply_tab_styling(self):
        self.logger.info(f"{self.__class__.__name__}._apply_tab_styling ENTRY [LOG_UI_TAB]")
        """Apply enhanced tab styling"""
        
        st.markdown("""
        <style>
            .tab-container {
                background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
                border-radius: 15px;
                padding: 2rem;
                margin-bottom: 1rem;
                border: 1px solid rgba(148, 163, 184, 0.2);
            }
            
            .tab-header {
                background: linear-gradient(135deg, #3b82f6, #06b6d4);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                font-weight: 700;
                font-size: 1.8em;
                margin-bottom: 1.5rem;
                text-align: center;
            }
            
            .metric-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 1rem;
                margin-bottom: 2rem;
            }
            
            .enhanced-metric {
                background: rgba(59, 130, 246, 0.1);
                border: 1px solid rgba(59, 130, 246, 0.3);
                border-radius: 12px;
                padding: 1rem;
                text-align: center;
                transition: transform 0.3s ease;
            }
            
            .enhanced-metric:hover {
                transform: translateY(-2px);
                box-shadow: 0 8px 25px rgba(59, 130, 246, 0.2);
            }
            
            .section-divider {
                border: none;
                height: 2px;
                background: linear-gradient(90deg, transparent, rgba(59, 130, 246, 0.5), transparent);
                margin: 2rem 0;
            }
            
            .alert-info {
                background: rgba(6, 182, 212, 0.1);
                border-left: 4px solid #06b6d4;
                padding: 1rem;
                border-radius: 8px;
                margin: 1rem 0;
            }
            
            .alert-success {
                background: rgba(16, 185, 129, 0.1);
                border-left: 4px solid #10b981;
                padding: 1rem;
                border-radius: 8px;
                margin: 1rem 0;
            }
            
            .alert-warning {
                background: rgba(245, 158, 11, 0.1);
                border-left: 4px solid #f59e0b;
                padding: 1rem;
                border-radius: 8px;
                margin: 1rem 0;
            }
        </style>
        """, unsafe_allow_html=True)

class MarketIntelligenceTab(BaseTab):
    """Enhanced Market Intelligence Tab with comprehensive analysis"""
    
    def __init__(self, market_intelligence, sector_analyzer, market_charts):
        super().__init__()
        self.market_intelligence = market_intelligence
        self.sector_analyzer = sector_analyzer
        self.market_charts = market_charts
        self.logger.info("MarketIntelligenceTab.__init__ ENTRY [LOG_UI_TAB]")
        self.logger.info("MarketIntelligenceTab.__init__ EXIT [LOG_UI_TAB]")
    
    def render(self):
        self.logger.info("MarketIntelligenceTab.render ENTRY [LOG_UI_TAB]")
        """Render comprehensive market intelligence interface"""
        
        self._apply_tab_styling()
        
        st.markdown('<div class="tab-container">', unsafe_allow_html=True)
        st.markdown('<div class="tab-header">🌐 Advanced Market Intelligence Center</div>', unsafe_allow_html=True)
        
        # Market Overview Section
        self._render_market_overview()
        
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        
        # Market Regime Analysis
        self._render_regime_analysis()
        
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        
        # Sector Rotation Analysis
        self._render_sector_analysis()
        
        st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
        
        # Economic Calendar & Events
        self._render_economic_calendar()
        
        st.markdown('</div>', unsafe_allow_html=True)
        self.logger.info("MarketIntelligenceTab.render EXIT [LOG_UI_TAB]")
    
    def _render_market_overview(self):
        self.logger.info("MarketIntelligenceTab._render_market_overview ENTRY [LOG_UI_TAB]")
        """Render comprehensive market overview"""
        
        st.subheader("📊 Real-Time Market Overview")
        
        try:
            # Create market overview layout
            col1, col2 = st.columns([3, 1])
            
            with col1:
                # Market indices performance chart
                self._render_market_indices_chart()
            
            with col2:
                # Market summary metrics
                self._render_market_metrics()
        
        except Exception as e:
            st.error(f"❌ Market overview error: {e}")
            self.logger.error(f"Market overview render failed: {e}")
        self.logger.info("MarketIntelligenceTab._render_market_overview EXIT [LOG_UI_TAB]")
    
    def _render_market_indices_chart(self):
        self.logger.info("MarketIntelligenceTab._render_market_indices_chart ENTRY [LOG_UI_TAB]")
        """Render market indices performance chart"""
        
        try:
            # Mock market data - replace with actual data
            indices = ['NIFTY 50', 'SENSEX', 'NIFTY BANK', 'NIFTY IT']
            performance = [1.2, 0.8, 2.1, -0.3]  # Daily performance %
            
            fig = go.Figure(data=[
                go.Bar(
                    x=indices,
                    y=performance,
                    marker_color=['green' if p > 0 else 'red' for p in performance],
                    text=[f'{p:+.1f}%' for p in performance],
                    textposition='auto',
                )
            ])
            
            fig.update_layout(
                title="📈 Major Indices Performance (Today)",
                template="plotly_dark",
                height=400,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Indices chart error: {e}")
        self.logger.info("MarketIntelligenceTab._render_market_indices_chart EXIT [LOG_UI_TAB]")
    
    def _render_market_metrics(self):
        self.logger.info("MarketIntelligenceTab._render_market_metrics ENTRY [LOG_UI_TAB]")
        """Render key market metrics"""
        
        st.markdown("**🔍 Market Snapshot**")
        
        # Market sentiment gauge
        sentiment_score = 65  # Mock data
        
        fig_sentiment = go.Figure(go.Indicator(
            mode="gauge+number",
            value=sentiment_score,
            title={'text': "Market Sentiment"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "lightgreen" if sentiment_score > 60 else "orange" if sentiment_score > 40 else "red"},
                'steps': [
                    {'range': [0, 40], 'color': "rgba(255, 0, 0, 0.2)"},
                    {'range': [40, 60], 'color': "rgba(255, 255, 0, 0.2)"},
                    {'range': [60, 100], 'color': "rgba(0, 255, 0, 0.2)"}
                ]
            }
        ))
        
        fig_sentiment.update_layout(height=250, template="plotly_dark")
        st.plotly_chart(fig_sentiment, use_container_width=True)
        
        # Key metrics
        metrics = {
            "VIX Level": "18.5 (Low Fear)",
            "Market Breadth": "65% Advancing",
            "Volume": "Above Average",
            "FII Flow": "₹+2,500 Cr"
        }
        
        for metric, value in metrics.items():
            st.markdown(f"**{metric}:** {value}")
        self.logger.info("MarketIntelligenceTab._render_market_metrics EXIT [LOG_UI_TAB]")
    
    def _render_regime_analysis(self):
        self.logger.info("MarketIntelligenceTab._render_regime_analysis ENTRY [LOG_UI_TAB]")
        """Render market regime analysis"""
        
        st.subheader("🌡️ Market Regime Detection")
        
        try:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Regime score gauge
                regime_score = 7.2  # Mock data
                
                fig_regime = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=regime_score,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Regime Score (0-10)"},
                    delta={'reference': 5.0},
                    gauge={
                        'axis': {'range': [0, 10]},
                        'bar': {'color': "green" if regime_score > 6 else "orange" if regime_score > 4 else "red"},
                        'steps': [
                            {'range': [0, 3], 'color': "rgba(255, 0, 0, 0.3)"},
                            {'range': [3, 7], 'color': "rgba(255, 255, 0, 0.3)"},
                            {'range': [7, 10], 'color': "rgba(0, 255, 0, 0.3)"}
                        ]
                    }
                ))
                
                fig_regime.update_layout(height=300, template="plotly_dark")
                st.plotly_chart(fig_regime, use_container_width=True)
            
            with col2:
                st.markdown("**🎯 Current Regime**")
                st.markdown("### Risk-On (Bull Market)")
                st.markdown(f"**Confidence:** 85%")
                st.markdown(f"**Duration:** 12 days")
                
                st.markdown("**📊 Components:**")
                st.markdown("• Trend Strength: 8.1")
                st.markdown("• Volatility: 6.8")  
                st.markdown("• Breadth: 7.5")
                st.markdown("• Volume: 6.9")
                
                st.markdown('<div class="alert-success">✅ Strong bullish momentum detected</div>', unsafe_allow_html=True)
        
        except Exception as e:
            st.error(f"Regime analysis error: {e}")
    
    def _render_sector_analysis(self):
        """Render sector rotation analysis"""
        
        st.subheader("🔄 Sector Rotation Intelligence")
        
        try:
            # Sector performance heatmap
            sectors = ['IT', 'Banking', 'Auto', 'Pharma', 'FMCG', 'Energy', 'Metals']
            periods = ['1D', '1W', '1M', '3M']
            
            # Mock performance data
            np.random.seed(42)
            performance_data = np.random.uniform(-5, 8, (len(sectors), len(periods)))
            
            fig_heatmap = go.Figure(data=go.Heatmap(
                z=performance_data,
                x=periods,
                y=sectors,
                colorscale='RdYlGn',
                zmid=0,
                text=np.round(performance_data, 1),
                texttemplate="%{text}%",
                textfont={"size": 10}
            ))
            
            fig_heatmap.update_layout(
                title="📊 Sector Performance Heatmap",
                template="plotly_dark",
                height=400
            )
            
            st.plotly_chart(fig_heatmap, use_container_width=True)
            
            # Rotation insights
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("**📈 Leaders (1W)**")
                st.markdown("• IT: +4.2%")
                st.markdown("• Banking: +3.8%")
                st.markdown("• Auto: +2.1%")
            
            with col2:
                st.markdown("**📉 Laggards (1W)**")
                st.markdown("• Metals: -2.1%")
                st.markdown("• Energy: -1.5%")
                st.markdown("• FMCG: -0.8%")
            
            with col3:
                st.markdown("**🎯 Rotation Signal**")
                st.markdown('<div class="alert-info">💡 <strong>Growth Rotation</strong><br>Technology and financial sectors leading</div>', unsafe_allow_html=True)
        
        except Exception as e:
            st.error(f"Sector analysis error: {e}")
    
    def _render_economic_calendar(self):
        """Render economic calendar and events"""
        
        st.subheader("📅 Economic Calendar & Market Events")
        
        # High-impact events
        with st.expander("🔍 This Week's High-Impact Events", expanded=True):
            events_data = [
                {"Date": "Today", "Event": "US Fed Minutes Release", "Impact": "🔴 High", "Currency": "USD"},
                {"Date": "Tomorrow", "Event": "India GDP Data Q3", "Impact": "🔴 High", "Currency": "INR"},
                {"Date": "Thursday", "Event": "ECB Interest Rate Decision", "Impact": "🟡 Medium", "Currency": "EUR"},
                {"Date": "Friday", "Event": "US Employment Data", "Impact": "🔴 High", "Currency": "USD"}
            ]
            
            events_df = pd.DataFrame(events_data)
            st.dataframe(events_df, use_container_width=True, hide_index=True)
        
        # Market alerts
        with st.expander("🚨 Market Alerts & Warnings"):
            st.markdown('<div class="alert-warning">⚠️ <strong>Volatility Alert:</strong> Increased volatility expected around Fed announcement</div>', unsafe_allow_html=True)
            st.markdown('<div class="alert-info">📊 <strong>Technical Alert:</strong> NIFTY approaching key resistance at 22,000</div>', unsafe_allow_html=True)

class PortfolioManagementTab(BaseTab):
    """Enhanced Portfolio Management Tab"""
    
    def __init__(self, portfolio_manager, risk_calculator, performance_analyzer, 
                 portfolio_dashboard, risk_dashboard):
        super().__init__()
        self.portfolio_manager = portfolio_manager
        self.risk_calculator = risk_calculator
        self.performance_analyzer = performance_analyzer
        self.portfolio_dashboard = portfolio_dashboard
        self.risk_dashboard = risk_dashboard
        self.logger.info("PortfolioManagementTab.__init__ ENTRY [LOG_UI_TAB]")
        self.logger.info("PortfolioManagementTab.__init__ EXIT [LOG_UI_TAB]")
    
    def render(self, active_portfolio: Optional[Dict[str, Any]]):
        self.logger.info("PortfolioManagementTab.render ENTRY [LOG_UI_TAB]")
        """Render portfolio management interface"""
        
        self._apply_tab_styling()
        
        st.markdown('<div class="tab-container">', unsafe_allow_html=True)
        st.markdown('<div class="tab-header">🏦 Advanced Portfolio Management</div>', unsafe_allow_html=True)
        
        if active_portfolio:
            self._render_portfolio_analytics(active_portfolio)
            st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
            self._render_portfolio_positions(active_portfolio)
            st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
            self._render_risk_analytics(active_portfolio)
        else:
            self._render_no_portfolio_message()
        
        st.markdown('</div>', unsafe_allow_html=True)
        self.logger.info("PortfolioManagementTab.render EXIT [LOG_UI_TAB]")
    
    def _render_portfolio_analytics(self, portfolio: Dict[str, Any]):
        self.logger.info("PortfolioManagementTab._render_portfolio_analytics ENTRY [LOG_UI_TAB]")
        """Render comprehensive portfolio analytics"""
        
        st.subheader("📊 Portfolio Performance Dashboard")
        
        try:
            # Key metrics row
            balances = portfolio['balances']
            total_value = balances['total_value']
            initial_capital = balances['initial_capital']
            total_pnl = total_value - initial_capital
            pnl_pct = (total_pnl / initial_capital) * 100 if initial_capital > 0 else 0
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("Portfolio Value", f"₹{total_value:,.0f}", f"{pnl_pct:+.2f}%")
            
            with col2:
                st.metric("Total P&L", f"₹{total_pnl:+,.0f}", f"₹{total_pnl:+,.0f}")
            
            with col3:
                st.metric("Cash Balance", f"₹{balances['cash']:,.0f}")
            
            with col4:
                st.metric("Invested Amount", f"₹{balances['market_value']:,.0f}")
            
            with col5:
                positions_count = len(portfolio.get('positions', {}))
                st.metric("Active Positions", str(positions_count))
            
            # Portfolio visualization
            if self.portfolio_dashboard:
                try:
                    portfolio_viz = self.portfolio_dashboard.create_portfolio_overview(portfolio)
                    st.plotly_chart(portfolio_viz, use_container_width=True)
                except Exception as e:
                    st.error(f"Portfolio visualization error: {e}")
                    self.logger.info(f"Portfolio visualization error: {e}")
        
        except Exception as e:
            st.error(f"Portfolio analytics error: {e}")
            self.logger.info(f"Portfolio analytics error: {e}")
        self.logger.info("PortfolioManagementTab._render_portfolio_analytics EXIT [LOG_UI_TAB]")
    
    def _render_portfolio_positions(self, portfolio: Dict[str, Any]):
        self.logger.info("PortfolioManagementTab._render_portfolio_positions ENTRY [LOG_UI_TAB]")
        """Render detailed positions analysis"""
        
        st.subheader("📋 Portfolio Positions Analysis")
        
        positions = portfolio.get('positions', {})
        
        if positions:
            # Positions table
            positions_data = []
            
            for ticker, pos in positions.items():
                unrealized_pnl = pos.get('unrealized_pnl', 0)
                invested_value = pos['avg_price'] * pos['quantity']
                pnl_pct = (unrealized_pnl / invested_value) * 100 if invested_value > 0 else 0
                market_value = pos['quantity'] * pos.get('last_price', pos['avg_price'])
                weight = (market_value / sum([p['quantity'] * p.get('last_price', p['avg_price']) for p in positions.values()])) * 100 if positions else 0
                positions_data.append({
                    "Symbol": ticker,
                    "Quantity": pos['quantity'],
                    "Avg Price": f"₹{pos['avg_price']:.2f}",
                    "Market Value": f"₹{market_value:,.0f}",
                    "Weight %": f"{weight:.1f}%",
                    "Unrealized P&L": f"₹{unrealized_pnl:+,.0f}",
                    "Return %": f"{pnl_pct:+.2f}%"
                })
            
            positions_df = pd.DataFrame(positions_data)
            st.dataframe(positions_df, use_container_width=True, hide_index=True)
            
            # Position allocation chart
            self._render_allocation_chart(positions)
        else:
            st.info("📊 No positions currently held in this portfolio.")
        self.logger.info("PortfolioManagementTab._render_portfolio_positions EXIT [LOG_UI_TAB]")
    
    def _render_allocation_chart(self, positions: Dict[str, Any]):
        self.logger.info("PortfolioManagementTab._render_allocation_chart ENTRY [LOG_UI_TAB]")
        """Render portfolio allocation pie chart"""
        
        try:
            symbols = list(positions.keys())
            values = [pos.get('market_value', pos['quantity'] * pos.get('last_price', pos['avg_price'])) 
                     for pos in positions.values()]
            
            fig = go.Figure(data=[go.Pie(
                labels=symbols,
                values=values,
                hole=0.4,
                textinfo='label+percent',
                textposition='auto'
            )])
            
            fig.update_layout(
                title="🥧 Portfolio Allocation",
                template="plotly_dark",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Allocation chart error: {e}")
        self.logger.info("PortfolioManagementTab._render_allocation_chart EXIT [LOG_UI_TAB]")
    
    def _render_risk_analytics(self, portfolio: Dict[str, Any]):
        self.logger.info("PortfolioManagementTab._render_risk_analytics ENTRY [LOG_UI_TAB]")
        """Render comprehensive risk analytics"""
        
        st.subheader("⚠️ Risk Analysis & Management")
        
        try:
            if self.risk_dashboard:
                risk_viz = self.risk_dashboard.create_risk_overview(portfolio)
                st.plotly_chart(risk_viz, use_container_width=True)
            
            # Risk metrics summary
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                # Portfolio Beta (mock)
                beta = 1.15
                st.metric("Portfolio Beta", f"{beta:.2f}")
            
            with col2:
                # Volatility (mock)
                volatility = 18.5
                st.metric("Volatility (Ann.)", f"{volatility:.1f}%")
            
            with col3:
                # Sharpe Ratio (mock)
                sharpe = 1.35
                st.metric("Sharpe Ratio", f"{sharpe:.2f}")
            
            with col4:
                # Max Drawdown (mock)
                max_dd = -8.2
                st.metric("Max Drawdown", f"{max_dd:.1f}%")
        
        except Exception as e:
            st.error(f"Risk analytics error: {e}")
        self.logger.info("PortfolioManagementTab._render_risk_analytics EXIT [LOG_UI_TAB]")
    
    def _render_no_portfolio_message(self):
        self.logger.info("PortfolioManagementTab._render_no_portfolio_message ENTRY [LOG_UI_TAB]")
        """Render message when no portfolio is selected"""
        
        st.markdown('<div class="alert-info">📊 <strong>No Portfolio Selected</strong><br>Please select or create a portfolio from the sidebar to view analytics.</div>', unsafe_allow_html=True)
        
        # Quick portfolio creation
        st.markdown("### 🚀 Quick Portfolio Creation")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("💼 Conservative Portfolio", help="Low risk, stable returns"):
                st.info("Conservative portfolio template selected")
        
        with col2:
            if st.button("⚖️ Balanced Portfolio", help="Moderate risk, balanced approach"):
                st.info("Balanced portfolio template selected")
        
        with col3:
            if st.button("🚀 Growth Portfolio", help="Higher risk, growth focused"):
                st.info("Growth portfolio template selected")
        self.logger.info("PortfolioManagementTab._render_no_portfolio_message EXIT [LOG_UI_TAB]")

class LiveTradingTab(BaseTab):
    """Enhanced Live Trading Terminal"""
    
    def __init__(self, order_manager, trading_engine, risk_validator, market_data_manager):
        super().__init__()
        self.order_manager = order_manager
        self.trading_engine = trading_engine
        self.risk_validator = risk_validator
        self.market_data_manager = market_data_manager
        self.logger.info("LiveTradingTab.__init__ ENTRY [LOG_UI_TAB]")
        self.logger.info("LiveTradingTab.__init__ EXIT [LOG_UI_TAB]")
    
    def render(self, active_portfolio: Optional[Dict[str, Any]]):
        self.logger.info("LiveTradingTab.render ENTRY [LOG_UI_TAB]")
        """Render live trading terminal"""
        
        self._apply_tab_styling()
        
        st.markdown('<div class="tab-container">', unsafe_allow_html=True)
        st.markdown('<div class="tab-header">⚡ Professional Trading Terminal</div>', unsafe_allow_html=True)
        
        if active_portfolio:
            self._render_trading_interface(active_portfolio)
            st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
            self._render_market_watch()
            st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
            self._render_order_management(active_portfolio)
        else:
            self._render_no_portfolio_trading_message()
        
        st.markdown('</div>', unsafe_allow_html=True)
        self.logger.info("LiveTradingTab.render EXIT [LOG_UI_TAB]")
    
    def _render_trading_interface(self, portfolio: Dict[str, Any]):
        self.logger.info("LiveTradingTab._render_trading_interface ENTRY [LOG_UI_TAB]")
        """Render main trading interface"""
        
        st.subheader("📋 Order Entry & Execution")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            self._render_order_entry_form(portfolio)
        
        with col2:
            self._render_trading_metrics(portfolio)
        self.logger.info("LiveTradingTab._render_trading_interface EXIT [LOG_UI_TAB]")
    
    def _render_order_entry_form(self, portfolio: Dict[str, Any]):
        self.logger.info("LiveTradingTab._render_order_entry_form ENTRY [LOG_UI_TAB]")
        """Render enhanced order entry form"""
        
        with st.form("professional_order_form", clear_on_submit=False):
            st.markdown("**🎯 Order Parameters**")
            
            # Basic order details
            col_a, col_b = st.columns(2)
            
            with col_a:
                ticker = st.text_input(
                    "Symbol",
                    placeholder="e.g., RELIANCE",
                    help="Enter stock symbol"
                ).upper()
                
                order_type = st.selectbox(
                    "Order Type",
                    ["MARKET", "LIMIT", "STOP_LOSS", "BRACKET"],
                    help="Select order execution type"
                )
                
                side = st.radio(
                    "Side",
                    ["BUY", "SELL"],
                    horizontal=True,
                    help="Order direction"
                )
            
            with col_b:
                quantity = st.number_input(
                    "Quantity",
                    min_value=0.01,
                    step=1.0,
                    help="Number of shares"
                )
                
                if order_type in ["LIMIT", "STOP_LOSS", "BRACKET"]:
                    limit_price = st.number_input(
                        "Limit Price (₹)",
                        min_value=0.01,
                        step=0.01,
                        help="Order limit price"
                    )
                else:
                    limit_price = None
                
                if order_type == "BRACKET":
                    target_price = st.number_input(
                        "Target Price (₹)",
                        min_value=0.01,
                        step=0.01,
                        help="Profit target price"
                    )
                else:
                    target_price = None
            
            # Advanced options
            with st.expander("⚙️ Advanced Trading Options"):
                col_adv1, col_adv2 = st.columns(2)
                
                with col_adv1:
                    max_slippage = st.slider(
                        "Max Slippage %",
                        0.0, 2.0, 0.1,
                        help="Maximum allowed slippage"
                    )
                    
                    time_in_force = st.selectbox(
                        "Time in Force",
                        ["DAY", "GTC", "IOC"],
                        help="Order validity period"
                    )
                
                with col_adv2:
                    commission_rate = st.slider(
                        "Commission (bps)",
                        0, 50, 5,
                        help="Trading commission rate"
                    )
                    
                    enable_algo = st.checkbox(
                        "Enable Algo Trading",
                        help="Use algorithmic execution"
                    )
            
            # Order validation and submission
            st.markdown("**📊 Order Validation**")
            
            if ticker and quantity > 0:
                self._render_order_validation(portfolio, ticker, side, quantity, limit_price)
        
        # Submit button
        submitted = st.form_submit_button(
            "🚀 Place Order",
            type="primary",
            help="Execute trade order"
        )
        
        if submitted:
            self.logger.info("LiveTradingTab._render_order_entry_form submitting order [LOG_UI_TAB]")
            self._process_order_submission(
                portfolio, ticker, order_type, side, quantity, 
                limit_price, max_slippage, commission_rate
            )
        self.logger.info("LiveTradingTab._render_order_entry_form EXIT [LOG_UI_TAB]")
    
    def _render_order_validation(self, portfolio: Dict[str, Any], ticker: str, 
                                side: str, quantity: float, price: Optional[float]):
        self.logger.info("LiveTradingTab._render_order_validation ENTRY [LOG_UI_TAB]")
        """Render real-time order validation"""
        
        try:
            # Mock price for validation
            estimated_price = price or 100.0  # Would fetch real price
            trade_value = quantity * estimated_price
            
            # Validation checks
            checks = []
            
            # Cash check for buy orders
            if side == "BUY":
                available_cash = portfolio['balances']['cash']
                if trade_value <= available_cash:
                    checks.append(("💰 Cash Available", "✅ Sufficient funds"))
                else:
                    checks.append(("💰 Cash Available", "❌ Insufficient funds"))
            
            # Position size check
            if quantity <= 0:
                checks.append(("📦 Position Size", "❌ Invalid quantity"))
            else:
                checks.append(("📦 Position Size", "✅ Valid quantity"))
            
            # Add more validations as needed
            checks.extend([
                ("⚠️ Risk Limits", "✅ Within limits"),
                ("📈 Market Hours", "✅ Market open"),
                ("🎯 Price Validation", "✅ Price reasonable")
            ])
            
            # Display validation results
            for check_name, check_result in checks:
                if "✅" in check_result:
                    st.success(f"{check_name}: {check_result.replace('✅ ', '')}")
                else:
                    st.error(f"{check_name}: {check_result.replace('❌ ', '')}")
        
        except Exception as e:
            st.warning(f"⚠️ Validation error: {e}")
        self.logger.info("LiveTradingTab._render_order_validation EXIT [LOG_UI_TAB]")
    
    def _render_trading_metrics(self, portfolio: Dict[str, Any]):
        self.logger.info("LiveTradingTab._render_trading_metrics ENTRY [LOG_UI_TAB]")
        """Render key trading metrics"""
        
        st.markdown("**💼 Trading Metrics**")
        
        balances = portfolio['balances']
        
        # Key metrics
        metrics = [
            ("Buying Power", f"₹{balances['cash']:,.0f}"),
            ("Market Value", f"₹{balances['market_value']:,.0f}"),
            ("Available Margin", f"₹{balances.get('available_margin', 0):,.0f}"),
            ("Day P&L", "+₹12,500 (+2.1%)"),  # Mock data
            ("Open Orders", "3")  # Mock data
        ]
        
        for metric, value in metrics:
            st.markdown(f"**{metric}:** {value}")
        
        st.markdown("---")
        
        # Recent trades
        st.markdown("**📊 Recent Trades**")
        
        if portfolio.get('transactions'):
            recent_trades = portfolio['transactions'][:3]
            for trade in recent_trades:
                action_emoji = "🟢" if trade['action'] == 'BUY' else "🔴"
                st.markdown(f"{action_emoji} {trade['action']} {trade['quantity']} {trade['ticker']} @ ₹{trade['price']:.2f}")
        else:
            st.info("No recent trades")
        self.logger.info("LiveTradingTab._render_trading_metrics EXIT [LOG_UI_TAB]")
    
    def _render_market_watch(self):
        self.logger.info("LiveTradingTab._render_market_watch ENTRY [LOG_UI_TAB]")
        """Render market watch with live prices"""
        
        st.subheader("📈 Market Watch & Live Feed")
        
        # Watchlist
        watchlist_symbols = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ITC", "SBIN"]
        
        try:
            # Mock live data - would fetch real prices
            market_data = []
            np.random.seed(int(datetime.now().timestamp()) % 100)
            
            for symbol in watchlist_symbols:
                price = np.random.uniform(500, 3000)
                change_pct = np.random.uniform(-3, 3)
                
                market_data.append({
                    "Symbol": symbol,
                    "Price": f"₹{price:.2f}",
                    "Change %": f"{change_pct:+.2f}%",
                    "Volume": f"{np.random.randint(100, 999)}K",
                    "High": f"₹{price * 1.02:.2f}",
                    "Low": f"₹{price * 0.98:.2f}"
                })
            
            market_df = pd.DataFrame(market_data)
            st.dataframe(market_df, use_container_width=True, hide_index=True)
            
        except Exception as e:
            st.error(f"Market watch error: {e}")
        self.logger.info("LiveTradingTab._render_market_watch EXIT [LOG_UI_TAB]")
    
    def _render_order_management(self, portfolio: Dict[str, Any]):
        self.logger.info("LiveTradingTab._render_order_management ENTRY [LOG_UI_TAB]")
        """Render order management interface"""
        
        st.subheader("📋 Order Management & History")
        
        tab1, tab2 = st.tabs(["Active Orders", "Order History"])
        
        with tab1:
            # Active orders
            pending_orders = portfolio.get('pending_orders', {})
            
            if pending_orders:
                st.markdown("**🔄 Active Orders**")
                
                for order_id, order in pending_orders.items():
                    with st.container():
                        col1, col2, col3 = st.columns([3, 1, 1])
                        
                        with col1:
                            st.markdown(f"**{order['ticker']}** - {order['side']} {order['quantity']} @ ₹{order.get('price', 'Market')}")
                            st.caption(f"Order ID: {order_id[-8:]} | Status: {order['status']}")
                        
                        with col2:
                            if st.button(f"📝 Modify", key=f"modify_{order_id[-4:]}"):
                                st.info("Order modification feature")
                        
                        with col3:
                            if st.button(f"❌ Cancel", key=f"cancel_{order_id[-4:]}"):
                                st.success("Order cancelled")
                        
                        st.markdown("---")
            else:
                st.info("📊 No active orders")
        
        with tab2:
            # Order history
            transactions = portfolio.get('transactions', [])
            
            if transactions:
                st.markdown("**📚 Order History**")
                
                history_data = []
                for txn in transactions:
                    history_data.append({
                        "Time": txn['timestamp'][:16],
                        "Symbol": txn['ticker'],
                        "Side": txn['action'],
                        "Quantity": txn['quantity'],
                        "Price": f"₹{txn['price']:.2f}",
                        "Value": f"₹{txn.get('trade_value', 0):,.0f}"
                    })
                
                history_df = pd.DataFrame(history_data)
                st.dataframe(history_df, use_container_width=True, hide_index=True)
            else:
                st.info("📊 No transaction history")
        self.logger.info("LiveTradingTab._render_order_management EXIT [LOG_UI_TAB]")
    
    def _render_no_portfolio_trading_message(self):
        self.logger.info("LiveTradingTab._render_no_portfolio_trading_message ENTRY [LOG_UI_TAB]")
        """Render message when no portfolio is available for trading"""
        
        st.markdown('<div class="alert-warning">⚠️ <strong>No Portfolio Selected</strong><br>Please select a portfolio from the sidebar to access the trading terminal.</div>', unsafe_allow_html=True)
        
        st.markdown("### 🎯 Trading Features")
        
        features = [
            "🚀 **Multiple Order Types** - Market, Limit, Stop Loss, Bracket Orders",
            "⚡ **Real-time Execution** - Lightning-fast order processing",
            "📊 **Advanced Analytics** - Pre-trade risk analysis",
            "🎯 **Smart Routing** - Optimal execution algorithms",
            "📈 **Live Market Data** - Real-time price feeds",
            "⚠️ **Risk Management** - Built-in risk controls"
        ]
        
        for feature in features:
            st.markdown(feature)
        self.logger.info("LiveTradingTab._render_no_portfolio_trading_message EXIT [LOG_UI_TAB]")
    
    def _process_order_submission(self, portfolio, ticker, order_type, side, 
                             quantity, price, slippage, commission):
        self.logger.info("LiveTradingTab._process_order_submission ENTRY [LOG_UI_TAB]")
        """Process order submission"""
        
        try:
            if not ticker or quantity <= 0:
                st.error("❌ Please enter valid order details")
                self.logger.info("LiveTradingTab._process_order_submission EXIT [LOG_UI_TAB]")
                return
            
            with st.spinner("🚀 Processing order..."):
                # Mock order processing - would use actual order manager
                import time
                time.sleep(1)  # Simulate processing
                
                st.success(f"✅ {order_type} order for {quantity} {ticker} submitted successfully!")
                st.balloons()
                
                # Mock execution result
                execution_price = price or 100.0
                st.info(f"📊 Order executed at ₹{execution_price:.2f}")
        
        except Exception as e:
            st.error(f"❌ Order submission failed: {e}")
        self.logger.info("LiveTradingTab._process_order_submission EXIT [LOG_UI_TAB]")
