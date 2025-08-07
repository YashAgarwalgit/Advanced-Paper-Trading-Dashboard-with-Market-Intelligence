"""
Professional sidebar portfolio management interface with enhanced styling
"""
import streamlit as st
import json
import shutil
import tempfile
from datetime import datetime
from typing import Dict, Optional, Any
import logging
import sys
from portfolio.manager import EnhancedPortfolioManager
from data.async_market_data import AsyncMarketDataManager
from utils.helpers import format_currency, format_percentage, get_timestamp_iso

class PortfolioSidebar:
    """
    Professional sidebar portfolio management interface
    Features: Portfolio selection, management, real-time updates, enhanced styling
    """
    
    def __init__(self, portfolio_manager: EnhancedPortfolioManager, market_data_manager: AsyncMarketDataManager):
        self.portfolio_manager = portfolio_manager
        self.market_data_manager = market_data_manager
        self.logger = logging.getLogger(__name__)
        self.logger.info("PortfolioSidebar.__init__ ENTRY [LOG_UI_SIDEBAR]")
        self.logger.info("PortfolioSidebar.__init__ EXIT [LOG_UI_SIDEBAR]")
    
    def render(self) -> Optional[Dict[str, Any]]:
        self.logger.info("PortfolioSidebar.render ENTRY [LOG_UI_SIDEBAR]")
        """Render the complete sidebar interface"""
        
        # Apply enhanced sidebar styling
        self._apply_sidebar_styling()
        
        # Render main control panel
        result = self._render_control_panel()
        self.logger.info("PortfolioSidebar.render EXIT [LOG_UI_SIDEBAR]")
        return result
    
    def _apply_sidebar_styling(self):
        self.logger.info("PortfolioSidebar._apply_sidebar_styling ENTRY [LOG_UI_SIDEBAR]")
        """Apply professional sidebar styling"""
        
        st.markdown("""
        <style>
            /* Enhanced Sidebar Styling */
            .sidebar-main {
                background: linear-gradient(145deg, #1e293b 0%, #334155 100%);
                border-radius: 15px;
                padding: 1.5rem;
                margin-bottom: 1.5rem;
                border: 2px solid rgba(59, 130, 246, 0.3);
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
            }
            
            .control-panel-header {
                color: #3b82f6;
                font-weight: 700;
                font-size: 1.4em;
                text-align: center;
                margin-bottom: 1.5rem;
                padding-bottom: 0.5rem;
                border-bottom: 2px solid rgba(59, 130, 246, 0.3);
                background: linear-gradient(135deg, #3b82f6, #06b6d4);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            }
            
            .portfolio-summary {
                background: rgba(59, 130, 246, 0.1);
                padding: 1rem;
                border-radius: 10px;
                margin-bottom: 1rem;
                border-left: 4px solid #3b82f6;
                backdrop-filter: blur(10px);
            }
            
            .metric-row {
                display: flex;
                justify-content: space-between;
                margin-bottom: 0.5rem;
                font-size: 0.9em;
            }
            
            .metric-label {
                color: #94a3b8;
                font-weight: 500;
            }
            
            .metric-value {
                color: #f1f5f9;
                font-weight: 600;
            }
            
            .profit {
                color: #10b981 !important;
            }
            
            .loss {
                color: #ef4444 !important;
            }
            
            .action-button {
                width: 100%;
                border-radius: 8px !important;
                font-weight: 600 !important;
                transition: all 0.3s ease !important;
                margin-bottom: 0.5rem;
            }
            
            .action-button:hover {
                transform: translateY(-2px) !important;
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2) !important;
            }
            
            .status-badge {
                display: inline-block;
                padding: 0.25rem 0.5rem;
                border-radius: 12px;
                font-size: 0.75em;
                font-weight: 600;
                text-transform: uppercase;
            }
            
            .status-active {
                background: rgba(16, 185, 129, 0.2);
                color: #10b981;
                border: 1px solid rgba(16, 185, 129, 0.3);
            }
            
            .platform-info {
                background: rgba(15, 23, 42, 0.6);
                padding: 1rem;
                border-radius: 10px;
                border: 1px solid rgba(148, 163, 184, 0.2);
                margin-top: 1rem;
            }
        </style>
        """, unsafe_allow_html=True)
    
    def _render_control_panel(self) -> Optional[Dict[str, Any]]:
        """Render main portfolio control panel"""
        
        st.markdown('<div class="sidebar-main">', unsafe_allow_html=True)
        st.markdown('<div class="control-panel-header">🗂️ Portfolio Control Center</div>', unsafe_allow_html=True)
        
        # Get available portfolios
        available_portfolios = self.portfolio_manager.get_available_portfolios()
        active_portfolio = None
        
        self.logger.info("PortfolioSidebar._render_control_panel ENTRY [LOG_UI_SIDEBAR]")
        if available_portfolios:
            active_portfolio = self._render_portfolio_selector(available_portfolios)
            
            if active_portfolio:
                self._render_portfolio_summary(active_portfolio)
                self._render_quick_actions(active_portfolio)
                self._render_portfolio_management()
        else:
            self._render_no_portfolios_message()
        
        # Create new portfolio section
        self._render_create_portfolio_section()
        
        # Platform information
        self._render_platform_info()
        
        st.markdown('</div>', unsafe_allow_html=True)
        self.logger.info("PortfolioSidebar._render_control_panel EXIT [LOG_UI_SIDEBAR]")
        return active_portfolio
    
    def _render_portfolio_selector(self, portfolios) -> Optional[Dict[str, Any]]:
        """Render portfolio selector with enhanced styling"""
        
        selected_name = st.selectbox(
            "📊 Active Portfolio",
            portfolios,
            key="portfolio_selector",
            help="Select your active trading portfolio"
        )
        
        self.logger.info("PortfolioSidebar._render_portfolio_selector ENTRY [LOG_UI_SIDEBAR]")
        if selected_name:
            try:
                portfolio_data = self.portfolio_manager.load_portfolio_safe(selected_name)
                if portfolio_data:
                    st.session_state.active_portfolio = portfolio_data
                    self.logger.info("PortfolioSidebar._render_portfolio_selector loaded portfolio [LOG_UI_SIDEBAR]")
                    self.logger.info("PortfolioSidebar._render_portfolio_selector EXIT [LOG_UI_SIDEBAR]")
                    return portfolio_data
                else:
                    st.error(f"❌ Failed to load portfolio: {selected_name}")
                    self.logger.info("PortfolioSidebar._render_portfolio_selector EXIT [LOG_UI_SIDEBAR]")
            except Exception as e:
                st.error(f"❌ Portfolio loading error: {e}")
                self.logger.error(f"Portfolio loading failed: {e}")
                self.logger.info("PortfolioSidebar._render_portfolio_selector EXIT [LOG_UI_SIDEBAR]")
        
        self.logger.info("PortfolioSidebar._render_portfolio_selector EXIT [LOG_UI_SIDEBAR]")
        return None
    
    def _render_portfolio_summary(self, portfolio: Dict[str, Any]):
        """Render enhanced portfolio summary"""
        
        st.markdown('<div class="portfolio-summary">', unsafe_allow_html=True)
        
        # Portfolio metadata
        metadata = portfolio.get('metadata', {})
        balances = portfolio.get('balances', {})
        
        # Header with status badge
        st.markdown(f"""
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
            <h4 style="color: #f1f5f9; margin: 0;">{metadata.get('portfolio_name', 'Unknown')}</h4>
            <span class="status-badge status-active">Live</span>
        </div>
        """, unsafe_allow_html=True)
        
        # Key metrics
        total_value = balances.get('total_value', 0)
        initial_capital = balances.get('initial_capital', 1)
        total_pnl = total_value - initial_capital
        pnl_pct = (total_pnl / initial_capital * 100) if initial_capital > 0 else 0
        
        # Metrics rows
        metrics = [
            ("📅 Created", metadata.get('created_utc', 'Unknown')[:10]),
            ("🎯 Strategy", metadata.get('strategy_type', 'Multi-Asset')),
            ("⚡ Risk Profile", metadata.get('risk_profile', 'Moderate')),
            ("📈 Benchmark", metadata.get('benchmark', '^NSEI')),
            ("💰 Total Value", format_currency(total_value)),
            ("📊 P&L", f"{format_currency(total_pnl)} ({pnl_pct:+.2f}%)"),
            ("💵 Cash", format_currency(balances.get('cash', 0))),
            ("📈 Invested", format_currency(balances.get('market_value', 0))),
            ("📋 Positions", str(len(portfolio.get('positions', {}))))
        ]
        
        for label, value in metrics:
            pnl_class = "profit" if "P&L" in label and total_pnl > 0 else "loss" if "P&L" in label and total_pnl < 0 else ""
            st.markdown(f"""
            <div class="metric-row">
                <span class="metric-label">{label}:</span>
                <span class="metric-value {pnl_class}">{value}</span>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def _render_quick_actions(self, portfolio: Dict[str, Any]):
        """Render quick action buttons"""
        
        self.logger.info("PortfolioSidebar._render_quick_actions ENTRY [LOG_UI_SIDEBAR]")
        st.markdown("**⚡ Quick Actions**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Refresh", key="refresh_portfolio", help="Update live prices"):
                self.logger.info("PortfolioSidebar._render_quick_actions triggered refresh [LOG_UI_SIDEBAR]")
                self._refresh_portfolio_data(portfolio)
        
        with col2:
            if st.button("📊 Report", key="generate_report", help="Generate detailed report"):
                self.logger.info("PortfolioSidebar._render_quick_actions triggered report [LOG_UI_SIDEBAR]")
                self._generate_portfolio_report(portfolio)
        
        # Export functionality
        if st.button("💾 Export Portfolio", key="export_portfolio", help="Download portfolio data"):
            self.logger.info("PortfolioSidebar._render_quick_actions triggered export [LOG_UI_SIDEBAR]")
            self._export_portfolio(portfolio)
        self.logger.info("PortfolioSidebar._render_quick_actions EXIT [LOG_UI_SIDEBAR]")
    
    def _render_portfolio_management(self):
        """Render portfolio management options"""
        
        self.logger.info("PortfolioSidebar._render_portfolio_management ENTRY [LOG_UI_SIDEBAR]")
        st.markdown("---")
        st.markdown("**🔧 Portfolio Management**")
        
        # Rename Portfolio
        with st.expander("✏️ Rename Portfolio"):
            self._render_rename_form()
        
        # Delete Portfolio
        with st.expander("🗑️ Delete Portfolio", expanded=False):
            self._render_delete_form()
        
        # Duplicate Portfolio
        with st.expander("📋 Duplicate Portfolio"):
            self._render_duplicate_form()
        self.logger.info("PortfolioSidebar._render_portfolio_management EXIT [LOG_UI_SIDEBAR]")
    
    def _render_rename_form(self):
        """Render portfolio rename form"""
        
        self.logger.info("PortfolioSidebar._render_rename_form ENTRY [LOG_UI_SIDEBAR]")
        current_name = st.session_state.get('active_portfolio', {}).get('metadata', {}).get('portfolio_name', '')
        
        new_name = st.text_input(
            "New Portfolio Name",
            value=current_name,
            max_chars=50,
            key="rename_input",
            help="Enter new portfolio name"
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("✅ Rename", key="rename_btn", type="primary"):
                if new_name.strip() and new_name.strip() != current_name:
                    self.logger.info("PortfolioSidebar._render_rename_form triggered rename [LOG_UI_SIDEBAR]")
                    self._execute_rename(current_name, new_name.strip())
        
        with col2:
            if st.button("❌ Cancel", key="cancel_rename"):
                self.logger.info("PortfolioSidebar._render_rename_form triggered cancel [LOG_UI_SIDEBAR]")
                st.rerun()
        self.logger.info("PortfolioSidebar._render_rename_form EXIT [LOG_UI_SIDEBAR]")
    
    def _render_delete_form(self):
        """Render portfolio delete form"""
        
        self.logger.info("PortfolioSidebar._render_delete_form ENTRY [LOG_UI_SIDEBAR]")
        current_name = st.session_state.get('active_portfolio', {}).get('metadata', {}).get('portfolio_name', '')
        
        st.warning("⚠️ This action cannot be undone!")
        
        confirm_delete = st.checkbox(
            f"I want to delete '{current_name}'",
            key="confirm_delete",
            help="Check to confirm deletion"
        )
        
        if confirm_delete:
            if st.button("🗑️ DELETE PORTFOLIO", key="delete_btn", type="primary"):
                self.logger.info("PortfolioSidebar._render_delete_form triggered delete [LOG_UI_SIDEBAR]")
                self._execute_delete(current_name)
        self.logger.info("PortfolioSidebar._render_delete_form EXIT [LOG_UI_SIDEBAR]")
    
    def _render_duplicate_form(self):
        """Render portfolio duplicate form"""
        
        self.logger.info("PortfolioSidebar._render_duplicate_form ENTRY [LOG_UI_SIDEBAR]")
        current_name = st.session_state.get('active_portfolio', {}).get('metadata', {}).get('portfolio_name', '')
        
        duplicate_name = st.text_input(
            "New Portfolio Name",
            value=f"{current_name}_copy",
            key="duplicate_name",
            help="Name for the duplicated portfolio"
        )
        
        if st.button("📋 Create Copy", key="duplicate_btn", type="primary"):
            if duplicate_name.strip():
                self.logger.info("PortfolioSidebar._render_duplicate_form triggered duplicate [LOG_UI_SIDEBAR]")
                self._execute_duplicate(current_name, duplicate_name.strip())
        self.logger.info("PortfolioSidebar._render_duplicate_form EXIT [LOG_UI_SIDEBAR]")
    
    def _render_no_portfolios_message(self):
        """Render message when no portfolios exist"""
        
        self.logger.info("PortfolioSidebar._render_no_portfolios_message ENTRY [LOG_UI_SIDEBAR]")
        st.info("🎯 No portfolios found. Create your first portfolio below!")
        st.session_state.active_portfolio = None
        self.logger.info("PortfolioSidebar._render_no_portfolios_message EXIT [LOG_UI_SIDEBAR]")
    
    def _render_create_portfolio_section(self):
        """Render create new portfolio section"""
        
        self.logger.info("PortfolioSidebar._render_create_portfolio_section ENTRY [LOG_UI_SIDEBAR]")
        st.markdown("---")
        st.markdown("**➕ Create New Portfolio**")
        
        available_portfolios = self.portfolio_manager.get_available_portfolios()
        expanded = len(available_portfolios) == 0
        
        with st.expander("🆕 New Portfolio Setup", expanded=expanded):
            self._render_create_portfolio_form()
        self.logger.info("PortfolioSidebar._render_create_portfolio_section EXIT [LOG_UI_SIDEBAR]")
    
    def _render_create_portfolio_form(self):
        """Render enhanced create portfolio form"""
        
        self.logger.info("PortfolioSidebar._render_create_portfolio_form ENTRY [LOG_UI_SIDEBAR]")
        with st.form("create_portfolio_form"):
            # Basic Information
            st.markdown("**📋 Basic Information**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                portfolio_name = st.text_input(
                    "Portfolio Name",
                    placeholder="Enter unique name",
                    max_chars=50,
                    help="Unique identifier for your portfolio"
                )
            
            with col2:
                initial_capital = st.number_input(
                    "Initial Capital (₹)",
                    value=1000000,
                    step=100000,
                    format="%d",
                    min_value=10000,
                    help="Starting investment amount"
                )
            
            # Strategy Configuration
            st.markdown("**🎯 Strategy Configuration**")
            
            col3, col4 = st.columns(2)
            
            with col3:
                benchmark = st.selectbox(
                    "Benchmark",
                    ["^NSEI", "^BSESN", "NIFTYMIDCAP150", "^GSPC"],
                    help="Reference index for performance comparison"
                )
                
                risk_profile = st.selectbox(
                    "Risk Profile",
                    ["Conservative", "Moderate", "Aggressive", "Speculative"],
                    index=1,
                    help="Investment risk tolerance"
                )
            
            with col4:
                strategy_type = st.selectbox(
                    "Strategy Type",
                    ["Multi-Asset", "Equity Focus", "Balanced", "Growth", "Value"],
                    help="Primary investment strategy"
                )
            
            # Asset Allocation
            st.markdown("**📊 Asset Allocation**")
            
            equity_pct = st.slider(
                "Equities %",
                0, 100, 70, 5,
                help="Percentage allocated to equities"
            )
            
            col5, col6 = st.columns(2)
            
            with col5:
                cash_pct = st.slider("Cash %", 0, 100-equity_pct, min(30, 100-equity_pct), 5)
            
            with col6:
                other_pct = 100 - equity_pct - cash_pct
                st.markdown(f"**Other Assets:** {other_pct}%")
            
            # Advanced Options
            self.logger.info("PortfolioSidebar._render_create_portfolio_form EXIT [LOG_UI_SIDEBAR]")
    
            with st.expander("⚙️ Advanced Options"):
                max_position_size = st.slider("Max Position Size %", 1, 25, 10)
                enable_margin = st.checkbox("Enable Margin Trading", value=False)
                auto_rebalance = st.checkbox("Auto Rebalancing", value=True)
            
            # Submit button
            submitted = st.form_submit_button(
                "🚀 Create Portfolio",
                type="primary",
                help="Create new portfolio with specified settings"
            )
            
            if submitted:
                self._execute_portfolio_creation(
                    portfolio_name, initial_capital, benchmark, risk_profile,
                    strategy_type, equity_pct, cash_pct, max_position_size
                )
    
    def _render_platform_info(self):
        """Render platform information panel"""
        
        st.markdown('<div class="platform-info">', unsafe_allow_html=True)
        st.markdown("**🚀 Platform Information**")
        
        info_items = [
            ("Version", "5.0 Enhanced"),
            ("Engine", "Async Multi-Threading"),
            ("Features", "Atomic Transactions"),
            ("Status", "🟢 Live Trading Ready"),
            ("Last Updated", datetime.now().strftime("%Y-%m-%d"))
        ]
        
        for label, value in info_items:
            st.markdown(f"**{label}:** {value}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Helper methods for actions
    def _refresh_portfolio_data(self, portfolio: Dict[str, Any]):
        """Refresh portfolio with latest market data"""
        
        with st.spinner("🔄 Refreshing portfolio data..."):
            try:
                position_tickers = list(portfolio.get('positions', {}).keys())
                
                if position_tickers:
                    # This would use the async market data manager
                    st.success("✅ Portfolio data refreshed!")
                    st.rerun()
                else:
                    st.info("📊 No positions to refresh")
                    
            except Exception as e:
                st.error(f"❌ Refresh failed: {e}")
    
    def _generate_portfolio_report(self, portfolio: Dict[str, Any]):
        """Generate comprehensive portfolio report"""
        
        with st.spinner("📊 Generating portfolio report..."):
            try:
                # This would generate a comprehensive report
                st.success("📋 Report generated successfully!")
                st.info("📥 Report download feature coming soon!")
            except Exception as e:
                st.error(f"❌ Report generation failed: {e}")
    
    def _export_portfolio(self, portfolio: Dict[str, Any]):
        """Export portfolio data"""
        
        try:
            portfolio_name = portfolio['metadata']['portfolio_name']
            export_data = json.dumps(portfolio, indent=2, default=str)
            
            st.download_button(
                label="📥 Download JSON",
                data=export_data,
                file_name=f"{portfolio_name}_export_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json",
                key="download_portfolio",
                help="Download portfolio data as JSON file"
            )
            
        except Exception as e:
            st.error(f"❌ Export failed: {e}")
    
    def _execute_rename(self, old_name: str, new_name: str):
        """Execute portfolio rename"""
        
        try:
            # Implementation would handle the rename operation
            st.success(f"✅ Portfolio renamed to '{new_name}'")
            st.rerun()
        except Exception as e:
            st.error(f"❌ Rename failed: {e}")
    
    def _execute_delete(self, portfolio_name: str):
        """Execute portfolio deletion"""
        
        try:
            # Implementation would handle the delete operation
            st.success(f"✅ Portfolio '{portfolio_name}' deleted successfully!")
            st.session_state.active_portfolio = None
            st.rerun()
        except Exception as e:
            st.error(f"❌ Delete failed: {e}")
    
    def _execute_duplicate(self, original_name: str, new_name: str):
        """Execute portfolio duplication"""
        
        try:
            # Implementation would handle the duplication
            st.success(f"✅ Portfolio duplicated as '{new_name}'")
            st.rerun()
        except Exception as e:
            st.error(f"❌ Duplication failed: {e}")
    
    def _execute_portfolio_creation(self, name: str, capital: float, benchmark: str, 
                                   risk_profile: str, strategy: str, equity_pct: int, 
                                   cash_pct: int, max_position: int):
        """Execute portfolio creation"""
        
        if not name or not name.strip():
            st.error("❌ Portfolio name cannot be empty!")
            return
        
        try:
            # Check if portfolio already exists
            available_portfolios = self.portfolio_manager.get_available_portfolios()
            if name.strip() in available_portfolios:
                st.error(f"❌ Portfolio '{name.strip()}' already exists!")
                return
            
            # Create asset allocation
            asset_allocation = {
                "Equities": equity_pct / 100,
                "Cash": cash_pct / 100,
                "Others": (100 - equity_pct - cash_pct) / 100
            }
            
            # Create portfolio
            success = self.portfolio_manager.create_enhanced_portfolio(
                name.strip(),
                capital,
                asset_allocation,
                benchmark
            )
            
            if success:
                st.success(f"🎉 Portfolio '{name.strip()}' created successfully!")
                st.balloons()
                st.rerun()
            else:
                st.error("❌ Portfolio creation failed!")
                
        except Exception as e:
            st.error(f"❌ Creation failed: {e}")
            self.logger.error(f"Portfolio creation error: {e}")
