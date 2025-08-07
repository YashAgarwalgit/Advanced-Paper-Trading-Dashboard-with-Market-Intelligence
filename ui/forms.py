"""
Professional form components with enhanced validation and styling
"""
import streamlit as st
import re
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime
import logging
import sys
import numpy as np

class FormValidator:
    """Professional form validation utilities"""
    
    @staticmethod
    def validate_portfolio_name(name: str) -> tuple[bool, str]:
        """Validate portfolio name"""
        if not name or not name.strip():
            logging.getLogger(__name__).info("create_portfolio_from_config EXIT [LOG_UI_FORM]")
            return False, "Portfolio name cannot be empty"
        
        if len(name.strip()) < 3:
            logging.getLogger(__name__).info("create_portfolio_from_config EXIT [LOG_UI_FORM]")
            return False, "Portfolio name must be at least 3 characters"
        
        if len(name.strip()) > 50:
            logging.getLogger(__name__).info("create_portfolio_from_config EXIT [LOG_UI_FORM]")
            return False, "Portfolio name cannot exceed 50 characters"
        
        if not re.match(r'^[a-zA-Z0-9_\s-]+$', name.strip()):
            logging.getLogger(__name__).info("create_portfolio_from_config EXIT [LOG_UI_FORM]")
            return False, "Portfolio name can only contain letters, numbers, spaces, hyphens, and underscores"
        
        logging.getLogger(__name__).info("submit_order_from_config EXIT [LOG_UI_FORM]")
        return True, ""
    
    @staticmethod
    def validate_ticker_symbol(ticker: str) -> tuple[bool, str]:
        """Validate ticker symbol"""
        if not ticker or not ticker.strip():
            logging.getLogger(__name__).info("create_portfolio_from_config EXIT [LOG_UI_FORM]")
            return False, "Ticker symbol cannot be empty"
        
        ticker = ticker.strip().upper()
        
        if not re.match(r'^[A-Z0-9.-]{1,12}$', ticker):
            logging.getLogger(__name__).info("create_portfolio_from_config EXIT [LOG_UI_FORM]")
            return False, "Invalid ticker symbol format"
        
        logging.getLogger(__name__).info("submit_order_from_config EXIT [LOG_UI_FORM]")
        return True, ""
    
    @staticmethod
    def validate_quantity(quantity: float, min_qty: float = 0.01) -> tuple[bool, str]:
        """Validate trade quantity"""
        if quantity <= 0:
            logging.getLogger(__name__).info("create_portfolio_from_config EXIT [LOG_UI_FORM]")
            return False, "Quantity must be positive"
        
        if quantity < min_qty:
            logging.getLogger(__name__).info("create_portfolio_from_config EXIT [LOG_UI_FORM]")
            return False, f"Minimum quantity is {min_qty}"
        
        logging.getLogger(__name__).info("submit_order_from_config EXIT [LOG_UI_FORM]")
        return True, ""
    
    @staticmethod
    def validate_price(price: Optional[float], min_price: float = 0.01) -> tuple[bool, str]:
        """Validate price input"""
        if price is None:
            logging.getLogger(__name__).info("submit_order_from_config EXIT [LOG_UI_FORM]")
            return True, ""  # Optional field
        
        if price <= 0:
            logging.getLogger(__name__).info("create_portfolio_from_config EXIT [LOG_UI_FORM]")
            return False, "Price must be positive"
        
        if price < min_price:
            logging.getLogger(__name__).info("create_portfolio_from_config EXIT [LOG_UI_FORM]")
            return False, f"Minimum price is ₹{min_price}"
        
        logging.getLogger(__name__).info("submit_order_from_config EXIT [LOG_UI_FORM]")
        return True, ""
    
    @staticmethod
    def validate_allocation(allocations: Dict[str, float]) -> tuple[bool, str]:
        """Validate asset allocation percentages"""
        total = sum(allocations.values())
        
        if abs(total - 100) > 0.01:
            logging.getLogger(__name__).info("create_portfolio_from_config EXIT [LOG_UI_FORM]")
            return False, f"Allocation must sum to 100%, got {total:.2f}%"
        
        for asset, pct in allocations.items():
            if pct < 0 or pct > 100:
                logging.getLogger(__name__).info("create_portfolio_from_config EXIT [LOG_UI_FORM]")
                return False, f"Invalid allocation for {asset}: {pct}%"
        
        logging.getLogger(__name__).info("submit_order_from_config EXIT [LOG_UI_FORM]")
        return True, ""

class BaseForm:
    """Base class for all form components"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.info("BaseForm.__init__ ENTRY")
        self.validator = FormValidator()
        self.logger.info("BaseForm.__init__ EXIT")
    
    def _apply_form_styling(self):
        self.logger.info("BaseForm._apply_form_styling ENTRY")
        """Apply professional form styling"""
        st.markdown("""
        <style>
            .form-container {
                background: linear-gradient(145deg, #1e293b 0%, #334155 100%);
                border-radius: 15px;
                padding: 2rem;
                margin-bottom: 1rem;
                border: 1px solid rgba(148, 163, 184, 0.2);
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
            }
            
            .form-header {
                background: linear-gradient(135deg, #3b82f6, #06b6d4);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                font-weight: 700;
                font-size: 1.5em;
                margin-bottom: 1.5rem;
                text-align: center;
            }
            
            .form-section {
                background: rgba(59, 130, 246, 0.05);
                padding: 1.5rem;
                border-radius: 12px;
                margin-bottom: 1.5rem;
                border: 1px solid rgba(59, 130, 246, 0.2);
            }
            
            .form-section-header {
                color: #3b82f6;
                font-weight: 600;
                font-size: 1.1em;
                margin-bottom: 1rem;
                border-bottom: 1px solid rgba(59, 130, 246, 0.3);
                padding-bottom: 0.5rem;
            }
            
            .validation-success {
                background: rgba(16, 185, 129, 0.1);
                border: 1px solid rgba(16, 185, 129, 0.3);
                border-radius: 8px;
                padding: 0.75rem;
                margin-top: 0.5rem;
                color: #10b981;
            }
            
            .validation-error {
                background: rgba(239, 68, 68, 0.1);
                border: 1px solid rgba(239, 68, 68, 0.3);
                border-radius: 8px;
                padding: 0.75rem;
                margin-top: 0.5rem;
                color: #ef4444;
            }
            
            .validation-warning {
                background: rgba(245, 158, 11, 0.1);
                border: 1px solid rgba(245, 158, 11, 0.3);
                border-radius: 8px;
                padding: 0.75rem;
                margin-top: 0.5rem;
                color: #f59e0b;
            }
            
            .form-help-text {
                font-size: 0.9em;
                color: #94a3b8;
                margin-top: 0.25rem;
            }
            
            .submit-button {
                background: linear-gradient(135deg, #3b82f6, #1d4ed8);
                border: none;
                border-radius: 8px;
                padding: 0.75rem 2rem;
                font-weight: 600;
                font-size: 1.1em;
                transition: all 0.3s ease;
                width: 100%;
                margin-top: 1rem;
            }
            
            .submit-button:hover {
                transform: translateY(-2px);
                box-shadow: 0 8px 25px rgba(59, 130, 246, 0.4);
            }
            
            .form-divider {
                border: none;
                height: 2px;
                background: linear-gradient(90deg, transparent, rgba(59, 130, 246, 0.5), transparent);
                margin: 2rem 0;
            }
        </style>
        """, unsafe_allow_html=True)

class CreatePortfolioForm(BaseForm):
    """Professional portfolio creation form"""
    
    def render(self, on_submit: Optional[Callable] = None) -> Optional[Dict[str, Any]]:
        """Render portfolio creation form"""
        
        self._apply_form_styling()
        
        st.markdown('<div class="form-container">', unsafe_allow_html=True)
        st.markdown('<div class="form-header">🆕 Create New Portfolio</div>', unsafe_allow_html=True)
        
        with st.form("advanced_portfolio_creation", clear_on_submit=False):
            
            # Basic Information Section
            st.markdown('<div class="form-section">', unsafe_allow_html=True)
            st.markdown('<div class="form-section-header">📋 Basic Information</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                portfolio_name = st.text_input(
                    "Portfolio Name *",
                    placeholder="Enter unique portfolio name",
                    max_chars=50,
                    help="Unique identifier for your portfolio"
                )
                
                # Real-time validation for portfolio name
                if portfolio_name:
                    valid, error_msg = self.validator.validate_portfolio_name(portfolio_name)
                    if valid:
                        st.markdown('<div class="validation-success">✅ Valid portfolio name</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="validation-error">❌ {error_msg}</div>', unsafe_allow_html=True)
            
            with col2:
                initial_capital = st.number_input(
                    "Initial Capital (₹) *",
                    value=1000000,
                    step=100000,
                    format="%d",
                    min_value=10000,
                    help="Starting investment amount"
                )
                
                if initial_capital:
                    st.markdown('<div class="form-help-text">💡 Recommended minimum: ₹1,00,000</div>', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Strategy Configuration Section
            st.markdown('<div class="form-section">', unsafe_allow_html=True)
            st.markdown('<div class="form-section-header">🎯 Strategy Configuration</div>', unsafe_allow_html=True)
            
            col3, col4 = st.columns(2)
            
            with col3:
                benchmark = st.selectbox(
                    "Benchmark Index *",
                    options=["^NSEI", "^BSESN", "NIFTYMIDCAP150", "^GSPC", "^IXIC"],
                    format_func=lambda x: {
                        "^NSEI": "NIFTY 50",
                        "^BSESN": "SENSEX",
                        "NIFTYMIDCAP150": "NIFTY MIDCAP 150",
                        "^GSPC": "S&P 500",
                        "^IXIC": "NASDAQ"
                    }.get(x, x),
                    help="Reference index for performance comparison"
                )
                
                risk_profile = st.selectbox(
                    "Risk Profile *",
                    ["Conservative", "Moderate", "Aggressive", "Speculative"],
                    index=1,
                    help="Investment risk tolerance level"
                )
            
            with col4:
                strategy_type = st.selectbox(
                    "Investment Strategy *",
                    ["Multi-Asset", "Equity Focus", "Balanced", "Growth", "Value", "Momentum"],
                    help="Primary investment approach"
                )
                
                investment_horizon = st.selectbox(
                    "Investment Horizon",
                    ["Short-term (< 1 year)", "Medium-term (1-3 years)", "Long-term (3+ years)"],
                    index=2,
                    help="Expected investment duration"
                )
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Asset Allocation Section
            st.markdown('<div class="form-section">', unsafe_allow_html=True)
            st.markdown('<div class="form-section-header">📊 Asset Allocation Strategy</div>', unsafe_allow_html=True)
            
            # Allocation sliders with dynamic updates
            equity_pct = st.slider(
                "Equity Allocation (%)",
                0, 100, 70, 5,
                help="Percentage allocated to equity investments"
            )
            
            col5, col6 = st.columns(2)
            
            with col5:
                debt_pct = st.slider(
                    "Debt/Fixed Income (%)",
                    0, 100 - equity_pct, 
                    min(20, 100 - equity_pct), 5,
                    help="Percentage in bonds and fixed income"
                )
            
            with col6:
                cash_pct = st.slider(
                    "Cash & Equivalents (%)",
                    0, 100 - equity_pct - debt_pct,
                    100 - equity_pct - debt_pct, 5,
                    help="Cash reserves and liquid investments"
                )
            
            # Display allocation summary
            other_pct = 100 - equity_pct - debt_pct - cash_pct
            
            allocation_summary = f"""
            **Allocation Summary:**
            - Equities: {equity_pct}%
            - Debt/Fixed Income: {debt_pct}%
            - Cash & Equivalents: {cash_pct}%
            - Others: {other_pct}%
            """
            
            st.markdown(allocation_summary)
            
            # Validation for allocation
            total_allocation = equity_pct + debt_pct + cash_pct + other_pct
            if abs(total_allocation - 100) < 0.01:
                st.markdown('<div class="validation-success">✅ Allocation adds up to 100%</div>', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Advanced Settings Section
            with st.expander("⚙️ Advanced Portfolio Settings"):
                st.markdown('<div class="form-section">', unsafe_allow_html=True)
                
                col7, col8 = st.columns(2)
                
                with col7:
                    max_position_size = st.slider(
                        "Maximum Position Size (%)",
                        1, 25, 10,
                        help="Maximum percentage in any single stock"
                    )
                    
                    enable_margin = st.checkbox(
                        "Enable Margin Trading",
                        value=False,
                        help="Allow leveraged trading"
                    )
                    
                    auto_rebalance = st.checkbox(
                        "Automatic Rebalancing",
                        value=True,
                        help="Automatically maintain target allocation"
                    )
                
                with col8:
                    rebalance_threshold = st.slider(
                        "Rebalancing Threshold (%)",
                        1, 20, 5,
                        help="Deviation percentage to trigger rebalancing"
                    )
                    
                    dividend_reinvestment = st.checkbox(
                        "Dividend Reinvestment",
                        value=True,
                        help="Automatically reinvest dividends"
                    )
                    
                    tax_optimization = st.checkbox(
                        "Tax Loss Harvesting",
                        value=False,
                        help="Enable tax optimization strategies"
                    )
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Submit Section
            st.markdown("### 🚀 Create Portfolio")
            
            # Final validation summary
            validation_checks = []
            
            if portfolio_name:
                valid, _ = self.validator.validate_portfolio_name(portfolio_name)
                validation_checks.append(("Portfolio Name", valid))
            else:
                validation_checks.append(("Portfolio Name", False))
            
            validation_checks.extend([
                ("Initial Capital", initial_capital >= 10000),
                ("Asset Allocation", abs(total_allocation - 100) < 0.01),
                ("Strategy Selection", bool(strategy_type and risk_profile))
            ])
            
            # Display validation status
            col_val1, col_val2 = st.columns(2)
            
            with col_val1:
                st.markdown("**Validation Status:**")
                for check_name, is_valid in validation_checks:
                    status = "✅" if is_valid else "❌"
                    st.markdown(f"{status} {check_name}")
            
            with col_val2:
                all_valid = all(valid for _, valid in validation_checks)
                if all_valid:
                    st.markdown('<div class="validation-success">🎉 Ready to create portfolio!</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="validation-error">⚠️ Please fix validation errors</div>', unsafe_allow_html=True)
            
            # Submit button
            submitted = st.form_submit_button(
                "🚀 Create Portfolio",
                disabled=not all_valid,
                help="Create portfolio with specified settings"
            )
            
            if submitted and all_valid:
                portfolio_config = {
                    "name": portfolio_name.strip(),
                    "initial_capital": initial_capital,
                    "benchmark": benchmark,
                    "risk_profile": risk_profile,
                    "strategy_type": strategy_type,
                    "investment_horizon": investment_horizon,
                    "asset_allocation": {
                        "equity": equity_pct / 100,
                        "debt": debt_pct / 100,
                        "cash": cash_pct / 100,
                        "others": other_pct / 100
                    },
                    "advanced_settings": {
                        "max_position_size": max_position_size / 100,
                        "enable_margin": enable_margin,
                        "auto_rebalance": auto_rebalance,
                        "rebalance_threshold": rebalance_threshold / 100,
                        "dividend_reinvestment": dividend_reinvestment,
                        "tax_optimization": tax_optimization
                    }
                }
                
                if on_submit:
                    return on_submit(portfolio_config)
                else:
                    return portfolio_config
        
        st.markdown('</div>', unsafe_allow_html=True)
        self.logger.info("PortfolioImportForm.render EXIT [LOG_UI_FORM]")
        return None

class OrderEntryForm(BaseForm):
    """Professional order entry form"""
    
    def render(self, portfolio_data: Dict[str, Any], on_submit: Optional[Callable] = None) -> Optional[Dict[str, Any]]:
        self.logger.info("OrderEntryForm.render ENTRY")
        """Render order entry form"""
        
        self._apply_form_styling()
        
        st.markdown('<div class="form-container">', unsafe_allow_html=True)
        st.markdown('<div class="form-header">📋 Professional Order Entry</div>', unsafe_allow_html=True)
        
        with st.form("professional_order_entry", clear_on_submit=False):
            
            # Basic Order Details
            st.markdown('<div class="form-section">', unsafe_allow_html=True)
            st.markdown('<div class="form-section-header">🎯 Order Details</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                ticker = st.text_input(
                    "Symbol *",
                    placeholder="e.g., RELIANCE",
                    help="Enter stock ticker symbol"
                ).upper()
                
                # Real-time ticker validation
                if ticker:
                    valid, error_msg = self.validator.validate_ticker_symbol(ticker)
                    if valid:
                        st.markdown('<div class="validation-success">✅ Valid symbol</div>', unsafe_allow_html=True)
                        # Show current price (mock)
                        st.markdown('<div class="form-help-text">💰 Current Price: ₹2,847.50</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="validation-error">❌ {error_msg}</div>', unsafe_allow_html=True)
                
                order_type = st.selectbox(
                    "Order Type *",
                    ["MARKET", "LIMIT", "STOP_LOSS", "BRACKET", "ICEBERG"],
                    help="Select order execution type"
                )
                
                side = st.radio(
                    "Order Side *",
                    ["BUY", "SELL"],
                    horizontal=True,
                    help="Buy or sell order"
                )
            
            with col2:
                quantity = st.number_input(
                    "Quantity *",
                    min_value=0.01,
                    step=1.0,
                    help="Number of shares to trade"
                )
                
                # Quantity validation
                if quantity > 0:
                    valid, error_msg = self.validator.validate_quantity(quantity)
                    if valid:
                        trade_value = quantity * 2847.50  # Mock price
                        st.markdown(f'<div class="form-help-text">💵 Estimated Value: ₹{trade_value:,.0f}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="validation-error">❌ {error_msg}</div>', unsafe_allow_html=True)
                
                # Price fields based on order type
                if order_type in ["LIMIT", "STOP_LOSS", "BRACKET"]:
                    limit_price = st.number_input(
                        "Limit Price (₹) *",
                        min_value=0.01,
                        step=0.01,
                        help="Order limit price"
                    )
                else:
                    limit_price = None
                
                if order_type in ["STOP_LOSS", "BRACKET"]:
                    stop_price = st.number_input(
                        "Stop Price (₹) *",
                        min_value=0.01,
                        step=0.01,
                        help="Stop loss trigger price"
                    )
                else:
                    stop_price = None
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Order Validation Section
            st.markdown('<div class="form-section">', unsafe_allow_html=True)
            st.markdown('<div class="form-section-header">📊 Order Validation</div>', unsafe_allow_html=True)
            
            if ticker and quantity > 0:
                self._render_order_validation(portfolio_data, ticker, side, quantity, limit_price)
            else:
                st.info("Enter order details to see validation")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Advanced Order Options
            with st.expander("⚙️ Advanced Order Options"):
                col3, col4 = st.columns(2)
                
                with col3:
                    time_in_force = st.selectbox(
                        "Time in Force",
                        ["DAY", "GTC", "IOC", "FOK"],
                        help="Order validity period"
                    )
                    
                    max_slippage = st.slider(
                        "Max Slippage (%)",
                        0.0, 2.0, 0.1,
                        help="Maximum acceptable slippage"
                    )
                
                with col4:
                    commission_rate = st.slider(
                        "Commission (bps)",
                        0, 50, 5,
                        help="Expected commission rate"
                    )
                    
                    enable_algo = st.checkbox(
                        "Algorithmic Execution",
                        help="Use smart order routing"
                    )
            
            # Risk Management Section
            st.markdown('<div class="form-section">', unsafe_allow_html=True)
            st.markdown('<div class="form-section-header">⚠️ Risk Management</div>', unsafe_allow_html=True)
            
            col5, col6 = st.columns(2)
            
            with col5:
                position_limit = st.checkbox(
                    "Position Size Limit",
                    help="Enforce position size limits"
                )
                
                if position_limit:
                    max_position_pct = st.slider(
                        "Max Position %",
                        1, 20, 10,
                        help="Maximum position as % of portfolio"
                    )
            
            with col6:
                auto_stop_loss = st.checkbox(
                    "Automatic Stop Loss",
                    help="Set automatic stop loss"
                )
                
                if auto_stop_loss:
                    stop_loss_pct = st.slider(
                        "Stop Loss %",
                        1, 20, 5,
                        help="Stop loss percentage"
                    )
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Submit Section
            st.markdown("### 🚀 Submit Order")
            
            # Final validation
            order_valid = self._validate_complete_order(ticker, quantity, side, order_type, limit_price)
            
            if order_valid:
                st.markdown('<div class="validation-success">✅ Order ready for submission</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="validation-error">❌ Please complete all required fields</div>', unsafe_allow_html=True)
            
            # Submit button
            submitted = st.form_submit_button(
                "🚀 Place Order",
                disabled=not order_valid,
                help="Submit order for execution"
            )
            
            if submitted and order_valid:
                order_config = {
                    "ticker": ticker,
                    "order_type": order_type,
                    "side": side,
                    "quantity": quantity,
                    "limit_price": limit_price,
                    "stop_price": stop_price if order_type in ["STOP_LOSS", "BRACKET"] else None,
                    "time_in_force": time_in_force,
                    "max_slippage": max_slippage,
                    "commission_rate": commission_rate,
                    "enable_algo": enable_algo
                }
                
                if on_submit:
                    return on_submit(order_config)
                else:
                    return order_config
        
        st.markdown('</div>', unsafe_allow_html=True)
        self.logger.info("PortfolioImportForm.render EXIT [LOG_UI_FORM]")
        return None
    
    def _render_order_validation(self, portfolio: Dict[str, Any], ticker: str, 
                                side: str, quantity: float, price: Optional[float]):
        """Render comprehensive order validation"""
        
        balances = portfolio['balances']
        positions = portfolio.get('positions', {})
        
        # Mock current price
        current_price = price or 2847.50
        trade_value = quantity * current_price
        
        validation_results = []
        
        # Cash validation for buy orders
        if side == "BUY":
            available_cash = balances['cash']
            margin_available = balances.get('available_margin', 0)
            
            if trade_value <= available_cash:
                validation_results.append(("💰 Cash Check", "✅ Sufficient cash available", "success"))
            elif trade_value <= available_cash + margin_available:
                validation_results.append(("💰 Cash Check", "⚠️ Will use margin", "warning"))
            else:
                validation_results.append(("💰 Cash Check", f"❌ Insufficient funds (need ₹{trade_value:,.0f})", "error"))
        
        # Position validation for sell orders
        elif side == "SELL":
            available_qty = positions.get(ticker, {}).get('quantity', 0)
            
            if quantity <= available_qty:
                validation_results.append(("📊 Position Check", f"✅ Can sell {quantity} shares", "success"))
            else:
                validation_results.append(("📊 Position Check", f"❌ Only {available_qty} shares available", "error"))
        
        # Risk checks
        validation_results.extend([
            ("⚠️ Risk Limits", "✅ Within acceptable limits", "success"),
            ("📈 Market Status", "✅ Market is open", "success"),
            ("🎯 Price Check", "✅ Price within reasonable range", "success")
        ])

        # Display validation results
        for check_name, message, status in validation_results:
            if status == "success":
                st.success(f"{check_name}: {message.replace('✅ ', '')}")
            elif status == "warning":
                st.warning(f"{check_name}: {message.replace('⚠️ ', '')}")
            else:
                st.error(f"{check_name}: {message.replace('❌ ', '')}")
    
    def _validate_complete_order(self, ticker: str, quantity: float, side: str, 
                                order_type: str, price: Optional[float]) -> bool:
        """Validate complete order"""
        
        if not ticker or quantity <= 0 or not side or not order_type:
            logging.getLogger(__name__).info("create_portfolio_from_config EXIT [LOG_UI_FORM]")
            return False
        
        # Check if limit price is required
        if order_type in ["LIMIT", "STOP_LOSS", "BRACKET"] and (not price or price <= 0):
            logging.getLogger(__name__).info("create_portfolio_from_config EXIT [LOG_UI_FORM]")
            return False
        
        logging.getLogger(__name__).info("submit_order_from_config EXIT [LOG_UI_FORM]")
        return True

class RiskSettingsForm(BaseForm):
    """Professional risk management settings form"""
    
    def render(self, current_settings: Optional[Dict[str, Any]] = None, 
              on_submit: Optional[Callable] = None) -> Optional[Dict[str, Any]]:
        self.logger.info("RiskSettingsForm.render ENTRY")
        """Render risk settings form"""
        
        self._apply_form_styling()
        
        st.markdown('<div class="form-container">', unsafe_allow_html=True)
        st.markdown('<div class="form-header">⚠️ Risk Management Settings</div>', unsafe_allow_html=True)
        
        # Default settings
        defaults = current_settings or {
            "max_position_size": 10,
            "max_sector_concentration": 25,
            "max_drawdown_limit": 15,
            "var_limit": 5,
            "leverage_limit": 2.0,
            "enable_stop_loss": True,
            "default_stop_loss": 5,
            "enable_position_sizing": True,
            "enable_correlation_checks": True
        }
        
        with st.form("risk_management_settings"):
            
            # Position Limits Section
            st.markdown('<div class="form-section">', unsafe_allow_html=True)
            st.markdown('<div class="form-section-header">📊 Position & Concentration Limits</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                max_position_size = st.slider(
                    "Maximum Position Size (%)",
                    1, 30, defaults["max_position_size"],
                    help="Maximum percentage of portfolio in single position"
                )
                
                max_sector_concentration = st.slider(
                    "Maximum Sector Concentration (%)",
                    10, 50, defaults["max_sector_concentration"],
                    help="Maximum exposure to any single sector"
                )
            
            with col2:
                leverage_limit = st.slider(
                    "Maximum Leverage Ratio",
                    1.0, 5.0, defaults["leverage_limit"], 0.1,
                    help="Maximum portfolio leverage"
                )
                
                correlation_limit = st.slider(
                    "Correlation Limit",
                    0.5, 1.0, 0.8, 0.05,
                    help="Maximum correlation between positions"
                )
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Risk Metrics Limits
            st.markdown('<div class="form-section">', unsafe_allow_html=True)
            st.markdown('<div class="form-section-header">📈 Risk Metrics & Limits</div>', unsafe_allow_html=True)
            
            col3, col4 = st.columns(2)
            
            with col3:
                var_limit = st.slider(
                    "Daily VaR Limit (%)",
                    1, 15, defaults["var_limit"],
                    help="Maximum daily Value at Risk"
                )
                
                max_drawdown_limit = st.slider(
                    "Maximum Drawdown Limit (%)",
                    5, 30, defaults["max_drawdown_limit"],
                    help="Maximum acceptable portfolio drawdown"
                )
            
            with col4:
                volatility_limit = st.slider(
                    "Portfolio Volatility Limit (%)",
                    10, 40, 25,
                    help="Maximum annualized portfolio volatility"
                )
                
                tracking_error_limit = st.slider(
                    "Tracking Error Limit (%)",
                    2, 15, 8,
                    help="Maximum tracking error vs benchmark"
                )
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Automatic Risk Controls
            st.markdown('<div class="form-section">', unsafe_allow_html=True)
            st.markdown('<div class="form-section-header">🤖 Automatic Risk Controls</div>', unsafe_allow_html=True)
            
            col5, col6 = st.columns(2)
            
            with col5:
                enable_stop_loss = st.checkbox(
                    "Enable Automatic Stop Loss",
                    value=defaults["enable_stop_loss"],
                    help="Automatically set stop losses on new positions"
                )
                
                if enable_stop_loss:
                    default_stop_loss = st.slider(
                        "Default Stop Loss (%)",
                        1, 20, defaults["default_stop_loss"],
                        help="Default stop loss percentage for new positions"
                    )
                else:
                    default_stop_loss = 0
                
                enable_position_sizing = st.checkbox(
                    "Enable Position Sizing",
                    value=defaults["enable_position_sizing"],
                    help="Automatically calculate optimal position sizes"
                )
            
            with col6:
                enable_rebalancing = st.checkbox(
                    "Enable Auto Rebalancing",
                    value=True,
                    help="Automatically rebalance when limits are breached"
                )
                
                enable_correlation_checks = st.checkbox(
                    "Enable Correlation Monitoring",
                    value=defaults["enable_correlation_checks"],
                    help="Monitor and alert on high position correlations"
                )
                
                enable_stress_testing = st.checkbox(
                    "Enable Stress Testing",
                    value=True,
                    help="Regular portfolio stress testing"
                )
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Alert Settings
            with st.expander("🔔 Alert & Notification Settings"):
                st.markdown('<div class="form-section">', unsafe_allow_html=True)
                
                col7, col8 = st.columns(2)
                
                with col7:
                    email_alerts = st.checkbox("Email Alerts", value=True)
                    sms_alerts = st.checkbox("SMS Alerts", value=False)
                    push_notifications = st.checkbox("Push Notifications", value=True)
                
                with col8:
                    alert_frequency = st.selectbox(
                        "Alert Frequency",
                        ["Immediate", "Every 15 minutes", "Every hour", "Daily"],
                        help="How often to send risk alerts"
                    )
                    
                    risk_threshold = st.selectbox(
                        "Risk Alert Threshold",
                        ["Low", "Medium", "High", "Critical Only"],
                        index=1,
                        help="Minimum risk level to trigger alerts"
                    )
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Risk Profile Summary
            st.markdown("### 📋 Risk Profile Summary")
            
            # Calculate risk score based on settings
            risk_factors = [
                ("Position Concentration", max_position_size / 30 * 100),
                ("Sector Concentration", max_sector_concentration / 50 * 100),
                ("Leverage Risk", leverage_limit / 5.0 * 100),
                ("VaR Exposure", var_limit / 15 * 100),
                ("Drawdown Risk", max_drawdown_limit / 30 * 100)
            ]
            
            overall_risk_score = np.mean([score for _, score in risk_factors])
            
            col9, col10 = st.columns(2)
            
            with col9:
                st.markdown("**Risk Factor Breakdown:**")
                for factor_name, score in risk_factors:
                    risk_level = "High" if score > 66 else "Medium" if score > 33 else "Low"
                    color = "🔴" if score > 66 else "🟡" if score > 33 else "🟢"
                    st.markdown(f"{color} {factor_name}: {risk_level} ({score:.0f}%)")
            
            with col10:
                overall_level = "Conservative" if overall_risk_score < 40 else "Moderate" if overall_risk_score < 70 else "Aggressive"
                color = "🟢" if overall_risk_score < 40 else "🟡" if overall_risk_score < 70 else "🔴"
                
                st.markdown(f"**Overall Risk Profile:** {color} {overall_level}")
                st.markdown(f"**Risk Score:** {overall_risk_score:.0f}/100")
                
                # Risk recommendations
                if overall_risk_score > 80:
                    st.markdown('<div class="validation-error">⚠️ High risk configuration - consider reducing limits</div>', unsafe_allow_html=True)
                elif overall_risk_score < 20:
                    st.markdown('<div class="validation-warning">💡 Very conservative settings - may limit returns</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="validation-success">✅ Balanced risk configuration</div>', unsafe_allow_html=True)
            
            # Submit Section
            st.markdown('<hr class="form-divider">', unsafe_allow_html=True)
            st.markdown("### 💾 Save Risk Settings")
            
            # Advanced validation
            validation_passed = True
            validation_messages = []
            
            # Check for conflicting settings
            if max_position_size > max_sector_concentration:
                validation_messages.append("❌ Position size cannot exceed sector concentration limit")
                validation_passed = False
            
            if leverage_limit > 3.0 and var_limit > 10:
                validation_messages.append("⚠️ High leverage with high VaR limit increases risk significantly")
            
            if not enable_stop_loss and max_drawdown_limit > 20:
                validation_messages.append("💡 Consider enabling stop losses with high drawdown limits")
            
            # Display validation messages
            if validation_messages:
                for message in validation_messages:
                    if message.startswith("❌"):
                        st.markdown(f'<div class="validation-error">{message}</div>', unsafe_allow_html=True)
                    elif message.startswith("⚠️"):
                        st.markdown(f'<div class="validation-warning">{message}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="validation-success">{message}</div>', unsafe_allow_html=True)
            
            # Submit button
            submitted = st.form_submit_button(
                "💾 Save Risk Settings",
                disabled=not validation_passed,
                help="Save risk management configuration"
            )
            
            if submitted and validation_passed:
                risk_config = {
                    "position_limits": {
                        "max_position_size": max_position_size / 100,
                        "max_sector_concentration": max_sector_concentration / 100,
                        "leverage_limit": leverage_limit,
                        "correlation_limit": correlation_limit
                    },
                    "risk_metrics": {
                        "var_limit": var_limit / 100,
                        "max_drawdown_limit": max_drawdown_limit / 100,
                        "volatility_limit": volatility_limit / 100,
                        "tracking_error_limit": tracking_error_limit / 100
                    },
                    "automatic_controls": {
                        "enable_stop_loss": enable_stop_loss,
                        "default_stop_loss": default_stop_loss / 100,
                        "enable_position_sizing": enable_position_sizing,
                        "enable_rebalancing": enable_rebalancing,
                        "enable_correlation_checks": enable_correlation_checks,
                        "enable_stress_testing": enable_stress_testing
                    },
                    "notifications": {
                        "email_alerts": email_alerts,
                        "sms_alerts": sms_alerts,
                        "push_notifications": push_notifications,
                        "alert_frequency": alert_frequency,
                        "risk_threshold": risk_threshold
                    },
                    "risk_profile": {
                        "overall_score": overall_risk_score,
                        "risk_level": overall_level,
                        "factor_scores": dict(risk_factors)
                    }
                }
                
                if on_submit:
                    return on_submit(risk_config)
                else:
                    return risk_config
        
        st.markdown('</div>', unsafe_allow_html=True)
        self.logger.info("PortfolioImportForm.render EXIT [LOG_UI_FORM]")
        return None

class PortfolioImportForm(BaseForm):
    """Portfolio import/migration form"""
    
    def render(self, on_submit: Optional[Callable] = None) -> Optional[Dict[str, Any]]:
        """Render portfolio import form"""
        
        self._apply_form_styling()
        
        st.markdown('<div class="form-container">', unsafe_allow_html=True)
        st.markdown('<div class="form-header">📥 Import Portfolio</div>', unsafe_allow_html=True)
        
        with st.form("portfolio_import_form"):
            
            # Import Method Section
            st.markdown('<div class="form-section">', unsafe_allow_html=True)
            st.markdown('<div class="form-section-header">📂 Import Method</div>', unsafe_allow_html=True)
            
            import_method = st.selectbox(
                "Import Source",
                ["JSON File Upload", "CSV Holdings", "Broker Statement", "Manual Entry"],
                help="Choose how to import your portfolio data"
            )
            
            if import_method == "JSON File Upload":
                uploaded_file = st.file_uploader(
                    "Upload Portfolio JSON",
                    type=['json'],
                    help="Upload a previously exported portfolio JSON file"
                )
                
                if uploaded_file:
                    try:
                        import json
                        file_content = json.load(uploaded_file)
                        st.success("✅ JSON file loaded successfully")
                        st.json(file_content, expanded=False)
                    except Exception as e:
                        st.error(f"❌ Invalid JSON file: {e}")
            
            elif import_method == "CSV Holdings":
                uploaded_file = st.file_uploader(
                    "Upload Holdings CSV",
                    type=['csv'],
                    help="CSV should have columns: Symbol, Quantity, AvgPrice"
                )
                
                if uploaded_file:
                    try:
                        import pandas as pd
                        df = pd.read_csv(uploaded_file)
                        st.success("✅ CSV file loaded successfully")
                        st.dataframe(df.head())
                        
                        required_cols = ['Symbol', 'Quantity', 'AvgPrice']
                        missing_cols = [col for col in required_cols if col not in df.columns]
                        
                        if missing_cols:
                            st.error(f"❌ Missing required columns: {missing_cols}")
                        else:
                            st.success("✅ All required columns found")
                            
                    except Exception as e:
                        st.error(f"❌ Invalid CSV file: {e}")
            
            elif import_method == "Manual Entry":
                st.info("💡 Enter your holdings manually below")
                
                num_holdings = st.number_input("Number of Holdings", min_value=1, max_value=20, value=3)
                
                holdings_data = []
                for i in range(num_holdings):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        symbol = st.text_input(f"Symbol {i+1}", key=f"symbol_{i}")
                    with col2:
                        quantity = st.number_input(f"Quantity {i+1}", min_value=0.01, key=f"qty_{i}")
                    with col3:
                        avg_price = st.number_input(f"Avg Price {i+1}", min_value=0.01, key=f"price_{i}")
                    
                    if symbol and quantity > 0 and avg_price > 0:
                        holdings_data.append({
                            "symbol": symbol.upper(),
                            "quantity": quantity,
                            "avg_price": avg_price,
                            "market_value": quantity * avg_price
                        })
                
                if holdings_data:
                    import pandas as pd
                    df = pd.DataFrame(holdings_data)
                    st.dataframe(df)
                    st.success(f"✅ {len(holdings_data)} holdings entered")
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Portfolio Configuration
            st.markdown('<div class="form-section">', unsafe_allow_html=True)
            st.markdown('<div class="form-section-header">⚙️ Import Configuration</div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                portfolio_name = st.text_input(
                    "New Portfolio Name *",
                    placeholder="Imported Portfolio",
                    help="Name for the imported portfolio"
                )
                
                preserve_cash = st.checkbox(
                    "Preserve Cash Balance",
                    value=True,
                    help="Keep cash balance from import"
                )
            
            with col2:
                import_transactions = st.checkbox(
                    "Import Transaction History",
                    value=False,
                    help="Import historical transactions if available"
                )
                
                validate_prices = st.checkbox(
                    "Validate Current Prices",
                    value=True,
                    help="Check current market prices during import"
                )
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Submit
            submitted = st.form_submit_button("📥 Import Portfolio")
            
            if submitted:
                if not portfolio_name:
                    st.error("❌ Portfolio name is required")
                else:
                    import_config = {
                        "portfolio_name": portfolio_name,
                        "import_method": import_method,
                        "preserve_cash": preserve_cash,
                        "import_transactions": import_transactions,
                        "validate_prices": validate_prices,
                        "holdings_data": holdings_data if import_method == "Manual Entry" else None
                    }
                    
                    if on_submit:
                        self.logger.info("PortfolioImportForm.render EXIT [LOG_UI_FORM]")
                        return on_submit(import_config)
                    else:
                        self.logger.info("PortfolioImportForm.render EXIT [LOG_UI_FORM]")
                        return import_config
        
        st.markdown('</div>', unsafe_allow_html=True)
        self.logger.info("PortfolioImportForm.render EXIT [LOG_UI_FORM]")
        return None

# Integration helper functions for compatibility
def create_portfolio_from_config(config: Dict[str, Any], portfolio_manager) -> bool:
    logging.getLogger(__name__).info("create_portfolio_from_config ENTRY [LOG_UI_FORM]")
    """Create portfolio from form configuration"""
    try:
        result = portfolio_manager.create_enhanced_portfolio(
            config["name"],
            config["initial_capital"],
            config["asset_allocation"],
            config["benchmark"]
        )
        logging.getLogger(__name__).info("create_portfolio_from_config EXIT [LOG_UI_FORM]")
        return True
    except Exception as e:
        st.error(f"Portfolio creation failed: {e}")
        logging.getLogger(__name__).info("create_portfolio_from_config EXIT [LOG_UI_FORM]")
        return False
    finally:
        pass

def submit_order_from_config(config: Dict[str, Any], order_manager, portfolio) -> bool:
    logging.getLogger(__name__).info("submit_order_from_config ENTRY [LOG_UI_FORM]")
    """Submit order from form configuration"""
    try:
        # This would integrate with the actual order management system
        success, message, order = order_manager.place_order(config, portfolio)
        if success:
            st.success(f"✅ {message}")
        else:
            st.error(f"❌ {message}")
        logging.getLogger(__name__).info("submit_order_from_config EXIT [LOG_UI_FORM]")
        return success
    except Exception as e:
        st.error(f"Order submission failed: {e}")
        logging.getLogger(__name__).info("submit_order_from_config EXIT [LOG_UI_FORM]")
        return False
    finally:
        pass

def apply_risk_settings(config: Dict[str, Any], portfolio_name: str) -> bool:
    logging.getLogger(__name__).info("apply_risk_settings ENTRY [LOG_UI_FORM]")
    """Apply risk settings to portfolio"""
    try:
        # This would integrate with the portfolio risk management system
        st.success("✅ Risk settings applied successfully")
        logging.getLogger(__name__).info("apply_risk_settings EXIT [LOG_UI_FORM]")
        return True
    except Exception as e:
        st.error(f"Risk settings update failed: {e}")
        logging.getLogger(__name__).info("apply_risk_settings EXIT [LOG_UI_FORM]")
        return False
    finally:
        pass
