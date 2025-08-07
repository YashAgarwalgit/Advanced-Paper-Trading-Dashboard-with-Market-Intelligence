"""
Enhanced Institutional Trading Platform V5.0 - Main Application Orchestrator
A next-generation quantitative trading and portfolio management system

Architecture:
- Async market data management
- Atomic portfolio transactions
- Professional order management
- Advanced risk analytics
- Real-time market intelligence
"""

import streamlit as st
import asyncio
import logging
import sys
from datetime import datetime
import sys
import os
import time
from pathlib import Path
from typing import Dict, Any

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Core system imports
from config import Config
from utils.helpers import get_timestamp_iso, format_currency, format_percentage
from utils.decorators import retry_on_failure as error_handler, measure_performance as performance_monitor
from utils.logging_config import setup_logging  

# Data management
from data.async_market_data import AsyncMarketDataManager

# Portfolio management
from portfolio.manager import EnhancedPortfolioManager
from portfolio.transaction_manager import PortfolioTransactionManager
from portfolio.analytics import PortfolioAnalytics, ModernPortfolioTheory

# Trading system
from trading.order_manager import ProfessionalOrderManager, OrderType, OrderSide
from trading.trading_engine import AdvancedTradingEngine
from trading.risk_checks import TradingRiskValidator

# Analytics
from analytics.risk_metrics import RiskMetricsCalculator
from analytics.performance import PerformanceAnalyzer
from analytics.factor_analysis import FactorAnalyzer

# Market intelligence
from market.regime_detection import MarketRegimeDetector
from market.sector_rotation import SectorRotationAnalyzer

# Visualization components
from visualization.portfolio_dashboard import PortfolioDashboard
from visualization.risk_dashboard import RiskDashboard
from visualization.market_charts import MarketCharts
from visualization.gauge_charts import GaugeCharts

# UI components
from ui.sidebar import PortfolioSidebar
from ui.tabs import MarketIntelligenceTab, PortfolioManagementTab, LiveTradingTab
from ui.forms import CreatePortfolioForm, OrderEntryForm, RiskSettingsForm

# Database layer
from database.connection import DatabaseConnection
from database.schema import DatabaseSchema
from database.queries import DatabaseQueries

print("🚀 STARTING: Imports completed successfully")

class InstitutionalTradingPlatform:
    def __init__(self):
        print("🚀 INIT: Platform initialization started")
        self.logger = self._setup_logging()
        self.logger.info("InstitutionalTradingPlatform.__init__ ENTRY")
        print("🚀 INIT: Logging setup completed")
        
        self.config = Config()
        print("🚀 INIT: Config loaded")
        
        # Core components
        self.market_data_manager = None
        self.portfolio_manager = None
        self.order_manager = None
        self.trading_engine = None
        self.risk_validator = None
        
        # Analytics components
        self.risk_calculator = None
        self.performance_analyzer = None
        self.factor_analyzer = None
        
        # Market intelligence
        self.market_intelligence = None
        self.regime_detector = None
        self.sector_analyzer = None
        
        # Visualization
        self.portfolio_dashboard = None
        self.risk_dashboard = None
        self.market_charts = None
        self.gauge_charts = None
        
        # Database
        self.db_manager = None
        
        # UI state
        self.ui_state = {
            'active_portfolio': None,
            'market_data': {},
            'regime_data': {},
            'last_refresh': None,
            'initialized': False
        }
        self.logger.info("InstitutionalTradingPlatform.__init__ EXIT")
        print("🚀 INIT: Platform initialization completed")

    
    def _setup_logging(self) -> logging.Logger:
        """Setup application logging"""
        return setup_logging(
            log_level=logging.INFO,
            log_file="logs/trading_platform.log"
        )
    
    async def initialize_platform(self):
        self.logger.info("InstitutionalTradingPlatform.initialize_platform ENTRY")
        """Initialize all platform components"""
        
        start_time = time.time()  # Manual performance tracking
        
        try:
            self.logger.info("Initializing Enhanced Institutional Trading Platform V5.0")
            
            # Initialize core components
            await self._initialize_core_components()
            
            # Initialize analytics
            self._initialize_analytics()
            
            # Initialize market intelligence
            self._initialize_market_intelligence()
            
            # Initialize visualization
            self._initialize_visualization()
            
            # Initialize database
            await self._initialize_database()
            
            # Setup Streamlit configuration
            self._setup_streamlit_config()
            
            # Load initial market data
            await self._load_initial_data()
            
            execution_time = time.time() - start_time
            self.logger.info(f"Platform initialization completed successfully in {execution_time:.4f}s")
            self.logger.info("InstitutionalTradingPlatform.initialize_platform EXIT")
            return True
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Platform initialization failed after {execution_time:.4f}s: {e}")
            self.logger.info("InstitutionalTradingPlatform.initialize_platform EXIT (error)")
            st.error(f"System initialization failed: {e}")
            return False
    
    async def _initialize_core_components(self):
        self.logger.info("InstitutionalTradingPlatform._initialize_core_components ENTRY")
        """Initialize core trading components"""
        
        # Market data manager with async capabilities
        self.market_data_manager = AsyncMarketDataManager(
            max_workers=self.config.MAX_WORKERS,
            timeout=self.config.REQUEST_TIMEOUT
        )
        
        # Portfolio management with atomic transactions
        self.portfolio_manager = EnhancedPortfolioManager(
            portfolios_dir=self.config.PORTFOLIOS_DIR
        )
        
        # Trading engine with advanced execution
        self.trading_engine = AdvancedTradingEngine()
        
        # Order manager with institutional features
        self.order_manager = ProfessionalOrderManager(
            self.market_data_manager
        )
        
        # Risk validation system
        self.risk_validator = TradingRiskValidator()
        
        self.logger.info("Core components initialized")
        self.logger.info("InstitutionalTradingPlatform._initialize_core_components EXIT")
    
    def _initialize_analytics(self):
        self.logger.info("InstitutionalTradingPlatform._initialize_analytics ENTRY")
        """Initialize analytics components"""
        
        self.risk_calculator = RiskMetricsCalculator(
            risk_free_rate=self.config.DEFAULT_RISK_FREE_RATE
        )
        
        self.performance_analyzer = PerformanceAnalyzer(
            risk_free_rate=self.config.DEFAULT_RISK_FREE_RATE
        )
        
        self.factor_analyzer = FactorAnalyzer()
        
        self.logger.info("Analytics components initialized")
        self.logger.info("InstitutionalTradingPlatform._initialize_analytics EXIT")
    
    def _initialize_market_intelligence(self):
        self.logger.info("InstitutionalTradingPlatform._initialize_market_intelligence ENTRY")
        """Initialize enhanced market intelligence system"""
        
        # Use the new enhanced system
        from market.intelligence import create_market_intelligence
        
        self.market_intelligence = create_market_intelligence(
            data_manager=self.market_data_manager
        )
        
        self.logger.info("Enhanced market intelligence system initialized")
        self.logger.info("InstitutionalTradingPlatform._initialize_market_intelligence EXIT")

    
    def _initialize_visualization(self):
        self.logger.info("InstitutionalTradingPlatform._initialize_visualization ENTRY")
        """Initialize visualization components"""
        
        self.portfolio_dashboard = PortfolioDashboard()
        self.risk_dashboard = RiskDashboard()
        self.market_charts = MarketCharts()
        self.gauge_charts = GaugeCharts()
        
        self.logger.info("Visualization components initialized")
        self.logger.info("InstitutionalTradingPlatform._initialize_visualization EXIT")
    
    async def _initialize_database(self):
        self.logger.info("InstitutionalTradingPlatform._initialize_database ENTRY")
        """Initialize database layer"""
        
        try:
            # Create database connection
            self.db_manager = DatabaseConnection(
                db_path=self.config.DATABASE_PATH,
                pool_size=self.config.CONNECTION_POOL_SIZE
            )
            await self.db_manager.initialize()
            
            schema = DatabaseSchema(self.db_manager)
            
            schema.initialize_schema()
            
            self.logger.info("Database initialized successfully")
            self.logger.info("InstitutionalTradingPlatform._initialize_database EXIT")
            
        except Exception as e:
            self.logger.warning(f"Database initialization failed: {e}")
            self.logger.info("InstitutionalTradingPlatform._initialize_database EXIT (error)")
            self.db_manager = None

    
    def _setup_streamlit_config(self):
        self.logger.info("InstitutionalTradingPlatform._setup_streamlit_config ENTRY")
        """Setup Streamlit page configuration and styling"""
        
        st.set_page_config(
            page_title="🚀 Institutional Trading Platform V5.0",
            page_icon="⚡",
            layout="wide",
            initial_sidebar_state="expanded",
            menu_items={
                'Get Help': 'https://github.com/your-repo/help',
                'Report a bug': 'https://github.com/your-repo/issues',
                'About': "Enhanced Institutional Trading Platform V5.0\n\nNext-generation quantitative trading system"
            }
        )
        
        # Load custom CSS
        self._load_custom_styling()
        self.logger.info("InstitutionalTradingPlatform._setup_streamlit_config EXIT")
    
    def _load_custom_styling(self):
        self.logger.info("InstitutionalTradingPlatform._load_custom_styling ENTRY")
        """Load custom CSS styling"""
        
        st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
            
            .stApp {
                font-family: 'Inter', sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            }
            
            .main-header {
                background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
                padding: 2rem;
                border-radius: 15px;
                color: white;
                text-align: center;
                margin-bottom: 2rem;
                box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            }
            
            .metric-card {
                background: rgba(255, 255, 255, 0.1);
                padding: 1.5rem;
                border-radius: 12px;
                border-left: 4px solid #00d4aa;
                backdrop-filter: blur(15px);
                box-shadow: 0 8px 25px rgba(0,0,0,0.1);
                margin-bottom: 1rem;
            }
            
            .status-indicator {
                display: inline-block;
                width: 12px;
                height: 12px;
                border-radius: 50%;
                margin-right: 8px;
                animation: pulse 2s infinite;
            }
            
            .status-live { background-color: #51cf66; }
            .status-warning { background-color: #ffd43b; }
            .status-error { background-color: #ff6b6b; }
            
            @keyframes pulse {
                0%, 100% { opacity: 1; }
                50% { opacity: 0.5; }
            }
            
            [data-testid="metric-container"] {
                background: rgba(255, 255, 255, 0.1);
                border: 1px solid rgba(255, 255, 255, 0.2);
                padding: 1rem;
                border-radius: 10px;
                box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            }
            
            .stDataFrame {
                border-radius: 10px;
                overflow: hidden;
                box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            }
        </style>
        """, unsafe_allow_html=True)
    
    async def _load_initial_data(self):
        self.logger.info("InstitutionalTradingPlatform._load_initial_data ENTRY")
        """Load ONLY real-time market data - strict no fallback policy"""
        
        start_time = time.time()
        
        try:
            # Essential tickers - optimized for reliability
            key_tickers = [
                "^NSEI", "^NSEBANK", "^GSPC", "USDINR=X", 
                "^CNXIT", "^CNXAUTO", "^CNXFMCG"
            ]
            
            # Set loading state immediately
            self.ui_state['loading'] = True
            self.ui_state['data_status'] = 'fetching_live_data'
            
            # Reduced attempts with faster timeout for reliability
            for attempt in range(2):  # Reduced from 3 to 2 attempts
                self.logger.info(f"Attempting live data fetch (attempt {attempt + 1}/2)")
                
                success, market_data = await self.market_data_manager.get_live_prices_async(
                    key_tickers, 
                    force_refresh=True
                )
                
                # Strict validation - only accept real prices
                if success and market_data and self._validate_real_data(market_data):
                    self.ui_state['market_data'] = market_data
                    self.ui_state['last_refresh'] = get_timestamp_iso()
                    self.ui_state['data_source'] = 'live_api'
                    self.ui_state['loading'] = False
                    
                    # Fix: Remove emoji from log message
                    self.logger.info(f"SUCCESS: Live data loaded for {len(market_data)} instruments")
                    self.logger.info("InstitutionalTradingPlatform._load_initial_data EXIT (success)")
                    break
                else:
                    self.logger.warning(f"Live data attempt {attempt + 1} failed")
                    if attempt < 1:
                        await asyncio.sleep(1)  # Reduced backoff time
            else:
                # All attempts failed - DO NOT use fallback
                self.logger.info("InstitutionalTradingPlatform._load_initial_data EXIT (all attempts failed)")
                raise Exception("CRITICAL: All live data sources failed")
            
            # Load real market intelligence with reduced scope
            regime_data = await self._load_real_regime_data_optimized()
            self.ui_state['regime_data'] = regime_data
            
            elapsed = time.time() - start_time
            self.logger.info(f"Real-time data initialization completed in {elapsed:.2f}s")
            
        except Exception as e:
            self.logger.error(f"CRITICAL: Real-time data loading failed: {e}")
            self.logger.info("InstitutionalTradingPlatform._load_initial_data EXIT (error)")
            
            # Show error to user instead of using fallback
            st.error("CRITICAL ERROR: Real-time market data unavailable")
            st.error("This institutional platform requires live data to function properly")
            st.info("Please check your internet connection and API access")
            
            # Stop execution - don't proceed with mock data
            st.stop()

    def _validate_real_data(self, market_data: Dict[str, float]) -> bool:
        self.logger.info("InstitutionalTradingPlatform._validate_real_data ENTRY")
        """
        Validate that market data contains only authentic, real-time prices
        
        Performs comprehensive validation to ensure:
        - Data exists and is not empty
        - All prices are valid numbers and positive
        - Prices fall within realistic market ranges
        - No mock or fallback data is accepted
        
        Args:
            market_data: Dictionary mapping tickers to current prices
            
        Returns:
            bool: True if all data passes validation, False otherwise
        """
        
        try:
            # Check if market data exists and is not empty
            if not market_data or len(market_data) == 0:
                self.logger.warning("Market data validation failed: Empty or None data received")
                return False
            
            # Define realistic price ranges for known instruments
            # These ranges are based on recent market levels and updated periodically
            validation_rules = {
                # Indian Market Indices
                "^NSEI": (15000, 30000),        # NIFTY 50 realistic range
                "^NSEBANK": (35000, 70000),     # NIFTY Bank realistic range  
                "^CNXIT": (25000, 45000),       # NIFTY IT realistic range
                "^CNXAUTO": (10000, 25000),     # NIFTY Auto realistic range
                "^CNXFMCG": (40000, 65000),     # NIFTY FMCG realistic range
                
                # Global Indices
                "^GSPC": (3000, 8000),          # S&P 500 realistic range
                "^DJI": (25000, 45000),         # Dow Jones realistic range
                "^IXIC": (10000, 20000),        # NASDAQ realistic range
                
                # Currency Pairs
                "USDINR=X": (60, 100),          # USD/INR realistic range
                "EURINR=X": (70, 110),          # EUR/INR realistic range
                
                # Volatility Indices
                "^VIX": (8, 80),                # VIX realistic range
                "^INDIAVIX": (8, 60),           # India VIX realistic range
                
                # Commodities (if included)
                "GC=F": (1500, 3000),          # Gold futures realistic range
                "CL=F": (30, 150),              # Crude Oil futures realistic range
            }
            
            # Validate each ticker-price pair
            for ticker, price in market_data.items():
                
                # 1. Type validation - ensure price is numeric
                if not isinstance(price, (int, float)):
                    self.logger.warning(f"Price validation failed for {ticker}: Non-numeric price {price}")
                    return False
                
                # 2. Convert to float for consistency
                try:
                    price_float = float(price)
                except (ValueError, TypeError):
                    self.logger.warning(f"Price validation failed for {ticker}: Cannot convert {price} to float")
                    return False
                
                # 3. Basic range validation - price must be positive
                if price_float <= 0:
                    self.logger.warning(f"Price validation failed for {ticker}: Non-positive price {price_float}")
                    return False
                
                # 4. Upper bound sanity check - no instrument should be above 10 million
                if price_float > 10_000_000:
                    self.logger.warning(f"Price validation failed for {ticker}: Unrealistic high price {price_float}")
                    return False
                
                # 5. Specific instrument validation using realistic ranges
                if ticker in validation_rules:
                    min_price, max_price = validation_rules[ticker]
                    if not (min_price <= price_float <= max_price):
                        self.logger.warning(
                            f"Price validation failed for {ticker}: Price {price_float} outside "
                            f"realistic range [{min_price}, {max_price}]"
                        )
                        return False
                
                # 6. Additional validation for unknown tickers
                else:
                    # For unknown tickers, apply conservative bounds
                    if price_float < 0.001:  # Too small (likely invalid)
                        self.logger.warning(f"Price validation failed for unknown ticker {ticker}: Too small {price_float}")
                        return False
                    
                    if price_float > 1_000_000:  # Too large (likely invalid)
                        self.logger.warning(f"Price validation failed for unknown ticker {ticker}: Too large {price_float}")
                        return False
            
            # 7. Data completeness check - ensure we have expected key instruments
            required_instruments = ["^NSEI", "^NSEBANK"]  # Minimum required for Indian market
            missing_instruments = [inst for inst in required_instruments if inst not in market_data]
            
            if missing_instruments:
                self.logger.warning(f"Price validation failed: Missing required instruments {missing_instruments}")
                return False
            
            # 8. Data freshness validation (optional enhancement)
            # Check if prices seem stale by comparing with expected market hours
            current_time = datetime.now()
            
            # Indian market hours: 9:15 AM to 3:30 PM IST (Monday-Friday)
            # Outside market hours, we still accept data but log it
            if current_time.weekday() < 5:  # Monday = 0, Sunday = 6
                market_open = current_time.replace(hour=9, minute=15, second=0, microsecond=0)
                market_close = current_time.replace(hour=15, minute=30, second=0, microsecond=0)
                
                if not (market_open <= current_time <= market_close):
                    self.logger.info("Market data received outside trading hours - may be pre-market or after-hours")
            
            # All validations passed
            self.logger.info(f"Market data validation successful: {len(market_data)} instruments validated")
            self.logger.info("InstitutionalTradingPlatform._validate_real_data EXIT (success)")
            
            # Log validated instruments for debugging
            self.logger.debug(f"Validated instruments: {list(market_data.keys())}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Market data validation failed with exception: {e}")
            self.logger.info("InstitutionalTradingPlatform._validate_real_data EXIT (error)")
            return False

    async def _load_real_regime_data_optimized(self):
        self.logger.info("InstitutionalTradingPlatform._load_real_regime_data_optimized ENTRY")
        """Load optimized regime data using enhanced intelligence"""
        
        try:
            # Use priority 1 for fastest loading
            report = await self.market_intelligence.get_comprehensive_market_intelligence(
                focus_region="india",
                priority_level=1  # Essential analysis only
            )
            
            # Update UI cache
            self.market_intelligence.update_ui_cache(report)
            
            result = {
                "regime_score": report.regime_data.regime_score,
                "regime_label": report.regime_data.regime_label,
                "confidence": report.regime_data.confidence,
                "market_condition": report.regime_data.market_condition.value,
                "sentiment_level": report.regime_data.sentiment_level.value,
                "loading": False,
                "data_source": "enhanced_intelligence_v2",
                "timestamp": report.timestamp,
                "performance": report.performance_metrics
            }
            
        except Exception as e:
            self.logger.error(f"Enhanced regime data loading failed: {e}")
            self.logger.info("InstitutionalTradingPlatform._load_real_regime_data_optimized EXIT (error)")
            raise Exception(f"Enhanced intelligence failed: {e}")

    async def _calculate_basic_regime_score(self, market_data: dict) -> float:
        self.logger.info("InstitutionalTradingPlatform._calculate_basic_regime_score ENTRY")
        """Calculate basic regime score from live market data only"""
        
        try:
            # Get key Indian market indicators
            nifty_price = market_data.get("^NSEI", 0)
            bank_nifty_price = market_data.get("^NSEBANK", 0)
            usd_inr = market_data.get("USDINR=X", 80)
            
            # Basic momentum calculation (simplified)
            base_score = 5.0  # Neutral
            
            # Market level adjustments based on real prices
            if nifty_price > 25000:  # Above recent highs
                base_score += 1.5
            elif nifty_price < 24000:  # Below recent support
                base_score -= 1.5
            
            if bank_nifty_price > 56000:  # Strong banking sector
                base_score += 1.0
            elif bank_nifty_price < 54000:  # Weak banking sector
                base_score -= 1.0
            
            # Currency impact
            if usd_inr < 85:  # Strong rupee
                base_score += 0.5
            elif usd_inr > 87.5:  # Weak rupee
                base_score -= 0.5
            
            result = max(0, min(10, base_score))
            self.logger.info("InstitutionalTradingPlatform._calculate_basic_regime_score EXIT (success)")
            return result
            
        except Exception as e:
            self.logger.error(f"Basic regime calculation failed: {e}")
            self.logger.info("InstitutionalTradingPlatform._calculate_basic_regime_score EXIT (error)")
            return 5.0  # Neutral default

    def _get_regime_label_from_score(self, score: float) -> str:
        self.logger.info("InstitutionalTradingPlatform._get_regime_label_from_score ENTRY")
        """Convert regime score to descriptive label"""
        
        if score >= 8:
            return "Strong Bullish"
        elif score >= 6.5:
            return "Bullish"
        elif score >= 5.5:
            return "Neutral Positive"
        elif score >= 4.5:
            return "Neutral"
        elif score >= 3:
            return "Bearish"
        else:
            self.logger.info("InstitutionalTradingPlatform._get_regime_label_from_score EXIT")
            return "Strong Bearish"


    async def _load_comprehensive_data_background(self):
        self.logger.info("InstitutionalTradingPlatform._load_comprehensive_data_background ENTRY")
        """Load comprehensive data in background with proper cleanup"""
        
        background_task = None
        try:
            await asyncio.sleep(2)  # Let UI stabilize first
            
            # Store task reference for cleanup
            background_task = asyncio.current_task()
            
            # Check if task was cancelled during sleep
            if background_task and background_task.cancelled():
                self.logger.info("Background data loading was cancelled")
                self.logger.info("InstitutionalTradingPlatform._load_comprehensive_data_background EXIT (cancelled)")
                return
            
            # Load market intelligence with cancellation checks
            regime_data = await self._load_regime_data_with_cancellation()
            if not background_task.cancelled():
                self.ui_state['regime_data'] = regime_data
            
            # Load additional market data
            extended_tickers = ["USDINR=X", "^NSEBANK", "^CNXIT"]
            success, extended_data = await self._load_extended_data_with_cancellation(extended_tickers)
            
            if success and not background_task.cancelled():
                self.ui_state['market_data'].update(extended_data)
            
            self.logger.info("Background data loading completed successfully")
            self.logger.info("InstitutionalTradingPlatform._load_comprehensive_data_background EXIT (success)")
            
        except asyncio.CancelledError:
            self.logger.info("Background data loading was cancelled gracefully")
            self.logger.info("InstitutionalTradingPlatform._load_comprehensive_data_background EXIT (cancelled graceful)")
            raise  # Re-raise to properly handle cancellation
        except Exception as e:
            self.logger.error(f"Background data loading failed: {e}")
            self.logger.info("InstitutionalTradingPlatform._load_comprehensive_data_background EXIT (error)")
        finally:
            # Mark task as completed
            if hasattr(self, '_background_tasks'):
                self._background_tasks.discard(background_task)

    async def _load_regime_data_with_cancellation(self):
        self.logger.info("InstitutionalTradingPlatform._load_regime_data_with_cancellation ENTRY")
        """Load regime data with cancellation support"""
        try:
            return await asyncio.wait_for(
                self.market_intelligence.get_comprehensive_market_intelligence(),
                timeout=10.0  # 10-second timeout
            )
        except asyncio.TimeoutError:
            self.logger.warning("Regime data loading timed out")
            self.logger.info("InstitutionalTradingPlatform._load_regime_data_with_cancellation EXIT (timeout)")
            return {"regime_score": 5.0, "regime_label": "Timeout"}
        except asyncio.CancelledError:
            raise

    async def _load_extended_data_with_cancellation(self, tickers):
        self.logger.info("InstitutionalTradingPlatform._load_extended_data_with_cancellation ENTRY")
        """Load extended data with cancellation support"""
        try:
            return await asyncio.wait_for(
                self.market_data_manager.get_live_prices_async(tickers),
                timeout=8.0  # 8-second timeout
            )
        except asyncio.TimeoutError:
            self.logger.warning("Extended data loading timed out")
            self.logger.info("InstitutionalTradingPlatform._load_extended_data_with_cancellation EXIT (timeout)")
            return False, {}
        except asyncio.CancelledError:
            raise

    def display_header(self):
        self.logger.info("InstitutionalTradingPlatform.display_header ENTRY")
        """Display professional application header"""
        
        print("🎨 HEADER: Starting header display...")
        
        st.markdown("""
        <div class="main-header">
            <h1>🚀 Enhanced Institutional Trading Platform V5.0</h1>
            <p>Next-Generation Quantitative Trading & Portfolio Management Suite</p>
            <p><small>Real-time Analytics • Advanced Risk Management • Institutional-Grade Execution</small></p>
        </div>
        """, unsafe_allow_html=True)
        
        print("🎨 HEADER: Header markdown rendered")
        self.logger.info("InstitutionalTradingPlatform.display_header EXIT")
        
        # Display key market indicators
        self._display_market_overview()
        print("🎨 HEADER: Market overview displayed")
    
    def _display_market_overview(self):
        self.logger.info("InstitutionalTradingPlatform._display_market_overview ENTRY")
        """Display market overview with fixed column spacing"""
        
        market_data = self.ui_state.get('market_data', {})
        regime_data = self.ui_state.get('regime_data', {})
        self.logger.info("InstitutionalTradingPlatform._display_market_overview EXIT")
        
        if market_data:
            # Fix: Use explicit equal column distribution
            col1, col2, col3, col4, col5, col6 = st.columns([1, 1, 1, 1, 1, 1])
            
            # Define indicators with explicit column assignments
            indicators = [
                ("NIFTY 50", "^NSEI", col1),
                ("NIFTY BANK", "^NSEBANK", col2),  # Fix: No gap here
                ("S&P 500", "^GSPC", col3),
                ("USD/INR", "USDINR=X", col4),
                ("NIFTY IT", "^CNXIT", col5)
            ]
            
            for name, ticker, col in indicators:
                if ticker in market_data:
                    with col:
                        price = market_data[ticker]
                        status = "LIVE" if regime_data.get('regime_score', 5) > 6 else "WATCH" if regime_data.get('regime_score', 5) > 4 else "ALERT"
                        st.metric(
                            label=f"{name}",
                            value=format_currency(price) if 'INR' in ticker else f"{price:,.2f}",
                            delta=status
                        )
            
            # Market regime in last column
            with col6:
                regime_score = regime_data.get('regime_score', 5.0)
                regime_label = regime_data.get('regime_label', 'Analyzing...')
                st.metric(
                    label="Market Regime",
                    value=f"{regime_score:.1f}/10",
                    delta=regime_label,
                    delta_color="normal"
                )

    def run_application(self):
        """Main application runner"""
        
        print("🚀 RUN: Application starting")
        
        try:
            print("🚀 RUN: Checking session state...")
            
            # Initialize platform if not already done
            if not hasattr(st.session_state, 'platform_initialized'):
                print("🚀 RUN: Session state check passed, initializing platform...")
                
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                try:
                    print("🚀 RUN: Starting async initialization...")
                    success = loop.run_until_complete(self.initialize_platform())
                    print(f"🚀 RUN: Async initialization result: {success}")
                    
                    if success:
                        st.session_state.platform_initialized = True
                        print("🚀 RUN: Platform initialized successfully")
                    else:
                        print("❌ RUN: Platform initialization failed")
                        st.error("❌ Platform initialization failed")
                        return
                except Exception as init_error:
                    print(f"❌ RUN: Initialization exception: {init_error}")
                    import traceback
                    traceback.print_exc()
                    st.error(f"Initialization failed: {init_error}")
                    return
                finally:
                    loop.close()
                    print("🚀 RUN: Event loop closed")
            else:
                print("🚀 RUN: Platform already initialized")
            
            print("🚀 RUN: About to display header...")
            
            # Test with minimal Streamlit call first
            st.success("✅ Platform Loaded Successfully!")
            st.write("📊 Institutional Trading Platform is running...")
            
            # Display header
            self.display_header()
            print("🚀 RUN: Header displayed successfully")
            
            st.markdown("---")
            
            print("🚀 RUN: About to create main layout...")
            # Create main layout with sidebar and tabs
            self._create_main_layout()
            print("🚀 RUN: Main layout created successfully")
            
        except Exception as e:
            print(f"❌ RUN: Critical application error: {e}")
            import traceback
            traceback.print_exc()
            st.error(f"Application error: {e}")
            st.text(f"Error details: {traceback.format_exc()}")
            
            # Display error recovery options
            st.markdown("### 🛠️ Error Recovery")
            if st.button("🔄 Restart Platform"):
                st.session_state.clear()
                st.rerun()
        
        print("🚀 RUN: Application method completed")
    
    def _create_main_layout(self):
        """Create main application layout with sidebar and tabs"""
        
        print("🎨 LAYOUT: Creating sidebar...")
        with st.sidebar:
            print("🎨 LAYOUT: Inside sidebar context...")
            
            # Ensure portfolio manager is initialized before passing to sidebar
            if self.portfolio_manager is None:
                st.warning("⚠️ Portfolio management temporarily unavailable")
                st.write("🔧 Portfolio Control Center - Initializing...")
            else:
                sidebar = PortfolioSidebar(
                    portfolio_manager=self.portfolio_manager,
                    market_data_manager=self.market_data_manager
                )
                active_portfolio = sidebar.render()
                
                # Update UI state
                if active_portfolio:
                    self.ui_state['active_portfolio'] = active_portfolio
            
            print("🎨 LAYOUT: Sidebar content added")
        
        print("🎨 LAYOUT: Creating main tabs...")
        # Create main content tabs
        tab1, tab2, tab3 = st.tabs([
            "🌐 **Market Intelligence**",
            "🏦 **Portfolio Management**",
            "⚡ **Live Trading**"
        ])
        print("🎨 LAYOUT: Tabs created successfully")
        
        # Market Intelligence Tab
        with tab1:
            st.write("📊 Market Intelligence Content")
            market_tab = MarketIntelligenceTab(
                market_intelligence=self.market_intelligence,
                sector_analyzer=self.sector_analyzer,
                market_charts=self.market_charts
            )
            market_tab.render()
            print("🎨 LAYOUT: Tab 1 content added")
        
        # Portfolio Management Tab
        with tab2:
            portfolio_tab = PortfolioManagementTab(
                portfolio_manager=self.portfolio_manager,
                risk_calculator=self.risk_calculator,
                performance_analyzer=self.performance_analyzer,
                portfolio_dashboard=self.portfolio_dashboard,
                risk_dashboard=self.risk_dashboard
            )
            portfolio_tab.render(self.ui_state.get('active_portfolio'))
        
        # Live Trading Tab
        with tab3:
            trading_tab = LiveTradingTab(
                order_manager=self.order_manager,
                trading_engine=self.trading_engine,
                risk_validator=self.risk_validator,
                market_data_manager=self.market_data_manager
            )
            trading_tab.render(self.ui_state.get('active_portfolio'))
    
    def cleanup_resources(self):
        """Cleanup application resources"""
        
        try:
            # Cancel pending order monitoring tasks
            if self.order_manager and hasattr(self.order_manager, 'cleanup_monitoring_task'):
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                try:
                    loop.run_until_complete(self.order_manager.cleanup_monitoring_task())
                finally:
                    loop.close()
            
            # Cleanup database connections
            if self.db_manager:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                try:
                    loop.run_until_complete(self.db_manager.close())
                finally:
                    loop.close()
            
            self.logger.info("Resource cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Resource cleanup error: {e}")

    def _cleanup_ui_state(self):
        """Safely cleanup UI state during shutdown"""
        
        try:
            if self.ui_state and isinstance(self.ui_state, dict):
                # Mark as shutting down
                self.ui_state['platform_status'] = 'shutting_down'
                
                # Clear sensitive data but keep structure
                self.ui_state['market_data'] = {}
                self.ui_state['regime_data'] = {}
            
        except Exception as e:
            self.logger.warning(f"UI state cleanup error: {e}")
            # Create minimal valid state if cleanup fails
            self.ui_state = {'platform_status': 'shutdown_error'}

    def handle_system_shutdown(self):
        """Handle graceful system shutdown with enhanced error handling"""
        
        self.logger.info("Initiating graceful shutdown...")
        
        try:
            # Clean UI state before saving
            self._cleanup_ui_state()
            
            # Save session data
            self._save_session_state()
            
            # Cleanup resources
            self.cleanup_resources()
            
            self.logger.info("Shutdown completed successfully")
            
        except Exception as e:
            self.logger.error(f"Shutdown error: {e}")
            # Continue with cleanup even if session save fails
            try:
                self.cleanup_resources()
            except Exception as cleanup_error:
                self.logger.error(f"Resource cleanup failed: {cleanup_error}")
    
    def _save_session_state(self):
        """Save current session state with robust error handling"""
        
        try:
            # Safe extraction of portfolio name with multiple fallbacks
            portfolio_name = None
            
            if self.ui_state and isinstance(self.ui_state, dict):
                active_portfolio = self.ui_state.get('active_portfolio')
                if active_portfolio and isinstance(active_portfolio, dict):
                    metadata = active_portfolio.get('metadata')
                    if metadata and isinstance(metadata, dict):
                        portfolio_name = metadata.get('portfolio_name')
            
            # Create session data with safe values
            session_data = {
                'last_session': get_timestamp_iso(),
                'active_portfolio': portfolio_name,
                'ui_preferences': {
                    'theme': 'dark',
                    'auto_refresh': True
                },
                'platform_status': 'shutdown',
                'shutdown_timestamp': get_timestamp_iso()
            }
            
            # Ensure config and session file path exist
            if not hasattr(self, 'config') or not self.config:
                self.logger.warning("Config not available during session save")
                return
            
            session_file = getattr(self.config, 'SESSION_STATE_FILE', 'session_state.json')
            
            # Save to file with error handling
            import json
            import os
            
            # Create directory if it doesn't exist
            session_dir = os.path.dirname(session_file)
            if session_dir and not os.path.exists(session_dir):
                os.makedirs(session_dir, exist_ok=True)
            
            with open(session_file, 'w') as f:
                json.dump(session_data, f, indent=2, default=str)
            
            self.logger.info(f"Session state saved successfully: {portfolio_name or 'No active portfolio'}")
            
        except Exception as e:
            self.logger.error(f"Failed to save session state: {e}")
            # Don't re-raise during shutdown to prevent cascading errors


def main():
    """Application entry point"""
    
    # Create and run the trading platform
    print("🚀 MAIN: Entry point reached")
    platform = InstitutionalTradingPlatform()
    
    try:
        # Setup cleanup handler
        import atexit
        atexit.register(platform.handle_system_shutdown)
        
        # Run main application
        print("🚀 MAIN: Platform created, starting application")
        platform.run_application()
        print("🚀 MAIN: Application completed")
        
    except KeyboardInterrupt:
        st.info("👋 Application terminated by user")
        platform.handle_system_shutdown()
        
    except Exception as e:
        st.error(f"❌ Critical application error: {e}")
        logging.error(f"Critical error: {e}", exc_info=True)
        platform.handle_system_shutdown()

if __name__ == "__main__":
    # Set up exception handling
    import sys
    import traceback
    
    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        
        logging.error(
            "Uncaught exception",
            exc_info=(exc_type, exc_value, exc_traceback)
        )
    
    sys.excepthook = handle_exception
    
    # Run application
    main()
