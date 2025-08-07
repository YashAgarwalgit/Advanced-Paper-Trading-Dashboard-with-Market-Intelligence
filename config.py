"""
Configuration management for institutional trading platform
"""
import os
from typing import Dict, Any


class Config:
    """Application configuration"""
    
    # Directory Configuration
    PORTFOLIOS_DIR = "enhanced_portfolios"
    CACHE_DIR = "cache"
    ANALYTICS_DIR = "analytics"
    LOGS_DIR = "logs"
    
    # Database Configuration
    DATABASE_PATH = "institutional_trading.db"
    DATABASE_URL = f"sqlite:///{DATABASE_PATH}"
    CONNECTION_POOL_SIZE = 10
    ENABLE_WAL = True
    BACKUP_INTERVAL_HOURS = 24
    
    # API Configuration
    REQUEST_TIMEOUT = 30.0
    MAX_RETRIES = 3
    RETRY_DELAYS = [1, 3, 5]
    MAX_WORKERS = 8  # ← MISSING: For AsyncMarketDataManager
    
    # Rate Limiting
    API_CALLS_PER_MINUTE = 60
    RATE_LIMIT_PERIOD = 60
    
    # Cache Configuration
    DEFAULT_CACHE_TTL = 300  # 5 minutes
    PRICE_CACHE_TTL = 60     # 1 minute
    TECHNICAL_CACHE_TTL = 1800  # 30 minutes
    
    # Trading Configuration
    MAX_ORDER_VALUE = 10_000_000  # ₹1 crore
    MAX_DAILY_TURNOVER = 50_000_000  # ₹5 crore
    DEFAULT_COMMISSION_BPS = 5
    DEFAULT_SLIPPAGE_PCT = 0.1
    
    # Risk Configuration
    DEFAULT_RISK_FREE_RATE = 0.07  # ← Fixed typo: was DEFAULT_RISK_RATE
    MAX_POSITION_WEIGHT = 0.10
    MAX_SECTOR_WEIGHT = 0.25
    
    # Session Management
    SESSION_STATE_FILE = "session_state.json"  # ← MISSING: For session persistence
    SESSION_TIMEOUT = 3600  # 1 hour
    
    # Async Configuration
    ASYNC_TIMEOUT = 30.0
    CONCURRENT_REQUESTS = 10
    
    # Market Data Configuration
    MARKET_DATA_SOURCES = ["yfinance", "alpha_vantage"]
    DEFAULT_DATA_SOURCE = "yfinance"
    
    # Portfolio Configuration
    MAX_PORTFOLIOS = 50
    DEFAULT_INITIAL_CAPITAL = 1000000  # ₹10 lakh
    
    # Logging Configuration
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s"
    LOG_MAX_BYTES = 10 * 1024 * 1024  # 10 MB
    LOG_BACKUP_COUNT = 5
    
    # UI Configuration
    STREAMLIT_PORT = 8501
    STREAMLIT_HOST = "localhost"
    
    # Security Configuration
    ENABLE_ENCRYPTION = False
    SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here")
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories"""
        directories = [
            cls.PORTFOLIOS_DIR, 
            cls.CACHE_DIR, 
            cls.ANALYTICS_DIR,
            cls.LOGS_DIR
        ]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    @classmethod
    def get_env_config(cls) -> Dict[str, Any]:
        """Get environment-specific configuration"""
        return {
            "debug": os.getenv("DEBUG", "False").lower() == "true",
            "log_level": os.getenv("LOG_LEVEL", cls.LOG_LEVEL),
            "database_url": os.getenv("DATABASE_URL", cls.DATABASE_URL),
            "max_workers": int(os.getenv("MAX_WORKERS", str(cls.MAX_WORKERS))),
            "request_timeout": float(os.getenv("REQUEST_TIMEOUT", str(cls.REQUEST_TIMEOUT)))
        }
    
    @classmethod
    def validate_config(cls):
        """Validate configuration settings"""
        issues = []
        
        # Check required attributes
        required_attrs = [
            'MAX_WORKERS', 'REQUEST_TIMEOUT', 'DATABASE_URL', 
            'SESSION_STATE_FILE', 'DEFAULT_RISK_FREE_RATE', 'PORTFOLIOS_DIR'
        ]
        
        for attr in required_attrs:
            if not hasattr(cls, attr):
                issues.append(f"Missing required attribute: {attr}")
        
        # Validate values
        if cls.MAX_WORKERS <= 0:
            issues.append("MAX_WORKERS must be positive")
        
        if cls.REQUEST_TIMEOUT <= 0:
            issues.append("REQUEST_TIMEOUT must be positive")
        
        return issues
    
    @classmethod
    def initialize(cls):
        """Initialize configuration and create directories"""
        cls.create_directories()
        
        # Validate configuration
        issues = cls.validate_config()
        if issues:
            raise ValueError(f"Configuration validation failed: {issues}")
        
        return cls()
