"""
Application-wide constants and enums
"""
from typing import Dict, Any, Optional
import logging
import os

# Set up module-level logger
logger = logging.getLogger(__name__)

# HTTP Headers for market data requests
def get_headers() -> Dict[str, str]:
    """
    Get standard HTTP headers for market data requests.
    
    Returns:
        Dict[str, str]: Dictionary of HTTP headers
    """
    logger.debug("Generating HTTP headers for market data requests")
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'
    }
    
    logger.debug("HTTP headers generated successfully")
    return headers

# Initialize headers with logging
logger.debug("Initializing application constants")
try:
    HEADERS = get_headers()
    logger.debug("HTTP headers initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize HTTP headers: {e}", exc_info=True)
    # Fallback to minimal headers if there's an error
    HEADERS = {
        'User-Agent': 'Mozilla/5.0',
        'Accept': '*/*'
    }
    logger.warning("Using fallback HTTP headers")

# Supported Asset Classes and Markets
SUPPORTED_ASSET_CLASSES = {
    "Equities": {"suffix": ".NS", "multiplier": 1},
    "ETFs": {"suffix": ".NS", "multiplier": 1},
    "Futures": {"suffix": ".NS", "multiplier": 1},
    "Indices": {"suffix": "", "multiplier": 1},
    "Currency": {"suffix": "=X", "multiplier": 1}
}

# Active NSE Tickers with metadata
ACTIVE_NSE_TICKERS = {
    # IT Sector
    "TCS.NS": {"name": "Tata Consultancy Services", "sector": "IT", "size": "Large"},
    "INFY.NS": {"name": "Infosys Limited", "sector": "IT", "size": "Large"},
    "WIPRO.NS": {"name": "Wipro Limited", "sector": "IT", "size": "Large"},
    "HCLTECH.NS": {"name": "HCL Technologies", "sector": "IT", "size": "Large"},
    "TECHM.NS": {"name": "Tech Mahindra", "sector": "IT", "size": "Large"},
    
    # Banking Sector
    "HDFCBANK.NS": {"name": "HDFC Bank", "sector": "Banking", "size": "Large"},
    "ICICIBANK.NS": {"name": "ICICI Bank", "sector": "Banking", "size": "Large"},
    "SBIN.NS": {"name": "State Bank of India", "sector": "Banking", "size": "Large"},
    "KOTAKBANK.NS": {"name": "Kotak Mahindra Bank", "sector": "Banking", "size": "Large"},
    
    # Other sectors
    "RELIANCE.NS": {"name": "Reliance Industries", "sector": "Energy", "size": "Large"},
    "MARUTI.NS": {"name": "Maruti Suzuki", "sector": "Auto", "size": "Large"},
    "HINDUNILVR.NS": {"name": "Hindustan Unilever", "sector": "FMCG", "size": "Large"}
}

# Default watchlist
DEFAULT_WATCHLIST = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ITC"]
