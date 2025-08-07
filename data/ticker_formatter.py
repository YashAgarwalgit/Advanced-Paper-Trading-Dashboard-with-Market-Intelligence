"""
Ticker formatting utilities for different markets and exchanges
"""
from typing import List, Dict, Optional
import logging
from utils.constants import SUPPORTED_ASSET_CLASSES

logging.basicConfig(level=logging.INFO, format='%(filename)s:%(funcName)s: %(message)s')

class TickerFormatter:
    """Handles ticker formatting for different exchanges and asset classes"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.info("TickerFormatter.__init__ ENTRY")
        logging.info(f"{__file__}:{self.__class__.__name__} initialized")
        self.logger.info("TickerFormatter.__init__ EXIT")
        self.asset_classes = SUPPORTED_ASSET_CLASSES
    
    def format_single_ticker(self, ticker: str, asset_class: str = "Equities") -> str:
        self.logger.info(f"TickerFormatter.format_single_ticker ENTRY: {ticker}, {asset_class}")
        """
        Format a single ticker symbol for yfinance
        
        Args:
            ticker: Raw ticker symbol
            asset_class: Asset class for formatting rules
            
        Returns:
            Formatted ticker symbol
        """
        if not ticker:
            return ticker
        
        ticker = ticker.upper().strip()
        
        # Handle special cases first
        if ticker.startswith('^') or '=' in ticker:
            return ticker  # Index or forex symbols
        
        # Get formatting rules for asset class
        rules = self.asset_classes.get(asset_class, self.asset_classes["Equities"])
        suffix = rules["suffix"]
        
        # Add suffix if not already present
        if suffix and not ticker.endswith(suffix):
            self.logger.info(f"TickerFormatter.format_single_ticker adding suffix: {suffix}")
            return f"{ticker}{suffix}"
        
        return ticker
    
    def format_tickers_batch(self, tickers: List[str], asset_class: str = "Equities") -> List[str]:
        self.logger.info(f"TickerFormatter.format_tickers_batch ENTRY: {tickers}, {asset_class}")
        """
        Format a batch of ticker symbols
        
        Args:
            tickers: List of raw ticker symbols
            asset_class: Asset class for formatting rules
            
        Returns:
            List of formatted ticker symbols
        """
        result = [self.format_single_ticker(ticker, asset_class) for ticker in tickers]
        self.logger.info(f"TickerFormatter.format_tickers_batch EXIT: {result}")
        return result
    
    def parse_ticker_components(self, ticker: str) -> Dict[str, Optional[str]]:
        self.logger.info(f"TickerFormatter.parse_ticker_components ENTRY: {ticker}")
        """
        Parse ticker into components
        
        Args:
            ticker: Formatted ticker symbol
            
        Returns:
            Dictionary with ticker components
        """
        components = {
            "symbol": None,
            "exchange": None,
            "suffix": None,
            "asset_type": "equity"
        }
        
        if not ticker:
            return components
        
        ticker = ticker.upper()
        
        # Handle indices
        if ticker.startswith('^'):
            components["symbol"] = ticker[1:]
            components["asset_type"] = "index"
            components["suffix"] = "^"
            return components
        
        # Handle forex
        if '=' in ticker:
            parts = ticker.split('=')
            components["symbol"] = parts[0]
            components["suffix"] = f"={parts[1]}" if len(parts) > 1 else "=X"
            components["asset_type"] = "currency"
            return components
        
        # Handle regular stocks with exchange suffix
        if '.NS' in ticker:
            components["symbol"] = ticker.replace('.NS', '')
            components["exchange"] = "NSE"
            components["suffix"] = ".NS"
        elif '.BO' in ticker:
            components["symbol"] = ticker.replace('.BO', '')
            components["exchange"] = "BSE"
            components["suffix"] = ".BO"
        else:
            components["symbol"] = ticker
        
        return components
    
    def is_valid_ticker(self, ticker: str) -> bool:
        """
        Validate ticker format
        
        Args:
            ticker: Ticker symbol to validate
            
        Returns:
            True if ticker format is valid
        """
        if not ticker or not isinstance(ticker, str):
            return False
        
        ticker = ticker.strip()
        
        # Basic validation
        if len(ticker) < 1 or len(ticker) > 20:
            return False
        
        # Allow alphanumeric characters and common symbols
        allowed_chars = set('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-=^')
        return all(c in allowed_chars for c in ticker.upper())
    
    def get_exchange_from_ticker(self, ticker: str) -> Optional[str]:
        """Extract exchange information from ticker"""
        components = self.parse_ticker_components(ticker)
        return components.get("exchange")
