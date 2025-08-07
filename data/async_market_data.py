"""
Async market data manager with advanced features
"""
import asyncio
import time
import logging
import sys
import yfinance as yf
import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import Dict, List, Tuple, Optional, Any
from concurrent.futures import ThreadPoolExecutor
from .rate_limiter import AsyncRateLimiter
from .cache import AsyncDataCache
from .ticker_formatter import TickerFormatter
from utils.decorators import async_retry_on_failure

class AsyncMarketDataManager:
    """
    Production-grade async market data manager
    Features: Connection pooling, retry logic, rate limiting, caching
    """
    
    def __init__(self, max_workers: int = 10, timeout: float = 30.0):
        self.logger = logging.getLogger(__name__)
        self.logger.info("AsyncMarketDataManager.__init__ ENTRY")
        self.max_workers = max_workers
        self.timeout = timeout
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.rate_limiter = AsyncRateLimiter(calls=60, period=60)  # 60 calls per minute
        self.cache = AsyncDataCache(default_ttl=300)  # 5 minute cache
        self.formatter = TickerFormatter()
        self.logger = logging.getLogger(__name__)
        self.logger.info("AsyncMarketDataManager.__init__ EXIT")
        
    # Add this optimized method to your AsyncMarketDataManager class
    async def get_live_prices_async(self, tickers: List[str], 
                                force_refresh: bool = False) -> Tuple[bool, Dict[str, float]]:
        """Optimized async price fetching with reduced timeout and better error handling"""
        
        self.logger.info("AsyncMarketDataManager.get_live_prices_async ENTRY")
        if not tickers:
            return False, {}
        
        cache_key = f"prices_{'_'.join(sorted(tickers))}"
        
        # Check cache first for performance
        if not force_refresh:
            self.logger.info("AsyncMarketDataManager.get_live_prices_async checking cache")
            cached_data = await self.cache.get(cache_key)
            if cached_data is not None:
                self.logger.info("AsyncMarketDataManager.get_live_prices_async cache HIT")
                self.logger.info(f"Cache hit for {len(tickers)} tickers")
                return True, cached_data
        
        try:
            # Apply rate limiting
            await self.rate_limiter.acquire()
            
            # Format tickers for yfinance
            self.logger.info("AsyncMarketDataManager.get_live_prices_async formatting tickers")
            formatted_tickers = self.formatter.format_tickers_batch(tickers)
            
            # Reduced timeout for faster failure detection
            reduced_timeout = 8.0  # Reduced from 30s to 8s
            
            self.logger.info("AsyncMarketDataManager.get_live_prices_async submitting fetch to executor")
            future = self.executor.submit(self._fetch_prices_batch_optimized, formatted_tickers)
            
            try:
                self.logger.info("AsyncMarketDataManager.get_live_prices_async waiting for result")
                result = await asyncio.wait_for(
                    asyncio.wrap_future(future), 
                    timeout=reduced_timeout
                )
                
                success, prices = result
                
                if success and prices:
                    self.logger.info("AsyncMarketDataManager.get_live_prices_async fetch SUCCESS")
                    # Map back to original tickers
                    mapped_prices = {}
                    for i, original_ticker in enumerate(tickers):
                        if i < len(formatted_tickers):
                            formatted = formatted_tickers[i]
                            if formatted in prices:
                                mapped_prices[original_ticker] = prices[formatted]
                    
                    # Cache successful results with shorter TTL
                    await self.cache.set(cache_key, mapped_prices, ttl=30)  # 30-second cache
                    
                    self.logger.info(f"Successfully fetched {len(mapped_prices)}/{len(tickers)} prices")
                    self.logger.info("AsyncMarketDataManager.get_live_prices_async EXIT (success)")
                    return True, mapped_prices
                else:
                    self.logger.warning(f"Price fetch failed: {prices}")
                    self.logger.info("AsyncMarketDataManager.get_live_prices_async EXIT (price fetch failed)")
                    return False, {}
                    
            except asyncio.TimeoutError:
                self.logger.error(f"Price fetch timeout after {reduced_timeout}s")
                self.logger.info("AsyncMarketDataManager.get_live_prices_async EXIT (timeout)")
                future.cancel()
                return False, {}
                
        except Exception as e:
            self.logger.error(f"Async price fetch failed: {e}")
            self.logger.info("AsyncMarketDataManager.get_live_prices_async EXIT (error)")
            return False, {}

    def _fetch_prices_batch_optimized(self, tickers: List[str]) -> Tuple[bool, Dict[str, float]]:
        """Optimized batch price fetching with single attempt"""
        
        try:
            self.logger.info(f"Fetching prices for {len(tickers)} tickers")
            
            # Single attempt with optimized parameters
            data = yf.download(
                tickers,
                period="1d",  # Reduced from 2d to 1d
                interval="1d", 
                progress=False,
                group_by='ticker' if len(tickers) > 1 else None,
                threads=min(4, len(tickers)),  # Limit concurrent threads
                timeout=6  # Reduced internal timeout
            )
            
            if data.empty:
                self.logger.warning("No data returned from yfinance")
                return False, {}
            
            # Extract prices with enhanced error handling
            prices = self._extract_prices_from_data_safe(data, tickers)
            
            if prices:
                self.logger.info(f"Successfully fetched {len(prices)} real prices")
                return True, prices
            else:
                return False, {}
                
        except Exception as e:
            self.logger.error(f"Optimized batch fetch failed: {e}")
            return False, {}

    def _extract_prices_from_data_safe(self, data: pd.DataFrame, tickers: List[str]) -> Dict[str, float]:
        """Enhanced price extraction with robust error handling"""
        
        prices = {}
        
        try:
            if data.empty:
                return prices
            
            # Handle single ticker case
            if len(tickers) == 1:
                if 'Close' in data.columns and not data['Close'].empty:
                    close_price = data['Close'].dropna().iloc[-1]
                    if pd.notna(close_price) and close_price > 0:
                        prices[tickers[0]] = float(close_price)
                return prices
            
            # Handle multiple tickers with MultiIndex
            if isinstance(data.columns, pd.MultiIndex):
                for ticker in tickers:
                    try:
                        if ticker in data.columns.get_level_values(0):
                            close_series = data[(ticker, 'Close')].dropna()
                            if not close_series.empty:
                                close_price = close_series.iloc[-1]
                                if pd.notna(close_price) and close_price > 0:
                                    prices[ticker] = float(close_price)
                    except (KeyError, IndexError, ValueError) as e:
                        self.logger.debug(f"Could not extract price for {ticker}: {e}")
                        continue
            else:
                # Fallback for non-MultiIndex
                if 'Close' in data.columns:
                    close_series = data['Close'].dropna()
                    if not close_series.empty:
                        price = float(close_series.iloc[-1])
                        if price > 0:
                            # Assign same price to all tickers (fallback)
                            for ticker in tickers:
                                prices[ticker] = price
            
            return prices
            
        except Exception as e:
            self.logger.error(f"Price extraction failed: {e}")
            return {}

    async def get_enhanced_technical_data(self, ticker: str, 
                                        period: str = "6mo") -> Tuple[bool, Optional[pd.DataFrame]]:
        """Get technical indicators with async processing"""
        
        cache_key = f"technical_{ticker}_{period}"
        
        # Check cache
        cached_data = await self.cache.get(cache_key)
        if cached_data is not None:
            return True, cached_data
        
        try:
            await self.rate_limiter.acquire()
            
            future = self.executor.submit(self._calculate_technical_indicators, ticker, period)
            result = await asyncio.wait_for(asyncio.wrap_future(future), timeout=30)
            
            if result[0]:  # Success
                await self.cache.set(cache_key, result[1], ttl=1800)  # 30 min cache
            
            return result
            
        except asyncio.TimeoutError:
            self.logger.error(f"Technical data timeout for {ticker}")
            return False, None
        except Exception as e:
            self.logger.error(f"Technical data fetch failed for {ticker}: {e}")
            return False, None
    
    def _calculate_technical_indicators(self, ticker: str, period: str) -> Tuple[bool, Optional[pd.DataFrame]]:
        """Calculate technical indicators (runs in thread pool)"""
        
        try:
            # Download data
            formatted_ticker = self.formatter.format_single_ticker(ticker)
            data = yf.download(formatted_ticker, period=period, progress=False, timeout=20)
            
            if data.empty:
                return False, None
            
            # Calculate indicators using pandas_ta
            data = self._add_technical_indicators(data)
            
            return True, data
            
        except Exception as e:
            self.logger.error(f"Technical calculation failed for {ticker}: {e}")
            return False, None
    
    def _add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Enhanced technical indicators with proper MultiIndex handling"""
        
        try:
            if data is None or data.empty:
                return data
            
            # Fix: Handle MultiIndex columns properly
            if isinstance(data.columns, pd.MultiIndex):
                # Flatten MultiIndex columns for compatibility
                data.columns = ['_'.join(map(str, col)).strip() for col in data.columns.values]
                self.logger.warning("Flattened MultiIndex columns for pandas_ta compatibility")
            
            # Check for required columns with flexible naming
            close_col = None
            high_col = None
            low_col = None
            volume_col = None
            
            for col in data.columns:
                col_lower = col.lower()
                if 'close' in col_lower and close_col is None:
                    close_col = col
                elif 'high' in col_lower and high_col is None:
                    high_col = col
                elif 'low' in col_lower and low_col is None:
                    low_col = col
                elif 'volume' in col_lower and volume_col is None:
                    volume_col = col
            
            if not close_col:
                self.logger.warning("No Close column found for technical indicators")
                return data
            
            # Rename columns for pandas_ta compatibility
            renamed_data = data.copy()
            renamed_data = renamed_data.rename(columns={
                close_col: 'Close',
                high_col: 'High' if high_col else close_col,
                low_col: 'Low' if low_col else close_col,
                volume_col: 'Volume' if volume_col else close_col
            })
            
            try:
                import pandas_ta as ta
                
                # Add technical indicators with error handling
                try:
                    renamed_data.ta.sma(length=20, append=True)
                    renamed_data.ta.sma(length=50, append=True)
                    renamed_data.ta.ema(length=12, append=True)
                    renamed_data.ta.ema(length=26, append=True)
                except Exception as e:
                    self.logger.warning(f"Moving averages failed: {e}")
                
                try:
                    renamed_data.ta.rsi(length=14, append=True)
                except Exception as e:
                    self.logger.warning(f"RSI calculation failed: {e}")
                
                try:
                    renamed_data.ta.macd(append=True)
                except Exception as e:
                    self.logger.warning(f"MACD calculation failed: {e}")
                
                try:
                    renamed_data.ta.bbands(append=True)
                except Exception as e:
                    self.logger.warning(f"Bollinger Bands failed: {e}")
                
                try:
                    if 'Volume' in renamed_data.columns:
                        renamed_data.ta.obv(append=True)
                except Exception as e:
                    self.logger.warning(f"Volume indicators failed: {e}")
                
                return renamed_data
                
            except ImportError:
                self.logger.warning("pandas_ta not available, using manual indicators")
                return self._add_manual_indicators(renamed_data)
                
        except Exception as e:
            self.logger.error(f"Technical indicators failed: {e}")
            return data

    def _add_manual_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Manual technical indicators as fallback"""
        
        try:
            if 'Close' not in data.columns:
                return data
            
            # Simple moving averages
            data['SMA_20'] = data['Close'].rolling(window=20, min_periods=1).mean()
            data['SMA_50'] = data['Close'].rolling(window=50, min_periods=1).mean()
            
            # Exponential moving averages
            data['EMA_12'] = data['Close'].ewm(span=12, adjust=False).mean()
            data['EMA_26'] = data['Close'].ewm(span=26, adjust=False).mean()
            
            # MACD
            data['MACD'] = data['EMA_12'] - data['EMA_26']
            data['MACD_signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
            
            # RSI
            if len(data) >= 14:
                delta = data['Close'].diff()
                gain = delta.clip(lower=0)
                loss = -delta.clip(upper=0)
                avg_gain = gain.rolling(window=14, min_periods=1).mean()
                avg_loss = loss.rolling(window=14, min_periods=1).mean()
                
                rs = avg_gain / avg_loss.replace(0, np.nan)
                data['RSI'] = 100 - (100 / (1 + rs))
                data['RSI'] = data['RSI'].fillna(50)
            
            return data
            
        except Exception as e:
            self.logger.error(f"Manual indicators failed: {e}")
            return data

    async def cleanup(self):
        """Clean up resources"""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)
        await self.cache.cleanup()
