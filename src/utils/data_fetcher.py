import yfinance as yf
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
import pickle

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

# Use absolute imports
from src.utils.logger import setup_logger
from config.config_loader import Config

# Set up logger
logger = setup_logger('data_fetcher')

class DataFetcher:
    """Handles all data fetching operations from Yahoo Finance API"""
    
    def __init__(self, cache_dir=None):
        """Initialize with optional cache directory"""
        if cache_dir is None:
            cache_dir = Path(__file__).parent.parent.parent / "data" / "cache"
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        
    def fetch_stock_data(self, symbol, start_date, end_date, interval='1d', use_cache=True):
        """
        Fetch stock data from Yahoo Finance or cache
        
        Args:
            symbol (str): Stock symbol
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
            interval (str): Data interval ('1d', '1h', etc.)
            use_cache (bool): Whether to use cached data if available
            
        Returns:
            DataFrame: Stock data
        """
        cache_file = self.cache_dir / f"{symbol}_{start_date}_{end_date}_{interval}.pkl"
        
        # Try to load from cache first if enabled
        if use_cache and cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    logger.info(f"Loading cached data for {symbol}")
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Failed to load from cache: {e}")
        
        # Fetch data from Yahoo Finance
        try:
            logger.info(f"Fetching data for {symbol} from {start_date} to {end_date}")
            data = yf.download(symbol, start=start_date, end=end_date, interval=interval)
            
            # Cache the data
            if use_cache:
                with open(cache_file, 'wb') as f:
                    pickle.dump(data, f)
                logger.info(f"Cached data for {symbol}")
            
            if data.empty:
                logger.warning(f"No data retrieved for {symbol}")
                return None
                
            return data
        
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return None
    
    def fetch_nifty50_stocks(self, start_date=None, end_date=None):
        """
        Fetch data for NIFTY 50 stocks defined in configuration
        
        Args:
            start_date (str): Start date (default: from config)
            end_date (str): End date (default: from config)
            
        Returns:
            dict: Dictionary of DataFrames with stock symbols as keys
        """
        if start_date is None:
            start_date = Config.BACKTEST_START_DATE
        if end_date is None:
            end_date = Config.BACKTEST_END_DATE
            
        stock_data = {}
        for symbol in Config.STOCK_SYMBOLS:
            data = self.fetch_stock_data(symbol, start_date, end_date)
            if data is not None:
                stock_data[symbol] = data
        
        logger.info(f"Successfully fetched data for {len(stock_data)} out of {len(Config.STOCK_SYMBOLS)} stocks")
        return stock_data
    
    @staticmethod
    def add_technical_indicators(data):
        """
        Add technical indicators to the dataframe
        
        Args:
            data (DataFrame): Stock price data
            
        Returns:
            DataFrame: Data with technical indicators added
        """
        # Make a copy of the dataframe to avoid modifying the original
        df = data.copy()
        
        # Check if we have a multi-index DataFrame from yfinance
        if isinstance(df.columns, pd.MultiIndex):
            # For multi-index DataFrame, extract the Close column properly
            close_series = df['Close']
            volume_series = df['Volume']
        else:
            # For single-level columns
            close_series = df['Close']
            volume_series = df['Volume']
        
        # Calculate RSI
        delta = close_series.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=Config.RSI_PERIOD).mean()
        avg_loss = loss.rolling(window=Config.RSI_PERIOD).mean()
        
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Calculate Moving Averages
        df['MA_Short'] = close_series.rolling(window=Config.MA_SHORT_PERIOD).mean()
        df['MA_Long'] = close_series.rolling(window=Config.MA_LONG_PERIOD).mean()
        
        # MACD
        df['EMA_12'] = close_series.ewm(span=12, adjust=False).mean()
        df['EMA_26'] = close_series.ewm(span=26, adjust=False).mean()
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Histogram'] = df['MACD'] - df['Signal_Line']
        
        # Bollinger Bands - using explicit series operations
        bb_middle = close_series.rolling(window=20).mean()
        rolling_std = close_series.rolling(window=20).std()
        
        df['BB_Middle'] = bb_middle
        df['BB_Upper'] = bb_middle + (rolling_std * 2)
        df['BB_Lower'] = bb_middle - (rolling_std * 2)
        
        # Volume indicators
        df['Volume_MA'] = volume_series.rolling(window=20).mean()
        
        return df