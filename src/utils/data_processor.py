import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

# Use absolute imports
from src.utils.data_fetcher import DataFetcher
from src.utils.logger import setup_logger
from src.strategies.rsi_ma_strategy import RSIMAStrategy
from config.config_loader import Config

# Set up logger
logger = setup_logger('data_processor')


# Consolidate this pattern into a utility function
def convert_multi_index_dataframe(df):
    """Convert MultiIndex DataFrame to standard DataFrame with consistent column names"""
    if not isinstance(df.columns, pd.MultiIndex):
        return df
        
    simple_df = pd.DataFrame(index=df.index)
    
    for col_name in ['Close', 'High', 'Low', 'Open', 'Volume']:
        for full_col in df.columns:
            if full_col[0] == col_name:
                simple_df[col_name] = df[full_col].values
                break
                
    return simple_df


# Add this function at the beginning of your process_stock method:

def process_stock(self, stock_data, symbol):
    """
    Process raw stock data and add technical indicators
    
    Args:
        stock_data (DataFrame): Raw stock data
        symbol (str): Stock symbol
        
    Returns:
        DataFrame: Processed stock data with technical indicators
    """
    # Debug the input data structure
    logger.info(f"Processing {symbol} data with shape {stock_data.shape}")
    logger.info(f"Column structure: {type(stock_data.columns)}")
    logger.info(f"Columns: {list(stock_data.columns)}")
    
    # Ensure we're working with a clean DataFrame
    if isinstance(stock_data.columns, pd.MultiIndex):
        logger.info("Converting MultiIndex DataFrame to simple DataFrame")
        # Get the basic OHLCV columns from the first level
        simple_data = pd.DataFrame()
        
        # Map common column names
        col_mapping = {
            'Open': ['Open', 'open'],
            'High': ['High', 'high'],
            'Low': ['Low', 'low'],
            'Close': ['Close', 'close', 'Adj Close', 'adj close'],
            'Volume': ['Volume', 'volume']
        }
        
        # Try to extract columns using various possible names
        for target_col, possible_cols in col_mapping.items():
            found = False
            for col in possible_cols:
                # Check if this column exists in any level of the MultiIndex
                for level in range(len(stock_data.columns.levels)):
                    if col in stock_data.columns.levels[level]:
                        # Find the full MultiIndex column that contains this name
                        for full_col in stock_data.columns:
                            if full_col[level] == col:
                                simple_data[target_col] = stock_data[full_col]
                                found = True
                                logger.info(f"Found {target_col} column as {full_col}")
                                break
                    if found:
                        break
                if found:
                    break
            
            if not found:
                logger.warning(f"Could not find {target_col} column in the data")
        
        # Use the simplified DataFrame for processing
        stock_data = simple_data
    
    # Continue with the rest of your processing...

class DataProcessor:
    """Process stock data for analysis and machine learning"""
    
    def __init__(self):
        """Initialize the data processor"""
        self.data_fetcher = DataFetcher()
        self.strategy = RSIMAStrategy()
        self.processed_data_dir = Path(__file__).parent.parent.parent / "data" / "processed"
        self.processed_data_dir.mkdir(exist_ok=True, parents=True)
        
    def process_stock_data(self, symbol, start_date=None, end_date=None):
        """
        Process stock data by fetching and adding technical indicators
        
        Args:
            symbol (str): Stock symbol
            start_date (str): Start date
            end_date (str): End date
            
        Returns:
            DataFrame: Processed stock data with technical indicators
        """
        if start_date is None:
            start_date = Config.BACKTEST_START_DATE
        if end_date is None:
            end_date = Config.BACKTEST_END_DATE
            
        # Fetch raw data
        data = self.data_fetcher.fetch_stock_data(symbol, start_date, end_date)
        if data is None:
            logger.error(f"Failed to fetch data for {symbol}")
            return None
            
        # Add technical indicators
        processed_data = self.data_fetcher.add_technical_indicators(data)
        
        # Generate trading signals using the strategy
        processed_data = self.strategy.generate_signals(processed_data)
        
        # Save processed data
        self.save_processed_data(processed_data, symbol)
        
        return processed_data
    
    def save_processed_data(self, df, symbol):
        """Save processed data to CSV file"""
        file_path = self.processed_data_dir / f"{symbol}_processed.csv"
        df.to_csv(file_path)
        logger.info(f"Saved processed data for {symbol} to {file_path}")
        
    def process_all_stocks(self):
        """Process all stocks defined in the configuration"""
        processed_data = {}
        for symbol in Config.STOCK_SYMBOLS:
            data = self.process_stock_data(symbol)
            if data is not None:
                processed_data[symbol] = data
                
        return processed_data