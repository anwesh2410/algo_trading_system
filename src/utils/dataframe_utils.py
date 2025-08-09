import pandas as pd
import numpy as np
import os
import sys

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from src.utils.logger import setup_logger

# Initialize logger
logger = setup_logger('dataframe_utils')

def standardize_dataframe(df):
    """
    Convert MultiIndex DataFrame to standard format with consistent column names
    
    Args:
        df: Input DataFrame, can be MultiIndex or standard
        
    Returns:
        DataFrame: Standardized DataFrame with consistent column names
    """
    # If not MultiIndex, return unchanged
    if not isinstance(df.columns, pd.MultiIndex):
        return df
    
    # Create new DataFrame with same index
    result = pd.DataFrame(index=df.index)
    
    # Mapping of target columns to possible names
    col_mapping = {
        'Open': ['Open', 'open'],
        'High': ['High', 'high'],
        'Low': ['Low', 'low'],
        'Close': ['Close', 'close', 'Adj Close', 'adj close'],
        'Volume': ['Volume', 'volume']
    }
    
    # Extract price columns
    for target, sources in col_mapping.items():
        for source in sources:
            found = False
            for col in df.columns:
                if col[0] == source or (len(col) > 1 and col[1] == source):
                    result[target] = df[col].values
                    found = True
                    break
            if found:
                break
    
    # Copy technical indicators
    tech_indicators = [
        'RSI', 'MA_Short', 'MA_Long', 'MACD', 'Signal_Line', 
        'MACD_Histogram', 'BB_Middle', 'BB_Upper', 'BB_Lower', 'Volume_MA',
        'RSI_Buy_Signal', 'MA_Crossover', 'Buy_Signal', 'Sell_Signal'
    ]
    
    for indicator in tech_indicators:
        for col in df.columns:
            if col[0] == indicator or (len(col) > 1 and col[1] == indicator):
                result[indicator] = df[col].values
                break
    
    logger.info(f"Standardized DataFrame with {len(result.columns)} columns")
    return result

def to_multiindex_dataframe(df, symbol):
    """
    Convert standard DataFrame to MultiIndex for consistency with yfinance format
    
    Args:
        df: Standard DataFrame
        symbol: Stock symbol for the second level
        
    Returns:
        DataFrame: MultiIndex DataFrame
    """
    # Create new MultiIndex columns
    price_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    tech_cols = [col for col in df.columns if col not in price_cols]
    
    # Create tuples for MultiIndex
    multi_columns = [(col, symbol) for col in price_cols] + [(col, '') for col in tech_cols]
    
    # Create new DataFrame with MultiIndex
    result = pd.DataFrame(index=df.index)
    
    # Copy data
    for col in price_cols:
        if col in df.columns:
            result[(col, symbol)] = df[col].values
        
    for col in tech_cols:
        if col in df.columns:
            result[(col, '')] = df[col].values
    
    # Set MultiIndex columns
    result.columns = pd.MultiIndex.from_tuples(multi_columns)
    
    logger.info(f"Converted DataFrame to MultiIndex format with {len(multi_columns)} columns")
    return result