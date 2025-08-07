import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

# Use absolute imports
from src.utils.logger import setup_logger
from config.config_loader import Config

# Set up logger
logger = setup_logger('rsi_ma_strategy')

class RSIMAStrategy:
    """
    Implementation of the RSI + Moving Average Crossover Strategy
    
    Buy Signal:
    - RSI < 30 (oversold condition)
    - Confirmation: 20-DMA crosses above 50-DMA
    
    Sell Signal:
    - RSI > 70 (overbought condition)
    - OR 20-DMA crosses below 50-DMA
    """
    
    def __init__(self, rsi_threshold=None, ma_short_period=None, ma_long_period=None):
        """Initialize the strategy with optional custom parameters"""
        self.rsi_buy_threshold = rsi_threshold or Config.RSI_BUY_THRESHOLD
        self.ma_short_period = ma_short_period or Config.MA_SHORT_PERIOD
        self.ma_long_period = ma_long_period or Config.MA_LONG_PERIOD
    
    def generate_signals(self, df):
        """
        Generate buy/sell signals based on the strategy
        
        Args:
            df (DataFrame): Processed stock data with technical indicators
            
        Returns:
            DataFrame: Data with strategy signals
        """
        # Make a copy to avoid modifying the original
        data = df.copy()
        
        # RSI Buy Signal
        data['RSI_Buy_Signal'] = 0
        data.loc[data['RSI'] < self.rsi_buy_threshold, 'RSI_Buy_Signal'] = 1
        
        # Moving Average Crossover Signal
        data['MA_Crossover'] = 0
        data['MA_Crossover'] = ((data['MA_Short'] > data['MA_Long']) & 
                                (data['MA_Short'].shift(1) <= data['MA_Long'].shift(1))).astype(int)
        
        # MODIFIED: Combined Buy Signal - Either RSI condition OR MA Crossover
        # This will generate more signals than requiring both conditions
        data['Buy_Signal'] = ((data['RSI_Buy_Signal'] == 1) | 
                             (data['MA_Crossover'] == 1)).astype(int)
        
        # Simple Sell Signal - RSI > 70 or MA_Short crosses below MA_Long
        data['Sell_Signal'] = 0
        data.loc[data['RSI'] > 70, 'Sell_Signal'] = 1
        data.loc[(data['MA_Short'] < data['MA_Long']) & 
                 (data['MA_Short'].shift(1) >= data['MA_Long'].shift(1)), 'Sell_Signal'] = 1
        
        return data