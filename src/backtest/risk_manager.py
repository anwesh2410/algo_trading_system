import pandas as pd
import numpy as np
import logging
from pathlib import Path
import os
import sys

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from src.utils.logger import setup_logger

# Initialize logger
logger = setup_logger('risk_manager')

class RiskManager:
    """
    Risk management module for trading strategies
    """
    
    def __init__(self, position_size=0.15, stop_loss_pct=0.03, take_profit_pct=0.06, max_drawdown_pct=0.15, max_position_count=5):
        """
        Initialize risk manager
        
        Args:
            position_size (float): Maximum position size as percentage of portfolio
            stop_loss_pct (float): Stop loss percentage
            take_profit_pct (float): Take profit percentage
            max_drawdown_pct (float): Maximum allowable drawdown percentage
            max_position_count (int): Maximum number of concurrent positions
        """
        # Store the parameters
        self.position_size = position_size
        self.stop_loss = stop_loss_pct
        self.take_profit = take_profit_pct
        self.max_drawdown = max_drawdown_pct
        self.max_position_count = max_position_count
        
        logger.info(f"Initialized Risk Manager with position size: {position_size}, " +
                   f"stop loss: {stop_loss_pct}, take profit: {take_profit_pct}, " +
                   f"max drawdown: {max_drawdown_pct}, max positions: {max_position_count}")
    
    def apply_position_sizing(self, portfolio_value, df):
        """
        Calculate position sizes based on portfolio value
        
        Args:
            portfolio_value (float): Current portfolio value
            df (DataFrame): DataFrame with signals
            
        Returns:
            DataFrame: Data with position sizes
        """
        # Make a copy to avoid modifying the original
        result = df.copy()
        
        # Calculate position size as fraction of portfolio
        position_value = portfolio_value * self.position_size
        
        # Add position sizes to buy signals
        result['Position_Size'] = 0
        buy_signals = result['Buy_Signal'] == 1
        if buy_signals.any():
            result.loc[buy_signals, 'Position_Size'] = position_value / result.loc[buy_signals, 'Close']
            
        logger.debug(f"Applied position sizing, average position: {result['Position_Size'].mean():.2f} shares")
        
        return result
        
    def apply_stop_loss_take_profit(self, df):
        """
        Apply stop loss and take profit to trading signals
        
        Args:
            df: DataFrame with trading signals
        
        Returns:
            DataFrame: Updated with stop loss/take profit signals
        """
        result = df.copy()
        
        # Check column structure to determine if using MultiIndex
        using_multi_index = isinstance(result.columns, pd.MultiIndex)
        
        # Helper function to access columns regardless of index type
        def get_col_value(row, col_name):
            if using_multi_index:
                # Try different variations of the column name in multi-index
                if (col_name, '') in row:
                    return row[(col_name, '')]
                elif col_name in [c[0] for c in row.index]:
                    # Find the first column that starts with the name
                    for col in row.index:
                        if col[0] == col_name:
                            return row[col]
                return None
            else:
                # Simple column access
                return row[col_name] if col_name in row else None
        
        # Helper function to check if a signal exists
        def has_signal(row, signal_name):
            val = get_col_value(row, signal_name)
            return val == 1 if val is not None else False
        
        # Initialize tracking variables
        in_position = False
        entry_price = 0
        stop_loss_price = 0
        take_profit_price = 0
        entry_index = 0
        
        # Iterate through the data
        for i in range(1, len(result)):
            current_row = result.iloc[i]
            prev_row = result.iloc[i-1]
            
            # Get Close price
            close_price = get_col_value(current_row, 'Close')
            if close_price is None or pd.isna(close_price):
                continue
                
            # Convert to float if it's a Series
            if isinstance(close_price, pd.Series):
                close_price = float(close_price.iloc[0])
            else:
                close_price = float(close_price)
            
            # Entry logic - Buy signal when not in position
            if not in_position and has_signal(prev_row, 'Buy_Signal'):
                in_position = True
                entry_price = close_price
                entry_index = i
                
                # Calculate stop loss and take profit levels
                stop_loss_price = entry_price * (1 - self.stop_loss)
                take_profit_price = entry_price * (1 + self.take_profit)
                
                # Log the entry
                logger.info(f"Entry at {result.index[i]}: {entry_price:.2f}, " +
                           f"SL: {stop_loss_price:.2f}, TP: {take_profit_price:.2f}")
            
            # Exit logic - only check if in a position
            elif in_position:
                # Check for stop loss hit
                if close_price <= stop_loss_price:
                    # Set exit signals
                    if using_multi_index:
                        if ('Sell_Signal', '') in result.columns:
                            result.at[result.index[i], ('Sell_Signal', '')] = 1
                        else:
                            # Find the appropriate column
                            for col in result.columns:
                                if col[0] == 'Sell_Signal':
                                    result.at[result.index[i], col] = 1
                                    break
                    else:
                        result.at[result.index[i], 'Sell_Signal'] = 1
                    
                    # Log the exit
                    logger.info(f"Stop loss hit at {result.index[i]}: {close_price:.2f} " +
                               f"(Entry: {entry_price:.2f}, Loss: {((close_price/entry_price)-1)*100:.2f}%)")
                    
                    # Reset position tracking
                    in_position = False
                    
                # Check for take profit hit
                elif close_price >= take_profit_price:
                    # Set exit signals
                    if using_multi_index:
                        if ('Sell_Signal', '') in result.columns:
                            result.at[result.index[i], ('Sell_Signal', '')] = 1
                        else:
                            # Find the appropriate column
                            for col in result.columns:
                                if col[0] == 'Sell_Signal':
                                    result.at[result.index[i], col] = 1
                                    break
                    else:
                        result.at[result.index[i], 'Sell_Signal'] = 1
                    
                    # Log the exit
                    logger.info(f"Take profit hit at {result.index[i]}: {close_price:.2f} " +
                               f"(Entry: {entry_price:.2f}, Gain: {((close_price/entry_price)-1)*100:.2f}%)")
                    
                    # Reset position tracking
                    in_position = False
                    
                # Check for existing sell signal
                elif has_signal(current_row, 'Sell_Signal'):
                    # Log the exit
                    logger.info(f"Strategy sell signal at {result.index[i]}: {close_price:.2f} " +
                               f"(Entry: {entry_price:.2f}, {((close_price/entry_price)-1)*100:.2f}%)")
                    
                    # Reset position tracking
                    in_position = False
        
        return result
        
    def limit_concurrent_positions(self, signals_data_dict, max_positions=None):
        """
        Limit the number of concurrent positions across multiple symbols
        
        Args:
            signals_data_dict (dict): Dictionary of signal DataFrames by symbol
            max_positions (int): Maximum allowed positions (default: self.max_position_count)
            
        Returns:
            dict: Modified signals data
        """
        if max_positions is None:
            max_positions = self.max_position_count
            
        # Initialize positions tracking
        current_positions = {}
        result = {}
        
        # Make copies of input DataFrames to avoid modification
        for symbol, df in signals_data_dict.items():
            result[symbol] = df.copy()
            
        # Get all dates across all DataFrames
        all_dates = set()
        for df in result.values():
            all_dates.update(df.index)
        
        all_dates = sorted(all_dates)
        
        # Process each date
        for date in all_dates:
            active_positions = len(current_positions)
            
            # Process exit signals first to free up position slots
            for symbol, entry_date in list(current_positions.items()):
                if date in result[symbol].index:
                    if result[symbol].loc[date, 'Sell_Signal'] == 1:
                        del current_positions[symbol]
                        logger.debug(f"{date}: Exited position in {symbol}")
            
            # Process entry signals if positions are available
            for symbol, df in result.items():
                if date in df.index and df.loc[date, 'Buy_Signal'] == 1:
                    if symbol not in current_positions and len(current_positions) < max_positions:
                        current_positions[symbol] = date
                        logger.debug(f"{date}: Entered position in {symbol}")
                    elif symbol not in current_positions:
                        # Too many positions, suppress buy signal
                        result[symbol].loc[date, 'Buy_Signal'] = 0
                        logger.debug(f"{date}: Suppressed buy signal for {symbol} (max positions reached)")
        
        return result
        
    def apply_risk_management(self, data, portfolio_value=100000):
        """
        Apply all risk management rules to a DataFrame
        
        Args:
            data (DataFrame): Input DataFrame with signals
            portfolio_value (float): Initial portfolio value
            
        Returns:
            DataFrame: Risk-managed DataFrame
        """
        logger.info("Applying risk management rules")
        
        # Apply rules in sequence
        df = self.apply_position_sizing(portfolio_value, data)
        df = self.apply_stop_loss_take_profit(df)
        
        logger.info("Risk management applied successfully")
        return df