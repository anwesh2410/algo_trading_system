import pandas as pd
import numpy as np
from src.utils.logger import setup_logger

# Initialize logger
logger = setup_logger('risk_management')

class RiskManagement:
    """
    Risk management strategies for trading
    """
    
    @staticmethod
    def apply_position_sizing(data, initial_capital=100000, risk_per_trade=0.02):
        """
        Apply position sizing based on risk management
        
        Args:
            data: DataFrame with trading signals
            initial_capital: Initial capital amount
            risk_per_trade: Risk per trade as a percentage of capital
        
        Returns:
            DataFrame: Data with position sizing applied
        """
        df = data.copy()
        
        # Initialize portfolio value
        df['Portfolio_Value'] = initial_capital
        df['Position_Size'] = 0.0
        
        # Calculate position size for each buy signal
        for i in range(1, len(df)):
            # Update portfolio value (carry forward)
            df.loc[df.index[i], 'Portfolio_Value'] = df.loc[df.index[i-1], 'Portfolio_Value']
            
            # If buy signal, calculate position size
            if df.iloc[i]['Buy_Signal'] == 1:
                # Calculate risk amount
                risk_amount = df.loc[df.index[i], 'Portfolio_Value'] * risk_per_trade
                
                # Use ATR or a percentage for stop loss
                if 'ATR' in df.columns:
                    stop_loss_amount = df.iloc[i]['ATR'] * 1.5  # 1.5 ATR stop loss
                else:
                    stop_loss_amount = df.iloc[i]['Close'] * 0.02  # 2% stop loss
                
                # Calculate position size
                max_position = risk_amount / stop_loss_amount
                position_value = min(max_position * df.iloc[i]['Close'], 
                                    df.loc[df.index[i], 'Portfolio_Value'] * 0.5)  # Max 50% of portfolio
                
                # Store position size
                df.loc[df.index[i], 'Position_Size'] = position_value
                
        logger.info(f"Applied position sizing to {len(df)} rows of data")
        return df
    
    @staticmethod
    def apply_stop_loss_take_profit(data, stop_loss_pct=0.02, take_profit_pct=0.05):
        """
        Apply stop loss and take profit rules to trading signals
        
        Args:
            data: DataFrame with trading signals
            stop_loss_pct: Stop loss percentage
            take_profit_pct: Take profit percentage
        
        Returns:
            DataFrame: Data with stop loss and take profit applied
        """
        df = data.copy()
        
        # Add columns for stop loss and take profit
        df['Stop_Loss'] = 0.0
        df['Take_Profit'] = 0.0
        df['Exit_Type'] = None
        
        in_position = False
        entry_price = 0
        entry_index = 0
        
        for i in range(1, len(df)):
            # If not in position and we get a buy signal
            if not in_position and df.iloc[i-1]['Buy_Signal'] == 1:
                in_position = True
                entry_price = df.iloc[i]['Open']
                entry_index = i
                
                # Calculate stop loss and take profit levels
                stop_loss = entry_price * (1 - stop_loss_pct)
                take_profit = entry_price * (1 + take_profit_pct)
                
                # Store in DataFrame
                df.loc[df.index[i], 'Stop_Loss'] = stop_loss
                df.loc[df.index[i], 'Take_Profit'] = take_profit
            
            # If in position, check for exit conditions
            if in_position:
                # Check if hit stop loss during the day
                if df.iloc[i]['Low'] <= df.iloc[entry_index]['Stop_Loss']:
                    df.loc[df.index[i], 'Sell_Signal'] = 1
                    df.loc[df.index[i], 'Exit_Type'] = 'Stop_Loss'
                    in_position = False
                
                # Check if hit take profit during the day
                elif df.iloc[i]['High'] >= df.iloc[entry_index]['Take_Profit']:
                    df.loc[df.index[i], 'Sell_Signal'] = 1
                    df.loc[df.index[i], 'Exit_Type'] = 'Take_Profit'
                    in_position = False
                
                # Check if regular sell signal
                elif df.iloc[i]['Sell_Signal'] == 1:
                    df.loc[df.index[i], 'Exit_Type'] = 'Signal'
                    in_position = False
        
        logger.info(f"Applied stop loss ({stop_loss_pct:.1%}) and take profit ({take_profit_pct:.1%}) rules")
        return df
    
    @staticmethod
    def apply_max_drawdown_protection(data, max_drawdown_pct=0.10, cooling_period=5):
        """
        Apply maximum drawdown protection - stop trading after hitting max drawdown
        
        Args:
            data: DataFrame with trading signals
            max_drawdown_pct: Maximum allowable drawdown before stopping
            cooling_period: Number of days to pause trading after hitting max drawdown
        
        Returns:
            DataFrame: Data with drawdown protection applied
        """
        df = data.copy()
        
        # Calculate portfolio value if not already present
        if 'Portfolio_Value' not in df.columns:
            # Initialize with placeholder - should be calculated elsewhere
            df['Portfolio_Value'] = 100000
        
        # Calculate drawdown
        df['Peak_Value'] = df['Portfolio_Value'].cummax()
        df['Drawdown'] = (df['Portfolio_Value'] - df['Peak_Value']) / df['Peak_Value']
        
        # Initialize cooldown
        cooling_down = False
        cooldown_counter = 0
        
        for i in range(1, len(df)):
            if cooling_down:
                # In cooldown period, disable trading
                df.loc[df.index[i], 'Buy_Signal'] = 0
                cooldown_counter += 1
                
                if cooldown_counter >= cooling_period:
                    cooling_down = False
                    cooldown_counter = 0
            else:
                # Check if max drawdown exceeded
                if df.iloc[i]['Drawdown'] <= -max_drawdown_pct:
                    cooling_down = True
                    df.loc[df.index[i], 'Buy_Signal'] = 0
                    logger.warning(f"Max drawdown ({max_drawdown_pct:.1%}) exceeded at {df.index[i]}, "
                                   f"cooling down for {cooling_period} days")
        
        return df