import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
import traceback

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from src.utils.logger import setup_logger
from src.utils.dataframe_utils import standardize_dataframe, to_multiindex_dataframe

# Initialize logger
logger = setup_logger('ml_strategy')

class MLEnhancedStrategy:
    """
    Enhanced trading strategy that combines technical indicators with ML predictions
    """
    
    def __init__(self, ml_predictor=None, rsi_buy=35, rsi_sell=65, ml_weight=0.6):
        """Initialize ML-enhanced strategy with parameters"""
        self.ml_predictor = ml_predictor
        self.rsi_buy = rsi_buy
        self.rsi_sell = rsi_sell
        self.ml_weight = ml_weight  # Weight given to ML predictions (0-1)
        logger.info(f"Initialized ML-enhanced strategy with RSI buy: {rsi_buy}, RSI sell: {rsi_sell}, ML weight: {ml_weight}")
        
    def generate_signals(self, df, ml_predictions=None):
        """
        Generate signals combining technical indicators and ML predictions
        
        Args:
            df (DataFrame): Stock data with technical indicators
            ml_predictions (array): ML model predictions
            
        Returns:
            DataFrame: Data with strategy signals
        """
        try:
            # Standardize the DataFrame to handle MultiIndex
            working_data = standardize_dataframe(df)
            
            # Make a copy to avoid SettingWithCopyWarning
            working_data = working_data.copy()
            
            # Initialize signals
            working_data['Buy_Signal'] = 0
            working_data['Sell_Signal'] = 0
            working_data['ML_Prediction'] = 0  # 1 for UP, -1 for DOWN
            
            # Add ML predictions if available
            if ml_predictions is not None and len(ml_predictions) > 0:
                # Map ML predictions (0=down, 1=up) to -1/1
                ml_pred_array = np.array(ml_predictions)
                ml_score = (ml_pred_array * 2) - 1  # Convert 0->-1, 1->1
                
                # Fill with zeros if not enough predictions
                if len(ml_pred_array) < len(working_data):
                    padding = np.zeros(len(working_data) - len(ml_pred_array))
                    ml_score = np.concatenate([padding, ml_score])
                    
                    # Ensure the predictions align with the most recent data
                    ml_score = ml_score[-len(working_data):]
                
                # Truncate if too many predictions
                if len(ml_score) > len(working_data):
                    ml_score = ml_score[-len(working_data):]
                    
                working_data['ML_Prediction'] = ml_score
                logger.info(f"Applied {len(ml_pred_array)} ML predictions")
            
            # Technical signals
            # RSI signals
            if 'RSI' in working_data.columns:
                working_data['RSI_Signal'] = 0
                working_data.loc[working_data['RSI'] < self.rsi_buy, 'RSI_Signal'] = 1
                working_data.loc[working_data['RSI'] > self.rsi_sell, 'RSI_Signal'] = -1
            
            # MA Crossover signal
            if 'MA_Short' in working_data.columns and 'MA_Long' in working_data.columns:
                working_data['MA_Signal'] = 0
                working_data.loc[working_data['MA_Short'] > working_data['MA_Long'], 'MA_Signal'] = 1
                working_data.loc[working_data['MA_Short'] < working_data['MA_Long'], 'MA_Signal'] = -1
            
            # MACD signal
            if 'MACD' in working_data.columns and 'Signal_Line' in working_data.columns:
                working_data['MACD_Signal'] = 0
                working_data.loc[working_data['MACD'] > working_data['Signal_Line'], 'MACD_Signal'] = 1
                working_data.loc[working_data['MACD'] < working_data['Signal_Line'], 'MACD_Signal'] = -1
            
            # Bollinger Band signals
            if all(col in working_data.columns for col in ['Close', 'BB_Lower', 'BB_Upper']):
                working_data['BB_Signal'] = 0
                working_data.loc[working_data['Close'] < working_data['BB_Lower'], 'BB_Signal'] = 1
                working_data.loc[working_data['Close'] > working_data['BB_Upper'], 'BB_Signal'] = -1
            
            # Calculate technical score - only use columns that exist
            tech_columns = [col for col in ['RSI_Signal', 'MA_Signal', 'MACD_Signal', 'BB_Signal'] 
                            if col in working_data.columns]
            
            if tech_columns:
                working_data['Tech_Score'] = working_data[tech_columns].mean(axis=1)
            else:
                working_data['Tech_Score'] = 0
            
            # Combine technical score with ML prediction
            working_data['Strategy_Score'] = (
                (1 - self.ml_weight) * working_data['Tech_Score'] + 
                self.ml_weight * working_data['ML_Prediction']
            )
            
            # Generate signals based on strategy score
            working_data['Buy_Signal'] = 0
            working_data['Sell_Signal'] = 0
            working_data.loc[working_data['Strategy_Score'] > 0.3, 'Buy_Signal'] = 1
            working_data.loc[working_data['Strategy_Score'] < -0.3, 'Sell_Signal'] = 1
            
            # If original data had MultiIndex, put signals back in that format
            if isinstance(df.columns, pd.MultiIndex):
                result = df.copy()
                symbol = df.columns[0][1] if len(df.columns) > 0 and len(df.columns[0]) > 1 else ""
                
                # Add the signals back to the original MultiIndex DataFrame
                result[('Buy_Signal', '')] = working_data['Buy_Signal'].values
                result[('Sell_Signal', '')] = working_data['Sell_Signal'].values
                result[('ML_Prediction', '')] = working_data['ML_Prediction'].values
                result[('Strategy_Score', '')] = working_data['Strategy_Score'].values
                
                return result
            else:
                return working_data
                
        except Exception as e:
            logger.error(f"Error applying ML predictions: {str(e)}")
            logger.error(traceback.format_exc())
            # Return the original data with basic signals
            return df