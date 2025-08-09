#!/usr/bin/env python3
# Live Trading Script - Sends real-time alerts

import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
import numpy as np
import traceback

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

# Import required modules
from src.utils.logger import setup_logger
from src.utils.data_fetcher import DataFetcher
from src.utils.data_processor import DataProcessor
from src.models.enhanced_ml_predictor import EnhancedMLPredictor
from src.strategies.ml_enhanced_strategy import MLEnhancedStrategy
from src.integrations.integrations_manager import IntegrationsManager
from config.config_loader import Config

# Set up logger
logger = setup_logger('live_trading')

class LiveTrading:
    """Live trading system with real-time alerts"""
    
    def __init__(self):
        """Initialize live trading system"""
        self.data_fetcher = DataFetcher()
        self.data_processor = DataProcessor()
        self.ml_models = {}
        self.integrations = IntegrationsManager.get_instance()
        self.portfolio_value = Config.INITIAL_CAPITAL
        self.current_positions = {}
        
        # Load ML models
        self._load_models()
        
    def _load_models(self):
        """Load pre-trained ML models"""
        logger.info("Loading pre-trained ML models")
        
        model_dir = Path(project_root) / "data" / "models"
        
        # Check for available models
        for symbol in Config.STOCK_SYMBOLS:
            model_name = f"{symbol}_enhanced_model"
            predictor = EnhancedMLPredictor(model_dir)
            
            if predictor.load_model(model_name):
                self.ml_models[symbol] = predictor
                logger.info(f"Loaded ML model for {symbol}")
            else:
                logger.warning(f"No model found for {symbol}")
                
        logger.info(f"Loaded {len(self.ml_models)} models")
        
    def check_for_signals(self):
        """Check for new trading signals and send alerts"""
        logger.info("Checking for new trading signals")
        
        # Send status update via Telegram
        self.integrations.telegram_helper.send_system_alert(
            "Checking for new trading signals", 
            "info"
        )
        
        # 1. Fetch latest data
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=100)).strftime('%Y-%m-%d')
        
        # 2. Process data for all symbols
        processed_data = {}
        for symbol in Config.STOCK_SYMBOLS:
            data = self.data_processor.process_stock_data(symbol, start_date, end_date)
            if data is not None:
                processed_data[symbol] = data
        
        if not processed_data:
            logger.error("No data was processed")
            return
            
        # 3. Generate signals and send alerts
        all_signals = []
        
        for symbol, data in processed_data.items():
            try:
                # If we have a model, use ML-enhanced signals
                if symbol in self.ml_models:
                    predictor = self.ml_models[symbol]
                    
                    # Generate ML prediction
                    pred = predictor.predict(data.iloc[-30:])
                    next_day_pred = "UP" if pred[-1] == 1 else "DOWN"
                    confidence = 0.75  # Example confidence value
                    
                    # Log prediction to Google Sheets
                    self.integrations.log_ml_prediction(
                        symbol=symbol,
                        prediction=next_day_pred,
                        confidence=confidence,
                        supporting_data={
                            "Time": datetime.now().strftime("%H:%M:%S"),
                            "Model": "Enhanced ML",
                            "Date": datetime.now().strftime("%Y-%m-%d")
                        }
                    )
                    
                    # Generate signals
                    ml_strategy = MLEnhancedStrategy(ml_predictor=predictor)
                    signals_data = ml_strategy.generate_signals(data, pred)
                    
                    # Check for buy/sell signals on the most recent day
                    latest_data = signals_data.iloc[-1]
                    
                    # Extract signals (handle both MultiIndex and regular DataFrame)
                    buy_signal = False
                    sell_signal = False
                    
                    if isinstance(latest_data.index, pd.MultiIndex):
                        if ('Buy_Signal', '') in latest_data and latest_data[('Buy_Signal', '')] == 1:
                            buy_signal = True
                        if ('Sell_Signal', '') in latest_data and latest_data[('Sell_Signal', '')] == 1:
                            sell_signal = True
                    else:
                        if 'Buy_Signal' in latest_data and latest_data['Buy_Signal'] == 1:
                            buy_signal = True
                        if 'Sell_Signal' in latest_data and latest_data['Sell_Signal'] == 1:
                            sell_signal = True
                    
                    # Get latest price
                    latest_price = latest_data['Close'] if 'Close' in latest_data else latest_data[('Close', symbol)]
                    
                    # Process signals
                    if buy_signal:
                        signal = {
                            "symbol": symbol,
                            "action": "BUY",
                            "price": float(latest_price),
                            "time": datetime.now(),
                            "prediction": next_day_pred,
                            "confidence": confidence
                        }
                        all_signals.append(signal)
                        
                        # Log to Google Sheets
                        self.integrations.log_trade(
                            symbol=symbol,
                            action="BUY",
                            price=float(latest_price),
                            quantity=int(self.portfolio_value * 0.1 / float(latest_price)),  # 10% allocation
                            strategy="ML Enhanced (Live)"
                        )
                        
                        # Send Telegram alert
                        self.integrations.telegram_helper.send_trade_signal(
                            symbol=symbol,
                            action="BUY",
                            price=float(latest_price),
                            quantity=int(self.portfolio_value * 0.1 / float(latest_price)),  # 10% allocation
                            signal_strength=confidence,
                            strategy="ML Enhanced (Live)"
                        )
                        
                        logger.info(f"BUY signal for {symbol} at {latest_price}")
                        
                    if sell_signal:
                        signal = {
                            "symbol": symbol,
                            "action": "SELL",
                            "price": float(latest_price),
                            "time": datetime.now(),
                            "prediction": next_day_pred,
                            "confidence": confidence
                        }
                        all_signals.append(signal)
                        
                        # Log to Google Sheets
                        self.integrations.log_trade(
                            symbol=symbol,
                            action="SELL",
                            price=float(latest_price),
                            quantity=int(self.portfolio_value * 0.1 / float(latest_price)),  # 10% allocation
                            strategy="ML Enhanced (Live)"
                        )
                        
                        # Send Telegram alert
                        self.integrations.telegram_helper.send_trade_signal(
                            symbol=symbol,
                            action="SELL",
                            price=float(latest_price),
                            quantity=int(self.portfolio_value * 0.1 / float(latest_price)),  # 10% allocation
                            signal_strength=confidence,
                            strategy="ML Enhanced (Live)"
                        )
                        
                        logger.info(f"SELL signal for {symbol} at {latest_price}")
                    
                    # Always send prediction alert
                    if not buy_signal and not sell_signal:
                        self.integrations.telegram_helper.send_ml_prediction(
                            symbol=symbol,
                            prediction=next_day_pred,
                            confidence=confidence,
                            supporting_data={
                                "RSI": f"{latest_data['RSI']:.1f}" if 'RSI' in latest_data else "N/A",
                                "MACD": f"{latest_data['MACD']:.3f}" if 'MACD' in latest_data else "N/A",
                                "Current Price": f"${float(latest_price):.2f}"
                            }
                        )
            except Exception as e:
                logger.error(f"Error processing signals for {symbol}: {str(e)}")
                logger.error(traceback.format_exc())
        
        # Send daily summary if we have signals
        if all_signals:
            # Calculate simple portfolio metrics
            buys = sum(1 for s in all_signals if s["action"] == "BUY")
            sells = sum(1 for s in all_signals if s["action"] == "SELL")
            
            # Send summary via Telegram
            summary_message = f"*Daily Trading Summary*\n\n" \
                             f"Total Signals: {len(all_signals)}\n" \
                             f"Buy Signals: {buys}\n" \
                             f"Sell Signals: {sells}\n\n" \
                             f"*Symbols with Signals:*\n" \
                             f"{', '.join(set(s['symbol'] for s in all_signals))}\n\n" \
                             f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                             
            self.integrations.telegram_helper.send_message(summary_message)
            
        logger.info("Finished checking for signals")
        return all_signals
        
    def run_live_trading(self):
        """Run live trading loop"""
        logger.info("Starting live trading mode")
        
        # Send startup alert
        self.integrations.telegram_helper.send_system_alert(
            "Live Trading System is now active", 
            "info"
        )
        
        try:
            while True:
                # Check market hours (9:15 AM to 3:30 PM IST on weekdays)
                now = datetime.now()
                
                # Skip weekends
                if now.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
                    logger.info("Weekend - Market closed")
                    # Sleep for 1 hour
                    time.sleep(3600)
                    continue
                    
                # Check if within trading hours (9:15 AM to 3:30 PM IST)
                market_open = now.replace(hour=9, minute=15, second=0)
                market_close = now.replace(hour=15, minute=30, second=0)
                
                if market_open <= now <= market_close:
                    logger.info("Market is open - checking for signals")
                    signals = self.check_for_signals()
                    
                    if signals:
                        logger.info(f"Generated {len(signals)} trading signals")
                    else:
                        logger.info("No trading signals generated")
                    
                    # Check again in 15 minutes during market hours
                    time.sleep(15 * 60)
                else:
                    logger.info("Outside market hours")
                    
                    # If before market open and same day, sleep until market open
                    if now < market_open:
                        sleep_seconds = (market_open - now).seconds + 5
                        logger.info(f"Sleeping until market open ({sleep_seconds} seconds)")
                        time.sleep(sleep_seconds)
                    else:
                        # If after market close, check if we have open positions to report
                        if hasattr(self, 'current_positions') and self.current_positions:
                            positions_message = "*End of Day - Open Positions*\n\n"
                            for symbol, position in self.current_positions.items():
                                positions_message += f"{symbol}: {position['quantity']} shares @ ${position['price']:.2f}\n"
                            
                            self.integrations.telegram_helper.send_message(positions_message)
                        
                        # Sleep for 1 hour
                        logger.info("Sleeping for 1 hour")
                        time.sleep(3600)
                    
        except KeyboardInterrupt:
            logger.info("Live trading stopped by user")
            self.integrations.telegram_helper.send_system_alert(
                "Live Trading System stopped by user", 
                "info"
            )
        except Exception as e:
            logger.error(f"Error in live trading: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Send error alert
            self.integrations.telegram_helper.send_system_alert(
                f"ERROR: Live trading system encountered an exception: {str(e)}", 
                "error"
            )

if __name__ == "__main__":
    live_trading = LiveTrading()
    live_trading.run_live_trading()