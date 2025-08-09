#!/usr/bin/env python3
# Enhanced Run Script - Uses all improved components

import os
import sys
from pathlib import Path
import pandas as pd
from datetime import datetime

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

# Import required modules
from src.utils.logger import setup_logger
from src.utils.data_processor import DataProcessor
from src.strategies.backtest import Backtest
from src.visualization.dashboard import PerformanceDashboard
from src.models.enhanced_ml_predictor import EnhancedMLPredictor
from src.strategies.ml_enhanced_strategy import MLEnhancedStrategy
from config.config_loader import Config
from src.integrations.integrations_manager import IntegrationsManager

# Set up logger
logger = setup_logger('enhanced_run')

def main():
    """Enhanced main execution function"""
    logger.info("Starting Enhanced Algorithmic Trading System")
    
    try:
        # Initialize integrations
        integrations = IntegrationsManager.get_instance()
        
        # Send system startup notification
        integrations.telegram_helper.send_system_alert(
            "Algo Trading System starting up - Enhanced Mode",
            "info"
        )
        
        # 1. Process data for all stocks
        logger.info("Processing stock data...")
        data_processor = DataProcessor()
        processed_data = data_processor.process_all_stocks()
        
        if not processed_data:
            logger.error("No data was processed. Exiting.")
            integrations.telegram_helper.send_system_alert(
                "ERROR: No data was processed. System shutting down.",
                "error"
            )
            return
            
        logger.info(f"Successfully processed data for {len(processed_data)} stocks")
        
        # 2. Train enhanced ML models
        logger.info("Training enhanced ML models...")
        ml_models = {}
        
        for symbol, data in processed_data.items():
            logger.info(f"Training enhanced model for {symbol}")
            try:
                predictor = EnhancedMLPredictor()
                accuracy = predictor.train_model(data)
                ml_models[symbol] = predictor
                logger.info(f"Model for {symbol} trained with accuracy: {accuracy:.4f}")
            except Exception as e:
                logger.error(f"Error training model for {symbol}: {str(e)}")
                
        # 3. Generate ML predictions
        logger.info("Generating ML predictions...")
        predictions = {}
        
        for symbol, predictor in ml_models.items():
            try:
                # Use last 30 days for prediction
                latest_data = processed_data[symbol].iloc[-30:].copy()
                pred = predictor.predict(latest_data)
                
                if pred is not None:
                    next_day = "UP" if pred[-1] == 1 else "DOWN"
                    confidence = 0.75  # Example confidence value, adjust as needed
                    predictions[symbol] = {
                        "next_day": next_day,
                        "confidence": confidence,
                        "predictions": pred
                    }
                    
                    # Log ML prediction to Google Sheets
                    integrations.log_ml_prediction(
                        symbol=symbol, 
                        prediction=next_day,
                        confidence=confidence,
                        supporting_data={
                            "Model": "Enhanced ML",
                            "Date": datetime.now().strftime("%Y-%m-%d")
                        }
                    )
                    
                    logger.info(f"Prediction for {symbol}: {next_day}")
            except Exception as e:
                logger.error(f"Error predicting for {symbol}: {str(e)}")
        
        # 4. Run ML-enhanced backtests
        logger.info("Running ML-enhanced backtests...")
        backtest_results = {}
        portfolio_value = Config.INITIAL_CAPITAL
        
        for symbol, data in processed_data.items():
            if symbol in ml_models:
                try:
                    # Create ML-enhanced strategy
                    ml_strategy = MLEnhancedStrategy(ml_predictor=ml_models[symbol])
                    
                    # Get predictions
                    pred = ml_models[symbol].predict(data)
                    
                    # Generate signals with ML enhancement
                    signals_data = ml_strategy.generate_signals(data, pred)
                    
                    # Run backtest
                    backtest = Backtest()
                    backtest_data, trades, metrics = backtest.run_backtest(signals_data, symbol)
                    
                    # Process trades for logging
                    if not trades.empty:
                        for _, trade in trades.iterrows():
                            # Log trade to integrations
                            integrations.log_trade(
                                symbol=symbol,
                                action="SELL",  # This is when we exit the trade
                                price=trade['Exit_Price'],
                                quantity=int(portfolio_value / trade['Entry_Price'] * 0.1),  # Example allocation
                                pnl=trade['PnL_Pct'],
                                portfolio_value=trade['Portfolio'],
                                strategy="ML Enhanced"
                            )
                    
                    # Store results
                    backtest_results[symbol] = {
                        'data': backtest_data,
                        'trades': trades,
                        'metrics': metrics
                    }
                    
                    logger.info(f"ML-enhanced backtest for {symbol} completed")
                    logger.info(f"Metrics: Win Rate={metrics['Win_Rate']:.2f}%, "
                                f"Total Return={metrics['Total_Return']:.2f}%, "
                                f"Sharpe={metrics['Sharpe_Ratio']:.2f}")
                except Exception as e:
                    logger.error(f"Error in ML-enhanced backtest for {symbol}: {str(e)}")
        
        # 5. Generate performance dashboard
        logger.info("Generating performance dashboard...")
        dashboard = PerformanceDashboard()
        dashboard_path = dashboard.create_dashboard(backtest_results)
        logger.info(f"Dashboard created at {dashboard_path}")
        
        # 6. Update performance metrics in Google Sheets
        logger.info("Updating performance data in Google Sheets")
        performance_data = {}
        for symbol, results in backtest_results.items():
            performance_data[symbol] = results['metrics']
        
        integrations.update_performance(
            performance_data,
            send_telegram=True,
            detailed_metrics={
                symbol: {
                    "Sharpe": results['metrics']['Sharpe_Ratio'],
                    "Max Drawdown": f"{results['metrics']['Max_Drawdown']:.2f}%",
                    "Trades": results['metrics']['Total_Trades']
                } for symbol, results in backtest_results.items()
            }
        )
        
        # 7. Calculate portfolio metrics and send daily summary
        total_portfolio_value = sum([
            results['trades']['Portfolio'].iloc[-1] if not results['trades'].empty else Config.INITIAL_CAPITAL
            for symbol, results in backtest_results.items()
        ])
        
        daily_change = (total_portfolio_value / (len(backtest_results) * Config.INITIAL_CAPITAL) - 1) * 100
        
        # Get top performers
        top_performers = []
        for symbol, results in backtest_results.items():
            if not results['trades'].empty:
                top_performers.append({
                    "symbol": symbol,
                    "return": results['metrics']['Total_Return']
                })
        
        # Sort by return
        top_performers = sorted(top_performers, key=lambda x: x["return"], reverse=True)[:3]
        
        # Send daily summary
        integrations.send_daily_summary(
            portfolio_value=total_portfolio_value,
            daily_change=daily_change,
            top_performers=top_performers,
            trades_executed=sum([len(results['trades']) for _, results in backtest_results.items()]),
            signals_generated=sum([
                (data['Buy_Signal'] == 1).sum() + (data['Sell_Signal'] == 1).sum()
                for symbol, data in processed_data.items()
            ])
        )
        
        # 8. Print summary
        print("\n" + "="*50)
        print("ENHANCED TRADING SYSTEM RESULTS")
        print("="*50)
        
        for symbol, results in backtest_results.items():
            metrics = results['metrics']
            trades = results['trades']
            
            print(f"\nResults for {symbol}:")
            print(f"  Total Trades: {metrics['Total_Trades']}")
            print(f"  Win Rate: {metrics['Win_Rate']:.2f}%")
            print(f"  Average Return per Trade: {metrics['Average_Return']:.2f}%")
            print(f"  Total Return: {metrics['Total_Return']:.2f}%")
            print(f"  Sharpe Ratio: {metrics['Sharpe_Ratio']:.2f}")
            print(f"  Maximum Drawdown: {metrics['Max_Drawdown']:.2f}%")
            
            if symbol in predictions:
                print(f"  Next day prediction: {predictions[symbol]['next_day']}")
            
            print("\n  Recent trades:")
            if not trades.empty:
                print(trades.tail(3)[['Entry_Date', 'Entry_Price', 'Exit_Date', 'Exit_Price', 'PnL_Pct']])
        
        print("\n" + "="*50)
        print("Performance dashboard available at:", dashboard_path)
        print("="*50)
        
        # Send completion notification
        integrations.telegram_helper.send_system_alert(
            "Algo Trading System execution completed successfully!",
            "info"
        )
        
    except Exception as e:
        logger.error(f"Error during execution: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        
        # Send error alert
        try:
            IntegrationsManager.get_instance().telegram_helper.send_system_alert(
                f"ERROR: System encountered an exception: {str(e)}",
                "error"
            )
        except:
            pass

if __name__ == "__main__":
    main()