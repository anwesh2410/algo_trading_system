import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from pathlib import Path

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Use absolute imports
from src.utils.logger import setup_logger
from src.utils.data_processor import DataProcessor
from src.strategies.backtest import Backtest
from config.config_loader import Config
from src.models.ml_predictor import MLPredictor
from src.strategies.ml_enhanced_strategy import MLEnhancedStrategy
from src.models.enhanced_ml_predictor import EnhancedMLPredictor
from src.backtest.risk_manager import RiskManager
from src.portfolio.portfolio_optimizer import PortfolioOptimizer
from src.integrations.integrations_manager import IntegrationsManager

# Initialize logger
logger = setup_logger('main')

def main():
    """Main execution function for the trading system"""
    logger.info("Starting Algorithmic Trading System - Test Mode")
    
    try:
        # Initialize integrations
        integrations = IntegrationsManager.get_instance()
        
        # Send system startup notification
        integrations.telegram_helper.send_system_alert(
            "Algo Trading System starting up - Test Mode",
            "info"
        )
        
        # Process data for all stocks
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
        
        # Run backtesting on processed data
        backtest_results = run_backtests(processed_data)
        
        # Log backtest results to Google Sheets
        performance_data = {}
        for symbol, results in backtest_results.items():
            performance_data[symbol] = results['metrics']
        
        integrations.update_performance(performance_data)
        
        # Print summary of results
        print_summary(backtest_results)
        
        # Analyze strategy signals
        try:
            analyze_no_trades(processed_data)
        except Exception as e:
            logger.error(f"Error in strategy analysis: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
        
        # Train ML models
        try:
            enhanced_ml_models = train_enhanced_ml_models(processed_data)
            logger.info(f"Successfully trained ML models for {len(enhanced_ml_models)} stocks")
            
            # Make predictions using trained models
            predictions = make_ml_predictions(enhanced_ml_models, processed_data)
            
            # Log ML predictions to integrations
            for symbol, pred_data in predictions.items():
                integrations.log_ml_prediction(
                    symbol=symbol,
                    prediction=pred_data['next_day'],
                    confidence=0.75,  # Example confidence
                    supporting_data={
                        "Bullish Rate": f"{pred_data['accuracy']*100:.2f}%",
                        "Recent Trend": ' → '.join(["UP" if p == 1 else "DOWN" for p in pred_data['prediction_array'][-3:]])
                    }
                )
            
            # Print predictions
            print_ml_predictions(predictions)
            
            # Run ML-enhanced backtests
            ml_backtest_results = run_ml_enhanced_backtests(processed_data, enhanced_ml_models)
            print_summary(ml_backtest_results, ml_enhanced=True)
            
            # Process trades for integrations
            for symbol, results in ml_backtest_results.items():
                trades = results['trades']
                if not trades.empty:
                    for _, trade in trades.iterrows():
                        # Log significant trades (>2% gain/loss)
                        if abs(trade['PnL_Pct']) > 2.0:
                            integrations.log_trade(
                                symbol=symbol,
                                action="SELL",  # This is when we exit the trade
                                price=trade['Exit_Price'],
                                quantity=int(trade['Portfolio'] / trade['Entry_Price'] * 0.1),  # Example allocation
                                pnl=trade['PnL_Pct'],
                                portfolio_value=trade['Portfolio'],
                                strategy="ML Enhanced"
                            )
                    
            # Update ML performance metrics in Google Sheets
            ml_performance_data = {}
            for symbol, results in ml_backtest_results.items():
                ml_performance_data[symbol] = results['metrics']
            
            # Log with "ML_" prefix to differentiate
            for symbol, metrics in ml_performance_data.items():
                metrics_with_prefix = {f"ML_{k}": v for k, v in metrics.items()}
                integrations.sheets_helper.update_performance({symbol: metrics_with_prefix})
            
            # Print ML-enhanced backtest summary
            print("\n" + "="*50)
            print("ML-ENHANCED BACKTEST RESULTS")
            print("="*50)
            
            for symbol, results in ml_backtest_results.items():
                metrics = results['metrics']
                print(f"\nML-Enhanced Results for {symbol}:")
                print(f"  Total Trades: {metrics['Total_Trades']}")
                print(f"  Win Rate: {metrics['Win_Rate']:.2f}%")
                print(f"  Average Return per Trade: {metrics['Average_Return']:.2f}%")
                print(f"  Total Return: {metrics['Total_Return']:.2f}%")
                print(f"  Sharpe Ratio: {metrics['Sharpe_Ratio']:.2f}")
                print(f"  Maximum Drawdown: {metrics['Max_Drawdown']:.2f}%")
            
            print("\n" + "="*50)
            
        except Exception as e:
            logger.error(f"Error in ML processing: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
        
        # Run backtests with risk management
        risk_managed_results = run_risk_managed_backtest(processed_data)
        
        # Optimize portfolio allocation
        portfolio_allocations = optimize_portfolio_allocation(risk_managed_results, processed_data)

        # Print portfolio allocations
        print("\n" + "="*50)
        print("PORTFOLIO RECOMMENDATIONS")
        print("="*50)

        # Send portfolio recommendations to Telegram
        portfolio_recommendations = []
        
        for risk_profile, allocation in portfolio_allocations.items():
            print(f"\n{risk_profile.capitalize()} Risk Profile:")
            print(f"  Strategy: {allocation['strategy']}")
            print(f"  Expected Annual Return: {allocation['expected_return']*100:.2f}%")
            print(f"  Expected Volatility: {allocation['expected_volatility']*100:.2f}%")
            print(f"  Sharpe Ratio: {allocation['sharpe_ratio']:.2f}")
            print("  Recommended Allocation:")
            
            # Format for Telegram
            weights_text = ""
            for symbol, weight in allocation['weights'].items():
                print(f"    {symbol}: {weight:.2f}%")
                weights_text += f"    {symbol}: {weight:.2f}%\n"
            
            portfolio_recommendations.append({
                "risk_profile": risk_profile.capitalize(),
                "expected_return": allocation['expected_return']*100,
                "sharpe_ratio": allocation['sharpe_ratio'],
                "weights": weights_text
            })

        # Send portfolio recommendations as Telegram alert
        for recommendation in portfolio_recommendations:
            # Use the safer formatting method
            message_parts = {
                "title": f"Portfolio Recommendation - {recommendation['risk_profile']} Risk",
                "sections": [
                    {"header": "Expected Return", "content": f"{recommendation['expected_return']:.2f}%"},
                    {"header": "Sharpe Ratio", "content": f"{recommendation['sharpe_ratio']:.2f}"},
                    {"header": "Allocation", "content": recommendation['weights']}
                ]
            }
            
            integrations.telegram_helper.send_message(
                integrations.telegram_helper.format_safe_message(message_parts)
            )

        print("\n" + "="*50)
        
        # Send system completion notification
        integrations.telegram_helper.send_system_alert(
            "Algo Trading System test execution completed successfully",
            "info"
        )
        
        logger.info("Test execution completed successfully")
        
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

def run_backtests(processed_data):
    """
    Run backtests on all processed stocks
    
    Args:
        processed_data (dict): Dictionary of processed stock data
        
    Returns:
        dict: Dictionary of backtest results
    """
    logger.info("Running backtests on processed data")
    
    backtest = Backtest()
    results = {}
    
    for symbol, data in processed_data.items():
        logger.info(f"Running backtest for {symbol}")
        
        # Run backtest
        backtest_data, trades, metrics = backtest.run_backtest(data, symbol)
        
        # Plot results if there were any trades
        if not trades.empty:
            backtest.plot_results(backtest_data, trades, symbol)
        
        results[symbol] = {
            'backtest_data': backtest_data,
            'trades': trades,
            'metrics': metrics
        }
        
        # Log significant trades to integrations
        integrations = IntegrationsManager.get_instance()
        if not trades.empty:
            for _, trade in trades.iterrows():
                # Only log significant trades (e.g., over 2% gain/loss)
                if abs(trade['PnL_Pct']) > 2.0:
                    integrations.log_trade(
                        symbol=symbol,
                        action="SELL", # This is when we exit the trade
                        price=trade['Exit_Price'],
                        quantity=int(Config.INITIAL_CAPITAL / trade['Entry_Price'] * 0.1),  # 10% allocation example
                        pnl=trade['PnL_Pct'],
                        portfolio_value=trade['Portfolio'],
                        strategy="Standard"
                    )
    
    return results

def print_summary(backtest_results, ml_enhanced=False):
    """Print a summary of all backtest results"""
    suffix = " (ML-enhanced)" if ml_enhanced else ""
    
    print("\n" + "="*50)
    print(f"BACKTEST RESULTS SUMMARY{suffix}")
    print("="*50)
    
    for symbol, result in backtest_results.items():
        metrics = result['metrics']
        trades = result['trades']
        
        print(f"\nResults for {symbol}:")
        print(f"  Total Trades: {metrics['Total_Trades']}")
        print(f"  Win Rate: {metrics['Win_Rate']:.2f}%")
        print(f"  Average Return per Trade: {metrics['Average_Return']:.2f}%")
        print(f"  Total Return: {metrics['Total_Return']:.2f}%")
        print(f"  Sharpe Ratio: {metrics['Sharpe_Ratio']:.2f}")
        print(f"  Maximum Drawdown: {metrics['Max_Drawdown']:.2f}%")
        
        if not trades.empty:
            print("  Recent trades:")
            print(trades.tail(3)[['Entry_Date', 'Entry_Price', 'Exit_Date', 'Exit_Price', 'PnL_Pct']])
        else:
            print("  No trades executed during backtesting period")
    
    print("\n" + "="*50)

def analyze_no_trades(processed_data):
    """Analyze strategy signals"""
    print("\n" + "="*50)
    print("STRATEGY ANALYSIS")
    print("="*50)
    
    for symbol, data in processed_data.items():
        # Check how many times RSI was below threshold
        rsi_opportunities = len(data[data['RSI'] < Config.RSI_BUY_THRESHOLD])
        
        # Check how many times MA crossover happened
        ma_crossovers = data['MA_Crossover'].sum()
        
        # Report signal statistics
        print(f"\nAnalysis for {symbol}:")
        print(f"  Days with RSI < {Config.RSI_BUY_THRESHOLD}: {rsi_opportunities}")
        print(f"  MA Crossover events: {ma_crossovers}")
        print(f"  Total buy signals generated: {data['Buy_Signal'].sum()}")
        print(f"  Total sell signals generated: {data['Sell_Signal'].sum()}")
    
    print("\n" + "="*50)

def train_ml_models(processed_data):
    """
    Train ML models for each stock
    
    Args:
        processed_data (dict): Dictionary of processed stock data
        
    Returns:
        dict: Dictionary of trained models
    """
    logger.info("Training machine learning models")
    
    models = {}
    
    for symbol, data in processed_data.items():
        logger.info(f"Training model for {symbol}")
        
        # Initialize and train model
        predictor = MLPredictor()
        accuracy = predictor.train_model(data)
        
        models[symbol] = predictor
        
        logger.info(f"Model for {symbol} trained with accuracy: {accuracy:.4f}")
    
    return models

def make_ml_predictions(models, processed_data):
    """
    Make predictions using trained ML models
    
    Args:
        models (dict): Dictionary of trained models
        processed_data (dict): Dictionary of processed stock data
        
    Returns:
        dict: Dictionary of prediction results
    """
    logger.info("Making predictions with ML models")
    
    predictions = {}
    
    for symbol, predictor in models.items():
        logger.info(f"Making predictions for {symbol}")
        
        # Get the latest data for prediction
        latest_data = processed_data[symbol].iloc[-30:].copy()  # Use last 30 days
        
        try:
            # Make predictions
            pred = predictor.predict(latest_data)
            
            if pred is not None and len(pred) > 0:
                # Last prediction is for the next trading day
                next_day_prediction = "UP" if pred[-1] == 1 else "DOWN"
                logger.info(f"Prediction for {symbol} next trading day: {next_day_prediction}")
                
                predictions[symbol] = {
                    'next_day': next_day_prediction,
                    'prediction_array': pred,
                    'accuracy': sum(pred == 1) / len(pred) if len(pred) > 0 else 0
                }
        except Exception as e:
            logger.error(f"Error making prediction for {symbol}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
    
    return predictions

def print_ml_predictions(predictions):
    """Print ML model predictions in a nicely formatted way"""
    print("\n" + "="*50)
    print("MACHINE LEARNING PREDICTIONS")
    print("="*50)
    
    for symbol, pred_data in predictions.items():
        print(f"\nPredictions for {symbol}:")
        print(f"  Next trading day: {pred_data['next_day']}")
        
        # Calculate bullish percentage
        bullish_pct = pred_data['accuracy'] * 100
        print(f"  Bullish prediction rate: {bullish_pct:.2f}%")
        
        # Show recent trend
        recent_preds = ["UP" if p == 1 else "DOWN" for p in pred_data['prediction_array'][-5:]]
        print(f"  Recent predictions (last 5 days): {' → '.join(recent_preds)}")
    
    print("\n" + "="*50)

def run_ml_enhanced_backtests(processed_data, ml_models):
    """
    Run backtests with ML-enhanced strategy
    
    Args:
        processed_data (dict): Dictionary of processed stock data
        ml_models (dict): Dictionary of trained ML models
        
    Returns:
        dict: Dictionary of backtest results
    """
    logger.info("Running ML-enhanced backtests")
    
    backtest_results = {}
    
    for symbol, data in processed_data.items():
        logger.info(f"Running ML-enhanced backtest for {symbol}")
        
        # Get ML model for this symbol
        if symbol in ml_models:
            ml_predictor = ml_models[symbol]
            
            try:
                # Make predictions
                predictions = ml_predictor.predict(data)
                
                # Create ML-enhanced strategy
                ml_strategy = MLEnhancedStrategy(
                    ml_predictor=ml_predictor,
                    rsi_buy=35,
                    rsi_sell=65,
                    ml_weight=0.6
                )
                
                # Generate signals using ML predictions
                signals_data = ml_strategy.generate_signals(data, predictions)
                
                # Run backtest with enhanced signals
                backtest = Backtest()
                backtest_data, trades, metrics = backtest.run_backtest(signals_data, symbol)
                
                # Store results
                backtest_results[symbol] = {
                    'data': backtest_data,
                    'trades': trades,
                    'metrics': metrics
                }
            except Exception as e:
                logger.error(f"Error in ML-enhanced backtest for {symbol}: {str(e)}")
                # Fall back to regular backtest
                backtest = Backtest()
                backtest_data, trades, metrics = backtest.run_backtest(data, symbol)
                backtest_results[symbol] = {
                    'data': backtest_data,
                    'trades': trades,
                    'metrics': metrics
                }
        else:
            logger.warning(f"No ML model for {symbol}, using standard backtest")
            backtest = Backtest()
            backtest_data, trades, metrics = backtest.run_backtest(data, symbol)
            backtest_results[symbol] = {
                'data': backtest_data,
                'trades': trades,
                'metrics': metrics
            }
    
    return backtest_results

def train_enhanced_ml_models(processed_data):
    """
    Train enhanced ML models on processed data
    
    Args:
        processed_data (dict): Dictionary of processed stock data
        
    Returns:
        dict: Dictionary of trained models
    """
    logger.info("Training enhanced machine learning models")
    
    models = {}
    
    for symbol, data in processed_data.items():
        logger.info(f"Training enhanced model for {symbol}")
        
        try:
            # Create and train model
            predictor = EnhancedMLPredictor()
            accuracy = predictor.train_model(data)
            
            models[symbol] = predictor
            logger.info(f"Enhanced model for {symbol} trained with accuracy: {accuracy:.4f}")
        except Exception as e:
            logger.error(f"Error training enhanced model for {symbol}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
    
    return models

def run_risk_managed_backtest(processed_data):
    """
    Run backtests with risk management
    
    Args:
        processed_data (dict): Dictionary of processed stock data
        
    Returns:
        dict: Dictionary of backtest results
    """
    logger.info("Running backtests with risk management")
    
    risk_manager = RiskManager(
        position_size=0.15,       # 15% of portfolio per position
        stop_loss_pct=0.03,       # 3% stop loss
        take_profit_pct=0.06,     # 6% take profit
        max_drawdown_pct=0.15,    # 15% maximum drawdown
        max_position_count=2      # 2 simultaneous positions max
    )
    
    backtest_results = {}
    
    # Apply risk management to each stock
    for symbol, data in processed_data.items():
        logger.info(f"Running risk-managed backtest for {symbol}")
        
        # Apply risk management rules
        risk_managed_data = risk_manager.apply_risk_management(data, portfolio_value=100000)
        
        # Run backtest with risk-managed data
        backtest = Backtest()
        backtest_data, trades, metrics = backtest.run_backtest(risk_managed_data, symbol)
        
        # Store results
        backtest_results[symbol] = {
            'data': backtest_data,
            'trades': trades,
            'metrics': metrics
        }



