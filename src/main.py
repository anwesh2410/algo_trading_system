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


# Initialize logger
logger = setup_logger('main')

def main():
    """Main execution function for the trading system"""
    logger.info("Starting Algorithmic Trading System - Test Mode")
    
    try:
        # Process data for all stocks
        data_processor = DataProcessor()
        processed_data = data_processor.process_all_stocks()
        
        if not processed_data:
            logger.error("No data was processed. Exiting.")
            return
            
        logger.info(f"Successfully processed data for {len(processed_data)} stocks")
        
        # Run backtesting on processed data
        backtest_results = run_backtests(processed_data)
        
        # Print summary of results
        print_summary(backtest_results)
        
        analyze_no_trades(processed_data)
        
        logger.info("Test execution completed successfully")
        
    except Exception as e:
        logger.error(f"Error during execution: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

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
    
    return results

def print_summary(backtest_results):
    """Print a summary of all backtest results"""
    print("\n" + "="*50)
    print("BACKTEST RESULTS SUMMARY")
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

if __name__ == "__main__":
    main()