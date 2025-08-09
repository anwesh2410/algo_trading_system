import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from pathlib import Path
import os
import sys
import traceback

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from src.utils.logger import setup_logger

# Initialize logger
logger = setup_logger('dashboard')

class PerformanceDashboard:
    """
    Generate visualization dashboard for trading performance
    """
    
    def __init__(self, output_dir=None):
        """Initialize dashboard"""
        if output_dir is None:
            output_dir = Path(__file__).parent.parent.parent / "data" / "dashboard"
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
    def plot_equity_curve(self, trades_df, symbol, initial_capital=100000):
        """Plot equity curve from trade results"""
        try:
            plt.figure(figsize=(12, 6))
            
            # Plot equity curve
            plt.plot(trades_df['Exit_Date'], trades_df['Portfolio'], marker='', linewidth=2)
            plt.axhline(y=initial_capital, color='r', linestyle='--', label='Initial Capital')
            
            # Format
            plt.title(f'{symbol} - Portfolio Equity Curve', fontsize=15)
            plt.xlabel('Date')
            plt.ylabel('Portfolio Value')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Format x-axis dates
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
            plt.gcf().autofmt_xdate()
            
            # Save
            plt.tight_layout()
            plt.savefig(self.output_dir / f"{symbol}_equity_curve.png")
            plt.close()
            
            logger.info(f"Saved equity curve for {symbol}")
        except Exception as e:
            logger.error(f"Error plotting equity curve: {str(e)}")
            logger.error(traceback.format_exc())
        
    def plot_monthly_returns(self, trades_df, symbol):
        """Plot monthly returns"""
        try:
            # Convert dates to datetime if they aren't already
            if not pd.api.types.is_datetime64_any_dtype(trades_df['Exit_Date']):
                trades_df = trades_df.copy()
                trades_df['Exit_Date'] = pd.to_datetime(trades_df['Exit_Date'])
            
            # Extract month and year
            trades_df['Month'] = trades_df['Exit_Date'].dt.strftime('%Y-%m')
            
            # Calculate monthly returns
            monthly_returns = trades_df.groupby('Month')['PnL_Pct'].sum()
            
            plt.figure(figsize=(12, 6))
            
            # Create bar chart
            bars = plt.bar(monthly_returns.index, monthly_returns.values)
            
            # Color bars based on return
            for i, bar in enumerate(bars):
                if monthly_returns.values[i] > 0:
                    bar.set_color('green')
                else:
                    bar.set_color('red')
            
            # Format
            plt.title(f'{symbol} - Monthly Returns', fontsize=15)
            plt.xlabel('Month')
            plt.ylabel('Return (%)')
            plt.grid(True, alpha=0.3, axis='y')
            
            # Rotate x labels
            plt.xticks(rotation=45)
            
            # Save
            plt.tight_layout()
            plt.savefig(self.output_dir / f"{symbol}_monthly_returns.png")
            plt.close()
            
            logger.info(f"Saved monthly returns for {symbol}")
        except Exception as e:
            logger.error(f"Error plotting monthly returns: {str(e)}")
            logger.error(traceback.format_exc())
        
    def plot_win_loss_distribution(self, trades_df, symbol):
        """Plot win/loss distribution"""
        try:
            plt.figure(figsize=(10, 6))
            
            # Create histogram of returns
            sns.histplot(trades_df['PnL_Pct'], bins=20, kde=True)
            
            # Add vertical line at 0
            plt.axvline(x=0, color='r', linestyle='--')
            
            # Format
            plt.title(f'{symbol} - Win/Loss Distribution', fontsize=15)
            plt.xlabel('Return per Trade (%)')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
            
            # Save
            plt.tight_layout()
            plt.savefig(self.output_dir / f"{symbol}_win_loss_dist.png")
            plt.close()
            
            logger.info(f"Saved win/loss distribution for {symbol}")
        except Exception as e:
            logger.error(f"Error plotting win/loss distribution: {str(e)}")
            logger.error(traceback.format_exc())
        
    def plot_drawdown(self, trades_df, symbol, initial_capital=100000):
        """Plot drawdown chart"""
        try:
            # Calculate drawdown
            portfolio = trades_df['Portfolio'].values
            peak = np.maximum.accumulate(portfolio)
            drawdown = 100 * (peak - portfolio) / peak
            
            plt.figure(figsize=(12, 6))
            
            # Create dates for x-axis
            dates = trades_df['Exit_Date']
            
            # Plot drawdown
            plt.fill_between(dates, drawdown, 0, color='red', alpha=0.3)
            plt.plot(dates, drawdown, color='red', label='Drawdown')
            
            # Format
            plt.title(f'{symbol} - Portfolio Drawdown', fontsize=15)
            plt.xlabel('Date')
            plt.ylabel('Drawdown (%)')
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            # Format x-axis dates
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
            plt.gcf().autofmt_xdate()
            
            # Save
            plt.tight_layout()
            plt.savefig(self.output_dir / f"{symbol}_drawdown.png")
            plt.close()
            
            logger.info(f"Saved drawdown chart for {symbol}")
        except Exception as e:
            logger.error(f"Error plotting drawdown: {str(e)}")
            logger.error(traceback.format_exc())
        
    def generate_summary_report(self, backtest_results):
        """Generate summary report for all symbols"""
        try:
            plt.figure(figsize=(12, 8))
            
            # Extract metrics
            symbols = list(backtest_results.keys())
            win_rates = [result['metrics']['Win_Rate'] for result in backtest_results.values()]
            returns = [result['metrics']['Total_Return'] for result in backtest_results.values()]
            trades = [result['metrics']['Total_Trades'] for result in backtest_results.values()]
            
            # Create 2x2 subplot
            plt.subplot(2, 2, 1)
            plt.bar(symbols, win_rates)
            plt.title('Win Rate (%)')
            plt.xticks(rotation=45)
            
            plt.subplot(2, 2, 2)
            plt.bar(symbols, returns)
            plt.title('Total Return (%)')
            plt.xticks(rotation=45)
            
            plt.subplot(2, 2, 3)
            plt.bar(symbols, trades)
            plt.title('Number of Trades')
            plt.xticks(rotation=45)
            
            plt.subplot(2, 2, 4)
            plt.pie([max(0.1, wr) for wr in win_rates], labels=symbols, autopct='%1.1f%%')
            plt.title('Win Rate Distribution')
            
            # Save
            plt.tight_layout()
            plt.savefig(self.output_dir / "performance_summary.png")
            plt.close()
            
            logger.info("Saved performance summary")
        except Exception as e:
            logger.error(f"Error generating summary report: {str(e)}")
            logger.error(traceback.format_exc())
        
    def create_dashboard(self, backtest_results, initial_capital=100000):
        """
        Create complete performance dashboard
        
        Args:
            backtest_results: Dictionary of backtest results
            initial_capital: Initial capital for each stock
        """
        logger.info("Creating performance dashboard")
        
        for symbol, results in backtest_results.items():
            trades_df = results['trades']
            if not trades_df.empty:
                # Make sure Exit_Date is in datetime format
                if not pd.api.types.is_datetime64_any_dtype(trades_df['Exit_Date']):
                    trades_df['Exit_Date'] = pd.to_datetime(trades_df['Exit_Date'])
                
                self.plot_equity_curve(trades_df, symbol, initial_capital)
                self.plot_monthly_returns(trades_df, symbol)
                self.plot_win_loss_distribution(trades_df, symbol)
                self.plot_drawdown(trades_df, symbol, initial_capital)
                logger.info(f"Created all charts for {symbol}")
        
        # Generate summary report
        self.generate_summary_report(backtest_results)
        
        logger.info("Performance dashboard creation complete")
        
        return self.output_dir

