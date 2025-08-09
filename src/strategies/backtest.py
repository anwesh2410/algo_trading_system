import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from pathlib import Path
import os
import sys

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from src.utils.logger import setup_logger
from config.config_loader import Config
from src.integrations.integrations_manager import IntegrationsManager

# Set up logger
logger = setup_logger('backtest')

class Backtest:
    """Backtesting engine for trading strategies"""
    
    def __init__(self, initial_capital=100000.0):
        """
        Initialize the backtester
        
        Args:
            initial_capital (float): Initial capital for the backtest
        """
        self.initial_capital = initial_capital
        self.results_dir = Path(__file__).parent.parent.parent / "data" / "results"
        self.results_dir.mkdir(exist_ok=True, parents=True)
        self.integrations = IntegrationsManager.get_instance()
    
    def run_backtest(self, df, symbol):
        """
        Run backtest on the processed data with signals
        
        Args:
            df (DataFrame): Processed data with buy/sell signals
            symbol (str): Stock symbol
            
        Returns:
            DataFrame: Backtest results
        """
        # Make a copy to avoid modifying the original
        data = df.copy()
        
        # Initialize columns for backtesting
        data['Position'] = 0  # 1: Long, 0: No position
        data['Entry_Price'] = 0.0
        data['Exit_Price'] = 0.0
        data['PnL'] = 0.0
        data['Cumulative_PnL'] = 0.0
        data['Portfolio_Value'] = self.initial_capital
        
        position = 0
        entry_price = 0
        portfolio = self.initial_capital
        trades = []
        
        # Iterate through the data to simulate trading
        for i in range(1, len(data)):
            # Check for buy signal when not in a position
            if position == 0 and data['Buy_Signal'].iloc[i] == 1:
                position = 1
                # Extract the actual Close price value - FIX HERE
                if isinstance(data['Close'].iloc[i], pd.Series):
                    entry_price = float(data['Close'].iloc[i].iloc[0])
                else:
                    entry_price = float(data['Close'].iloc[i])
                
                trade_date = data.index[i]
                data.at[data.index[i], 'Position'] = position
                data.at[data.index[i], 'Entry_Price'] = entry_price
                
                logger.info(f"BUY signal for {symbol} at {trade_date}: {entry_price:.2f}")
                
                # Log buy signal to Google Sheets (optional for backtesting)
                if Config.LOG_BACKTEST_SIGNALS_TO_INTEGRATIONS:
                    self.integrations.log_trade(
                        symbol=symbol,
                        action="BUY",
                        price=entry_price,
                        quantity=int(portfolio * 0.1 / entry_price),  # Example allocation
                        portfolio_value=portfolio,
                        strategy="Backtest"
                    )
            
            # Check for sell signal when in a position
            elif position == 1 and data['Sell_Signal'].iloc[i] == 1:
                # Extract the actual Close price value - FIX HERE
                if isinstance(data['Close'].iloc[i], pd.Series):
                    exit_price = float(data['Close'].iloc[i].iloc[0])
                else:
                    exit_price = float(data['Close'].iloc[i])
                
                exit_date = data.index[i]
                pnl = ((exit_price - entry_price) / entry_price) * 100  # Percentage P&L
                portfolio *= (1 + (pnl / 100))
                
                # Log trade details
                trades.append({
                    'Symbol': symbol,
                    'Entry_Date': trade_date,
                    'Entry_Price': entry_price,
                    'Exit_Date': exit_date,
                    'Exit_Price': exit_price,
                    'PnL_Pct': pnl,
                    'Portfolio': portfolio
                })
                
                # Update data - using .at instead of .iloc to avoid SettingWithCopyWarning
                data.at[data.index[i], 'Position'] = 0
                data.at[data.index[i], 'Exit_Price'] = exit_price
                data.at[data.index[i], 'PnL'] = pnl
                data.at[data.index[i], 'Portfolio_Value'] = portfolio
                
                logger.info(f"SELL signal for {symbol} at {exit_date}: {exit_price:.2f}, P&L: {pnl:.2f}%")
                
                # Log sell signal to integrations for significant trades
                if Config.LOG_BACKTEST_SIGNALS_TO_INTEGRATIONS and abs(pnl) > 2.0:
                    self.process_trade(symbol, trade_date, entry_price, exit_date, exit_price, pnl, portfolio)
                
                # Reset position
                position = 0
                entry_price = 0
    
        # Close any open positions at the end of the backtest
        if position == 1:
            if isinstance(data['Close'].iloc[-1], pd.Series):
                exit_price = float(data['Close'].iloc[-1].iloc[0])
            else:
                exit_price = float(data['Close'].iloc[-1])
            
            exit_date = data.index[-1]
            pnl = ((exit_price - entry_price) / entry_price) * 100  # Percentage P&L
            portfolio *= (1 + (pnl / 100))
            
            trades.append({
                'Symbol': symbol,
                'Entry_Date': trade_date,
                'Entry_Price': entry_price,
                'Exit_Date': exit_date,
                'Exit_Price': exit_price,
                'PnL_Pct': pnl,
                'Portfolio': portfolio
            })
            
            logger.info(f"Closing position for {symbol} at end of backtest: {exit_price:.2f}, P&L: {pnl:.2f}%")
        
        # Create a DataFrame with all trades
        trades_df = pd.DataFrame(trades)
        
        # Save trades to CSV
        if not trades_df.empty:
            trades_file = self.results_dir / f"{symbol}_trades.csv"
            trades_df.to_csv(trades_file)
            logger.info(f"Saved trade results for {symbol} to {trades_file}")
        
        # Calculate backtest metrics
        metrics = self.calculate_metrics(trades_df, data)
        
        return data, trades_df, metrics
    
    def process_trade(self, symbol, entry_date, entry_price, exit_date, exit_price, pnl_pct, portfolio_value):
        """Process and log a completed trade"""
        # Log trade to Google Sheets
        self.integrations.log_trade(
            symbol=symbol,
            action="SELL",  # This is when we exit the trade
            price=exit_price,
            quantity=int(portfolio_value / entry_price * 0.1),  # 10% allocation
            pnl=pnl_pct,
            portfolio_value=portfolio_value,
            strategy="Backtest"
        )
        
        # Send Telegram alert for significant trades
        self.integrations.telegram_helper.send_trade_signal(
            symbol=symbol,
            action="CLOSED",
            price=exit_price,
            quantity=int(portfolio_value / entry_price * 0.1),  # 10% allocation
            signal_strength=abs(pnl_pct) / 5.0,  # Normalize to 0-1 scale (assuming 5% is strong)
            strategy="Backtest"
        )
    
    def calculate_metrics(self, trades_df, data):
        """
        Calculate performance metrics for the backtest
        
        Args:
            trades_df (DataFrame): DataFrame with trade details
            data (DataFrame): Price data with signals
            
        Returns:
            dict: Performance metrics
        """
        if trades_df.empty:
            logger.warning("No trades executed during backtest")
            return {
                'Total_Trades': 0,
                'Win_Rate': 0,
                'Average_Return': 0,
                'Total_Return': 0,
                'Sharpe_Ratio': 0,
                'Max_Drawdown': 0
            }
        
        # Calculate metrics
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['PnL_Pct'] > 0])
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        avg_return = trades_df['PnL_Pct'].mean() if not trades_df.empty else 0
        total_return = ((trades_df['Portfolio'].iloc[-1] / self.initial_capital) - 1) * 100 if not trades_df.empty else 0
        
        # Calculate Sharpe Ratio (annualized)
        if not trades_df.empty:
            returns = trades_df['PnL_Pct'].values
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) != 0 else 0
        else:
            sharpe_ratio = 0
        
        # Calculate Maximum Drawdown
        if not trades_df.empty:
            portfolio_series = trades_df['Portfolio']
            rolling_max = portfolio_series.cummax()
            drawdowns = (portfolio_series - rolling_max) / rolling_max * 100
            max_drawdown = drawdowns.min() if not drawdowns.empty else 0
        else:
            max_drawdown = 0
        
        metrics = {
            'Total_Trades': total_trades,
            'Win_Rate': win_rate,
            'Average_Return': avg_return,
            'Total_Return': total_return,
            'Sharpe_Ratio': sharpe_ratio,
            'Max_Drawdown': max_drawdown
        }
        
        logger.info(f"Backtest Metrics: {metrics}")
        return metrics
    
    def plot_results(self, data, trades_df, symbol):
        """
        Plot backtest results including price, signals, and portfolio value
        
        Args:
            data (DataFrame): Price data with signals
            trades_df (DataFrame): DataFrame with trade details
            symbol (str): Stock symbol
        """
        if trades_df.empty:
            logger.warning(f"No trades to plot for {symbol}")
            return
        
        plt.figure(figsize=(14, 10))
        
        # Plot 1: Price with Buy/Sell signals
        plt.subplot(2, 1, 1)
        plt.title(f"{symbol} - Price Chart with Signals")
        plt.plot(data.index, data['Close'], label='Close Price', color='blue', alpha=0.5)
        
        # Plot MA lines
        plt.plot(data.index, data['MA_Short'], label=f"{Config.MA_SHORT_PERIOD}-DMA", color='orange', alpha=0.7)
        plt.plot(data.index, data['MA_Long'], label=f"{Config.MA_LONG_PERIOD}-DMA", color='purple', alpha=0.7)
        
        # Plot Buy signals
        buy_signals = data[data['Buy_Signal'] == 1]
        plt.scatter(buy_signals.index, buy_signals['Close'], color='green', label='Buy Signal', marker='^', s=100)
        
        # Plot Sell signals
        sell_signals = data[data['Sell_Signal'] == 1]
        plt.scatter(sell_signals.index, sell_signals['Close'], color='red', label='Sell Signal', marker='v', s=100)
        
        plt.legend()
        plt.grid(True)
        
        # Plot 2: Portfolio Value
        plt.subplot(2, 1, 2)
        plt.title(f"{symbol} - Portfolio Value")
        plt.plot(trades_df['Exit_Date'], trades_df['Portfolio'], marker='o', label='Portfolio Value')
        plt.axhline(y=self.initial_capital, color='r', linestyle='--', label='Initial Capital')
        plt.legend()
        plt.grid(True)
        
        # Save the plot
        plot_file = self.results_dir / f"{symbol}_backtest_plot.png"
        plt.tight_layout()
        plt.savefig(plot_file)
        plt.close()
        
        logger.info(f"Saved backtest plot for {symbol} to {plot_file}")