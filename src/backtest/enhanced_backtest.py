import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import sys
from pathlib import Path
import traceback

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from src.utils.logger import setup_logger
from src.utils.dataframe_utils import standardize_dataframe

# Initialize logger
logger = setup_logger('enhanced_backtest')

class EnhancedBacktest:
    """Enhanced backtesting system with risk management"""
    
    def __init__(self, initial_capital=100000, position_size=0.2, stop_loss=0.02, 
                 take_profit=0.05, max_drawdown=0.2, results_dir=None):
        """
        Initialize backtest with risk management parameters
        
        Args:
            initial_capital: Starting capital
            position_size: Maximum position size as fraction of capital (0-1)
            stop_loss: Stop loss as fraction of position (0-1)
            take_profit: Take profit as fraction of position (0-1)
            max_drawdown: Maximum allowed drawdown (0-1)
            results_dir: Directory to save results
        """
        self.initial_capital = initial_capital
        self.position_size = position_size
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.max_drawdown = max_drawdown
        
        if results_dir is None:
            results_dir = Path(__file__).parent.parent.parent / "data" / "results"
            
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True, parents=True)
        
        logger.info(f"Initialized enhanced backtest with: capital={initial_capital}, "
                   f"position_size={position_size}, stop_loss={stop_loss}, "
                   f"take_profit={take_profit}, max_drawdown={max_drawdown}")
        
    def apply_risk_management(self, data):
        """
        Apply risk management rules to the trading signals
        
        Args:
            data: DataFrame with buy/sell signals
            
        Returns:
            DataFrame: Data with risk-managed signals
        """
        try:
            # Standardize DataFrame format
            df = standardize_dataframe(data).copy()
            
            # Check if required columns exist
            required_cols = ['Open', 'High', 'Low', 'Close', 'Buy_Signal', 'Sell_Signal']
            missing = [col for col in required_cols if col not in df.columns]
            
            if missing:
                logger.error(f"Missing required columns: {missing}")
                return data
                
            # Add risk management columns
            df['Stop_Loss'] = 0
            df['Take_Profit'] = 0
            df['Position_Size'] = self.position_size
            
            # Apply stop loss and take profit
            in_position = False
            entry_price = 0
            
            for i in range(1, len(df)):
                if not in_position and df.iloc[i-1]['Buy_Signal'] == 1:
                    # Enter position
                    in_position = True
                    entry_price = df.iloc[i]['Open']
                    
                elif in_position:
                    current_close = df.iloc[i]['Close']
                    current_low = df.iloc[i]['Low']
                    current_high = df.iloc[i]['High']
                    
                    # Check if stop loss was hit during the day
                    if current_low <= entry_price * (1 - self.stop_loss):
                        df.loc[df.index[i], 'Stop_Loss'] = 1
                        df.loc[df.index[i], 'Sell_Signal'] = 1
                        in_position = False
                        logger.info(f"Stop loss triggered at {df.index[i]}: Entry: {entry_price}, Low: {current_low}")
                        
                    # Check if take profit was hit during the day    
                    elif current_high >= entry_price * (1 + self.take_profit):
                        df.loc[df.index[i], 'Take_Profit'] = 1
                        df.loc[df.index[i], 'Sell_Signal'] = 1
                        in_position = False
                        logger.info(f"Take profit triggered at {df.index[i]}: Entry: {entry_price}, High: {current_high}")
                        
                    # Normal exit signal    
                    elif df.iloc[i]['Sell_Signal'] == 1:
                        in_position = False
                
                # Check if we already have a sell signal from original strategy
                if df.iloc[i]['Sell_Signal'] == 1:
                    in_position = False
                    
            logger.info(f"Risk management applied. Stop loss: {df['Stop_Loss'].sum()}, Take profit: {df['Take_Profit'].sum()}")
                    
            return df
            
        except Exception as e:
            logger.error(f"Error applying risk management: {str(e)}")
            logger.error(traceback.format_exc())
            return data
    
    def run_backtest(self, data, symbol):
        """
        Run backtest with risk management
        
        Args:
            data: DataFrame with price data and signals
            symbol: Stock symbol
            
        Returns:
            tuple: Backtest data, trades, metrics
        """
        try:
            # Apply risk management
            df = self.apply_risk_management(data)
            
            # Standardize DataFrame
            df = standardize_dataframe(df).copy()
            
            # Initialize portfolio metrics
            df['Position'] = 0
            df['Cash'] = self.initial_capital
            df['Holdings'] = 0
            df['Portfolio'] = self.initial_capital
            
            # Track trades
            trades = []
            current_position = 0
            entry_date = None
            entry_price = 0
            
            for i in range(1, len(df)):
                # Update today's position and cash based on yesterday
                df.loc[df.index[i], 'Position'] = current_position
                df.loc[df.index[i], 'Cash'] = df.iloc[i-1]['Cash']
                
                # Check for buy signal
                if df.iloc[i-1]['Buy_Signal'] == 1 and current_position == 0:
                    # Calculate position size
                    available_cash = df.iloc[i-1]['Cash']
                    max_position = int(available_cash * self.position_size / df.iloc[i]['Open'])
                    
                    if max_position > 0:
                        # Enter position
                        current_position = max_position
                        entry_price = df.iloc[i]['Open']
                        entry_date = df.index[i]
                        
                        # Update cash
                        new_cash = available_cash - (current_position * entry_price)
                        df.loc[df.index[i], 'Cash'] = new_cash
                        df.loc[df.index[i], 'Position'] = current_position
                        
                        logger.info(f"BUY signal for {symbol} at {entry_date}: {entry_price:.2f}, Shares: {current_position}")
                    else:
                        logger.warning(f"Insufficient cash to enter position at {df.index[i]}")
                
                # Check for sell signal
                elif (df.iloc[i-1]['Sell_Signal'] == 1 or df.iloc[i-1]['Stop_Loss'] == 1 or 
                      df.iloc[i-1]['Take_Profit'] == 1) and current_position > 0:
                    
                    # Calculate exit details
                    exit_price = df.iloc[i]['Open']
                    exit_date = df.index[i]
                    pnl_pct = (exit_price - entry_price) / entry_price * 100
                    pnl_amount = current_position * (exit_price - entry_price)
                    
                    # Update cash
                    new_cash = df.iloc[i-1]['Cash'] + (current_position * exit_price)
                    df.loc[df.index[i], 'Cash'] = new_cash
                    
                    # Log the trade
                    exit_reason = "Sell Signal"
                    if df.iloc[i-1]['Stop_Loss'] == 1:
                        exit_reason = "Stop Loss"
                    elif df.iloc[i-1]['Take_Profit'] == 1:
                        exit_reason = "Take Profit"
                        
                    trade = {
                        'Entry_Date': entry_date,
                        'Entry_Price': entry_price,
                        'Exit_Date': exit_date,
                        'Exit_Price': exit_price,
                        'Shares': current_position,
                        'PnL_Pct': pnl_pct,
                        'PnL_Amount': pnl_amount,
                        'Exit_Reason': exit_reason
                    }
                    
                    trades.append(trade)
                    logger.info(f"{exit_reason} for {symbol} at {exit_date}: {exit_price:.2f}, P&L: {pnl_pct:.2f}%")
                    
                    # Reset position
                    current_position = 0
                    df.loc[df.index[i], 'Position'] = 0
                
                # Calculate holdings value and total portfolio
                df.loc[df.index[i], 'Holdings'] = current_position * df.iloc[i]['Close']
                df.loc[df.index[i], 'Portfolio'] = df.iloc[i]['Cash'] + df.iloc[i]['Holdings']
            
            # Convert trades to DataFrame
            trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
            
            # Calculate metrics
            metrics = self._calculate_metrics(df, trades_df)
            
            # Save results
            self._save_results(df, trades_df, metrics, symbol)
            
            # Plot results
            self._plot_backtest(df, trades_df, symbol)
            
            return df, trades_df, metrics
            
        except Exception as e:
            logger.error(f"Error in backtest: {str(e)}")
            logger.error(traceback.format_exc())
            return data, pd.DataFrame(), {}
            
    def _calculate_metrics(self, df, trades_df):
        """Calculate backtest performance metrics"""
        metrics = {}
        
        try:
            # Basic trade metrics
            metrics['Total_Trades'] = len(trades_df)
            
            if len(trades_df) > 0:
                # Win rate
                winning_trades = trades_df[trades_df['PnL_Pct'] > 0]
                metrics['Win_Rate'] = len(winning_trades) / len(trades_df) * 100
                
                # Returns
                metrics['Average_Return'] = trades_df['PnL_Pct'].mean()
                metrics['Total_Return'] = ((df['Portfolio'].iloc[-1] / self.initial_capital) - 1) * 100
                
                # Drawdown calculation
                portfolio_value = df['Portfolio']
                running_max = portfolio_value.cummax()
                drawdown = (portfolio_value - running_max) / running_max * 100
                metrics['Max_Drawdown'] = drawdown.min()
                
                # Annualized metrics
                days = (df.index[-1] - df.index[0]).days
                if days > 0:
                    years = days / 365
                    metrics['Annualized_Return'] = ((1 + metrics['Total_Return']/100) ** (1/years) - 1) * 100
                    
                    # Sharpe ratio (assuming 0% risk-free rate for simplicity)
                    daily_returns = df['Portfolio'].pct_change().dropna()
                    annualized_volatility = daily_returns.std() * np.sqrt(252)
                    metrics['Sharpe_Ratio'] = (metrics['Annualized_Return']/100) / annualized_volatility if annualized_volatility > 0 else 0
                    
                    # Sortino ratio (downside risk only)
                    downside_returns = daily_returns[daily_returns < 0]
                    downside_volatility = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0.0001
                    metrics['Sortino_Ratio'] = (metrics['Annualized_Return']/100) / downside_volatility if downside_volatility > 0 else 0
                
                # Analysis by exit reason
                exit_reasons = trades_df['Exit_Reason'].value_counts()
                for reason, count in exit_reasons.items():
                    metrics[f'Exits_{reason.replace(" ", "_")}'] = count
                    
                logger.info(f"Backtest Metrics: {metrics}")
            else:
                logger.warning("No trades to calculate metrics")
                metrics = {
                    'Total_Trades': 0,
                    'Win_Rate': 0,
                    'Average_Return': 0,
                    'Total_Return': 0,
                    'Max_Drawdown': 0,
                    'Sharpe_Ratio': 0
                }
                
        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            metrics = {
                'Total_Trades': len(trades_df),
                'Win_Rate': 0,
                'Average_Return': 0,
                'Total_Return': 0,
                'Max_Drawdown': 0,
                'Sharpe_Ratio': 0,
                'Error': str(e)
            }
            
        return metrics
        
    def _save_results(self, df, trades_df, metrics, symbol):
        """Save backtest results to disk"""
        try:
            # Save trades
            if not trades_df.empty:
                trades_path = self.results_dir / f"{symbol}_trades.csv"
                trades_df.to_csv(trades_path)
                logger.info(f"Saved trade results for {symbol} to {trades_path}")
                
            # Save performance metrics
            metrics_path = self.results_dir / f"{symbol}_metrics.csv"
            pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
            logger.info(f"Saved metrics for {symbol} to {metrics_path}")
                
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
            
    def _plot_backtest(self, df, trades_df, symbol):
        """Create and save backtest visualization plots"""
        try:
            # Main backtest plot
            fig, axes = plt.subplots(3, 1, figsize=(12, 14), gridspec_kw={'height_ratios': [3, 1, 1]})
            
            # Price chart with buy/sell annotations
            ax1 = axes[0]
            ax1.plot(df.index, df['Close'], label='Close Price', color='blue', linewidth=1)
            
            # Add buy/sell markers
            for _, trade in trades_df.iterrows():
                # Buy marker
                ax1.scatter(trade['Entry_Date'], trade['Entry_Price'], color='green', s=100, marker='^', zorder=2)
                
                # Sell marker
                marker_color = 'red'
                if trade['Exit_Reason'] == 'Stop Loss':
                    marker_color = 'orange'
                elif trade['Exit_Reason'] == 'Take Profit':
                    marker_color = 'purple'
                    
                ax1.scatter(trade['Exit_Date'], trade['Exit_Price'], color=marker_color, s=100, marker='v', zorder=2)
            
            # Format
            ax1.set_title(f'{symbol} - Backtest Results', fontsize=15)
            ax1.set_ylabel('Price', fontsize=12)
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # Portfolio equity curve
            ax2 = axes[1]
            ax2.plot(df.index, df['Portfolio'], label='Portfolio Value', color='green', linewidth=1.5)
            ax2.axhline(y=self.initial_capital, color='red', linestyle='--', alpha=0.5)
            ax2.set_ylabel('Portfolio Value', fontsize=12)
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            
            # Drawdown chart
            ax3 = axes[2]
            portfolio_value = df['Portfolio']
            running_max = portfolio_value.cummax()
            drawdown = (portfolio_value - running_max) / running_max * 100
            ax3.fill_between(df.index, drawdown, 0, color='red', alpha=0.3, label='Drawdown')
            ax3.set_ylabel('Drawdown (%)', fontsize=12)
            ax3.set_xlabel('Date', fontsize=12)
            ax3.grid(True, alpha=0.3)
            ax3.legend()
            
            # Format dates on x-axis
            for ax in axes:
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
                
            plt.tight_layout()
            
            # Save plot
            plot_path = self.results_dir / f"{symbol}_backtest_plot.png"
            plt.savefig(plot_path)
            plt.close()
            
            logger.info(f"Saved backtest plot for {symbol} to {plot_path}")
            
            # Monte Carlo simulation if we have trades
            if len(trades_df) > 5:
                self.monte_carlo_simulation(trades_df, symbol)
                
        except Exception as e:
            logger.error(f"Error plotting backtest: {str(e)}")
            logger.error(traceback.format_exc())
            
    def monte_carlo_simulation(self, trades_df, symbol, n_simulations=1000):
        """
        Perform Monte Carlo simulation on trade results
        
        Args:
            trades_df: DataFrame of trades
            symbol: Stock symbol
            n_simulations: Number of simulations
        
        Returns:
            dict: Simulation results
        """
        if trades_df.empty:
            logger.warning("Cannot run Monte Carlo simulation with empty trades")
            return None
            
        try:
            # Extract returns
            returns = trades_df['PnL_Pct'].values / 100  # Convert percent to decimal
            
            if len(returns) < 5:
                logger.warning(f"Too few trades ({len(returns)}) for meaningful Monte Carlo simulation")
                return None
            
            # Run simulations
            final_values = []
            paths = []
            
            for _ in range(n_simulations):
                # Shuffle returns
                np.random.shuffle(returns)
                
                # Calculate path
                path = [1]
                for r in returns:
                    path.append(path[-1] * (1 + r))
                
                # Store results    
                final_values.append(path[-1])
                paths.append(path)
                
            # Calculate statistics
            mean_final = np.mean(final_values)
            median_final = np.median(final_values)
            percentile_5 = np.percentile(final_values, 5)
            percentile_95 = np.percentile(final_values, 95)
            
            # Plot results
            plt.figure(figsize=(10, 6))
            
            # Plot some sample paths
            for i in range(min(20, n_simulations)):
                plt.plot(paths[i], color='blue', alpha=0.1)
                
            # Plot average path
            avg_path = np.mean(np.array(paths), axis=0)
            plt.plot(avg_path, color='red', linewidth=2, label='Average Path')
            
            # Plot 5th and 95th percentile paths
            plt.axhline(y=percentile_5, color='orange', linestyle='--', label='5th Percentile')
            plt.axhline(y=percentile_95, color='green', linestyle='--', label='95th Percentile')
            plt.axhline(y=1, color='black', linestyle='-', label='Initial Value')
            
            plt.title(f'{symbol} - Monte Carlo Simulation ({n_simulations} runs)')
            plt.xlabel('Trade Number')
            plt.ylabel('Portfolio Value (multiple of initial)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save the plot
            plot_path = self.results_dir / f"{symbol}_monte_carlo.png"
            plt.savefig(plot_path)
            plt.close()
            
            logger.info(f"Monte Carlo simulation for {symbol} saved to {plot_path}")
            
            # Save simulation results
            mc_results = {
                'mean_final': mean_final,
                'median_final': median_final,
                'percentile_5': percentile_5,
                'percentile_95': percentile_95,
                'downside_risk': 1 - percentile_5,
                'upside_potential': percentile_95 - 1
            }
            
            results_path = self.results_dir / f"{symbol}_monte_carlo_results.csv"
            pd.DataFrame([mc_results]).to_csv(results_path, index=False)
            
            return mc_results
            
        except Exception as e:
            logger.error(f"Error in Monte Carlo simulation: {str(e)}")
            logger.error(traceback.format_exc())
            return None