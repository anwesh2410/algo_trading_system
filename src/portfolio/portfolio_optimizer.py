import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from src.utils.logger import setup_logger

# Initialize logger
logger = setup_logger('portfolio_optimizer')

class PortfolioOptimizer:
    """Portfolio optimization using Mean-Variance and other methods"""
    
    def __init__(self, output_dir=None):
        """Initialize the portfolio optimizer"""
        if output_dir is None:
            output_dir = Path(__file__).parent.parent.parent / "data" / "portfolio"
            
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
    def calculate_returns(self, price_data):
        """Calculate daily returns from price data"""
        returns = {}
        
        for symbol, df in price_data.items():
            # Ensure we have a DataFrame with Close prices
            if isinstance(df, pd.DataFrame) and 'Close' in df.columns:
                returns[symbol] = df['Close'].pct_change().dropna()
            elif isinstance(df.columns, pd.MultiIndex):
                # Find the Close column in MultiIndex
                for col in df.columns:
                    if col[0] == 'Close':
                        returns[symbol] = df[col].pct_change().dropna()
                        break
        
        # Combine all returns into a single DataFrame
        if returns:
            returns_df = pd.DataFrame(returns)
            returns_df = returns_df.dropna()  # Remove rows with NaN
            logger.info(f"Calculated returns for {len(returns)} symbols over {len(returns_df)} days")
            return returns_df
        else:
            logger.error("Could not calculate returns from provided data")
            return pd.DataFrame()
            
    def optimize_portfolio(self, returns, target_return=0.10, risk_free_rate=0.02, method='efficient_frontier'):
        """
        Optimize portfolio weights
        
        Args:
            returns: DataFrame of stock returns
            target_return: Target annual return
            risk_free_rate: Risk-free rate
            method: Optimization method (efficient_frontier, max_sharpe, min_volatility)
            
        Returns:
            dict: Optimal weights for each stock
        """
        if returns.empty:
            logger.error("Cannot optimize empty returns")
            return {}
            
        # Calculate expected returns and covariance
        mu = returns.mean() * 252  # Annualized
        sigma = returns.cov() * 252  # Annualized
        
        # Number of assets
        n = len(returns.columns)
        
        # Initial guess (equal weight)
        x0 = np.ones(n) / n
        
        if method == 'max_sharpe':
            # Maximize Sharpe ratio
            def neg_sharpe(weights):
                port_return = np.sum(weights * mu)
                port_volatility = np.sqrt(np.dot(weights.T, np.dot(sigma, weights)))
                sharpe = (port_return - risk_free_rate) / port_volatility
                return -sharpe  # Negative since we're minimizing
                
            constraints = (
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # Weights sum to 1
            )
            
            bounds = tuple((0, 1) for i in range(n))
            
            result = minimize(neg_sharpe, x0, method='SLSQP', bounds=bounds, constraints=constraints)
            
        elif method == 'min_volatility':
            # Minimize volatility
            def volatility(weights):
                return np.sqrt(np.dot(weights.T, np.dot(sigma, weights)))
                
            constraints = (
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # Weights sum to 1
            )
            
            bounds = tuple((0, 1) for i in range(n))
            
            result = minimize(volatility, x0, method='SLSQP', bounds=bounds, constraints=constraints)
            
        else:  # efficient_frontier
            # Target return with minimum variance
            constraints = (
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Weights sum to 1
                {'type': 'eq', 'fun': lambda x: np.sum(x * mu) - target_return}  # Target return
            )
            
            bounds = tuple((0, 1) for i in range(n))
            
            # Objective function (minimize variance)
            def objective(x):
                return np.dot(x.T, np.dot(sigma, x))
                
            try:
                result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
            except:
                logger.warning(f"Could not achieve target return {target_return}. Switching to max Sharpe ratio.")
                return self.optimize_portfolio(returns, target_return, risk_free_rate, method='max_sharpe')
        
        # Return optimal weights
        weights = {stock: weight for stock, weight in zip(returns.columns, result['x'])}
        
        # Calculate portfolio statistics
        port_return = np.sum(result['x'] * mu)
        port_volatility = np.sqrt(np.dot(result['x'].T, np.dot(sigma, result['x'])))
        sharpe = (port_return - risk_free_rate) / port_volatility
        
        logger.info(f"Optimized portfolio ({method}):")
        logger.info(f"Expected annual return: {port_return*100:.2f}%")
        logger.info(f"Annual volatility: {port_volatility*100:.2f}%")
        logger.info(f"Sharpe ratio: {sharpe:.2f}")
        
        # Log the weights
        for stock, weight in weights.items():
            logger.info(f"{stock}: {weight*100:.2f}%")
            
        return weights
        
    def plot_efficient_frontier(self, returns, risk_free_rate=0.02, n_portfolios=1000):
        """Plot the efficient frontier and optimal portfolios"""
        if returns.empty:
            logger.error("Cannot plot efficient frontier with empty returns")
            return
            
        # Calculate expected returns and covariance
        mu = returns.mean() * 252  # Annualized
        sigma = returns.cov() * 252  # Annualized
        
        # Number of assets
        n = len(returns.columns)
        
        # Generate random portfolios
        np.random.seed(42)
        weights = np.random.random((n_portfolios, n))
        weights = weights / np.sum(weights, axis=1)[:, np.newaxis]
        
        # Calculate portfolio metrics
        port_returns = np.dot(weights, mu)
        port_volatilities = []
        
        for i in range(n_portfolios):
            port_volatilities.append(np.sqrt(np.dot(weights[i].T, np.dot(sigma, weights[i]))))
            
        port_volatilities = np.array(port_volatilities)
        sharpe_ratios = (port_returns - risk_free_rate) / port_volatilities
        
        # Find the portfolio with maximum Sharpe ratio
        max_sharpe_idx = np.argmax(sharpe_ratios)
        max_sharpe_return = port_returns[max_sharpe_idx]
        max_sharpe_volatility = port_volatilities[max_sharpe_idx]
        
        # Find the portfolio with minimum volatility
        min_vol_idx = np.argmin(port_volatilities)
        min_vol_return = port_returns[min_vol_idx]
        min_vol_volatility = port_volatilities[min_vol_idx]
        
        # Plot the efficient frontier
        plt.figure(figsize=(10, 8))
        plt.scatter(port_volatilities, port_returns, c=sharpe_ratios, cmap='viridis', alpha=0.5)
        plt.colorbar(label='Sharpe Ratio')
        plt.scatter(max_sharpe_volatility, max_sharpe_return, marker='*', color='r', s=500, label='Maximum Sharpe')
        plt.scatter(min_vol_volatility, min_vol_return, marker='*', color='g', s=500, label='Minimum Volatility')
        
        # Add individual assets
        for i, symbol in enumerate(returns.columns):
            asset_vol = np.sqrt(sigma.iloc[i, i])
            asset_ret = mu[i]
            plt.scatter(asset_vol, asset_ret, marker='o', s=100, label=symbol)
            
        plt.title('Portfolio Optimization - Efficient Frontier', fontsize=15)
        plt.xlabel('Annual Volatility (%)', fontsize=12)
        plt.ylabel('Annual Return (%)', fontsize=12)
        plt.legend()
        plt.grid(alpha=0.3)
        
        # Save the plot
        output_path = self.output_dir / "efficient_frontier.png"
        plt.savefig(output_path)
        plt.close()
        
        logger.info(f"Efficient frontier plot saved to {output_path}")
        
        return {
            'max_sharpe': {
                'return': max_sharpe_return,
                'volatility': max_sharpe_volatility,
                'sharpe': sharpe_ratios[max_sharpe_idx],
                'weights': {symbol: weight for symbol, weight in zip(returns.columns, weights[max_sharpe_idx])}
            },
            'min_volatility': {
                'return': min_vol_return,
                'volatility': min_vol_volatility,
                'sharpe': sharpe_ratios[min_vol_idx],
                'weights': {symbol: weight for symbol, weight in zip(returns.columns, weights[min_vol_idx])}
            }
        }