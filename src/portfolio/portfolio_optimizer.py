import numpy as np
import pandas as pd
import logging
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Initialize logger
logger = logging.getLogger('portfolio_optimizer')

class PortfolioOptimizer:
    """
    Portfolio optimization using Modern Portfolio Theory (MPT)
    """
    
    def __init__(self):
        """Initialize the portfolio optimizer"""
        self.risk_profiles = {
            'conservative': {'target_return': 0.08, 'risk_tolerance': 'low'},
            'moderate': {'target_return': 0.12, 'risk_tolerance': 'medium'},
            'aggressive': {'target_return': 0.18, 'risk_tolerance': 'high'}
        }
    
    def calculate_returns(self, data):
        """
        Calculate daily returns from price data
        
        Args:
            data: Dictionary of DataFrames with price data
            
        Returns:
            DataFrame of returns
        """
        returns_dict = {}
        
        for symbol, df in data.items():
            # Make sure we have a 'Close' column
            if 'Close' in df.columns:
                returns_dict[symbol] = df['Close'].pct_change().dropna()
            elif ('Close', '') in df.columns:  # Handle MultiIndex columns
                returns_dict[symbol] = df[('Close', '')].pct_change().dropna()
            else:
                logger.warning(f"No Close price found for {symbol}, skipping in returns calculation")
        
        # Create a DataFrame with aligned dates
        returns_df = pd.DataFrame(returns_dict)
        returns_df = returns_df.dropna()  # Remove rows with NaN
        
        return returns_df
    
    def calculate_expected_returns(self, returns_df):
        """
        Calculate expected returns (annualized)
        
        Args:
            returns_df: DataFrame of daily returns
            
        Returns:
            Series of expected returns
        """
        # Annualize returns (assuming 252 trading days)
        return returns_df.mean() * 252
    
    def calculate_covariance_matrix(self, returns_df):
        """
        Calculate covariance matrix (annualized)
        
        Args:
            returns_df: DataFrame of daily returns
            
        Returns:
            DataFrame of covariance matrix
        """
        # Annualize covariance (assuming 252 trading days)
        return returns_df.cov() * 252
    
    def portfolio_return(self, weights, expected_returns):
        """
        Calculate portfolio expected return
        
        Args:
            weights: Array of weights
            expected_returns: Series of expected returns
            
        Returns:
            Expected portfolio return
        """
        return np.sum(weights * expected_returns)
    
    def portfolio_volatility(self, weights, cov_matrix):
        """
        Calculate portfolio volatility (standard deviation)
        
        Args:
            weights: Array of weights
            cov_matrix: Covariance matrix
            
        Returns:
            Portfolio volatility
        """
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    
    def negative_sharpe_ratio(self, weights, expected_returns, cov_matrix, risk_free_rate=0.02):
        """
        Calculate negative Sharpe ratio (to minimize)
        
        Args:
            weights: Array of weights
            expected_returns: Series of expected returns
            cov_matrix: Covariance matrix
            risk_free_rate: Risk-free rate (default: 0.02 or 2%)
            
        Returns:
            Negative Sharpe ratio
        """
        p_ret = self.portfolio_return(weights, expected_returns)
        p_vol = self.portfolio_volatility(weights, cov_matrix)
        sharpe = (p_ret - risk_free_rate) / p_vol
        
        return -sharpe  # Negative because we want to maximize Sharpe, but minimize function
    
    def optimize_portfolio(self, expected_returns, cov_matrix, risk_free_rate=0.02, 
                         target_return=None, target_volatility=None):
        """
        Find the optimal portfolio weights
        
        Args:
            expected_returns: Series of expected returns
            cov_matrix: Covariance matrix
            risk_free_rate: Risk-free rate
            target_return: Target portfolio return (optional)
            target_volatility: Target portfolio volatility (optional)
            
        Returns:
            Dictionary with optimization results
        """
        num_assets = len(expected_returns)
        
        # Initial guess (equal weight)
        init_weights = np.ones(num_assets) / num_assets
        
        # Constraints
        bounds = tuple((0, 1) for _ in range(num_assets))  # Each weight between 0-100%
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]  # Weights sum to 1
        
        if target_return is not None:
            constraints.append({
                'type': 'eq',
                'fun': lambda x: self.portfolio_return(x, expected_returns) - target_return
            })
        
        if target_volatility is not None:
            constraints.append({
                'type': 'eq',
                'fun': lambda x: self.portfolio_volatility(x, cov_matrix) - target_volatility
            })
        
        # Optimize!
        result = minimize(
            self.negative_sharpe_ratio,
            init_weights,
            args=(expected_returns, cov_matrix, risk_free_rate),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        # Get optimal weights
        weights = result['x']
        
        # Calculate portfolio metrics
        expected_return = self.portfolio_return(weights, expected_returns)
        volatility = self.portfolio_volatility(weights, cov_matrix)
        sharpe_ratio = (expected_return - risk_free_rate) / volatility
        
        # Create dictionary with results
        weights_dict = {asset: weight for asset, weight in zip(expected_returns.index, weights)}
        
        return {
            'weights': weights_dict,
            'expected_return': expected_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio
        }
    
    def generate_efficient_frontier(self, expected_returns, cov_matrix, risk_free_rate=0.02, points=100):
        """
        Generate the efficient frontier
        
        Args:
            expected_returns: Series of expected returns
            cov_matrix: Covariance matrix
            risk_free_rate: Risk-free rate
            points: Number of points on the frontier
            
        Returns:
            Dictionary with efficient frontier data
        """
        # Find min and max returns
        min_ret = expected_returns.min()
        max_ret = expected_returns.max()
        
        # Create range of target returns
        target_returns = np.linspace(min_ret, max_ret, points)
        
        # Calculate efficient frontier
        efficient_portfolios = []
        
        for target_return in target_returns:
            portfolio = self.optimize_portfolio(
                expected_returns, 
                cov_matrix, 
                risk_free_rate=risk_free_rate,
                target_return=target_return
            )
            efficient_portfolios.append(portfolio)
        
        # Extract volatility and return
        volatilities = [p['volatility'] for p in efficient_portfolios]
        returns = [p['expected_return'] for p in efficient_portfolios]
        
        return {
            'volatilities': volatilities,
            'returns': returns,
            'portfolios': efficient_portfolios
        }
    
    def generate_portfolio_recommendations(self, returns_df):
        """
        Generate portfolio recommendations for different risk profiles
        
        Args:
            returns_df: DataFrame of daily returns
            
        Returns:
            List of dictionaries with portfolio recommendations
        """
        expected_returns = self.calculate_expected_returns(returns_df)
        cov_matrix = self.calculate_covariance_matrix(returns_df)
        
        # Generate recommendations for each risk profile
        recommendations = []
        
        for profile, params in self.risk_profiles.items():
            target_return = params['target_return']
            
            # Find optimal portfolio with target return
            try:
                portfolio = self.optimize_portfolio(
                    expected_returns, 
                    cov_matrix,
                    target_return=target_return
                )
                
                # Create recommendation
                recommendation = {
                    'risk_profile': profile,
                    'expected_return': portfolio['expected_return'] * 100,  # Convert to percentage
                    'volatility': portfolio['volatility'] * 100,  # Convert to percentage
                    'sharpe_ratio': portfolio['sharpe_ratio'],
                    'weights': {asset: weight * 100 for asset, weight in portfolio['weights'].items()}  # Convert to percentage
                }
                
                recommendations.append(recommendation)
                
            except Exception as e:
                logger.error(f"Could not generate recommendation for {profile}: {str(e)}")
        
        return recommendations
    
    def plot_efficient_frontier(self, returns_df, risk_free_rate=0.02, save_path=None):
        """
        Plot the efficient frontier
        
        Args:
            returns_df: DataFrame of daily returns
            risk_free_rate: Risk-free rate
            save_path: Path to save the plot (optional)
            
        Returns:
            None
        """
        expected_returns = self.calculate_expected_returns(returns_df)
        cov_matrix = self.calculate_covariance_matrix(returns_df)
        
        # Generate efficient frontier
        frontier = self.generate_efficient_frontier(expected_returns, cov_matrix, risk_free_rate)
        
        # Find optimal portfolio (max Sharpe ratio)
        optimal_portfolio = self.optimize_portfolio(expected_returns, cov_matrix, risk_free_rate)
        
        # Create plot
        plt.figure(figsize=(12, 8))
        
        # Plot efficient frontier
        plt.plot(frontier['volatilities'], frontier['returns'], 'b-', linewidth=3)
        
        # Plot individual assets
        for i, asset in enumerate(returns_df.columns):
            asset_volatility = np.sqrt(cov_matrix.iloc[i, i])
            asset_return = expected_returns[i]
            plt.scatter(asset_volatility, asset_return, marker='o', s=100, label=asset)
        
        # Plot optimal portfolio
        plt.scatter(
            optimal_portfolio['volatility'],
            optimal_portfolio['expected_return'],
            marker='*',
            color='r',
            s=300,
            label='Optimal Portfolio'
        )
        
        # Capital Market Line
        max_sharpe_ratio = optimal_portfolio['sharpe_ratio']
        max_sharpe_volatility = optimal_portfolio['volatility']
        max_sharpe_return = optimal_portfolio['expected_return']
        
        # Plot CML
        volatility_range = np.linspace(0, max(frontier['volatilities']) * 1.2, 100)
        cml_returns = risk_free_rate + max_sharpe_ratio * volatility_range
        plt.plot(volatility_range, cml_returns, 'r-', label='Capital Market Line')
        
        # Plot settings
        plt.grid(True)
        plt.xlabel('Volatility (Standard Deviation)', fontsize=12)
        plt.ylabel('Expected Return', fontsize=12)
        plt.title('Efficient Frontier', fontsize=16)
        plt.legend()
        
        # Save plot if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved efficient frontier plot to {save_path}")
        
        plt.close()