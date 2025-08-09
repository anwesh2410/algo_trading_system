import os
from datetime import datetime
from src.integrations.google_sheets import GoogleSheetsIntegration
from config.config_loader import Config
from src.utils.logger import setup_logger

# Set up logger
logger = setup_logger('google_sheets_helper')

class GoogleSheetsHelper:
    """
    Helper class to simplify Google Sheets integration
    """
    
    _instance = None
    
    @classmethod
    def get_instance(cls):
        """Get singleton instance"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        """Initialize Google Sheets integration"""
        credentials_file = Config.GOOGLE_SHEETS_CREDENTIALS_FILE
        spreadsheet_id = Config.GOOGLE_SHEETS_SPREADSHEET_ID
        
        if credentials_file and os.path.exists(credentials_file):
            if spreadsheet_id:
                self.gs = GoogleSheetsIntegration(credentials_file, spreadsheet_id)
                logger.info(f"Google Sheets integration initialized with spreadsheet ID: {spreadsheet_id}")
            else:
                logger.warning("Google Sheets spreadsheet ID not provided in config")
                self.gs = None
        else:
            logger.warning(f"Google Sheets credentials file not found: {credentials_file}")
            self.gs = None
    
    def log_trade(self, symbol, action, price, quantity, pnl=None, portfolio_value=None, 
                  strategy=None, signal_strength=None, notes=None):
        """
        Log a trade to Google Sheets
        
        Args:
            symbol: Stock ticker symbol
            action: BUY or SELL
            price: Trade price
            quantity: Number of shares
            pnl: Profit/loss (optional)
            portfolio_value: Total portfolio value (optional)
            strategy: Trading strategy used (optional)
            signal_strength: Signal strength (optional)
            notes: Additional notes (optional)
        """
        if not self.gs:
            logger.warning("Google Sheets integration not available")
            return False
            
        trade_data = {
            "symbol": symbol,
            "action": action,
            "price": price,
            "quantity": quantity,
            "pnl": pnl,
            "portfolio_value": portfolio_value,
            "strategy": strategy,
            "signal_strength": signal_strength,
            "notes": notes
        }
        
        return self.gs.log_trade(trade_data)
    
    def update_performance(self, performance_data):
        """
        Update performance metrics in Google Sheets
        
        Args:
            performance_data: Dictionary mapping symbols to their performance metrics
            {
                "AAPL": {
                    "Win_Rate": 65.5,
                    "Total_Trades": 20,
                    "Total_Return": 12.5,
                    "Average_Return": 0.625,
                    "Sharpe_Ratio": 1.2,
                    "Max_Drawdown": 5.2
                },
                "MSFT": {...}
            }
        """
        if not self.gs:
            logger.warning("Google Sheets integration not available")
            return False
            
        return self.gs.update_performance(performance_data)
    
    def log_ml_prediction(self, symbol, prediction, confidence=None, 
                         actual_result=None, accuracy=None):
        """
        Log ML prediction to Google Sheets
        
        Args:
            symbol: Stock ticker symbol
            prediction: UP or DOWN (or 1/0)
            confidence: Prediction confidence (optional)
            actual_result: Actual outcome once known (optional)
            accuracy: Prediction accuracy (optional)
        """
        if not self.gs:
            logger.warning("Google Sheets integration not available")
            return False
            
        # Convert numerical prediction to UP/DOWN if needed
        if isinstance(prediction, (int, float)):
            prediction = "UP" if prediction == 1 else "DOWN"
            
        prediction_data = {
            "symbol": symbol,
            "prediction": prediction,
            "confidence": confidence,
            "actual_result": actual_result,
            "accuracy": accuracy
        }
        
        return self.gs.log_ml_prediction(prediction_data)
    
    def update_portfolio(self, positions):
        """
        Update portfolio positions and allocations
        
        Args:
            positions: List of portfolio positions
            [
                {
                    "symbol": "AAPL",
                    "allocation": 25.5,
                    "position_size": 100,
                    "current_value": 16500.25,
                    "profit_loss": 500.75
                },
                {...}
            ]
        """
        if not self.gs:
            logger.warning("Google Sheets integration not available")
            return False
            
        return self.gs.update_portfolio(positions)
    
    def update_risk_metrics(self, var_value, cvar_value, beta, volatility, correlation_matrix):
        """
        Update risk metrics in Google Sheets
        
        Args:
            var_value: Value at Risk (95%)
            cvar_value: Conditional Value at Risk
            beta: Portfolio beta
            volatility: Portfolio volatility
            correlation_matrix: Correlation matrix as pandas DataFrame
        """
        if not self.gs:
            logger.warning("Google Sheets integration not available")
            return False
            
        risk_data = {
            "var": var_value,
            "cvar": cvar_value,
            "beta": beta,
            "volatility": volatility,
            "correlation_matrix": correlation_matrix
        }
        
        return self.gs.update_risk_metrics(risk_data)
    
    def update_dashboard(self, portfolio_value=None, best_performer=None, 
                        worst_performer=None, open_positions=None, today_signals=None):
        """
        Update the dashboard with key metrics
        
        Args:
            portfolio_value: Current portfolio value (optional)
            best_performer: Dict with best performer info (optional) 
                           {"symbol": "AAPL", "return": 12.5}
            worst_performer: Dict with worst performer info (optional)
                            {"symbol": "MSFT", "return": -5.2}
            open_positions: Number of open positions (optional)
            today_signals: List of today's signals (optional)
                          [{"symbol": "AAPL", "action": "BUY"}, ...]
        """
        if not self.gs:
            logger.warning("Google Sheets integration not available")
            return False
            
        dashboard_data = {
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        if portfolio_value is not None:
            dashboard_data["portfolio_value"] = portfolio_value
            
        if best_performer is not None:
            dashboard_data["best_performer"] = best_performer
            
        if worst_performer is not None:
            dashboard_data["worst_performer"] = worst_performer
            
        if open_positions is not None:
            dashboard_data["open_positions"] = open_positions
            
        if today_signals is not None:
            dashboard_data["today_signals"] = today_signals
            
        return self.gs.update_dashboard(dashboard_data)