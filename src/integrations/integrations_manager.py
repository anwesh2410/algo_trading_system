from datetime import datetime
import pandas as pd
from src.integrations.google_sheets_helper import GoogleSheetsHelper
from src.integrations.telegram_helper import TelegramHelper
from src.utils.logger import setup_logger

# Set up logger
logger = setup_logger('integrations_manager')

class IntegrationsManager:
    """
    Combined manager for all external integrations
    """
    
    _instance = None
    
    @classmethod
    def get_instance(cls):
        """Get singleton instance"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        """Initialize all integrations"""
        # Get integration helper instances
        self.sheets_helper = GoogleSheetsHelper.get_instance()
        self.telegram_helper = TelegramHelper.get_instance()
        logger.info("Initialized integrations manager")
        
    def log_trade(self, symbol, action, price, quantity, pnl=None, portfolio_value=None, 
                  strategy=None, signal_strength=None, notes=None, send_telegram=True):
        """
        Log a trade to all enabled integrations
        
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
            send_telegram: Whether to send Telegram alert (default: True)
        """
        logger.info(f"Logging {action} trade for {symbol} at ${price}")
        
        # Log to Google Sheets
        self.sheets_helper.log_trade(
            symbol=symbol,
            action=action,
            price=price,
            quantity=quantity,
            pnl=pnl,
            portfolio_value=portfolio_value,
            strategy=strategy,
            signal_strength=signal_strength,
            notes=notes
        )
        
        # Send Telegram alert if enabled
        if send_telegram:
            self.telegram_helper.send_trade_signal(
                symbol=symbol,
                action=action,
                price=price,
                quantity=quantity,
                signal_strength=signal_strength,
                strategy=strategy
            )
    
    def log_ml_prediction(self, symbol, prediction, confidence=None, 
                        supporting_data=None, chart_path=None, 
                        actual_result=None, accuracy=None, send_telegram=True):
        """
        Log ML prediction to all enabled integrations
        
        Args:
            symbol: Stock ticker symbol
            prediction: UP or DOWN (or 1/0)
            confidence: Prediction confidence (optional)
            supporting_data: Dict with supporting data (optional)
            chart_path: Path to chart image (optional)
            actual_result: Actual outcome once known (optional)
            accuracy: Prediction accuracy (optional)
            send_telegram: Whether to send Telegram alert (default: True)
        """
        # Standardize prediction format
        pred_str = prediction
        if isinstance(prediction, (int, float)):
            pred_str = "UP" if prediction == 1 else "DOWN"
            
        logger.info(f"Logging {pred_str} prediction for {symbol}" + 
                   (f" with {confidence:.2f} confidence" if confidence else ""))
        
        # Log to Google Sheets
        self.sheets_helper.log_ml_prediction(
            symbol=symbol,
            prediction=pred_str,
            confidence=confidence,
            actual_result=actual_result,
            accuracy=accuracy
        )
        
        # Send Telegram alert if enabled
        if send_telegram:
            self.telegram_helper.send_ml_prediction(
                symbol=symbol,
                prediction=pred_str,
                confidence=confidence,
                supporting_data=supporting_data,
                chart_path=chart_path
            )
    
    def update_performance(self, performance_data, send_telegram=True, detailed_metrics=None):
        """
        Update performance metrics in all enabled integrations
        
        Args:
            performance_data: Dict mapping symbols to their performance metrics
            send_telegram: Whether to send Telegram alerts for each symbol
            detailed_metrics: Additional metrics for Telegram alerts
        """
        logger.info(f"Updating performance data for {len(performance_data)} symbols")
        
        # Normalize performance data to ensure required keys exist
        normalized_data = {}
        for symbol, metrics in performance_data.items():
            normalized_metrics = metrics.copy()
            
            # Ensure required keys exist
            required_keys = ['Win_Rate', 'Total_Trades', 'Total_Return', 'Max_Drawdown']
            for key in required_keys:
                if key not in normalized_metrics:
                    # Try common alternative keys
                    alternatives = {
                        'Win_Rate': ['win_rate', 'winRate', 'win_ratio'],
                        'Total_Trades': ['total_trades', 'num_trades', 'trades'],
                        'Total_Return': ['total_return', 'return', 'returns'],
                        'Max_Drawdown': ['max_drawdown', 'drawdown', 'maximum_drawdown']
                    }
                    
                    # Check for alternatives
                    found = False
                    for alt in alternatives.get(key, []):
                        if alt in normalized_metrics:
                            normalized_metrics[key] = normalized_metrics[alt]
                            found = True
                            break
                    
                    # If still not found, set default value
                    if not found:
                        normalized_metrics[key] = 0
            
            normalized_data[symbol] = normalized_metrics
        
        # Update Google Sheets
        self.sheets_helper.update_performance(normalized_data)
        
        # Send Telegram alerts if enabled
        if send_telegram:
            for symbol, metrics in normalized_data.items():
                # Use detailed metrics if provided, otherwise use basic metrics
                other_metrics = None
                if detailed_metrics and symbol in detailed_metrics:
                    other_metrics = detailed_metrics[symbol]
                
                self.telegram_helper.send_performance_update(
                    symbol=symbol,
                    total_return=metrics['Total_Return'],
                    win_rate=metrics['Win_Rate'],
                    other_metrics=other_metrics
                )
    
    def update_portfolio(self, positions, portfolio_value=None, daily_change=None, 
                       send_telegram=True, update_dashboard=True):
        """
        Update portfolio information in all enabled integrations
        
        Args:
            positions: List of portfolio positions
            portfolio_value: Total portfolio value (optional)
            daily_change: Daily change percentage (optional)
            send_telegram: Whether to send Telegram alert
            update_dashboard: Whether to update the dashboard in Google Sheets
        """
        logger.info(f"Updating portfolio data with {len(positions)} positions")
        
        # Calculate portfolio value if not provided
        if portfolio_value is None and positions:
            portfolio_value = sum(pos["current_value"] for pos in positions)
        
        # Update Google Sheets portfolio tab
        self.sheets_helper.update_portfolio(positions)
        
        # Update Google Sheets dashboard
        if update_dashboard:
            self.sheets_helper.update_dashboard(
                portfolio_value=portfolio_value,
                open_positions=len(positions)
            )
        
        # Send Telegram alert if enabled
        if send_telegram and portfolio_value is not None and daily_change is not None:
            # Convert positions to format expected by Telegram helper
            telegram_positions = [
                {
                    "symbol": pos["symbol"],
                    "allocation": pos["allocation"],
                    "return": (pos["profit_loss"] / (pos["current_value"] - pos["profit_loss"])) * 100 
                        if pos["current_value"] > pos["profit_loss"] else 0
                }
                for pos in positions
            ]
            
            self.telegram_helper.send_portfolio_update(
                portfolio_value=portfolio_value,
                daily_change=daily_change,
                positions=telegram_positions
            )
    
    def send_daily_summary(self, portfolio_value, daily_change, top_performers=None,
                         trades_executed=None, signals_generated=None, update_sheets=True):
        """
        Send daily summary to all enabled integrations
        
        Args:
            portfolio_value: Current portfolio value
            daily_change: Daily change percentage
            top_performers: List of top performers [{"symbol": str, "return": float}]
            trades_executed: Number of trades executed today
            signals_generated: Number of signals generated today
            update_sheets: Whether to update Google Sheets dashboard
        """
        logger.info("Sending daily summary")
        
        # Send Telegram summary
        self.telegram_helper.send_daily_summary(
            portfolio_value=portfolio_value,
            daily_change=daily_change,
            top_performers=top_performers,
            trades_executed=trades_executed,
            signals_generated=signals_generated
        )
        
        # Update Google Sheets dashboard
        if update_sheets:
            # Find best and worst performers
            best_performer = {"symbol": None, "return": -float('inf')}
            worst_performer = {"symbol": None, "return": float('inf')}
            
            if top_performers:
                for performer in top_performers:
                    if performer["return"] > best_performer["return"]:
                        best_performer = performer
                    if performer["return"] < worst_performer["return"]:
                        worst_performer = performer
            
            # Update dashboard
            self.sheets_helper.update_dashboard(
                portfolio_value=portfolio_value,
                best_performer=best_performer if best_performer["symbol"] else None,
                worst_performer=worst_performer if worst_performer["symbol"] else None
            )
    
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
        logger.info("Updating risk metrics")
        
        # Update Google Sheets
        self.sheets_helper.update_risk_metrics(
            var_value=var_value,
            cvar_value=cvar_value,
            beta=beta,
            volatility=volatility,
            correlation_matrix=correlation_matrix
        )
    
    def send_system_alert(self, message, alert_type="info"):
        """
        Send system alert
        
        Args:
            message: Alert message
            alert_type: Type of alert - "info", "warning", or "error"
        """
        logger.info(f"Sending system {alert_type} alert: {message}")
        
        # Send via Telegram
        self.telegram_helper.send_system_alert(message, alert_type)