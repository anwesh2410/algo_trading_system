import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd
from datetime import datetime
import os
import sys
import time
import traceback

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from src.utils.logger import setup_logger

# Initialize logger
logger = setup_logger('google_sheets')

class GoogleSheetsIntegration:
    """
    Enhanced integration with Google Sheets for comprehensive trade logging,
    performance metrics, and ML predictions
    """
    
    def __init__(self, credentials_file, spreadsheet_id):
        """
        Initialize Google Sheets connection
        
        Args:
            credentials_file: Path to Google API credentials JSON
            spreadsheet_id: ID of the Google Sheet (from URL)
        """
        self.credentials_file = credentials_file
        self.spreadsheet_id = spreadsheet_id
        self.client = None
        self.spreadsheet = None
        self.initialized = False
        self.worksheets = {}
        
        # Try to initialize connection
        self._initialize_connection()
        
    def _initialize_connection(self):
        """Initialize connection to Google Sheets with retry logic"""
        max_retries = 3
        retry_delay = 2  # seconds
        
        for attempt in range(max_retries):
            try:
                # Define the scope
                scope = [
                    'https://spreadsheets.google.com/feeds',
                    'https://www.googleapis.com/auth/drive'
                ]
                
                # Check if credentials file exists
                if not os.path.exists(self.credentials_file):
                    logger.error(f"Credentials file not found: {self.credentials_file}")
                    return False
                
                # Authenticate
                credentials = ServiceAccountCredentials.from_json_keyfile_name(self.credentials_file, scope)
                self.client = gspread.authorize(credentials)
                
                # Open spreadsheet by ID
                if self.spreadsheet_id:
                    try:
                        self.spreadsheet = self.client.open_by_key(self.spreadsheet_id)
                        logger.info(f"Successfully opened spreadsheet by ID: {self.spreadsheet_id}")
                        
                        # Set flag to true BEFORE setting up sheets
                        self.initialized = True
                        
                        # Setup sheets
                        self.setup_sheets()
                        return True
                    except Exception as e:
                        logger.error(f"Error opening spreadsheet by ID: {str(e)}")
                        logger.error(f"Make sure the service account has access to this spreadsheet")
                else:
                    logger.error("No spreadsheet ID provided")
                    return False
                
            except Exception as e:
                logger.error(f"Attempt {attempt+1}/{max_retries} failed: {str(e)}")
                logger.error(traceback.format_exc())
                
                if attempt < max_retries - 1:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
        
        logger.error(f"Failed to initialize Google Sheets integration after {max_retries} attempts")
        return False
            
    def setup_sheets(self):
        """Create worksheets if they don't exist"""
        if not self.initialized:
            logger.warning("Google Sheets not initialized. Cannot set up sheets.")
            return
            
        try:
            # Get existing worksheet names
            existing_worksheets = {ws.title: ws for ws in self.spreadsheet.worksheets()}
            
            # Store worksheets in a dict for easy access
            self.worksheets = existing_worksheets
            
            # Define required worksheets with their headers
            required_worksheets = {
                "Dashboard": ["Last Updated", "Total Portfolio Value", "Total Return", "Best Performing Stock", "Worst Performing Stock", "Open Positions", "Today's Signals"],
                "Trade Log": ["Date", "Symbol", "Action", "Price", "Quantity", "P&L", "Portfolio Value", "Strategy", "Signal Strength", "Notes"],
                "Performance": ["Symbol", "Win Rate", "Total Trades", "Total Return", "Avg Return/Trade", "Sharpe Ratio", "Max Drawdown", "Last Updated"],
                "ML Predictions": ["Date", "Symbol", "Prediction", "Confidence", "Actual Result", "Accuracy"],
                "Portfolio": ["Date", "Symbol", "Allocation %", "Position Size", "Current Value", "Profit/Loss"],
                "Risk Metrics": ["Date", "VaR (95%)", "CVaR", "Portfolio Beta", "Portfolio Volatility", "Correlation Matrix"],
            }
            
            for sheet_name, headers in required_worksheets.items():
                if sheet_name not in existing_worksheets:
                    # Create new worksheet
                    ws = self.spreadsheet.add_worksheet(title=sheet_name, rows=1000, cols=len(headers))
                    # Add headers
                    ws.append_row(headers)
                    logger.info(f"Created {sheet_name} worksheet")
                    self.worksheets[sheet_name] = ws
                else:
                    logger.info(f"{sheet_name} worksheet already exists")
                    
            # Update dashboard with timestamp
            if "Dashboard" in self.worksheets:
                dashboard = self.worksheets["Dashboard"]
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                # Check if we need to update existing row or create new one
                try:
                    cell = dashboard.find("Last Updated")
                    if cell:
                        dashboard.update_cell(cell.row, 2, timestamp)
                except gspread.exceptions.CellNotFound:
                    dashboard.append_row(["Last Updated", timestamp])
                    
        except Exception as e:
            logger.error(f"Error setting up sheets: {str(e)}")
            logger.error(traceback.format_exc())
    
    def _get_worksheet(self, name):
        """Get a worksheet by name with error handling"""
        if not self.initialized:
            logger.warning(f"Google Sheets not initialized. Cannot access {name} worksheet.")
            return None
            
        try:
            # Try to get from cache first
            if name in self.worksheets:
                return self.worksheets[name]
                
            # If not in cache, try to get from spreadsheet
            worksheet = self.spreadsheet.worksheet(name)
            self.worksheets[name] = worksheet
            return worksheet
        except gspread.exceptions.WorksheetNotFound:
            logger.error(f"Worksheet {name} not found")
            return None
        except Exception as e:
            logger.error(f"Error accessing {name} worksheet: {str(e)}")
            return None
    
    def log_trade(self, trade_data):
        """
        Log a trade to Google Sheets
        
        Args:
            trade_data: Dict with trade information
            {
                "symbol": str,
                "action": str ("BUY" or "SELL"),
                "price": float,
                "quantity": int,
                "pnl": float (optional),
                "portfolio_value": float (optional),
                "strategy": str (optional),
                "signal_strength": float (optional),
                "notes": str (optional)
            }
        """
        worksheet = self._get_worksheet("Trade Log")
        if not worksheet:
            logger.error(f"Cannot log trade: Trade Log worksheet not available")
            return False
            
        try:
            # Create row data
            row = [
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                trade_data["symbol"],
                trade_data["action"],
                trade_data["price"],
                trade_data["quantity"],
                trade_data.get("pnl", ""),
                trade_data.get("portfolio_value", ""),
                trade_data.get("strategy", ""),
                trade_data.get("signal_strength", ""),
                trade_data.get("notes", "")
            ]
            
            # Append the row
            worksheet.append_row(row)
            logger.info(f"Logged {trade_data['action']} trade for {trade_data['symbol']}")
            
            # Update Dashboard with latest trade
            self.update_dashboard({"latest_trade": trade_data})
            
            return True
            
        except Exception as e:
            logger.error(f"Error logging trade: {str(e)}")
            return False
    
    def update_performance(self, performance_data):
        """Update performance metrics in Google Sheets"""
        worksheet = self._get_worksheet("Performance")
        if not worksheet:
            logger.error(f"Cannot update performance: Performance worksheet not available")
            return False
            
        try:
            # Clear existing data (except header)
            if worksheet.row_count > 1:
                worksheet.delete_rows(2, worksheet.row_count)
                
            # Add updated metrics
            for symbol, metrics in performance_data.items():
                # Check for required keys and provide defaults if missing
                win_rate = metrics.get('Win_Rate', 0.0)
                if isinstance(win_rate, str) and win_rate.endswith('%'):
                    win_rate = float(win_rate.rstrip('%'))
                    
                total_trades = metrics.get('Total_Trades', 0)
                
                total_return = metrics.get('Total_Return', 0.0)
                if isinstance(total_return, str) and total_return.endswith('%'):
                    total_return = float(total_return.rstrip('%'))
                    
                max_drawdown = metrics.get('Max_Drawdown', 0.0)
                if isinstance(max_drawdown, str) and max_drawdown.endswith('%'):
                    max_drawdown = float(max_drawdown.rstrip('%'))
                
                # Format the row data
                row = [
                    symbol,
                    f"{win_rate:.2f}%",
                    total_trades,
                    f"{total_return:.2f}%",
                    f"{max_drawdown:.2f}%"
                ]
                
                # Add additional metrics if available
                if 'Sharpe_Ratio' in metrics:
                    sharpe = metrics['Sharpe_Ratio']
                    if len(row) <= 5:  # Make sure we have enough columns
                        row.append(f"{sharpe:.2f}")
                
                worksheet.append_row(row)
                
            logger.info(f"Updated performance data for {len(performance_data)} symbols")
            return True
            
        except Exception as e:
            logger.error(f"Error updating performance: {str(e)}")
            return False
    
    def log_ml_prediction(self, prediction_data):
        """
        Log ML prediction to Google Sheets
        
        Args:
            prediction_data: Dict with prediction information
            {
                "symbol": str,
                "prediction": str ("UP" or "DOWN"),
                "confidence": float (0-1),
                "actual_result": str (optional),
                "accuracy": float (optional)
            }
        """
        worksheet = self._get_worksheet("ML Predictions")
        if not worksheet:
            logger.error(f"Cannot log ML prediction: ML Predictions worksheet not available")
            return False
            
        try:
            # Create row data
            row = [
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                prediction_data["symbol"],
                prediction_data["prediction"],
                prediction_data.get("confidence", ""),
                prediction_data.get("actual_result", ""),
                prediction_data.get("accuracy", "")
            ]
            
            # Append the row
            worksheet.append_row(row)
            logger.info(f"Logged {prediction_data['prediction']} prediction for {prediction_data['symbol']}")
            
            # Update dashboard with latest prediction
            self.update_dashboard({"latest_prediction": prediction_data})
            
            return True
            
        except Exception as e:
            logger.error(f"Error logging ML prediction: {str(e)}")
            return False
            
    def update_portfolio(self, portfolio_data):
        """
        Update portfolio allocation and performance
        
        Args:
            portfolio_data: List of portfolio positions
            [
                {
                    "symbol": str,
                    "allocation": float,
                    "position_size": float,
                    "current_value": float,
                    "profit_loss": float
                }
            ]
        """
        worksheet = self._get_worksheet("Portfolio")
        if not worksheet:
            logger.error(f"Cannot update portfolio: Portfolio worksheet not available")
            return False
            
        try:
            # Clear existing data (except header)
            if worksheet.row_count > 1:
                worksheet.delete_rows(2, worksheet.row_count)
                
            timestamp = datetime.now().strftime("%Y-%m-%d")
            total_value = 0
            
            # Add updated portfolio data
            for position in portfolio_data:
                row = [
                    timestamp,
                    position["symbol"],
                    f"{position['allocation']:.2f}%",
                    position["position_size"],
                    position["current_value"],
                    position["profit_loss"]
                ]
                worksheet.append_row(row)
                total_value += position["current_value"]
            
            # Update dashboard with portfolio value
            self.update_dashboard({"portfolio_value": total_value})
                
            logger.info(f"Updated portfolio data with {len(portfolio_data)} positions")
            return True
            
        except Exception as e:
            logger.error(f"Error updating portfolio: {str(e)}")
            return False
            
    def update_risk_metrics(self, risk_data):
        """
        Update risk metrics
        
        Args:
            risk_data: Dict with risk metrics
            {
                "var": float,
                "cvar": float,
                "beta": float,
                "volatility": float,
                "correlation_matrix": pd.DataFrame
            }
        """
        worksheet = self._get_worksheet("Risk Metrics")
        if not worksheet:
            logger.error(f"Cannot update risk metrics: Risk Metrics worksheet not available")
            return False
            
        try:
            # Clear existing data (except header)
            if worksheet.row_count > 1:
                worksheet.delete_rows(2, worksheet.row_count)
                
            timestamp = datetime.now().strftime("%Y-%m-%d")
            
            # Add risk metrics
            row = [
                timestamp,
                f"{risk_data['var']:.2f}%",
                f"{risk_data['cvar']:.2f}%",
                risk_data['beta'],
                f"{risk_data['volatility']:.2f}%",
                str(risk_data['correlation_matrix'].to_dict())
            ]
            worksheet.append_row(row)
                
            logger.info(f"Updated risk metrics data")
            return True
            
        except Exception as e:
            logger.error(f"Error updating risk metrics: {str(e)}")
            return False
    
    def update_dashboard(self, data):
        """
        Update the dashboard with key metrics
        
        Args:
            data: Dict with dashboard data
            {
                "portfolio_value": float (optional),
                "best_performer": dict (optional),
                "worst_performer": dict (optional),
                "latest_trade": dict (optional),
                "latest_prediction": dict (optional),
                "open_positions": int (optional),
                "today_signals": list (optional),
                "last_updated": str (optional)
            }
        """
        dashboard = self._get_worksheet("Dashboard")
        if not dashboard:
            logger.error(f"Cannot update dashboard: Dashboard worksheet not available")
            return False
            
        try:
            # Update portfolio value if provided
            if "portfolio_value" in data:
                try:
                    cell = dashboard.find("Total Portfolio Value")
                    if cell:
                        dashboard.update_cell(cell.row, 2, f"${data['portfolio_value']:.2f}")
                    else:
                        dashboard.append_row(["Total Portfolio Value", f"${data['portfolio_value']:.2f}"])
                except Exception as e:
                    logger.error(f"Error updating portfolio value on dashboard: {str(e)}")
            
            # Update best performer if provided
            if "best_performer" in data and data["best_performer"]["symbol"]:
                try:
                    cell = dashboard.find("Best Performing Stock")
                    if cell:
                        dashboard.update_cell(cell.row, 2, 
                                             f"{data['best_performer']['symbol']} ({data['best_performer']['return']:.2f}%)")
                    else:
                        dashboard.append_row(["Best Performing Stock", 
                                             f"{data['best_performer']['symbol']} ({data['best_performer']['return']:.2f}%)"])
                except Exception as e:
                    logger.error(f"Error updating best performer on dashboard: {str(e)}")
            
            # Update worst performer if provided
            if "worst_performer" in data and data["worst_performer"]["symbol"]:
                try:
                    cell = dashboard.find("Worst Performing Stock")
                    if cell:
                        dashboard.update_cell(cell.row, 2, 
                                             f"{data['worst_performer']['symbol']} ({data['worst_performer']['return']:.2f}%)")
                    else:
                        dashboard.append_row(["Worst Performing Stock", 
                                             f"{data['worst_performer']['symbol']} ({data['worst_performer']['return']:.2f}%)"])
                except Exception as e:
                    logger.error(f"Error updating worst performer on dashboard: {str(e)}")
            
            # Update open positions count if provided
            if "open_positions" in data:
                try:
                    cell = dashboard.find("Open Positions")
                    if cell:
                        dashboard.update_cell(cell.row, 2, data['open_positions'])
                    else:
                        dashboard.append_row(["Open Positions", data['open_positions']])
                except Exception as e:
                    logger.error(f"Error updating open positions on dashboard: {str(e)}")
            
            # Update today's signals if provided
            if "today_signals" in data and data["today_signals"]:
                try:
                    cell = dashboard.find("Today's Signals")
                    signals_text = ", ".join([f"{s['symbol']} ({s['action']})" for s in data["today_signals"]])
                    if cell:
                        dashboard.update_cell(cell.row, 2, signals_text)
                    else:
                        dashboard.append_row(["Today's Signals", signals_text])
                except Exception as e:
                    logger.error(f"Error updating today's signals on dashboard: {str(e)}")
            
            # Update last updated timestamp if provided
            if "last_updated" in data:
                try:
                    cell = dashboard.find("Last Updated")
                    if cell:
                        dashboard.update_cell(cell.row, 2, data['last_updated'])
                    else:
                        dashboard.append_row(["Last Updated", data['last_updated']])
                except Exception as e:
                    logger.error(f"Error updating timestamp on dashboard: {str(e)}")
            
            logger.info("Updated dashboard successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error updating dashboard: {str(e)}")
            return False