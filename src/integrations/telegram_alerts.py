import os
import sys
import requests
import time
from datetime import datetime
from pathlib import Path

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from src.utils.logger import setup_logger

# Initialize logger
logger = setup_logger('telegram_alerts')

class TelegramAlert:
    """
    Enhanced class to send trading alerts to a Telegram bot
    with chart images and formatted messages
    """
    
    def __init__(self, bot_token, chat_id):
        """
        Initialize Telegram bot connection
        
        Args:
            bot_token: Telegram bot token
            chat_id: Chat ID to send messages to
        """
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.api_base = "https://api.telegram.org"  # Add this line to define api_base
        
        # Validate credentials
        if not bot_token:
            logger.error("Bot token not provided")
        if not chat_id:
            logger.error("Chat ID not provided")
    
    def send_message(self, message, parse_mode='Markdown', retry_count=3, retry_delay=2):
        """
        Send a text message to the Telegram chat
        
        Args:
            message: Text message to send
            parse_mode: Message formatting (Markdown or HTML)
            retry_count: Number of retries on failure
            retry_delay: Delay between retries in seconds
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.bot_token or not self.chat_id:
            logger.error("Telegram credentials not configured")
            return False
        
        # Sanitize message for Markdown
        if parse_mode == 'Markdown':
            message = self._escape_markdown(message)
        
        # Send message with retry logic
        for attempt in range(retry_count):
            try:
                url = f"{self.api_base}/bot{self.bot_token}/sendMessage"
                params = {
                    'chat_id': self.chat_id,
                    'text': message,
                    'parse_mode': parse_mode
                }
                
                response = requests.post(url, data=params)
                response.raise_for_status()
                
                logger.info(f"Message sent successfully")
                return True
                
            except requests.exceptions.RequestException as e:
                logger.error(f"HTTP error {e}")
                if attempt < retry_count - 1:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    logger.error(f"Failed to send message after {retry_count} attempts")
                    return False
                    
            except Exception as e:
                logger.error(f"Error sending message: {str(e)}")
                return False
        
        return False
    
    def _escape_markdown(self, text):
        """
        Escape Markdown special characters
        
        Args:
            text: Text to escape
        
        Returns:
            str: Escaped text
        """
        # Basic Markdown escaping
        # This is a simplified approach, more advanced escaping might be needed
        escape_chars = ['_', '*', '`', '[']
        for char in escape_chars:
            text = text.replace(char, '\\' + char)
        
        return text

    # The rest of your methods would be updated similarly to use self.api_base
    # For example:
    
    def send_photo(self, photo_path=None, photo_url=None, caption=None, retry_count=3, retry_delay=2):
        """
        Send a photo to the Telegram chat
        
        Args:
            photo_path: Path to photo file (local)
            photo_url: URL of the photo (remote)
            caption: Photo caption
            retry_count: Number of retries on failure
            retry_delay: Delay between retries in seconds
            
        Returns:
            bool: True if successful, False otherwise
        """
        if not self.bot_token or not self.chat_id:
            logger.error("Telegram credentials not configured")
            return False
            
        # Check if we have a photo to send
        if not photo_path and not photo_url:
            logger.error("No photo provided (path or URL)")
            return False
            
        # Send photo with retry logic
        for attempt in range(retry_count):
            try:
                url = f"{self.api_base}/bot{self.bot_token}/sendPhoto"
                
                if photo_url:
                    # Send photo by URL
                    params = {
                        'chat_id': self.chat_id,
                        'photo': photo_url,
                        'caption': caption or ''
                    }
                    response = requests.post(url, data=params)
                else:
                    # Send photo by file upload
                    with open(photo_path, 'rb') as photo:
                        params = {
                            'chat_id': self.chat_id,
                            'caption': caption or ''
                        }
                        files = {'photo': photo}
                        response = requests.post(url, data=params, files=files)
                
                response.raise_for_status()
                logger.info(f"Photo sent successfully")
                return True
                
            except requests.exceptions.RequestException as e:
                logger.error(f"HTTP error {e}")
                if attempt < retry_count - 1:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    logger.error(f"Failed to send photo after {retry_count} attempts")
                    return False
                    
            except Exception as e:
                logger.error(f"Error sending photo: {str(e)}")
                return False
        
        return False
    
    def send_system_alert(self, alert_type, message):
        """
        Send a system alert message to Telegram
        
        Args:
            alert_type: Type of alert - "info", "warning", or "error"
            message: Alert message
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Set emoji based on alert type
        emoji = "ℹ️"  # info (default)
        if alert_type.lower() == "warning":
            emoji = "⚠️"
        elif alert_type.lower() == "error":
            emoji = "🚨"
        
        # Format the message
        formatted_message = f"{emoji} *SYSTEM {alert_type.upper()}*\n\n{message}\n\nTime: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        # Send the message
        return self.send_message(formatted_message)
    
    def send_trade_alert(self, symbol, action, price, timestamp=None, quantity=None, signal_strength=None, strategy=None):
        """
        Send a trade alert to Telegram
        
        Args:
            symbol: Stock symbol
            action: BUY or SELL
            price: Trade price
            timestamp: Time of the trade (default: current time)
            quantity: Number of shares (optional)
            signal_strength: Signal strength indicator (optional)
            strategy: Strategy name (optional)
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Format the message
        emoji = "🟢" if action == "BUY" else "🔴"
        timestamp = timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        message = f"{emoji} *{action} SIGNAL: {symbol}*\n\n"
        message += f"Price: ${price:.2f}\n"
        message += f"Time: {timestamp}\n"
        
        if quantity:
            message += f"Quantity: {quantity}\n"
            
        if signal_strength:
            strength_percent = signal_strength * 100 if signal_strength <= 1 else signal_strength
            message += f"Signal Strength: {strength_percent:.1f}%\n"
            
        if strategy:
            message += f"Strategy: {strategy}\n"
            
        # Send the message
        return self.send_message(message)

    def send_performance_update(self, symbol, total_return, win_rate, other_metrics=None):
        """
        Send performance update for a symbol
        
        Args:
            symbol: Stock ticker symbol
            total_return: Total return percentage
            win_rate: Win rate percentage
            other_metrics: Dict with additional metrics (optional)
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Format the message
        emoji = "📈" if total_return >= 0 else "📉"
        message = f"{emoji} *Performance Update: {symbol}*\n\n"
        message += f"Total Return: {total_return:.2f}%\n"
        message += f"Win Rate: {win_rate:.2f}%\n"
        
        # Add additional metrics if provided
        if other_metrics:
            message += "\n*Additional Metrics:*\n"
            for metric, value in other_metrics.items():
                message += f"{metric}: {value}\n"
        
        # Add timestamp
        message += f"\nTime: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        # Send the message
        return self.send_message(message)

    def send_ml_prediction(self, symbol, prediction, confidence=None, supporting_data=None, chart_path=None):
        """
        Send ML prediction alert
        
        Args:
            symbol: Stock ticker symbol
            prediction: UP or DOWN (or 1/0)
            confidence: Prediction confidence (optional)
            supporting_data: Dict with supporting data (optional)
            chart_path: Path to chart image (optional)
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Convert numerical prediction to UP/DOWN if needed
        if isinstance(prediction, (int, float)):
            prediction = "UP" if prediction > 0.5 or prediction == 1 else "DOWN"
        
        # Format the message
        emoji = "🔮" 
        direction_emoji = "↗️" if prediction == "UP" else "↘️"
        
        message = f"{emoji} *ML Prediction: {symbol}*\n\n"
        message += f"Direction: {direction_emoji} {prediction}\n"
        
        if confidence is not None:
            conf_val = confidence * 100 if confidence <= 1 else confidence
            message += f"Confidence: {conf_val:.1f}%\n"
        
        # Add supporting data if provided
        if supporting_data:
            message += "\n*Supporting Data:*\n"
            for key, value in supporting_data.items():
                message += f"{key}: {value}\n"
        
        # Add timestamp
        message += f"\nTime: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        # If we have a chart, send it with the message as caption
        if chart_path and os.path.exists(chart_path):
            return self.send_photo(photo_path=chart_path, caption=message)
        else:
            # Otherwise just send the text message
            return self.send_message(message)

    def send_daily_summary(self, summary_data):
        """
        Send daily summary
        
        Args:
            summary_data: Dictionary with summary information
                {
                    "date": str,
                    "portfolio_value": float,
                    "daily_change": float,
                    "top_performers": list of dicts,
                    "trades_executed": int,
                    "signals_generated": int
                }
        
        Returns:
            bool: True if successful, False otherwise
        """
        # Format the message
        emoji = "📊"
        message = f"{emoji} *Daily Trading Summary*\n\n"
        
        # Add date
        date = summary_data.get("date", datetime.now().strftime("%Y-%m-%d"))
        message += f"Date: {date}\n\n"
        
        # Add portfolio info
        if "portfolio_value" in summary_data:
            message += f"Portfolio Value: ${summary_data['portfolio_value']:.2f}\n"
            
        if "daily_change" in summary_data:
            change = summary_data["daily_change"]
            change_emoji = "📈" if change >= 0 else "📉"
            message += f"Daily Change: {change_emoji} {change:.2f}%\n\n"
        
        # Add top performers
        if "top_performers" in summary_data and summary_data["top_performers"]:
            message += "*Top Performers:*\n"
            for performer in summary_data["top_performers"][:3]:  # Top 3
                message += f"  {performer['symbol']}: {performer['return']:.2f}%\n"
            message += "\n"
        
        # Add activity stats
        if "trades_executed" in summary_data:
            message += f"Trades Executed: {summary_data['trades_executed']}\n"
            
        if "signals_generated" in summary_data:
            message += f"Signals Generated: {summary_data['signals_generated']}\n"
        
        # Send the message
        return self.send_message(message)

    def send_portfolio_update(self, portfolio_value, daily_change, positions):
        """
        Send portfolio update
        
        Args:
            portfolio_value: Current portfolio value
            daily_change: Daily change percentage
            positions: List of positions [{"symbol": str, "allocation": float, "return": float}]
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Format the message
        emoji = "💼"
        message = f"{emoji} *Portfolio Update*\n\n"
        
        # Add portfolio info
        message += f"Total Value: ${portfolio_value:.2f}\n"
        
        # Add daily change
        change_emoji = "📈" if daily_change >= 0 else "📉"
        message += f"Daily Change: {change_emoji} {daily_change:.2f}%\n\n"
        
        # Add positions
        if positions:
            message += "*Current Positions:*\n"
            for pos in positions:
                ret = pos.get("return", 0)
                ret_emoji = "📈" if ret >= 0 else "📉"
                message += f"  {pos['symbol']} ({pos['allocation']:.1f}%): {ret_emoji} {ret:.2f}%\n"
        
        # Add timestamp
        message += f"\nTime: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        # Send the message
        return self.send_message(message)

    def send_chart(self, symbol, chart_path=None, chart_buffer=None, caption=None):
        """
        Send a chart image
        
        Args:
            symbol: Stock ticker symbol
            chart_path: Path to chart image
            chart_buffer: Buffer with chart image (optional, alternative to path)
            caption: Chart caption (optional)
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Create default caption if not provided
        if not caption:
            caption = f"Chart for {symbol} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        # Send chart
        if chart_path and os.path.exists(chart_path):
            return self.send_photo(photo_path=chart_path, caption=caption)
        elif chart_buffer:
            # This would need additional implementation to handle buffers
            logger.warning("Sending chart from buffer not implemented")
            return False
        else:
            logger.warning(f"Chart for {symbol} not found")
            return False