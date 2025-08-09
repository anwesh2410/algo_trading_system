import os
from datetime import datetime
from src.integrations.telegram_alerts import TelegramAlert
from config.config_loader import Config
from src.utils.logger import setup_logger

# Set up logger
logger = setup_logger('telegram_helper')

class TelegramHelper:
    """
    Helper class to simplify Telegram alerts integration
    """
    
    _instance = None
    
    @classmethod
    def get_instance(cls):
        """Get singleton instance"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        """Initialize Telegram integration"""
        bot_token = Config.TELEGRAM_BOT_TOKEN
        chat_id = Config.TELEGRAM_CHAT_ID
        
        if bot_token and chat_id:
            self.telegram = TelegramAlert(bot_token, chat_id)
            logger.info("Telegram alerts integration initialized")
        else:
            logger.warning("Telegram credentials not configured")
            self.telegram = None
    
    def send_message(self, message):
        """
        Send a simple text message
        
        Args:
            message: Text message to send
        """
        if not self.telegram:
            logger.warning("Telegram integration not available")
            return False
            
        return self.telegram.send_message(message)
    
    def send_trade_signal(self, symbol, action, price, quantity=None, 
                         signal_strength=None, strategy=None):
        """
        Send trade signal alert
        
        Args:
            symbol: Stock ticker symbol
            action: BUY or SELL
            price: Trade price
            quantity: Trade quantity (optional)
            signal_strength: Signal strength (optional)
            strategy: Trading strategy used (optional)
        """
        if not self.telegram:
            logger.warning("Telegram integration not available")
            return False
            
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return self.telegram.send_trade_alert(
            symbol=symbol,
            action=action,
            price=price,
            timestamp=timestamp,
            quantity=quantity,
            signal_strength=signal_strength,
            strategy=strategy
        )
    
    def send_performance_update(self, symbol, total_return, win_rate, other_metrics=None):
        """
        Send performance update for a symbol
        
        Args:
            symbol: Stock ticker symbol
            total_return: Total return percentage
            win_rate: Win rate percentage
            other_metrics: Dict with additional metrics (optional)
        """
        if not self.telegram:
            logger.warning("Telegram integration not available")
            return False
            
        return self.telegram.send_performance_update(
            symbol=symbol,
            total_return=total_return,
            win_rate=win_rate,
            other_metrics=other_metrics
        )
    
    def send_ml_prediction(self, symbol, prediction, confidence=None, 
                          supporting_data=None, chart_path=None):
        """
        Send ML prediction alert
        
        Args:
            symbol: Stock ticker symbol
            prediction: UP or DOWN (or 1/0)
            confidence: Prediction confidence (optional)
            supporting_data: Dict with supporting data (optional)
            chart_path: Path to chart image (optional)
        """
        if not self.telegram:
            logger.warning("Telegram integration not available")
            return False
            
        # Convert numerical prediction to UP/DOWN if needed
        if isinstance(prediction, (int, float)):
            prediction = "UP" if prediction == 1 else "DOWN"
            
        return self.telegram.send_ml_prediction(
            symbol=symbol,
            prediction=prediction,
            confidence=confidence,
            supporting_data=supporting_data,
            chart_path=chart_path
        )
    
    def send_portfolio_update(self, portfolio_value, daily_change, positions):
        """
        Send portfolio update
        
        Args:
            portfolio_value: Current portfolio value
            daily_change: Daily change percentage
            positions: List of positions [{"symbol": str, "allocation": float, "return": float}]
        """
        if not self.telegram:
            logger.warning("Telegram integration not available")
            return False
            
        return self.telegram.send_portfolio_update(
            portfolio_value=portfolio_value,
            daily_change=daily_change,
            positions=positions
        )
    
    def send_chart(self, symbol, chart_path=None, chart_buffer=None, caption=None):
        """
        Send a chart image
        
        Args:
            symbol: Stock ticker symbol
            chart_path: Path to chart image
            chart_buffer: Buffer with chart image (optional, alternative to path)
            caption: Chart caption (optional)
        """
        if not self.telegram:
            logger.warning("Telegram integration not available")
            return False
            
        return self.telegram.send_chart(
            symbol=symbol,
            chart_path=chart_path,
            chart_buffer=chart_buffer,
            caption=caption
        )
    
    def send_daily_summary(self, portfolio_value, daily_change, 
                          top_performers=None, trades_executed=None, signals_generated=None):
        """
        Send daily summary
        
        Args:
            portfolio_value: Current portfolio value
            daily_change: Daily change percentage
            top_performers: List of top performers [{"symbol": str, "return": float}] (optional)
            trades_executed: Number of trades executed today (optional)
            signals_generated: Number of signals generated today (optional)
        """
        if not self.telegram:
            logger.warning("Telegram integration not available")
            return False
            
        summary_data = {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "portfolio_value": portfolio_value,
            "daily_change": daily_change
        }
        
        if top_performers is not None:
            summary_data["top_performers"] = top_performers
            
        if trades_executed is not None:
            summary_data["trades_executed"] = trades_executed
            
        if signals_generated is not None:
            summary_data["signals_generated"] = signals_generated
            
        return self.telegram.send_daily_summary(summary_data)
    
    def send_system_alert(self, message, alert_type="info"):
        """
        Send system alert
        
        Args:
            message: Alert message
            alert_type: Type of alert - "info", "warning", or "error"
        """
        if not self.telegram:
            logger.warning("Telegram integration not available")
            return False
            
        return self.telegram.send_system_alert(alert_type, message)
    
    def format_safe_message(self, message_parts):
        """
        Create a safely formatted message from parts
        
        Args:
            message_parts: Dictionary with message parts
                {
                    "title": "Message Title",
                    "sections": [
                        {"header": "Section 1", "content": "Content 1"},
                        {"header": "Section 2", "content": "Content 2"}
                    ],
                    "footer": "Message Footer"
                }
        
        Returns:
            str: Safely formatted message
        """
        if not self.telegram:
            logger.warning("Telegram integration not available")
            return ""
        
        # Start with the title
        message = f"{message_parts.get('title', '')}\n\n"
        
        # Add each section
        for section in message_parts.get('sections', []):
            if 'header' in section:
                message += f"• {section['header']}: "
            if 'content' in section:
                message += f"{section['content']}\n"
        
        # Add footer
        if 'footer' in message_parts:
            message += f"\n{message_parts['footer']}"
        
        return message