import os
import logging
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env file
env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)

class Config:
    # API Configuration
    TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
    GOOGLE_SHEETS_CREDENTIALS_FILE = os.getenv('GOOGLE_SHEETS_CREDENTIALS_FILE')
    GOOGLE_SHEETS_SPREADSHEET_ID = os.getenv('GOOGLE_SHEETS_SPREADSHEET_ID')
    
    # Trading Parameters
    BACKTEST_START_DATE = os.getenv('BACKTEST_START_DATE', '2023-01-01')
    BACKTEST_END_DATE = os.getenv('BACKTEST_END_DATE', '2023-12-31')
    RSI_PERIOD = int(os.getenv('RSI_PERIOD', 14))
    RSI_BUY_THRESHOLD = int(os.getenv('RSI_BUY_THRESHOLD', 30))
    MA_SHORT_PERIOD = int(os.getenv('MA_SHORT_PERIOD', 20))
    MA_LONG_PERIOD = int(os.getenv('MA_LONG_PERIOD', 50))
    
    # Stock List
    STOCK_SYMBOLS = os.getenv('STOCK_SYMBOLS', 'RELIANCE.NS,TCS.NS,HDFCBANK.NS').split(',')
    
    # System Configuration
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    
    @classmethod
    def get_log_level(cls):
        """Convert string log level to logging level."""
        levels = {
            'DEBUG': logging.DEBUG,
            'INFO': logging.INFO,
            'WARNING': logging.WARNING,
            'ERROR': logging.ERROR,
            'CRITICAL': logging.CRITICAL
        }
        return levels.get(cls.LOG_LEVEL, logging.INFO)