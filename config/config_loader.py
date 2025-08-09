import os
from pathlib import Path
from dotenv import load_dotenv
from typing import List, Optional
import logging

# Load environment variables from .env file
env_path = Path(__file__).parent / '.env'
load_dotenv(dotenv_path=env_path)

class Config:
    # API Configuration
    TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
    TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
    GOOGLE_SHEETS_CREDENTIALS_FILE = os.getenv('GOOGLE_SHEETS_CREDENTIALS_FILE', '')
    GOOGLE_SHEETS_SPREADSHEET_ID = os.getenv('GOOGLE_SHEETS_SPREADSHEET_ID', '')
    
    # Trading parameters
    STOCK_SYMBOLS = os.getenv('SYMBOLS', 'AAPL,MSFT,GOOG').split(',')
    BACKTEST_START_DATE = os.getenv('DATA_START_DATE', '2020-01-01')
    BACKTEST_END_DATE = os.getenv('DATA_END_DATE', '')  # Empty means today
    INITIAL_CAPITAL = float(os.getenv('INITIAL_CAPITAL', '100000'))
    
    # Strategy parameters
    RSI_PERIOD = int(os.getenv('RSI_PERIOD', '14'))
    RSI_BUY_THRESHOLD = float(os.getenv('RSI_BUY_THRESHOLD', '30'))
    MA_SHORT_PERIOD = int(os.getenv('MA_SHORT_PERIOD', '20'))
    MA_LONG_PERIOD = int(os.getenv('MA_LONG_PERIOD', '50'))
    
    # Integration options
    LOG_BACKTEST_SIGNALS_TO_INTEGRATIONS = os.getenv('LOG_BACKTEST_SIGNALS', 'False').lower() == 'true'
    
    @staticmethod
    def get_log_level():
        """Get logging level from environment or default to INFO"""
        level = os.getenv('LOG_LEVEL', 'INFO').upper()
        return getattr(logging, level, logging.INFO)

# Update the template creation function
def create_env_template():
    """Create .env file template if it doesn't exist"""
    env_path = Path(__file__).parent.parent / ".env"
    
    if not env_path.exists():
        template = """# Algo Trading System Configuration
# Trading parameters
SYMBOLS=AAPL,MSFT,GOOG
DATA_START_DATE=2020-01-01
DATA_END_DATE=
INITIAL_CAPITAL=100000

# Integration credentials
GOOGLE_SHEETS_CREDENTIALS_FILE=
GOOGLE_SHEETS_SPREADSHEET_ID=
GOOGLE_SHEETS_SPREADSHEET_NAME=AlgoTradingSystem
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=
"""
        with open(env_path, 'w') as f:
            f.write(template)
        
        logger.info(f"Created .env template at {env_path}")

class ConfigurationError(Exception):
    """Custom exception for configuration errors"""
    pass

def get_required_env(key: str) -> str:
    """Get required environment variable or raise error"""
    value = os.getenv(key)
    if not value:
        raise ConfigurationError(f"Required environment variable {key} is not set")
    return value

def get_optional_env(key: str, default: str) -> str:
    """Get optional environment variable with default"""
    return os.getenv(key, default)

# Configuration variables with proper validation
try:
    TELEGRAM_BOT_TOKEN = get_required_env('TELEGRAM_BOT_TOKEN')
    TELEGRAM_CHAT_ID = get_required_env('TELEGRAM_CHAT_ID')
    
    STOCK_SYMBOLS = get_optional_env('STOCK_SYMBOLS', 'AAPL,MSFT,GOOG').split(',')
    STOCK_SYMBOLS = [symbol.strip() for symbol in STOCK_SYMBOLS]  # Clean whitespace
    
    BACKTEST_START_DATE = get_optional_env('BACKTEST_START_DATE', '2020-01-01')
    BACKTEST_END_DATE = get_optional_env('BACKTEST_END_DATE', '2024-01-01')
    
    INITIAL_CAPITAL = float(get_optional_env('INITIAL_CAPITAL', '10000'))
    RISK_FREE_RATE = float(get_optional_env('RISK_FREE_RATE', '0.02'))
    
    GOOGLE_CREDENTIALS_PATH = get_optional_env('GOOGLE_CREDENTIALS_PATH', '')
    
except ConfigurationError as e:
    logging.error(f"Configuration error: {e}")
    raise
except ValueError as e:
    logging.error(f"Invalid configuration value: {e}")
    raise ConfigurationError(f"Invalid configuration value: {e}")

# Validation
if INITIAL_CAPITAL <= 0:
    raise ConfigurationError("INITIAL_CAPITAL must be positive")

if not (0 <= RISK_FREE_RATE <= 1):
    raise ConfigurationError("RISK_FREE_RATE must be between 0 and 1")

if len(STOCK_SYMBOLS) == 0:
    raise ConfigurationError("At least one stock symbol must be provided")

print("Configuration loaded successfully")