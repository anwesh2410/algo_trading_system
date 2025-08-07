import logging
import os
from datetime import datetime
from pathlib import Path
import sys

# Add the project root to path to find the config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config_loader import Config

# Create logs directory if it doesn't exist
log_dir = Path(__file__).parent.parent.parent / "logs"
log_dir.mkdir(exist_ok=True)

# Configure logging
def setup_logger(name):
    log_filename = log_dir / f"{datetime.now().strftime('%Y%m%d')}_{name}.log"
    
    logger = logging.getLogger(name)
    logger.setLevel(Config.get_log_level())
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(Config.get_log_level())
    
    # File handler
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(Config.get_log_level())
    
    # Create formatter and add it to handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger