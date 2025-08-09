#!/usr/bin/env python3
"""
Test script for fixed Google Sheets integration
"""

import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from config.config_loader import Config
from src.integrations.google_sheets import GoogleSheetsIntegration
from src.utils.logger import setup_logger

# Set up logger
logger = setup_logger('test_sheets')

def main():
    """Test Google Sheets integration with fixed ID"""
    # Display configuration
    print(f"Credentials file: {Config.GOOGLE_SHEETS_CREDENTIALS_FILE}")
    print(f"Spreadsheet ID: {Config.GOOGLE_SHEETS_SPREADSHEET_ID}")
    
    # Verify credentials file exists
    if not os.path.exists(Config.GOOGLE_SHEETS_CREDENTIALS_FILE):
        print(f"❌ ERROR: Credentials file not found at {Config.GOOGLE_SHEETS_CREDENTIALS_FILE}")
        return
    
    # Initialize Google Sheets integration
    print("\nTrying to connect to Google Sheets...")
    gs = GoogleSheetsIntegration(
        Config.GOOGLE_SHEETS_CREDENTIALS_FILE,
        Config.GOOGLE_SHEETS_SPREADSHEET_ID
    )
    
    if not gs.initialized:
        print("❌ Failed to initialize Google Sheets integration")
        return
    
    print("✅ Successfully connected to Google Sheets!")
    
    # Test logging a trade
    print("\nLogging a test trade...")
    trade_data = {
        "symbol": "TEST",
        "action": "BUY",
        "price": 150.25,
        "quantity": 10
    }
    
    if gs.log_trade(trade_data):
        print("✅ Successfully logged test trade")
    else:
        print("❌ Failed to log test trade")
    
    # Test logging ML prediction
    print("\nLogging a test ML prediction...")
    prediction_data = {
        "symbol": "TEST",
        "prediction": "UP",
        "confidence": 0.85
    }
    
    if gs.log_ml_prediction(prediction_data):
        print("✅ Successfully logged test prediction")
    else:
        print("❌ Failed to log test prediction")
    
    print("\nTest completed. Check your Google Sheet to verify the data was added.")
    print(f"Sheet URL: https://docs.google.com/spreadsheets/d/{Config.GOOGLE_SHEETS_SPREADSHEET_ID}")

if __name__ == "__main__":
    main()