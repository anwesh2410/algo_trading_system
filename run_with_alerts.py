#!/usr/bin/env python3
"""
Run the trading system with Telegram alerts and Google Sheets logging
"""

import os
import sys
import argparse
from datetime import datetime

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from src.main import main as run_main
from enhanced_run import main as run_enhanced
from live_trading import LiveTrading
from src.integrations.integrations_manager import IntegrationsManager
from src.utils.logger import setup_logger

# Set up logger
logger = setup_logger('run_with_alerts')

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Run Algo Trading System with alerts')
    parser.add_argument('--mode', choices=['main', 'enhanced', 'live'], default='enhanced',
                       help='Execution mode (main, enhanced, or live)')
    parser.add_argument('--telegram', action='store_true', 
                       help='Enable Telegram alerts')
    parser.add_argument('--sheets', action='store_true',
                       help='Enable Google Sheets logging')
    
    args = parser.parse_args()
    
    # Initialize integrations
    integrations = IntegrationsManager.get_instance()
    
    # Send startup message
    if args.telegram:
        integrations.telegram_helper.send_system_alert(
            f"Starting Algo Trading System in {args.mode} mode",
            "info"
        )
    
    # Run in selected mode
    try:
        if args.mode == 'main':
            logger.info("Running in main mode")
            run_main()
        elif args.mode == 'enhanced':
            logger.info("Running in enhanced mode")
            run_enhanced()
        elif args.mode == 'live':
            logger.info("Running in live trading mode")
            live_trading = LiveTrading()
            live_trading.run_live_trading()
    except Exception as e:
        logger.error(f"Error in execution: {str(e)}")
        
        if args.telegram:
            integrations.telegram_helper.send_system_alert(
                f"ERROR: System encountered an exception: {str(e)}",
                "error"
            )
    
    # Send completion message
    if args.telegram:
        integrations.telegram_helper.send_system_alert(
            f"Algo Trading System {args.mode} mode execution completed",
            "info"
        )

if __name__ == "__main__":
    main()