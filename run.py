# File: /home/anwesh2410/IITJ/Proj/Assignments/Algo-Trading System/algo_trading_system/run.py

import os
import sys
from pathlib import Path

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

# Now directly import the main function
from src.main import main

if __name__ == "__main__":
    main()