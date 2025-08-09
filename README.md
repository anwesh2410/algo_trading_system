# Algorithmic Trading System

A comprehensive algorithmic trading system with ML-enhanced strategies, risk management, portfolio optimization, and real-time alerts through Telegram and Google Sheets.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the System](#running-the-system)
- [Trading Methodology](#trading-methodology)
- [External Integrations](#external-integrations)
- [Performance Metrics](#performance-metrics)
- [Project Structure](#project-structure)
- [System Flow](#system-flow)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

This algorithmic trading system combines traditional technical analysis indicators with machine learning to generate buy/sell signals, manage risk, optimize portfolios, and deliver real-time alerts. The system can backtest strategies on historical data and run in live trading mode with external integrations.

---

## Features

- **Multiple Trading Strategies**
  - RSI and Moving Average crossover strategy
  - ML-enhanced prediction models
  - Risk-managed strategies with stop-loss and take-profit

- **Advanced Risk Management**
  - Position sizing
  - Stop-loss and take-profit levels
  - Maximum drawdown limits
  - Portfolio diversification

- **Machine Learning Integration**
  - Price movement prediction
  - Feature engineering with technical indicators
  - Model training and evaluation

- **Comprehensive Backtesting**
  - Performance metrics calculation
  - Trade visualization
  - Strategy comparison

- **Portfolio Optimization**
  - Modern Portfolio Theory implementation
  - Risk-adjusted returns
  - Multiple risk profiles

- **Real-time Alerts and Reporting**
  - Telegram notifications
  - Google Sheets integration
  - Trade logging
  - Performance tracking

---

## System Architecture

The system follows a modular architecture with the following components:

1. **Data Processing**: Fetches and processes historical stock data
2. **Strategy Generation**: Applies trading strategies to generate signals
3. **Machine Learning**: Enhances signals with predictive models
4. **Backtesting**: Evaluates strategy performance on historical data
5. **Risk Management**: Applies risk controls and position sizing
6. **Portfolio Optimization**: Optimizes asset allocation
7. **External Integrations**: Sends alerts and logs results

---

## Installation

### Prerequisites
- Python 3.8+
- pip package manager
- Git

### Step 1: Clone the repository
```bash
git clone https://github.com/yourusername/algo-trading-system.git
cd algo-trading-system
```

### Step 2: Create and activate virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Set up integrations

**Telegram**
1. Create a bot using BotFather on Telegram
2. Get your chat ID using the IDBot

**Google Sheets**
1. Create a Google Cloud project
2. Enable Google Sheets API
3. Create service account and download credentials JSON file

---

## Configuration

Create a `.env` file in the project root directory:

```
# API Configuration
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_CHAT_ID=your_chat_id_here
GOOGLE_SHEETS_CREDENTIALS_FILE=path/to/your/credentials.json
GOOGLE_SHEETS_SPREADSHEET_ID=your_spreadsheet_id_here

# Trading parameters
SYMBOLS=AAPL,MSFT,GOOG
DATA_START_DATE=2020-01-01
INITIAL_CAPITAL=100000

# Strategy parameters
RSI_PERIOD=14
RSI_BUY_THRESHOLD=30
MA_SHORT_PERIOD=20
MA_LONG_PERIOD=50

# Integration options
LOG_BACKTEST_SIGNALS=True
```

---

## Running the System

### Basic Run
```bash
python -m src.main
```

### Enhanced Run with ML models
```bash
python enhanced_run.py
```

### Live Trading with Alerts
```bash
python live_trading.py
```

### Run with Integrations
```bash
./run_with_alerts.py --telegram --sheets
```

### Available Command-line Options
```
--mode {main,enhanced,live}  Execution mode (default: enhanced)
--telegram                   Enable Telegram alerts
--sheets                     Enable Google Sheets logging
```

### Testing Integrations
```bash
python test_all_integrations.py
```

---

## Trading Methodology

### Data Processing
1. Fetch historical price data
2. Clean and handle missing values
3. Feature engineering with technical indicators
4. Normalize data for ML models

### Trading Strategies

**1. RSI-MA Crossover Strategy**
- RSI for overbought/oversold detection
- Moving average crossovers for trend detection
- Combined signals for entry/exit points

**2. ML-Enhanced Strategy**
- ML models to predict price movement
- Combine technical indicators with ML predictions
- Adjust parameters dynamically based on prediction confidence

**3. Risk-Managed Strategy**
- Stop-loss and take-profit levels
- Drawdown-based position exits
- Volatility-based position sizing

### Machine Learning Models
- **Feature Generation**: RSI, MACD, Bollinger Bands, price patterns, volume
- **Model Selection**: Random Forest, Gradient Boosting, Ensembles
- **Training/Evaluation**: Time-series aware split, cross-validation, metrics tracking

### Risk Management
- Volatility-based position sizing
- Stop-loss/take-profit
- Drawdown control
- Diversification

---

## External Integrations

### Telegram Alerts
- Trade signals
- Performance updates
- ML predictions
- Daily summaries
- Portfolio updates

### Google Sheets
- Trade logs with P&L
- Performance metrics
- ML prediction logs
- Portfolio allocation
- KPI dashboard

---

## Performance Metrics
- **Win Rate**: % of profitable trades
- **Average Return per Trade**
- **Total Return**
- **Sharpe Ratio**
- **Maximum Drawdown**
- **Volatility**
- **Profit Factor**

---

## Project Structure
```
algo_trading_system/
├── config/
│   └── config_loader.py
├── data/
│   ├── models/
│   └── results/
├── src/
│   ├── backtest/
│   ├── integrations/
│   ├── models/
│   ├── portfolio/
│   ├── strategies/
│   ├── utils/
│   ├── visualization/
│   └── main.py
├── tests/
├── enhanced_run.py
├── live_trading.py
├── run_with_alerts.py
├── test_all_integrations.py
├── requirements.txt
└── README.md
```

---

## System Flow
1. **Data Acquisition**
2. **Signal Generation**
3. **Risk Assessment**
4. **Execution (Backtest/Live)**
5. **Performance Analysis**
6. **External Communication**

---

## Troubleshooting

**Google Sheets Issues**
- Check credentials path & permissions
- Verify spreadsheet ID

**Telegram Issues**
- Validate bot token & chat ID
- Start a chat with the bot

**Data Processing Errors**
- Check internet & stock symbol validity
- Ensure correct date format

**Missing Dependencies**
```bash
pip install -r requirements.txt
```

---

## Contributing
1. Fork the repo
2. Create a feature branch
3. Commit changes
4. Push branch
5. Open a Pull Request

---

## License
This project is licensed under the MIT License - see the LICENSE file for details.

---

**Created by Ale Anwesh**
