# Algorithmic Trading System

An automated trading system that fetches stock data, implements trading strategies, and logs results to Google Sheets with ML-based predictions.

## Features
- Data ingestion from Yahoo Finance API for NIFTY 50 stocks
- RSI + Moving Average crossover trading strategy
- ML predictions for next-day price movements
- Automated trade logging to Google Sheets
- Performance analytics and backtesting
- Telegram alerts for trading signals

## Setup Instructions
1. Clone this repository
2. Install requirements: `pip install -r requirements.txt`
3. Set up Google Sheets API credentials (see below)
4. Run the system: `python src/main.py`

## Google Sheets API Setup
[Instructions for setting up Google Sheets API will be here]

## Project Structure
- `src/`: Source code
- `data/`: Data storage
- `config/`: Configuration files
- `notebooks/`: Jupyter notebooks for analysis
- `tests/`: Unit tests

## Trading Strategy
The implemented strategy uses:
- RSI < 30 as buy signal
- 20-DMA crossing above 50-DMA as confirmation
- Backtested for 6 months

## ML Model
A Decision Tree model predicts next-day price movements using technical indicators.