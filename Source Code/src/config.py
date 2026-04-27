"""
Configuration settings for the quantitative trading project.
"""
import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PLOTS_DIR = PROJECT_ROOT / "plots"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# Create directories if they don't exist
for dir_path in [DATA_DIR, PLOTS_DIR, MODELS_DIR, RESULTS_DIR]:
    dir_path.mkdir(exist_ok=True)

# Data file paths
SPOT_DATA_PATH = DATA_DIR / "nifty_spot_5min.csv"
FUTURES_DATA_PATH = DATA_DIR / "nifty_futures_5min.csv"
OPTIONS_DATA_PATH = DATA_DIR / "nifty_options_5min.csv"
MERGED_DATA_PATH = DATA_DIR / "nifty_merged_5min.csv"
FEATURES_DATA_PATH = DATA_DIR / "nifty_features_5min.csv"
CLEANING_REPORT_PATH = DATA_DIR / "data_cleaning_report.txt"

# Trading parameters
RISK_FREE_RATE = 0.065  # 6.5% as specified
TRADING_DAYS_PER_YEAR = 252
MINUTES_PER_DAY = 75  # 9:15 AM to 3:30 PM = 6.25 hours = 375 min / 5 = 75 candles

# EMA parameters
EMA_FAST = 5
EMA_SLOW = 15

# HMM parameters
HMM_N_STATES = 3
HMM_TRAIN_RATIO = 0.7  # 70% for training

# Backtesting parameters
TRAIN_RATIO = 0.7
TEST_RATIO = 0.3

# ML parameters
LSTM_SEQUENCE_LENGTH = 10
ML_CONFIDENCE_THRESHOLD = 0.5

# Regime labels
REGIME_UPTREND = 1
REGIME_DOWNTREND = -1
REGIME_SIDEWAYS = 0
