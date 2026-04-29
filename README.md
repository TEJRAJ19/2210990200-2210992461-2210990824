# NIFTY 50 Quantitative Trading Strategy

## 📋 Project Information

| Field | Details |
|-------|---------|
| **Project Title** | NIFTY 50 Quantitative Trading Strategy |
| **Project Type** | Research Paper |
| **Roll Numbers** | 2210990200, 2210992461, 2210990824 |
| **Team Members** | Avinash Kumar Jha, Tej Raj, Shivam Laul |
| **Submission Status** | ✅ Submitted |

## 👥 Team Details

| Sr. No. | Name | Roll Number |
|---------|------|-------------|
| 1 | Avinash Kumar Jha | 2210990200 |
| 2 | Tej Raj | 2210992461 |
| 3 | Shivam Laul | 2210990824 |

## 📄 Submission Status

| Document | Status |
|----------|--------|
| Research Paper | ✅ Submitted |
| Final Project Report | ✅ Submitted |
| Source Code | ✅ Submitted |

---

## 📝 About the Project

A comprehensive quantitative trading system for NIFTY 50 featuring regime detection, machine learning trade filtering, and in-depth performance analysis.

## 🎯 Key Results

| Metric | Baseline | XGBoost-Filtered |
|--------|----------|------------------|
| **Total Return** | 10,072 pts | **14,261 pts** |
| **Sharpe Ratio** | 4.18 | **6.37** |
| **Win Rate** | 51.1% | 100% |
| **Max Drawdown** | 0.8% | 0% |

## 📊 Features

- **Data Pipeline**: Yahoo Finance integration + synthetic 5-minute candle generation
- **Feature Engineering**: 101 features including EMAs, Options Greeks, IV, PCR
- **Regime Detection**: 3-state Hidden Markov Model (Uptrend, Downtrend, Sideways)
- **Trading Strategy**: EMA crossover with regime filter
- **ML Enhancement**: XGBoost and LSTM for trade prediction
- **Trade Analysis**: Outlier detection, statistical tests, pattern recognition

## 🏗️ Project Structure

```
NIFTY-50-Quantitative-Trading-Strategy (2210990200, 2210992461, 2210990824)/
│
├── Source Code/
│   ├── src/
│   │   ├── __init__.py
│   │   ├── config.py              # Configuration
│   │   ├── data_fetcher.py        # Data acquisition
│   │   ├── data_cleaner.py        # Data cleaning
│   │   ├── data_merger.py         # Data merging
│   │   ├── feature_engineering.py # Feature engineering
│   │   ├── regime_detection.py    # HMM implementation
│   │   ├── strategy.py           # Trading strategy
│   │   ├── backtester.py         # Performance metrics
│   │   ├── ml_models.py          # XGBoost & LSTM
│   │   ├── ml_backtester.py      # ML-enhanced backtesting
│   │   ├── trade_analysis.py     # Outlier detection
│   │   ├── insights_summary.py   # Insights generation
│   │   └── enhanced_ml.py        # Ensemble model
│   ├── main.py                    # Main execution script
│   ├── run_ml_demo.py             # ML demo runner
│   ├── requirements.txt           # Dependencies
│   ├── plots/                     # Visualizations
│   └── results/                   # Reports and analysis
│
├── Report and PPT/
│   └── Quantitative_Trading_Abstract_Final_Formatted.docx
│
├── IPR Submission Proof/
│   └── 2210990200,2210992461,2210990824_Research__paper.docx
│
└── README.md
```

## 🚀 Quick Start

```bash
# Navigate to source code
cd "Source Code"

# Install dependencies
pip install -r requirements.txt

# Run the complete pipeline
python main.py
```

## 📈 Pipeline Components

### 1. Data Acquisition
- Fetches NIFTY 50 daily data from Yahoo Finance
- Generates realistic 5-minute intraday candles
- Creates futures (cost-of-carry) and options chain data

### 2. Feature Engineering
- **Technical Indicators**: EMA(5), EMA(15), ATR, Volatility
- **Options Greeks**: Delta, Gamma, Theta, Vega, Rho (Black-Scholes, r=6.5%)
- **Derived Features**: IV spread, Put-Call Ratio, Futures Basis, Gamma Exposure

### 3. Regime Detection
- Hidden Markov Model with 3 states
- Features: IV, PCR, Greeks, Basis, Returns
- 70/30 train/test split

### 4. Trading Strategy
- Long: EMA(5) > EMA(15) AND Regime = Uptrend
- Short: EMA(5) < EMA(15) AND Regime = Downtrend
- No trades in Sideways regime

### 5. Machine Learning
- **XGBoost**: Time-series cross-validation, 56% accuracy
- **LSTM**: Bidirectional architecture, 10-candle sequences
- **Ensemble**: XGBoost + RandomForest + GradientBoosting

### 6. Trade Analysis
- Outlier identification (Z-score > 3)
- Feature importance analysis
- Time-of-day patterns
- Regime distribution

## 📊 Visualizations

| Plot | Description |
|------|-------------|
| `regime_overlay.png` | Price chart with regime color coding |
| `transition_matrix.png` | Regime transition probabilities |
| `equity_curve.png` | Portfolio equity over time |
| `correlation_heatmap.png` | Feature correlations |
| `pnl_vs_duration.png` | Trade PnL vs duration scatter |

## 🔬 Key Insights

1. **1.65% of trades generate 27% of total profit**
2. **Outliers earn 7x more** than normal trades
3. **Best trading time**: Mid-morning (10:15-12:00) with 70% win rate
4. **Key predictors**: ATR (+282%), EMA gap (+189%), Volatility (+200%)

## 📋 Requirements

- Python 3.8+
- pandas, numpy, matplotlib, seaborn
- scikit-learn, xgboost
- tensorflow (for LSTM)
- hmmlearn, yfinance

## 🤝 Contact

Created as a quantitative trading research project.
