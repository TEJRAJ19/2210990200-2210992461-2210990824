"""
Main Execution Script
Runs the complete quantitative trading pipeline.
"""

import sys
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(__file__).replace('main.py', 'src'))

from pathlib import Path

def main():
    """
    Run the complete quantitative trading pipeline.
    
    Pipeline:
    1. Data Fetching & Generation
    2. Data Cleaning
    3. Data Merging
    4. Feature Engineering
    5. Regime Detection
    6. Trading Strategy & Backtesting
    7. ML Model Training
    8. ML-Enhanced Backtesting
    9. Trade Analysis
    """
    print("=" * 70)
    print("QUANTITATIVE TRADING STRATEGY DEVELOPMENT")
    print("=" * 70)
    print()
    
    # Import modules after path setup
    from src.config import (
        DATA_DIR, PLOTS_DIR, MODELS_DIR, RESULTS_DIR,
        FEATURES_DATA_PATH
    )
    
    # Ensure directories exist
    for dir_path in [DATA_DIR, PLOTS_DIR, MODELS_DIR, RESULTS_DIR]:
        dir_path.mkdir(exist_ok=True)
    
    # ============================================================
    # PART 1: DATA ACQUISITION AND ENGINEERING
    # ============================================================
    print("\n" + "=" * 70)
    print("PART 1: DATA ACQUISITION AND ENGINEERING")
    print("=" * 70)
    
    # Task 1.1: Data Fetching
    print("\n>>> Task 1.1: Data Fetching")
    from src.data_fetcher import fetch_and_save_all_data
    spot_df, futures_df, options_df = fetch_and_save_all_data()
    
    # Task 1.2: Data Cleaning
    print("\n>>> Task 1.2: Data Cleaning")
    from src.data_cleaner import clean_all_data
    spot_df, futures_df, options_df = clean_all_data()
    
    # Task 1.3: Data Merging
    print("\n>>> Task 1.3: Data Merging")
    from src.data_merger import merge_all_data
    merged_df = merge_all_data()
    
    # ============================================================
    # PART 2: FEATURE ENGINEERING
    # ============================================================
    print("\n" + "=" * 70)
    print("PART 2: FEATURE ENGINEERING")
    print("=" * 70)
    
    from src.feature_engineering import engineer_all_features
    features_df = engineer_all_features()
    
    # ============================================================
    # PART 3: REGIME DETECTION
    # ============================================================
    print("\n" + "=" * 70)
    print("PART 3: REGIME DETECTION")
    print("=" * 70)
    
    from src.regime_detection import detect_regimes_and_visualize
    features_df, regime_detector = detect_regimes_and_visualize()
    
    # ============================================================
    # PART 4: TRADING STRATEGY
    # ============================================================
    print("\n" + "=" * 70)
    print("PART 4: TRADING STRATEGY")
    print("=" * 70)
    
    import pandas as pd
    features_df = pd.read_csv(FEATURES_DATA_PATH)
    features_df['datetime'] = pd.to_datetime(features_df['datetime'])
    
    from src.strategy import run_strategy
    from src.backtester import run_full_backtest, split_train_test
    
    # Task 4.1 & 4.2: Strategy Implementation & Backtesting
    print("\n>>> Task 4.1 & 4.2: Strategy Implementation & Backtesting")
    results_df, trades_df, strategy = run_strategy(features_df, use_regime_filter=True)
    
    # Run full backtest with train/test split
    from src.backtester import (
        calculate_metrics, plot_equity_curve, 
        plot_trade_distribution, generate_backtest_report
    )
    
    train_df, test_df = split_train_test(features_df)
    
    print("\n--- Training Period ---")
    train_results, train_trades, _ = run_strategy(train_df, use_regime_filter=True)
    train_equity = train_results['cumulative_pnl'] + 100000
    train_metrics = calculate_metrics(train_trades, train_equity)
    generate_backtest_report(train_metrics, train_trades, "Training")
    
    print("\n--- Testing Period ---")
    test_results, test_trades, _ = run_strategy(test_df, use_regime_filter=True)
    test_equity = test_results['cumulative_pnl'] + 100000
    test_metrics = calculate_metrics(test_trades, test_equity)
    generate_backtest_report(test_metrics, test_trades, "Testing")
    
    print("\n--- Full Period ---")
    full_results, full_trades, _ = run_strategy(features_df, use_regime_filter=True)
    full_equity = full_results['cumulative_pnl'] + 100000
    full_metrics = calculate_metrics(full_trades, full_equity)
    generate_backtest_report(full_metrics, full_trades, "Full")
    
    # Generate plots
    plot_equity_curve(full_results, "Equity Curve - Full Period")
    plot_trade_distribution(full_trades)
    
    # ============================================================
    # PART 5: MACHINE LEARNING ENHANCEMENT
    # ============================================================
    print("\n" + "=" * 70)
    print("PART 5: MACHINE LEARNING ENHANCEMENT")
    print("=" * 70)
    
    from src.ml_models import train_all_models
    
    if len(full_trades) >= 50:
        xgb_model, lstm_model = train_all_models(features_df, full_trades)
        
        # ML-Enhanced Backtesting
        print("\n>>> Task 5.3: ML-Enhanced Backtest")
        from src.ml_backtester import run_ml_enhanced_backtest
        ml_results = run_ml_enhanced_backtest(features_df, xgb_model, lstm_model)
    else:
        print(f"Not enough trades ({len(full_trades)}) for ML training. Need at least 50.")
        xgb_model, lstm_model = None, None
    
    # ============================================================
    # PART 6: HIGH-PERFORMANCE TRADE ANALYSIS
    # ============================================================
    print("\n" + "=" * 70)
    print("PART 6: HIGH-PERFORMANCE TRADE ANALYSIS")
    print("=" * 70)
    
    from src.trade_analysis import run_full_trade_analysis
    
    if len(full_trades) > 0:
        outliers, analysis = run_full_trade_analysis(features_df, full_trades)
        
        # Task 6.3: Insights Summary
        print("\n>>> Task 6.3: Generating Insights Summary")
        from src.insights_summary import run_insights_summary
        insights, insights_report = run_insights_summary(features_df, full_trades)
    
    # ============================================================
    # FINAL SUMMARY
    # ============================================================
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETE!")
    print("=" * 70)
    print("\nDeliverables generated:")
    print(f"  [DATA] Data files in: {DATA_DIR}")
    print(f"  [PLOTS] Plots in: {PLOTS_DIR}")
    print(f"  [MODELS] Models in: {MODELS_DIR}")
    print(f"  [RESULTS] Results in: {RESULTS_DIR}")
    print("\nKey files:")
    print("  - nifty_spot_5min.csv")
    print("  - nifty_futures_5min.csv")
    print("  - nifty_options_5min.csv")
    print("  - nifty_merged_5min.csv")
    print("  - nifty_features_5min.csv")
    print("  - data_cleaning_report.txt")
    print("  - regime_overlay.png")
    print("  - transition_matrix.png")
    print("  - equity_curve_full.png")
    print("  - backtest_report_*.txt")
    print("  - strategy_comparison.csv")
    print("  - trade_analysis_report.txt")
    print("=" * 70)


if __name__ == "__main__":
    main()
