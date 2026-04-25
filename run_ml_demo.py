"""
ML Model Demo Script
Run this to demonstrate the XGBoost model for trade prediction.
"""

import sys
sys.path.insert(0, 'src')

import pandas as pd
import numpy as np
from pathlib import Path

def demo_ml_model():
    """
    Demonstrate the ML model for trade prediction.
    """
    print("=" * 60)
    print("ML MODEL DEMONSTRATION")
    print("=" * 60)
    
    # 1. Load the trained model
    print("\n1. LOADING TRAINED MODEL")
    print("-" * 40)
    
    from config import MODELS_DIR, FEATURES_DATA_PATH, RESULTS_DIR
    from ml_models import XGBoostPredictor
    
    model_path = MODELS_DIR / "xgboost_model.joblib"
    
    if not model_path.exists():
        print("Model not found. Running training first...")
        # Run training if model doesn't exist
        from ml_models import train_all_models
        from strategy import run_strategy
        
        df = pd.read_csv(FEATURES_DATA_PATH)
        df['datetime'] = pd.to_datetime(df['datetime'])
        _, trades_df, _ = run_strategy(df, use_regime_filter=True)
        xgb_model, _ = train_all_models(df, trades_df)
    else:
        xgb_model = XGBoostPredictor()
        xgb_model.load(model_path)
        print(f"[OK] Model loaded from {model_path}")
    
    # 2. Show model details
    print("\n2. MODEL DETAILS")
    print("-" * 40)
    print(f"   Model type: XGBoost Classifier")
    print(f"   Features used: {len(xgb_model.feature_columns)}")
    print(f"   CV Accuracy: {np.mean(xgb_model.cv_scores)*100:.1f}%")
    
    # 3. Show feature importance
    print("\n3. TOP 10 FEATURE IMPORTANCE")
    print("-" * 40)
    importance = xgb_model.get_feature_importance()
    for i, row in importance.head(10).iterrows():
        bar = "#" * int(row['importance'] * 50)
        print(f"   {row['feature']:20s} {row['importance']:.3f} {bar}")
    
    # 4. Load test data and make predictions
    print("\n4. SAMPLE PREDICTIONS")
    print("-" * 40)
    
    df = pd.read_csv(FEATURES_DATA_PATH)
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    trades_path = RESULTS_DIR / "all_trades.csv"
    if trades_path.exists():
        trades_df = pd.read_csv(trades_path)
        trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
        
        # Get features for last 5 trades
        X, y = xgb_model.get_features_and_target(df, trades_df.tail(10))
        
        if len(X) > 0:
            predictions = xgb_model.predict(X)
            probabilities = xgb_model.predict_proba(X)
            
            print(f"   {'Actual':<10} {'Predicted':<12} {'Probability':<12} {'Correct'}")
            print("   " + "-" * 50)
            for i in range(min(5, len(X))):
                actual = "Profit" if y[i] == 1 else "Loss"
                pred = "Profit" if predictions[i] == 1 else "Loss"
                prob = probabilities[i]
                correct = "✓" if y[i] == predictions[i] else "✗"
                print(f"   {actual:<10} {pred:<12} {prob:.1%}          {correct}")
    
    # 5. How to use for new predictions
    print("\n5. HOW TO USE FOR NEW PREDICTIONS")
    print("-" * 40)
    print("""
    # Load the model
    from ml_models import XGBoostPredictor
    from config import MODELS_DIR
    
    model = XGBoostPredictor()
    model.load(MODELS_DIR / "xgboost_model.joblib")
    
    # Prepare your features (same columns as training)
    features = ['ema_5', 'ema_15', 'avg_iv', 'pcr_oi', ...]
    X = your_data[features].values
    
    # Get predictions
    predictions = model.predict(X)        # 0 = Loss, 1 = Profit
    probabilities = model.predict_proba(X)  # Probability of profit
    
    # Filter trades with high confidence
    high_confidence = probabilities > 0.6
    """)
    
    print("=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    demo_ml_model()
