"""
Enhanced ML Models for Higher Accuracy
Targets 70%+ accuracy using ensemble methods, SMOTE, and feature selection.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.feature_selection import SelectKBest, f_classif, RFE
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings('ignore')

from config import FEATURES_DATA_PATH, MODELS_DIR, RESULTS_DIR


class EnhancedPredictor:
    """
    Enhanced ML predictor with ensemble methods and class balancing.
    Targets 70%+ accuracy.
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.selected_features = None
        self.ensemble_model = None
        self.cv_scores = []
    
    def get_features_and_target(self, df: pd.DataFrame, trades_df: pd.DataFrame):
        """Prepare features and target with additional engineered features."""
        # Extended feature set for better prediction
        self.feature_columns = [
            # Core features
            'ema_5', 'ema_15', 'ema_diff',
            'avg_iv', 'iv_spread', 'pcr_oi', 'pcr_volume',
            'atm_call_delta', 'atm_call_gamma', 'atm_call_vega', 'atm_call_theta',
            'atm_put_delta', 'atm_put_gamma', 'atm_put_vega',
            'futures_basis', 'spot_returns',
            'delta_neutral_ratio', 'gamma_exposure',
            'atr_14', 'volatility_20',
            # Time features
            'hour', 'minute', 'day_of_week',
            # Lag features
            'spot_return_lag_1', 'spot_return_lag_2', 'spot_return_lag_3',
            'volume_lag_1', 'volume_lag_2',
            # Regime
            'regime',
            # Additional derived
            'spot_high', 'spot_low', 'spot_close', 'spot_volume'
        ]
        
        self.feature_columns = [col for col in self.feature_columns if col in df.columns]
        
        X_list = []
        y_list = []
        
        for _, trade in trades_df.iterrows():
            entry_time = trade['entry_time']
            candle_mask = df['datetime'] == entry_time
            
            if not candle_mask.any():
                continue
            
            candle_idx = df[candle_mask].index[0]
            
            if candle_idx > 2:  # Need some history
                # Get features from previous candles
                feature_row = df.loc[candle_idx - 1, self.feature_columns].values
                
                # Add momentum features
                prev_returns = df.loc[candle_idx-3:candle_idx-1, 'spot_returns'].values
                if len(prev_returns) == 3:
                    momentum = np.sum(prev_returns)
                    avg_return = np.mean(prev_returns)
                    return_std = np.std(prev_returns) if len(prev_returns) > 1 else 0
                    
                    feature_row = np.append(feature_row, [momentum, avg_return, return_std])
                else:
                    feature_row = np.append(feature_row, [0, 0, 0])
                
                X_list.append(feature_row)
                y_list.append(1 if trade['pnl'] > 0 else 0)
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        # Update feature columns with new features
        self.feature_columns = self.feature_columns + ['momentum', 'avg_return', 'return_std']
        
        return X, y
    
    def apply_smote(self, X, y):
        """Apply SMOTE for class balancing with more aggressive oversampling."""
        try:
            from imblearn.over_sampling import SMOTE
            # More aggressive oversampling
            smote = SMOTE(random_state=42, k_neighbors=min(5, sum(y==1)-1, sum(y==0)-1), 
                         sampling_strategy=1.0)
            X_balanced, y_balanced = smote.fit_resample(X, y)
            print(f"  SMOTE: {len(X)} -> {len(X_balanced)} samples")
            return X_balanced, y_balanced
        except:
            print("  SMOTE not available, using class weights instead")
            return X, y
    
    def select_features(self, X, y, n_features=15):
        """Select top features using statistical tests."""
        selector = SelectKBest(f_classif, k=min(n_features, X.shape[1]))
        X_selected = selector.fit_transform(X, y)
        
        # Get selected feature indices
        selected_mask = selector.get_support()
        self.selected_features = [self.feature_columns[i] for i in range(len(selected_mask)) if selected_mask[i]]
        
        print(f"  Selected {len(self.selected_features)} features")
        return X_selected, selector
    
    def train(self, X: np.ndarray, y: np.ndarray, n_splits: int = 10):
        """Train ensemble model with advanced techniques."""
        print("=" * 60)
        print("ENHANCED MODEL TRAINING (Target: 70% Accuracy)")
        print("=" * 60)
        
        # Handle NaN values
        X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Feature selection - use fewer features to reduce overfitting
        print("\n1. Feature Selection")
        X_selected, self.selector = self.select_features(X_scaled, y, n_features=15)
        
        # Class balancing
        print("\n2. Class Balancing")
        try:
            X_balanced, y_balanced = self.apply_smote(X_selected, y)
        except:
            X_balanced, y_balanced = X_selected, y
        
        # Create ensemble of models
        print("\n3. Building Ensemble Model")
        
        # Individual models with STRONG regularization to avoid overfitting
        xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=3,  # Simpler trees
            learning_rate=0.1,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_alpha=1.0,  # Strong L1
            reg_lambda=5.0,  # Strong L2
            min_child_weight=10,
            scale_pos_weight=1.2,
            random_state=42,
            eval_metric='logloss'
        )
        
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=4,  # Simpler
            min_samples_split=10,
            min_samples_leaf=5,
            max_features='sqrt',
            class_weight='balanced',
            random_state=42
        )
        
        gb_model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            subsample=0.7,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42
        )
        
        # Voting ensemble
        self.ensemble_model = VotingClassifier(
            estimators=[
                ('xgb', xgb_model),
                ('rf', rf_model),
                ('gb', gb_model)
            ],
            voting='soft',
            weights=[2, 1, 1]  # Weight XGBoost higher
        )
        
        # Stratified cross-validation for consistent results
        print("\n4. Cross-Validation (Stratified)")
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        self.cv_scores = []
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_balanced, y_balanced)):
            X_train, X_val = X_balanced[train_idx], X_balanced[val_idx]
            y_train, y_val = y_balanced[train_idx], y_balanced[val_idx]
            
            # Clone and train ensemble
            from sklearn.base import clone
            fold_model = clone(self.ensemble_model)
            fold_model.fit(X_train, y_train)
            
            y_pred = fold_model.predict(X_val)
            accuracy = accuracy_score(y_val, y_pred)
            self.cv_scores.append(accuracy)
            print(f"  Fold {fold + 1}: Accuracy = {accuracy:.1%}")
        
        mean_cv = np.mean(self.cv_scores)
        print(f"\n  >>> MEAN CV ACCURACY: {mean_cv:.1%} (+/- {np.std(self.cv_scores):.1%})")
        
        # Train final model on all balanced data
        print("\n5. Training Final Model")
        self.ensemble_model.fit(X_balanced, y_balanced)
        
        # Store the selector for prediction
        self.X_balanced_shape = X_balanced.shape
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
        X_scaled = self.scaler.transform(X)
        X_selected = self.selector.transform(X_scaled)
        return self.ensemble_model.predict(X_selected)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities."""
        X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
        X_scaled = self.scaler.transform(X)
        X_selected = self.selector.transform(X_scaled)
        return self.ensemble_model.predict_proba(X_selected)[:, 1]
    
    def save(self, filepath=None):
        """Save model."""
        if filepath is None:
            filepath = MODELS_DIR / "enhanced_model.joblib"
        
        joblib.dump({
            'model': self.ensemble_model,
            'scaler': self.scaler,
            'selector': self.selector,
            'feature_columns': self.feature_columns,
            'selected_features': self.selected_features,
            'cv_scores': self.cv_scores
        }, filepath)
        print(f"[OK] Saved enhanced model to {filepath}")
    
    def load(self, filepath=None):
        """Load model."""
        if filepath is None:
            filepath = MODELS_DIR / "enhanced_model.joblib"
        
        data = joblib.load(filepath)
        self.ensemble_model = data['model']
        self.scaler = data['scaler']
        self.selector = data['selector']
        self.feature_columns = data['feature_columns']
        self.selected_features = data['selected_features']
        self.cv_scores = data['cv_scores']
        return self


def train_enhanced_model(df: pd.DataFrame, trades_df: pd.DataFrame):
    """Train the enhanced model and report results."""
    
    # Create and train
    model = EnhancedPredictor()
    X, y = model.get_features_and_target(df, trades_df)
    
    print(f"\nDataset: {len(X)} samples")
    print(f"Class distribution: {sum(y==1)} profitable, {sum(y==0)} unprofitable")
    
    model.train(X, y)
    model.save()
    
    # Final evaluation on training data (for reference)
    y_pred = model.predict(X)
    train_acc = accuracy_score(y, y_pred)
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  Cross-Validation Accuracy: {np.mean(model.cv_scores):.1%}")
    print(f"  Training Accuracy: {train_acc:.1%}")
    print(f"  Selected Features: {model.selected_features[:5]}...")
    print("=" * 60)
    
    return model


if __name__ == "__main__":
    from strategy import run_strategy
    
    print("Loading data...")
    df = pd.read_csv(FEATURES_DATA_PATH)
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    print("Running strategy...")
    _, trades_df, _ = run_strategy(df, use_regime_filter=True)
    
    print(f"\nTraining enhanced model on {len(trades_df)} trades...")
    model = train_enhanced_model(df, trades_df)
