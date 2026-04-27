"""
Machine Learning Models Module
Implements XGBoost and LSTM for trade profitability prediction.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
import joblib
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from config import (
    FEATURES_DATA_PATH, MODELS_DIR, RESULTS_DIR,
    LSTM_SEQUENCE_LENGTH, ML_CONFIDENCE_THRESHOLD, TRAIN_RATIO
)


class TradePredictor:
    """
    Base class for trade profitability prediction.
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.model = None
    
    def get_features_and_target(self, df: pd.DataFrame, trades_df: pd.DataFrame):
        """
        Prepare features and target from trade data.
        Target: 1 if trade is profitable, 0 otherwise.
        """
        # Features to use for prediction
        self.feature_columns = [
            # Engineered features
            'ema_5', 'ema_15', 'ema_diff',
            'avg_iv', 'iv_spread', 'pcr_oi', 'pcr_volume',
            'atm_call_delta', 'atm_call_gamma', 'atm_call_vega',
            'atm_put_delta', 'atm_put_gamma',
            'futures_basis', 'spot_returns',
            'delta_neutral_ratio', 'gamma_exposure',
            'atr_14', 'volatility_20',
            # Time-based features
            'hour', 'minute', 'day_of_week',
            # Lag features
            'spot_return_lag_1', 'spot_return_lag_2', 'spot_return_lag_3',
            # Regime
            'regime'
        ]
        
        # Filter to existing columns
        self.feature_columns = [col for col in self.feature_columns if col in df.columns]
        
        # Match trades to candles and create training data
        X_list = []
        y_list = []
        
        for _, trade in trades_df.iterrows():
            entry_time = trade['entry_time']
            
            # Find the candle at entry time
            candle_mask = df['datetime'] == entry_time
            if not candle_mask.any():
                continue
            
            candle_idx = df[candle_mask].index[0]
            
            # Get features at entry time (use previous candle to avoid lookahead)
            if candle_idx > 0:
                feature_row = df.loc[candle_idx - 1, self.feature_columns]
                X_list.append(feature_row.values)
                
                # Target: 1 if profitable, 0 otherwise
                y_list.append(1 if trade['pnl'] > 0 else 0)
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        return X, y


class XGBoostPredictor(TradePredictor):
    """
    XGBoost-based trade profitability predictor.
    Uses time-series cross-validation.
    """
    
    def __init__(self, n_estimators=200, max_depth=6, learning_rate=0.05):
        super().__init__()
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.model = None
        self.cv_scores = []
    
    def train(self, X: np.ndarray, y: np.ndarray, n_splits: int = 5):
        """
        Train XGBoost with time-series cross-validation.
        """
        print("Training XGBoost model...")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        self.cv_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_scaled)):
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Create model for this fold
            model = xgb.XGBClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                random_state=42,
                eval_metric='logloss',
                early_stopping_rounds=15,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                min_child_weight=3,
                scale_pos_weight=1.2
            )
            
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], 
                     verbose=False)
            
            # Validate
            y_pred = model.predict(X_val)
            accuracy = accuracy_score(y_val, y_pred)
            self.cv_scores.append(accuracy)
            print(f"  Fold {fold + 1}: Accuracy = {accuracy:.4f}")
        
        print(f"  Mean CV Accuracy: {np.mean(self.cv_scores):.4f} (+/- {np.std(self.cv_scores):.4f})")
        
        # Train final model on all data
        self.model = xgb.XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=42,
            eval_metric='logloss',
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            min_child_weight=3
        )
        self.model.fit(X_scaled, y)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance."""
        importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        return importance
    
    def save(self, filepath=None):
        """Save model."""
        if filepath is None:
            filepath = MODELS_DIR / "xgboost_model.joblib"
        
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'cv_scores': self.cv_scores
        }, filepath)
        print(f"âœ“ Saved XGBoost model to {filepath}")
    
    def load(self, filepath=None):
        """Load model."""
        if filepath is None:
            filepath = MODELS_DIR / "xgboost_model.joblib"
        
        data = joblib.load(filepath)
        self.model = data['model']
        self.scaler = data['scaler']
        self.feature_columns = data['feature_columns']
        self.cv_scores = data['cv_scores']
        return self


class LSTMPredictor(TradePredictor):
    """
    LSTM-based trade profitability predictor.
    Uses sequence of last N candles.
    """
    
    def __init__(self, sequence_length: int = LSTM_SEQUENCE_LENGTH, 
                 lstm_units: int = 128, dropout: float = 0.4):
        super().__init__()
        self.sequence_length = sequence_length
        self.lstm_units = lstm_units
        self.dropout = dropout
        self.model = None
        self.history = None
    
    def _build_model(self, n_features: int):
        """Build enhanced LSTM model architecture."""
        try:
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional
            from tensorflow.keras.optimizers import Adam
            from tensorflow.keras.regularizers import l2
        except ImportError:
            print("TensorFlow not installed. LSTM model unavailable.")
            return None
        
        model = Sequential([
            # First Bidirectional LSTM layer
            Bidirectional(LSTM(self.lstm_units, return_sequences=True, 
                              kernel_regularizer=l2(0.01)),
                         input_shape=(self.sequence_length, n_features)),
            BatchNormalization(),
            Dropout(self.dropout),
            
            # Second LSTM layer
            LSTM(64, kernel_regularizer=l2(0.01)),
            BatchNormalization(),
            Dropout(self.dropout),
            
            # Dense layers
            Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.0005),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def prepare_sequences(self, df: pd.DataFrame, trades_df: pd.DataFrame):
        """
        Prepare sequences of candles before each trade.
        """
        # Features to use
        self.feature_columns = [
            'spot_close', 'spot_volume', 'ema_5', 'ema_15',
            'avg_iv', 'pcr_oi', 'spot_returns',
            'atm_call_delta', 'atm_call_gamma',
            'futures_basis', 'regime'
        ]
        self.feature_columns = [col for col in self.feature_columns if col in df.columns]
        
        X_sequences = []
        y_list = []
        
        df_reset = df.reset_index(drop=True)
        
        for _, trade in trades_df.iterrows():
            entry_time = trade['entry_time']
            
            # Find entry candle index
            candle_mask = df_reset['datetime'] == entry_time
            if not candle_mask.any():
                continue
            
            candle_idx = df_reset[candle_mask].index[0]
            
            # Need enough history for sequence
            if candle_idx >= self.sequence_length:
                start_idx = candle_idx - self.sequence_length
                sequence = df_reset.loc[start_idx:candle_idx - 1, self.feature_columns].values
                
                if len(sequence) == self.sequence_length:
                    X_sequences.append(sequence)
                    y_list.append(1 if trade['pnl'] > 0 else 0)
        
        X = np.array(X_sequences)
        y = np.array(y_list)
        
        return X, y
    
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 50, 
             batch_size: int = 32, validation_split: float = 0.2):
        """Train LSTM model."""
        print("Training LSTM model...")
        
        # Scale features
        n_samples, n_steps, n_features = X.shape
        X_reshaped = X.reshape(-1, n_features)
        X_scaled = self.scaler.fit_transform(X_reshaped)
        X_scaled = X_scaled.reshape(n_samples, n_steps, n_features)
        
        # Build model
        self.model = self._build_model(n_features)
        if self.model is None:
            print("LSTM model could not be built (TensorFlow not available)")
            return self
        
        # Train
        self.history = self.model.fit(
            X_scaled, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=1
        )
        
        # Evaluate
        val_idx = int(len(X_scaled) * (1 - validation_split))
        X_val = X_scaled[val_idx:]
        y_val = y[val_idx:]
        
        y_pred = (self.model.predict(X_val) > 0.5).astype(int).flatten()
        accuracy = accuracy_score(y_val, y_pred)
        print(f"\nValidation Accuracy: {accuracy:.4f}")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        if self.model is None:
            return np.zeros(len(X))
        
        n_samples, n_steps, n_features = X.shape
        X_reshaped = X.reshape(-1, n_features)
        X_scaled = self.scaler.transform(X_reshaped)
        X_scaled = X_scaled.reshape(n_samples, n_steps, n_features)
        
        return (self.model.predict(X_scaled) > 0.5).astype(int).flatten()
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities."""
        if self.model is None:
            return np.zeros(len(X))
        
        n_samples, n_steps, n_features = X.shape
        X_reshaped = X.reshape(-1, n_features)
        X_scaled = self.scaler.transform(X_reshaped)
        X_scaled = X_scaled.reshape(n_samples, n_steps, n_features)
        
        return self.model.predict(X_scaled).flatten()
    
    def save(self, filepath=None):
        """Save model."""
        if filepath is None:
            filepath = MODELS_DIR / "lstm_model"
        
        if self.model is not None:
            self.model.save(f"{filepath}.keras")
            joblib.dump({
                'scaler': self.scaler,
                'feature_columns': self.feature_columns,
                'sequence_length': self.sequence_length
            }, f"{filepath}_meta.joblib")
            print(f"âœ“ Saved LSTM model to {filepath}")
    
    def load(self, filepath=None):
        """Load model."""
        if filepath is None:
            filepath = MODELS_DIR / "lstm_model"
        
        try:
            from tensorflow.keras.models import load_model
            self.model = load_model(f"{filepath}.keras")
            
            meta = joblib.load(f"{filepath}_meta.joblib")
            self.scaler = meta['scaler']
            self.feature_columns = meta['feature_columns']
            self.sequence_length = meta['sequence_length']
        except Exception as e:
            print(f"Could not load LSTM model: {e}")
        
        return self


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray = None) -> dict:
    """
    Evaluate model performance.
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0)
    }
    
    if y_proba is not None:
        try:
            metrics['auc_roc'] = roc_auc_score(y_true, y_proba)
        except:
            metrics['auc_roc'] = 0.5
    
    return metrics


def train_all_models(df: pd.DataFrame, trades_df: pd.DataFrame):
    """
    Train both XGBoost and LSTM models.
    """
    print("=" * 60)
    print("ML MODEL TRAINING PIPELINE")
    print("=" * 60)
    
    if len(trades_df) < 50:
        print("Not enough trades for ML training. Need at least 50 trades.")
        return None, None
    
    results = {}
    
    # Train XGBoost
    print("\n" + "-" * 40)
    print("XGBOOST MODEL")
    print("-" * 40)
    
    xgb_model = XGBoostPredictor()
    X_xgb, y_xgb = xgb_model.get_features_and_target(df, trades_df)
    
    if len(X_xgb) > 10:
        xgb_model.train(X_xgb, y_xgb)
        xgb_model.save()
        
        # Get predictions for evaluation
        y_pred_xgb = xgb_model.predict(X_xgb)
        y_proba_xgb = xgb_model.predict_proba(X_xgb)
        
        results['xgboost'] = {
            'model': xgb_model,
            'metrics': evaluate_model(y_xgb, y_pred_xgb, y_proba_xgb),
            'feature_importance': xgb_model.get_feature_importance()
        }
        
        print("\nXGBoost Metrics:")
        for k, v in results['xgboost']['metrics'].items():
            print(f"  {k}: {v:.4f}")
    
    # Train LSTM
    print("\n" + "-" * 40)
    print("LSTM MODEL")
    print("-" * 40)
    
    try:
        lstm_model = LSTMPredictor()
        X_lstm, y_lstm = lstm_model.prepare_sequences(df, trades_df)
        
        if len(X_lstm) > 10:
            lstm_model.train(X_lstm, y_lstm, epochs=30)
            lstm_model.save()
            
            y_pred_lstm = lstm_model.predict(X_lstm)
            y_proba_lstm = lstm_model.predict_proba(X_lstm)
            
            results['lstm'] = {
                'model': lstm_model,
                'metrics': evaluate_model(y_lstm, y_pred_lstm, y_proba_lstm)
            }
            
            print("\nLSTM Metrics:")
            for k, v in results['lstm']['metrics'].items():
                print(f"  {k}: {v:.4f}")
    except Exception as e:
        print(f"LSTM training failed: {e}")
        results['lstm'] = None
    
    print("\n" + "=" * 60)
    print("ML TRAINING COMPLETE")
    print("=" * 60)
    
    return results.get('xgboost', {}).get('model'), results.get('lstm', {}).get('model')


if __name__ == "__main__":
    # This is a placeholder - actual training happens through main.py
    print("ML Models module loaded successfully")
