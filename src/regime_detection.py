"""
Regime Detection Module
Implements Hidden Markov Model for market regime classification.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

from config import (
    FEATURES_DATA_PATH, MODELS_DIR, PLOTS_DIR,
    HMM_N_STATES, HMM_TRAIN_RATIO,
    REGIME_UPTREND, REGIME_DOWNTREND, REGIME_SIDEWAYS
)


class RegimeDetector:
    """
    Hidden Markov Model based regime detector.
    Classifies market into 3 regimes: Uptrend, Downtrend, Sideways.
    """
    
    def __init__(self, n_states: int = HMM_N_STATES):
        self.n_states = n_states
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.regime_mapping = None  # Will be set after fitting
    
    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Prepare features for HMM training.
        Uses options-based features as specified in requirements.
        """
        self.feature_columns = [
            'avg_iv',           # Average IV
            'iv_spread',        # IV Spread
            'pcr_oi',           # PCR (OI-based)
            'atm_call_delta',   # ATM Delta
            'atm_call_gamma',   # ATM Gamma
            'atm_call_vega',    # ATM Vega
            'futures_basis',    # Futures Basis
            'spot_returns'      # Spot Returns
        ]
        
        # Ensure all features exist
        missing = [col for col in self.feature_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Missing features: {missing}")
        
        # Extract features
        X = df[self.feature_columns].copy()
        
        # Handle inf and nan
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(method='ffill').fillna(0)
        
        return X.values
    
    def fit(self, df: pd.DataFrame, train_ratio: float = HMM_TRAIN_RATIO):
        """
        Fit HMM model on training data.
        """
        print("Fitting HMM Regime Detector...")
        
        # Prepare features
        X = self.prepare_features(df)
        
        # Split train/test
        train_size = int(len(X) * train_ratio)
        X_train = X[:train_size]
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Initialize and fit HMM
        self.model = hmm.GaussianHMM(
            n_components=self.n_states,
            covariance_type="full",
            n_iter=100,
            random_state=42
        )
        
        self.model.fit(X_train_scaled)
        print(f"âœ“ HMM fitted on {train_size} training samples")
        
        # Predict regimes on training data
        train_regimes = self.model.predict(X_train_scaled)
        
        # Map regimes to Uptrend/Downtrend/Sideways based on returns
        self._map_regimes(df[:train_size], train_regimes)
        
        return self
    
    def _map_regimes(self, df: pd.DataFrame, regimes: np.ndarray):
        """
        Map HMM states to meaningful regime labels based on average returns.
        """
        df_temp = df.copy()
        df_temp['raw_regime'] = regimes
        
        # Calculate average return per regime
        regime_returns = df_temp.groupby('raw_regime')['spot_returns'].mean()
        
        # Sort regimes by average return
        sorted_regimes = regime_returns.sort_values()
        
        # Map: lowest return = downtrend, highest = uptrend, middle = sideways
        self.regime_mapping = {}
        regime_labels = list(sorted_regimes.index)
        
        if len(regime_labels) == 3:
            self.regime_mapping[regime_labels[0]] = REGIME_DOWNTREND  # -1
            self.regime_mapping[regime_labels[1]] = REGIME_SIDEWAYS   # 0
            self.regime_mapping[regime_labels[2]] = REGIME_UPTREND    # +1
        else:
            # Fallback for different number of states
            for i, label in enumerate(regime_labels):
                if i == 0:
                    self.regime_mapping[label] = REGIME_DOWNTREND
                elif i == len(regime_labels) - 1:
                    self.regime_mapping[label] = REGIME_UPTREND
                else:
                    self.regime_mapping[label] = REGIME_SIDEWAYS
        
        print(f"Regime mapping: {self.regime_mapping}")
        print(f"Average returns per regime: {regime_returns.to_dict()}")
    
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict regimes for entire dataset.
        """
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        X = self.prepare_features(df)
        X_scaled = self.scaler.transform(X)
        
        raw_regimes = self.model.predict(X_scaled)
        
        # Map to meaningful labels
        regimes = np.array([self.regime_mapping[r] for r in raw_regimes])
        
        return regimes
    
    def get_transition_matrix(self) -> np.ndarray:
        """Get the transition probability matrix."""
        if self.model is None:
            raise ValueError("Model not fitted.")
        return self.model.transmat_
    
    def get_state_means(self) -> np.ndarray:
        """Get the mean values for each state."""
        if self.model is None:
            raise ValueError("Model not fitted.")
        return self.model.means_
    
    def save(self, filepath=None):
        """Save the model to disk."""
        if filepath is None:
            filepath = MODELS_DIR / "regime_detector.joblib"
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'regime_mapping': self.regime_mapping,
            'n_states': self.n_states
        }
        joblib.dump(model_data, filepath)
        print(f"âœ“ Model saved to {filepath}")
    
    def load(self, filepath=None):
        """Load model from disk."""
        if filepath is None:
            filepath = MODELS_DIR / "regime_detector.joblib"
        
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_columns = model_data['feature_columns']
        self.regime_mapping = model_data['regime_mapping']
        self.n_states = model_data['n_states']
        print(f"âœ“ Model loaded from {filepath}")
        return self


def plot_regime_overlay(df: pd.DataFrame, save_path=None):
    """
    Create price chart with color-coded regime overlay.
    """
    fig, ax = plt.subplots(figsize=(15, 7))
    
    # Plot price
    ax.plot(df['datetime'], df['spot_close'], color='black', linewidth=0.8, alpha=0.7)
    
    # Color regions by regime
    regime_colors = {
        REGIME_UPTREND: 'green',
        REGIME_DOWNTREND: 'red',
        REGIME_SIDEWAYS: 'yellow'
    }
    
    for regime, color in regime_colors.items():
        mask = df['regime'] == regime
        ax.fill_between(
            df['datetime'],
            df['spot_close'].min(),
            df['spot_close'].max(),
            where=mask,
            alpha=0.3,
            color=color,
            label=f"Regime {regime}"
        )
    
    ax.set_xlabel('Date')
    ax.set_ylabel('NIFTY 50 Price')
    ax.set_title('NIFTY 50 with Regime Overlay')
    ax.legend(loc='upper left')
    
    plt.tight_layout()
    
    if save_path is None:
        save_path = PLOTS_DIR / "regime_overlay.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"âœ“ Saved regime overlay plot to {save_path}")


def plot_transition_matrix(transition_matrix: np.ndarray, save_path=None):
    """
    Create heatmap of transition probabilities.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    labels = ['Downtrend (-1)', 'Sideways (0)', 'Uptrend (+1)']
    
    sns.heatmap(
        transition_matrix,
        annot=True,
        fmt='.3f',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels,
        ax=ax
    )
    
    ax.set_xlabel('To State')
    ax.set_ylabel('From State')
    ax.set_title('Regime Transition Probability Matrix')
    
    plt.tight_layout()
    
    if save_path is None:
        save_path = PLOTS_DIR / "transition_matrix.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"âœ“ Saved transition matrix plot to {save_path}")


def plot_regime_statistics(df: pd.DataFrame, save_path=None):
    """
    Create box plots of IV and Greeks distribution per regime.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    features = ['avg_iv', 'iv_spread', 'pcr_oi', 
                'atm_call_delta', 'atm_call_gamma', 'spot_returns']
    titles = ['Average IV', 'IV Spread', 'PCR (OI)', 
              'ATM Call Delta', 'ATM Call Gamma', 'Spot Returns']
    
    regime_order = [REGIME_DOWNTREND, REGIME_SIDEWAYS, REGIME_UPTREND]
    
    for idx, (feature, title) in enumerate(zip(features, titles)):
        ax = axes[idx // 3, idx % 3]
        
        data = [df[df['regime'] == r][feature].dropna() for r in regime_order]
        bp = ax.boxplot(data, labels=['Down', 'Side', 'Up'])
        
        ax.set_title(title)
        ax.set_xlabel('Regime')
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Feature Distribution by Regime', fontsize=14)
    plt.tight_layout()
    
    if save_path is None:
        save_path = PLOTS_DIR / "regime_statistics.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"âœ“ Saved regime statistics plot to {save_path}")


def plot_duration_histogram(df: pd.DataFrame, save_path=None):
    """
    Create histogram of regime durations.
    """
    # Calculate regime durations
    df_temp = df.copy()
    df_temp['regime_change'] = df_temp['regime'] != df_temp['regime'].shift(1)
    df_temp['regime_group'] = df_temp['regime_change'].cumsum()
    
    durations = df_temp.groupby(['regime_group', 'regime']).size().reset_index(name='duration')
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    regime_names = {
        REGIME_DOWNTREND: 'Downtrend',
        REGIME_SIDEWAYS: 'Sideways',
        REGIME_UPTREND: 'Uptrend'
    }
    
    colors = {
        REGIME_DOWNTREND: 'red',
        REGIME_SIDEWAYS: 'yellow',
        REGIME_UPTREND: 'green'
    }
    
    for idx, regime in enumerate([REGIME_DOWNTREND, REGIME_SIDEWAYS, REGIME_UPTREND]):
        ax = axes[idx]
        regime_durations = durations[durations['regime'] == regime]['duration']
        
        ax.hist(regime_durations, bins=30, color=colors[regime], alpha=0.7, edgecolor='black')
        ax.set_title(f'{regime_names[regime]} Duration')
        ax.set_xlabel('Duration (candles)')
        ax.set_ylabel('Frequency')
        ax.axvline(regime_durations.mean(), color='black', linestyle='--', 
                   label=f'Mean: {regime_durations.mean():.1f}')
        ax.legend()
    
    plt.suptitle('Regime Duration Distribution', fontsize=14)
    plt.tight_layout()
    
    if save_path is None:
        save_path = PLOTS_DIR / "duration_histogram.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"âœ“ Saved duration histogram to {save_path}")


def detect_regimes_and_visualize(input_path=FEATURES_DATA_PATH):
    """
    Main function to detect regimes and create all visualizations.
    """
    print("=" * 60)
    print("Regime Detection Pipeline")
    print("=" * 60)
    
    # Load features
    print("\nLoading features...")
    df = pd.read_csv(input_path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    print(f"Loaded {len(df)} rows")
    
    # Initialize and fit detector
    detector = RegimeDetector(n_states=3)
    detector.fit(df)
    
    # Predict regimes
    print("\nPredicting regimes...")
    regimes = detector.predict(df)
    df['regime'] = regimes
    
    # Print regime distribution
    print("\nRegime Distribution:")
    regime_counts = df['regime'].value_counts().sort_index()
    for regime, count in regime_counts.items():
        pct = count / len(df) * 100
        name = {REGIME_UPTREND: 'Uptrend', REGIME_DOWNTREND: 'Downtrend', 
                REGIME_SIDEWAYS: 'Sideways'}[regime]
        print(f"  {name} ({regime}): {count} ({pct:.1f}%)")
    
    # Save updated data
    df.to_csv(input_path, index=False)
    print(f"\nâœ“ Saved data with regimes to {input_path}")
    
    # Save model
    detector.save()
    
    # Create visualizations
    print("\nCreating visualizations...")
    plot_regime_overlay(df)
    plot_transition_matrix(detector.get_transition_matrix())
    plot_regime_statistics(df)
    plot_duration_histogram(df)
    
    print("\n" + "=" * 60)
    print("REGIME DETECTION COMPLETE")
    print("=" * 60)
    
    return df, detector


if __name__ == "__main__":
    detect_regimes_and_visualize()
