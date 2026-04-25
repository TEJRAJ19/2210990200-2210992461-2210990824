"""
Feature Engineering Module
Creates EMA indicators, Options Greeks, and derived features.
"""

import pandas as pd
import numpy as np
from scipy.stats import norm
from tqdm import tqdm

from config import (
    MERGED_DATA_PATH, FEATURES_DATA_PATH,
    EMA_FAST, EMA_SLOW, RISK_FREE_RATE
)


def calculate_ema(series: pd.Series, period: int) -> pd.Series:
    """
    Calculate Exponential Moving Average.
    
    Args:
        series: Price series
        period: EMA period
    
    Returns:
        EMA series
    """
    return series.ewm(span=period, adjust=False).mean()


def calculate_greeks(
    spot: float,
    strike: float,
    time_to_expiry: float,
    volatility: float,
    option_type: str = 'call',
    risk_free_rate: float = RISK_FREE_RATE
) -> dict:
    """
    Calculate Black-Scholes Greeks.
    
    Args:
        spot: Current spot price
        strike: Strike price
        time_to_expiry: Time to expiry in years
        volatility: Implied volatility
        option_type: 'call' or 'put'
        risk_free_rate: Risk-free interest rate
    
    Returns:
        Dictionary with Delta, Gamma, Theta, Vega, Rho
    """
    if time_to_expiry <= 0:
        time_to_expiry = 1/365  # Minimum 1 day
    
    if volatility <= 0:
        volatility = 0.01  # Minimum volatility
    
    S = spot
    K = strike
    T = time_to_expiry
    r = risk_free_rate
    sigma = volatility
    
    # Calculate d1 and d2
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    # Standard normal PDF and CDF
    n_d1 = norm.pdf(d1)
    N_d1 = norm.cdf(d1)
    N_d2 = norm.cdf(d2)
    N_neg_d1 = norm.cdf(-d1)
    N_neg_d2 = norm.cdf(-d2)
    
    if option_type.lower() == 'call':
        delta = N_d1
        theta = (-(S * n_d1 * sigma) / (2 * np.sqrt(T)) - 
                 r * K * np.exp(-r * T) * N_d2) / 365
        rho = K * T * np.exp(-r * T) * N_d2 / 100
    else:  # put
        delta = N_d1 - 1
        theta = (-(S * n_d1 * sigma) / (2 * np.sqrt(T)) + 
                 r * K * np.exp(-r * T) * N_neg_d2) / 365
        rho = -K * T * np.exp(-r * T) * N_neg_d2 / 100
    
    # Gamma and Vega are same for calls and puts
    gamma = n_d1 / (S * sigma * np.sqrt(T))
    vega = S * n_d1 * np.sqrt(T) / 100
    
    return {
        'delta': delta,
        'gamma': gamma,
        'theta': theta,
        'vega': vega,
        'rho': rho
    }


def calculate_days_to_expiry(datetime_series: pd.Series) -> pd.Series:
    """Calculate days to next monthly expiry."""
    from datetime import timedelta, datetime as dt
    
    days_list = []
    for datetime_val in datetime_series:
        datetime_val = pd.to_datetime(datetime_val)
        year, month = datetime_val.year, datetime_val.month
        
        # Get last day of month
        if month == 12:
            next_month = dt(year + 1, 1, 1)
        else:
            next_month = dt(year, month + 1, 1)
        last_day = next_month - timedelta(days=1)
        
        # Find last Thursday
        days_until_thursday = (last_day.weekday() - 3) % 7
        last_thursday = last_day - timedelta(days=days_until_thursday)
        
        # If past this month's expiry, use next month
        if datetime_val.date() > last_thursday.date():
            if month == 12:
                month = 1
                year += 1
            else:
                month += 1
            
            if month == 12:
                next_month = dt(year + 1, 1, 1)
            else:
                next_month = dt(year, month + 1, 1)
            last_day = next_month - timedelta(days=1)
            days_until_thursday = (last_day.weekday() - 3) % 7
            last_thursday = last_day - timedelta(days=days_until_thursday)
        
        days = max(1, (last_thursday.date() - datetime_val.date()).days)
        days_list.append(days)
    
    return pd.Series(days_list, index=datetime_series.index)


def add_ema_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add EMA indicators to dataframe."""
    print("Calculating EMA indicators...")
    
    df['ema_5'] = calculate_ema(df['spot_close'], EMA_FAST)
    df['ema_15'] = calculate_ema(df['spot_close'], EMA_SLOW)
    df['ema_diff'] = df['ema_5'] - df['ema_15']
    df['ema_cross'] = np.sign(df['ema_diff'])
    
    # Crossover signals
    df['ema_cross_prev'] = df['ema_cross'].shift(1)
    df['bullish_cross'] = (df['ema_cross'] == 1) & (df['ema_cross_prev'] == -1)
    df['bearish_cross'] = (df['ema_cross'] == -1) & (df['ema_cross_prev'] == 1)
    
    return df


def add_greeks_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate and add Greeks for ATM options."""
    print("Calculating Options Greeks...")
    
    # Initialize columns
    greek_cols = ['atm_call_delta', 'atm_call_gamma', 'atm_call_theta', 
                  'atm_call_vega', 'atm_call_rho',
                  'atm_put_delta', 'atm_put_gamma', 'atm_put_theta',
                  'atm_put_vega', 'atm_put_rho']
    
    for col in greek_cols:
        df[col] = np.nan
    
    # Calculate days to expiry
    days_to_expiry = calculate_days_to_expiry(df['datetime'])
    time_to_expiry = days_to_expiry / 365.0
    
    # Calculate Greeks for each row
    for idx in tqdm(df.index, desc="Calculating Greeks"):
        spot = df.loc[idx, 'spot_close']
        strike = df.loc[idx, 'atm_strike']
        T = time_to_expiry[idx]
        call_iv = df.loc[idx, 'atm_call_iv']
        put_iv = df.loc[idx, 'atm_put_iv']
        
        # Call Greeks
        call_greeks = calculate_greeks(spot, strike, T, call_iv, 'call')
        df.loc[idx, 'atm_call_delta'] = call_greeks['delta']
        df.loc[idx, 'atm_call_gamma'] = call_greeks['gamma']
        df.loc[idx, 'atm_call_theta'] = call_greeks['theta']
        df.loc[idx, 'atm_call_vega'] = call_greeks['vega']
        df.loc[idx, 'atm_call_rho'] = call_greeks['rho']
        
        # Put Greeks
        put_greeks = calculate_greeks(spot, strike, T, put_iv, 'put')
        df.loc[idx, 'atm_put_delta'] = put_greeks['delta']
        df.loc[idx, 'atm_put_gamma'] = put_greeks['gamma']
        df.loc[idx, 'atm_put_theta'] = put_greeks['theta']
        df.loc[idx, 'atm_put_vega'] = put_greeks['vega']
        df.loc[idx, 'atm_put_rho'] = put_greeks['rho']
    
    return df


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add all derived features as specified in requirements."""
    print("Calculating derived features...")
    
    # 1. Average IV = (call_iv + put_iv) / 2
    df['avg_iv'] = (df['atm_call_iv'] + df['atm_put_iv']) / 2
    
    # 2. IV Spread = call_iv - put_iv
    df['iv_spread'] = df['atm_call_iv'] - df['atm_put_iv']
    
    # 3. PCR (OI-based) = total_put_oi / total_call_oi
    put_oi_cols = [col for col in df.columns if 'put_oi' in col]
    call_oi_cols = [col for col in df.columns if 'call_oi' in col]
    df['total_put_oi'] = df[put_oi_cols].sum(axis=1)
    df['total_call_oi'] = df[call_oi_cols].sum(axis=1)
    df['pcr_oi'] = df['total_put_oi'] / df['total_call_oi'].replace(0, 1)
    
    # 4. PCR (Volume-based) = total_put_volume / total_call_volume
    put_vol_cols = [col for col in df.columns if 'put_volume' in col]
    call_vol_cols = [col for col in df.columns if 'call_volume' in col]
    df['total_put_volume'] = df[put_vol_cols].sum(axis=1)
    df['total_call_volume'] = df[call_vol_cols].sum(axis=1)
    df['pcr_volume'] = df['total_put_volume'] / df['total_call_volume'].replace(0, 1)
    
    # 5. Futures Basis = (futures_close - spot_close) / spot_close
    df['futures_basis'] = (df['futures_close'] - df['spot_close']) / df['spot_close']
    
    # 6. Returns (spot and futures)
    df['spot_returns'] = df['spot_close'].pct_change()
    df['futures_returns'] = df['futures_close'].pct_change()
    
    # 7. Delta Neutral Ratio = abs(call_delta) / abs(put_delta)
    df['delta_neutral_ratio'] = np.abs(df['atm_call_delta']) / np.abs(df['atm_put_delta']).replace(0, 0.001)
    
    # 8. Gamma Exposure = spot_close Ã— gamma Ã— open_interest
    df['gamma_exposure'] = df['spot_close'] * df['atm_call_gamma'] * df['total_call_oi']
    
    # Additional useful features
    # ATR (Average True Range)
    df['tr'] = np.maximum(
        df['spot_high'] - df['spot_low'],
        np.maximum(
            np.abs(df['spot_high'] - df['spot_close'].shift(1)),
            np.abs(df['spot_low'] - df['spot_close'].shift(1))
        )
    )
    df['atr_14'] = df['tr'].rolling(window=14).mean()
    
    # Volatility (rolling std of returns)
    df['volatility_20'] = df['spot_returns'].rolling(window=20).std() * np.sqrt(252 * 75)
    
    # Time-based features
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['hour'] = df['datetime'].dt.hour
    df['minute'] = df['datetime'].dt.minute
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['is_first_hour'] = ((df['hour'] == 9) & (df['minute'] >= 15)) | (df['hour'] == 10)
    df['is_last_hour'] = (df['hour'] >= 14) & (df['minute'] >= 30)
    
    # Lag features
    for lag in [1, 2, 3, 5, 10]:
        df[f'spot_return_lag_{lag}'] = df['spot_returns'].shift(lag)
        df[f'volume_lag_{lag}'] = df['spot_volume'].shift(lag)
    
    # Fill NaN from calculations
    df = df.fillna(method='ffill').fillna(0)
    
    return df


def engineer_all_features(input_path=MERGED_DATA_PATH, output_path=FEATURES_DATA_PATH):
    """
    Main function to engineer all features.
    """
    print("=" * 60)
    print("Feature Engineering Pipeline")
    print("=" * 60)
    
    # Load merged data
    print("\nLoading merged data...")
    df = pd.read_csv(input_path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    print(f"Loaded {len(df)} rows")
    
    # Add EMA features
    df = add_ema_features(df)
    print(f"âœ“ Added EMA features: ema_5, ema_15, ema_diff, crossover signals")
    
    # Add Greeks
    df = add_greeks_features(df)
    print(f"âœ“ Added Greeks: Delta, Gamma, Theta, Vega, Rho for ATM options")
    
    # Add derived features
    df = add_derived_features(df)
    print(f"âœ“ Added derived features: IV, PCR, Basis, Returns, Gamma Exposure, etc.")
    
    # Save features
    df.to_csv(output_path, index=False)
    print(f"\nâœ“ Saved features to {output_path}")
    
    # Summary
    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING SUMMARY")
    print("=" * 60)
    print(f"Total rows: {len(df)}")
    print(f"Total features: {len(df.columns)}")
    print(f"New features added: EMA(5,15), Greeks, IV metrics, PCR, Basis, etc.")
    print("=" * 60)
    
    return df


if __name__ == "__main__":
    engineer_all_features()
