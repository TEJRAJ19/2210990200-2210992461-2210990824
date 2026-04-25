"""
Data Fetcher Module
Fetches NIFTY Spot, Futures, and Options data for the trading system.
Uses Yahoo Finance for daily data and generates realistic 5-minute intraday data.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from config import (
    DATA_DIR, SPOT_DATA_PATH, FUTURES_DATA_PATH, OPTIONS_DATA_PATH,
    TRADING_DAYS_PER_YEAR, MINUTES_PER_DAY, RISK_FREE_RATE
)


def fetch_nifty_daily_data(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch NIFTY 50 daily data from Yahoo Finance.
    
    Args:
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
    
    Returns:
        DataFrame with daily OHLCV data
    """
    print("Fetching NIFTY 50 daily data from Yahoo Finance...")
    ticker = yf.Ticker("^NSEI")
    df = ticker.history(start=start_date, end=end_date)
    df.reset_index(inplace=True)
    df.columns = ['datetime', 'open', 'high', 'low', 'close', 'volume', 'dividends', 'stock_splits']
    df = df[['datetime', 'open', 'high', 'low', 'close', 'volume']]
    df['datetime'] = pd.to_datetime(df['datetime']).dt.tz_localize(None)
    print(f"Fetched {len(df)} days of daily data")
    return df


def generate_intraday_from_daily(daily_df: pd.DataFrame, candles_per_day: int = 75) -> pd.DataFrame:
    """
    Generate realistic 5-minute intraday data from daily OHLC.
    Uses geometric Brownian motion within daily range constraints.
    
    Args:
        daily_df: DataFrame with daily OHLC data
        candles_per_day: Number of 5-minute candles per trading day (75 for NSE)
    
    Returns:
        DataFrame with 5-minute OHLCV data
    """
    print("Generating 5-minute intraday data from daily data...")
    intraday_data = []
    
    for idx, row in tqdm(daily_df.iterrows(), total=len(daily_df), desc="Generating intraday"):
        date = row['datetime']
        daily_open = row['open']
        daily_high = row['high']
        daily_low = row['low']
        daily_close = row['close']
        daily_volume = row['volume']
        
        # Generate price path using constrained random walk
        prices = _generate_price_path(
            daily_open, daily_high, daily_low, daily_close, candles_per_day
        )
        
        # Distribute volume with U-shaped pattern (higher at open and close)
        volumes = _distribute_volume(daily_volume, candles_per_day)
        
        # Create 5-minute candles
        for i in range(candles_per_day):
            # Calculate time: Trading hours 9:15 to 15:30
            minutes_from_start = i * 5
            candle_time = datetime.combine(
                date.date(),
                datetime.strptime("09:15", "%H:%M").time()
            ) + timedelta(minutes=minutes_from_start)
            
            # Get OHLC for this candle
            if i < candles_per_day - 1:
                candle_prices = prices[i * 4: (i + 1) * 4 + 1]
            else:
                candle_prices = prices[i * 4:]
            
            if len(candle_prices) > 0:
                candle_open = candle_prices[0]
                candle_close = candle_prices[-1]
                candle_high = max(candle_prices)
                candle_low = min(candle_prices)
            else:
                candle_open = candle_close = candle_high = candle_low = prices[-1]
            
            intraday_data.append({
                'datetime': candle_time,
                'open': round(candle_open, 2),
                'high': round(candle_high, 2),
                'low': round(candle_low, 2),
                'close': round(candle_close, 2),
                'volume': int(volumes[i])
            })
    
    df = pd.DataFrame(intraday_data)
    print(f"Generated {len(df)} intraday candles")
    return df


def _generate_price_path(
    open_price: float, 
    high_price: float, 
    low_price: float, 
    close_price: float,
    n_candles: int
) -> np.ndarray:
    """
    Generate a realistic intraday price path that respects daily OHLC constraints.
    """
    n_points = n_candles * 4 + 1  # 4 points per candle for OHLC
    
    # Start with Brownian bridge from open to close
    t = np.linspace(0, 1, n_points)
    
    # Generate random walk
    np.random.seed(int(open_price) % 10000)  # Reproducible but varied
    increments = np.random.normal(0, 0.001, n_points - 1)
    random_walk = np.cumsum(increments)
    random_walk = np.insert(random_walk, 0, 0)
    
    # Create Brownian bridge
    bridge = random_walk - t * random_walk[-1]
    
    # Linear interpolation from open to close
    linear_path = open_price + (close_price - open_price) * t
    
    # Combine with scaled bridge
    price_range = high_price - low_price
    prices = linear_path + bridge * price_range * 2
    
    # Find touch points for high and low
    high_idx = int(n_points * np.random.uniform(0.2, 0.8))
    low_idx = int(n_points * np.random.uniform(0.2, 0.8))
    while abs(high_idx - low_idx) < n_points * 0.1:
        low_idx = int(n_points * np.random.uniform(0.2, 0.8))
    
    # Scale and shift to ensure high/low are touched
    current_max = np.max(prices)
    current_min = np.min(prices)
    current_range = current_max - current_min
    
    if current_range > 0:
        # Scale to fit within daily range
        prices = (prices - current_min) / current_range * (high_price - low_price) + low_price
    
    # Ensure open and close are correct
    prices[0] = open_price
    prices[-1] = close_price
    
    return prices


def _distribute_volume(total_volume: int, n_candles: int) -> np.ndarray:
    """
    Distribute volume across candles with U-shaped pattern.
    Higher volume at market open and close, lower in mid-day.
    """
    # U-shaped distribution
    t = np.linspace(0, 1, n_candles)
    u_shape = 2 * (t - 0.5) ** 2 + 0.5
    
    # Add some randomness
    noise = np.random.uniform(0.8, 1.2, n_candles)
    weights = u_shape * noise
    
    # Normalize and distribute volume
    weights = weights / weights.sum()
    volumes = weights * total_volume
    
    return volumes.astype(int)


def generate_futures_data(spot_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate NIFTY Futures data from spot data.
    Futures trade at a premium/discount based on cost of carry.
    
    Args:
        spot_df: DataFrame with spot 5-minute data
    
    Returns:
        DataFrame with futures 5-minute data including Open Interest
    """
    print("Generating Futures data with realistic basis...")
    futures_df = spot_df.copy()
    
    # Calculate days to monthly expiry (last Thursday of month)
    futures_df['date'] = pd.to_datetime(futures_df['datetime']).dt.date
    
    # Generate basis based on cost of carry model
    # Futures Price = Spot * e^(r * t)
    # Add some noise for realism
    
    days_to_expiry = _calculate_days_to_expiry(futures_df['datetime'])
    time_to_expiry = days_to_expiry / 365.0
    
    # Cost of carry with some noise
    np.random.seed(42)
    noise = np.random.normal(0, 0.001, len(futures_df))
    carry = np.exp(RISK_FREE_RATE * time_to_expiry) + noise
    
    # Apply to all price columns
    for col in ['open', 'high', 'low', 'close']:
        futures_df[col] = (spot_df[col] * carry).round(2)
    
    # Generate Open Interest pattern
    base_oi = 10000000  # 1 crore base OI
    trend = np.linspace(0.8, 1.2, len(futures_df))  # Slight uptrend
    daily_pattern = np.tile(
        np.concatenate([np.linspace(1, 1.1, 37), np.linspace(1.1, 1, 38)]), 
        len(futures_df) // 75 + 1
    )[:len(futures_df)]
    noise = np.random.uniform(0.95, 1.05, len(futures_df))
    
    futures_df['open_interest'] = (base_oi * trend * daily_pattern * noise).astype(int)
    
    # Keep volume from spot
    futures_df['volume'] = (spot_df['volume'] * 0.8).astype(int)  # Futures typically lower volume
    
    futures_df = futures_df[['datetime', 'open', 'high', 'low', 'close', 'volume', 'open_interest']]
    print(f"Generated {len(futures_df)} futures candles")
    return futures_df


def _calculate_days_to_expiry(datetime_series: pd.Series) -> np.ndarray:
    """Calculate days to monthly expiry (last Thursday of month)."""
    days_to_expiry = []
    
    for dt in datetime_series:
        dt = pd.to_datetime(dt)
        # Find last Thursday of current month
        year, month = dt.year, dt.month
        
        # Get last day of month
        if month == 12:
            next_month = datetime(year + 1, 1, 1)
        else:
            next_month = datetime(year, month + 1, 1)
        last_day = next_month - timedelta(days=1)
        
        # Find last Thursday
        days_until_thursday = (last_day.weekday() - 3) % 7
        last_thursday = last_day - timedelta(days=days_until_thursday)
        
        # If we're past this month's expiry, use next month's
        if dt.date() > last_thursday.date():
            # Move to next month
            if month == 12:
                next_month_num = 1
                next_year = year + 1
            else:
                next_month_num = month + 1
                next_year = year
            
            # Calculate last day of next month
            if next_month_num == 12:
                month_after = datetime(next_year + 1, 1, 1)
            else:
                month_after = datetime(next_year, next_month_num + 1, 1)
            last_day = month_after - timedelta(days=1)
            days_until_thursday = (last_day.weekday() - 3) % 7
            last_thursday = last_day - timedelta(days=days_until_thursday)
        
        days = max(1, (last_thursday.date() - dt.date()).days)
        days_to_expiry.append(days)
    
    return np.array(days_to_expiry)


def generate_options_data(spot_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate NIFTY Options Chain data.
    Creates ATM, ATMÂ±1, ATMÂ±2 strikes for both Call and Put options.
    
    Args:
        spot_df: DataFrame with spot 5-minute data
    
    Returns:
        DataFrame with options data including LTP, IV, Volume, OI for each strike
    """
    print("Generating Options Chain data...")
    options_data = []
    
    strike_gap = 50  # NIFTY options have 50-point strike intervals
    
    for idx, row in tqdm(spot_df.iterrows(), total=len(spot_df), desc="Generating options"):
        spot_price = row['close']
        dt = row['datetime']
        
        # Calculate ATM strike
        atm_strike = round(spot_price / strike_gap) * strike_gap
        
        # Generate data for ATM, ATMÂ±1, ATMÂ±2
        strikes = [
            atm_strike - 2 * strike_gap,
            atm_strike - strike_gap,
            atm_strike,
            atm_strike + strike_gap,
            atm_strike + 2 * strike_gap
        ]
        
        # Days to expiry for IV calculation
        days_to_exp = _calculate_days_to_expiry(pd.Series([dt]))[0]
        
        row_data = {'datetime': dt, 'spot_close': spot_price, 'atm_strike': atm_strike}
        
        for i, strike in enumerate(strikes):
            strike_label = ['atm_m2', 'atm_m1', 'atm', 'atm_p1', 'atm_p2'][i]
            
            # Generate realistic IV (higher for OTM options - volatility smile)
            moneyness = abs(strike - spot_price) / spot_price
            base_iv = 0.15 + moneyness * 0.5  # Base IV increases with moneyness
            call_iv = base_iv + np.random.uniform(-0.02, 0.02)
            put_iv = base_iv + np.random.uniform(-0.02, 0.02) + 0.01  # Put IV slightly higher
            
            # Generate LTP based on Black-Scholes approximation
            call_ltp, put_ltp = _approximate_option_prices(
                spot_price, strike, days_to_exp, call_iv, put_iv
            )
            
            # Generate volume and OI
            atm_distance = abs(i - 2)  # Distance from ATM
            volume_multiplier = 1 / (1 + atm_distance * 0.5)  # Higher volume at ATM
            
            base_volume = 50000
            base_oi = 500000
            
            call_volume = int(base_volume * volume_multiplier * np.random.uniform(0.5, 1.5))
            put_volume = int(base_volume * volume_multiplier * np.random.uniform(0.5, 1.5))
            call_oi = int(base_oi * volume_multiplier * np.random.uniform(0.8, 1.2))
            put_oi = int(base_oi * volume_multiplier * np.random.uniform(0.8, 1.2))
            
            row_data[f'{strike_label}_call_ltp'] = round(call_ltp, 2)
            row_data[f'{strike_label}_put_ltp'] = round(put_ltp, 2)
            row_data[f'{strike_label}_call_iv'] = round(call_iv, 4)
            row_data[f'{strike_label}_put_iv'] = round(put_iv, 4)
            row_data[f'{strike_label}_call_volume'] = call_volume
            row_data[f'{strike_label}_put_volume'] = put_volume
            row_data[f'{strike_label}_call_oi'] = call_oi
            row_data[f'{strike_label}_put_oi'] = put_oi
        
        options_data.append(row_data)
    
    df = pd.DataFrame(options_data)
    print(f"Generated {len(df)} options chain records")
    return df


def _approximate_option_prices(
    spot: float, 
    strike: float, 
    days_to_expiry: int,
    call_iv: float,
    put_iv: float
) -> tuple:
    """
    Approximate option prices using simplified Black-Scholes.
    """
    from scipy.stats import norm
    
    T = days_to_expiry / 365.0
    r = RISK_FREE_RATE
    
    if T <= 0:
        T = 1/365.0
    
    # Call option
    sigma = call_iv
    d1 = (np.log(spot / strike) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = spot * norm.cdf(d1) - strike * np.exp(-r * T) * norm.cdf(d2)
    
    # Put option
    sigma = put_iv
    d1 = (np.log(spot / strike) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    put_price = strike * np.exp(-r * T) * norm.cdf(-d2) - spot * norm.cdf(-d1)
    
    return max(0.05, call_price), max(0.05, put_price)


def fetch_and_save_all_data(start_date: str = "2024-01-15", end_date: str = "2025-01-15"):
    """
    Main function to fetch/generate all required data and save to CSV.
    
    Args:
        start_date: Start date for data
        end_date: End date for data
    """
    print("=" * 60)
    print("NIFTY Data Fetcher - Starting Data Pipeline")
    print("=" * 60)
    
    # Step 1: Fetch daily data from Yahoo Finance
    daily_df = fetch_nifty_daily_data(start_date, end_date)
    
    # Step 2: Generate 5-minute spot data
    spot_df = generate_intraday_from_daily(daily_df)
    spot_df.to_csv(SPOT_DATA_PATH, index=False)
    print(f"âœ“ Saved spot data to {SPOT_DATA_PATH}")
    
    # Step 3: Generate futures data
    futures_df = generate_futures_data(spot_df)
    futures_df.to_csv(FUTURES_DATA_PATH, index=False)
    print(f"âœ“ Saved futures data to {FUTURES_DATA_PATH}")
    
    # Step 4: Generate options data
    options_df = generate_options_data(spot_df)
    options_df.to_csv(OPTIONS_DATA_PATH, index=False)
    print(f"âœ“ Saved options data to {OPTIONS_DATA_PATH}")
    
    print("=" * 60)
    print("Data Pipeline Complete!")
    print(f"Spot records: {len(spot_df)}")
    print(f"Futures records: {len(futures_df)}")
    print(f"Options records: {len(options_df)}")
    print("=" * 60)
    
    return spot_df, futures_df, options_df


if __name__ == "__main__":
    fetch_and_save_all_data()
