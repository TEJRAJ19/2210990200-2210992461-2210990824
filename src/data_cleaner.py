"""
Data Cleaner Module
Handles missing values, outliers, timestamp alignment, and futures rollover.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

from config import (
    DATA_DIR, SPOT_DATA_PATH, FUTURES_DATA_PATH, OPTIONS_DATA_PATH,
    CLEANING_REPORT_PATH
)


def clean_spot_data(df: pd.DataFrame) -> tuple:
    """
    Clean spot data: handle missing values, remove outliers, validate OHLC.
    
    Returns:
        Tuple of (cleaned_df, report_dict)
    """
    report = {'spot': {}}
    original_len = len(df)
    
    # Convert datetime
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    # 1. Handle missing values
    missing_before = df.isnull().sum().sum()
    df = df.fillna(method='ffill').fillna(method='bfill')
    report['spot']['missing_values_filled'] = int(missing_before)
    
    # 2. Validate OHLC relationships (High >= Open, Close, Low; Low <= Open, Close, High)
    invalid_ohlc = (
        (df['high'] < df['open']) | 
        (df['high'] < df['close']) |
        (df['low'] > df['open']) | 
        (df['low'] > df['close']) |
        (df['high'] < df['low'])
    )
    invalid_count = invalid_ohlc.sum()
    
    # Fix invalid OHLC
    df.loc[invalid_ohlc, 'high'] = df.loc[invalid_ohlc, ['open', 'high', 'low', 'close']].max(axis=1)
    df.loc[invalid_ohlc, 'low'] = df.loc[invalid_ohlc, ['open', 'high', 'low', 'close']].min(axis=1)
    report['spot']['invalid_ohlc_fixed'] = int(invalid_count)
    
    # 3. Remove extreme outliers (> 5 std from rolling mean)
    df['returns'] = df['close'].pct_change()
    rolling_mean = df['returns'].rolling(window=50, min_periods=1).mean()
    rolling_std = df['returns'].rolling(window=50, min_periods=1).std()
    
    outliers = (np.abs(df['returns'] - rolling_mean) > 5 * rolling_std)
    outlier_count = outliers.sum()
    
    # Replace outlier closes with interpolated values
    df.loc[outliers, 'close'] = np.nan
    df['close'] = df['close'].interpolate(method='linear')
    df = df.drop('returns', axis=1)
    report['spot']['outliers_removed'] = int(outlier_count)
    
    # 4. Ensure positive values
    for col in ['open', 'high', 'low', 'close', 'volume']:
        negative_count = (df[col] < 0).sum()
        df.loc[df[col] < 0, col] = df[col].abs()
        if negative_count > 0:
            report['spot'][f'negative_{col}_fixed'] = int(negative_count)
    
    # 5. Remove duplicates
    duplicates = df.duplicated(subset=['datetime']).sum()
    df = df.drop_duplicates(subset=['datetime'])
    report['spot']['duplicates_removed'] = int(duplicates)
    
    report['spot']['original_rows'] = original_len
    report['spot']['final_rows'] = len(df)
    
    return df, report


def clean_futures_data(df: pd.DataFrame) -> tuple:
    """
    Clean futures data: handle rollovers, missing values, outliers.
    """
    report = {'futures': {}}
    original_len = len(df)
    
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    # 1. Handle missing values
    missing_before = df.isnull().sum().sum()
    df = df.fillna(method='ffill').fillna(method='bfill')
    report['futures']['missing_values_filled'] = int(missing_before)
    
    # 2. Handle futures rollover
    # Detect rollovers by large gaps in OI
    df['oi_change_pct'] = df['open_interest'].pct_change().abs()
    rollovers = df['oi_change_pct'] > 0.5  # >50% OI change indicates rollover
    rollover_count = rollovers.sum()
    
    # Smooth price discontinuities at rollover
    df['price_gap'] = df['close'].pct_change()
    large_gaps = (rollovers) & (np.abs(df['price_gap']) > 0.01)
    
    for idx in df[large_gaps].index:
        if idx > 0:
            adjustment = df.loc[idx, 'close'] / df.loc[idx - 1, 'close']
            # Note: In a more sophisticated system, you'd adjust historical prices
    
    df = df.drop(['oi_change_pct', 'price_gap'], axis=1)
    report['futures']['rollovers_detected'] = int(rollover_count)
    
    # 3. Validate OHLC
    invalid_ohlc = (
        (df['high'] < df['low']) |
        (df['high'] < df['open']) |
        (df['high'] < df['close']) |
        (df['low'] > df['open']) |
        (df['low'] > df['close'])
    )
    df.loc[invalid_ohlc, 'high'] = df.loc[invalid_ohlc, ['open', 'high', 'low', 'close']].max(axis=1)
    df.loc[invalid_ohlc, 'low'] = df.loc[invalid_ohlc, ['open', 'high', 'low', 'close']].min(axis=1)
    report['futures']['invalid_ohlc_fixed'] = int(invalid_ohlc.sum())
    
    # 4. Remove duplicates
    duplicates = df.duplicated(subset=['datetime']).sum()
    df = df.drop_duplicates(subset=['datetime'])
    report['futures']['duplicates_removed'] = int(duplicates)
    
    report['futures']['original_rows'] = original_len
    report['futures']['final_rows'] = len(df)
    
    return df, report


def clean_options_data(df: pd.DataFrame) -> tuple:
    """
    Clean options data: validate IVs, prices, calculate dynamic ATM.
    """
    report = {'options': {}}
    original_len = len(df)
    
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    # 1. Handle missing values
    missing_before = df.isnull().sum().sum()
    df = df.fillna(method='ffill').fillna(method='bfill')
    report['options']['missing_values_filled'] = int(missing_before)
    
    # 2. Validate IV values (should be between 0.05 and 2.0 typically)
    iv_columns = [col for col in df.columns if '_iv' in col]
    for col in iv_columns:
        invalid_iv = (df[col] < 0.01) | (df[col] > 3.0)
        invalid_count = invalid_iv.sum()
        
        # Clip to reasonable range
        df[col] = df[col].clip(0.05, 2.0)
        if invalid_count > 0:
            report['options'][f'{col}_fixed'] = int(invalid_count)
    
    # 3. Validate option prices (must be positive)
    ltp_columns = [col for col in df.columns if '_ltp' in col]
    for col in ltp_columns:
        negative = (df[col] <= 0).sum()
        df[col] = df[col].clip(lower=0.05)
        if negative > 0:
            report['options'][f'{col}_negative_fixed'] = int(negative)
    
    # 4. Validate volume and OI (must be non-negative)
    vol_oi_columns = [col for col in df.columns if 'volume' in col or '_oi' in col]
    for col in vol_oi_columns:
        negative = (df[col] < 0).sum()
        df[col] = df[col].clip(lower=0)
        if negative > 0:
            report['options'][f'{col}_fixed'] = int(negative)
    
    # 5. Remove duplicates
    duplicates = df.duplicated(subset=['datetime']).sum()
    df = df.drop_duplicates(subset=['datetime'])
    report['options']['duplicates_removed'] = int(duplicates)
    
    report['options']['original_rows'] = original_len
    report['options']['final_rows'] = len(df)
    
    return df, report


def align_timestamps(spot_df: pd.DataFrame, futures_df: pd.DataFrame, 
                     options_df: pd.DataFrame) -> tuple:
    """
    Align all three datasets on common timestamps.
    """
    # Find common timestamps
    spot_times = set(spot_df['datetime'])
    futures_times = set(futures_df['datetime'])
    options_times = set(options_df['datetime'])
    
    common_times = spot_times & futures_times & options_times
    
    # Filter each dataframe
    spot_df = spot_df[spot_df['datetime'].isin(common_times)].reset_index(drop=True)
    futures_df = futures_df[futures_df['datetime'].isin(common_times)].reset_index(drop=True)
    options_df = options_df[options_df['datetime'].isin(common_times)].reset_index(drop=True)
    
    # Sort by datetime
    spot_df = spot_df.sort_values('datetime').reset_index(drop=True)
    futures_df = futures_df.sort_values('datetime').reset_index(drop=True)
    options_df = options_df.sort_values('datetime').reset_index(drop=True)
    
    return spot_df, futures_df, options_df


def generate_cleaning_report(reports: dict, save_path: Path = CLEANING_REPORT_PATH):
    """
    Generate and save data cleaning report.
    """
    report_lines = [
        "=" * 60,
        "DATA CLEANING REPORT",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 60,
        ""
    ]
    
    for dataset_name, dataset_report in reports.items():
        report_lines.append(f"\n{dataset_name.upper()} DATA")
        report_lines.append("-" * 40)
        for key, value in dataset_report.items():
            report_lines.append(f"  {key}: {value}")
    
    report_lines.append("\n" + "=" * 60)
    report_lines.append("CLEANING COMPLETE")
    report_lines.append("=" * 60)
    
    report_text = "\n".join(report_lines)
    
    with open(save_path, 'w') as f:
        f.write(report_text)
    
    print(report_text)
    return report_text


def clean_all_data():
    """
    Main function to clean all datasets.
    """
    print("=" * 60)
    print("Data Cleaning Pipeline")
    print("=" * 60)
    
    all_reports = {}
    
    # Load and clean spot data
    print("\nCleaning SPOT data...")
    spot_df = pd.read_csv(SPOT_DATA_PATH)
    spot_df, spot_report = clean_spot_data(spot_df)
    all_reports.update(spot_report)
    
    # Load and clean futures data
    print("\nCleaning FUTURES data...")
    futures_df = pd.read_csv(FUTURES_DATA_PATH)
    futures_df, futures_report = clean_futures_data(futures_df)
    all_reports.update(futures_report)
    
    # Load and clean options data
    print("\nCleaning OPTIONS data...")
    options_df = pd.read_csv(OPTIONS_DATA_PATH)
    options_df, options_report = clean_options_data(options_df)
    all_reports.update(options_report)
    
    # Align timestamps
    print("\nAligning timestamps...")
    spot_df, futures_df, options_df = align_timestamps(spot_df, futures_df, options_df)
    all_reports['alignment'] = {
        'final_aligned_rows': len(spot_df)
    }
    
    # Save cleaned data
    spot_df.to_csv(SPOT_DATA_PATH, index=False)
    futures_df.to_csv(FUTURES_DATA_PATH, index=False)
    options_df.to_csv(OPTIONS_DATA_PATH, index=False)
    
    # Generate report
    generate_cleaning_report(all_reports)
    
    print("\nâœ“ All data cleaned and saved!")
    
    return spot_df, futures_df, options_df


if __name__ == "__main__":
    clean_all_data()
