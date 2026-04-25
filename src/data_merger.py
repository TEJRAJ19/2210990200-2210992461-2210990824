"""
Data Merger Module
Merges spot, futures, and options data on timestamp.
"""

import pandas as pd
from pathlib import Path

from config import (
    SPOT_DATA_PATH, FUTURES_DATA_PATH, OPTIONS_DATA_PATH, MERGED_DATA_PATH
)


def merge_all_data(
    spot_path: Path = SPOT_DATA_PATH,
    futures_path: Path = FUTURES_DATA_PATH,
    options_path: Path = OPTIONS_DATA_PATH,
    output_path: Path = MERGED_DATA_PATH
) -> pd.DataFrame:
    """
    Merge spot, futures, and options data on timestamp.
    
    Returns:
        Merged DataFrame with all data
    """
    print("=" * 60)
    print("Data Merging Pipeline")
    print("=" * 60)
    
    # Load data
    print("\nLoading datasets...")
    spot_df = pd.read_csv(spot_path)
    futures_df = pd.read_csv(futures_path)
    options_df = pd.read_csv(options_path)
    
    # Convert datetime columns
    spot_df['datetime'] = pd.to_datetime(spot_df['datetime'])
    futures_df['datetime'] = pd.to_datetime(futures_df['datetime'])
    options_df['datetime'] = pd.to_datetime(options_df['datetime'])
    
    print(f"  Spot rows: {len(spot_df)}")
    print(f"  Futures rows: {len(futures_df)}")
    print(f"  Options rows: {len(options_df)}")
    
    # Rename columns to avoid conflicts
    spot_df = spot_df.rename(columns={
        'open': 'spot_open',
        'high': 'spot_high',
        'low': 'spot_low',
        'close': 'spot_close',
        'volume': 'spot_volume'
    })
    
    futures_df = futures_df.rename(columns={
        'open': 'futures_open',
        'high': 'futures_high',
        'low': 'futures_low',
        'close': 'futures_close',
        'volume': 'futures_volume',
        'open_interest': 'futures_oi'
    })
    
    # Merge spot and futures
    print("\nMerging spot and futures...")
    merged_df = pd.merge(spot_df, futures_df, on='datetime', how='inner')
    print(f"  After spot+futures merge: {len(merged_df)} rows")
    
    # Merge with options
    print("Merging with options...")
    # Drop spot_close from options as it's redundant
    if 'spot_close' in options_df.columns:
        options_df = options_df.drop('spot_close', axis=1)
    
    merged_df = pd.merge(merged_df, options_df, on='datetime', how='inner')
    print(f"  After full merge: {len(merged_df)} rows")
    
    # Sort by datetime
    merged_df = merged_df.sort_values('datetime').reset_index(drop=True)
    
    # Save merged data
    merged_df.to_csv(output_path, index=False)
    print(f"\nâœ“ Saved merged data to {output_path}")
    
    # Summary
    print("\n" + "=" * 60)
    print("MERGE SUMMARY")
    print("=" * 60)
    print(f"Total rows: {len(merged_df)}")
    print(f"Total columns: {len(merged_df.columns)}")
    print(f"Date range: {merged_df['datetime'].min()} to {merged_df['datetime'].max()}")
    print(f"Trading days: {merged_df['datetime'].dt.date.nunique()}")
    print("=" * 60)
    
    return merged_df


def get_merged_data_summary(df: pd.DataFrame) -> dict:
    """
    Generate summary statistics for merged data.
    """
    summary = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'date_range': {
            'start': df['datetime'].min().strftime('%Y-%m-%d'),
            'end': df['datetime'].max().strftime('%Y-%m-%d')
        },
        'trading_days': df['datetime'].dt.date.nunique(),
        'spot_price_range': {
            'min': df['spot_close'].min(),
            'max': df['spot_close'].max(),
            'mean': df['spot_close'].mean()
        },
        'futures_price_range': {
            'min': df['futures_close'].min(),
            'max': df['futures_close'].max(),
            'mean': df['futures_close'].mean()
        },
        'missing_values': df.isnull().sum().sum()
    }
    return summary


if __name__ == "__main__":
    merged_df = merge_all_data()
    summary = get_merged_data_summary(merged_df)
    print("\nData Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
