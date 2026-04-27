"""
Trade Analysis Module
Analyzes high-performance trades and identifies patterns.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict, List

from config import RESULTS_DIR, PLOTS_DIR


def identify_outlier_trades(trades_df: pd.DataFrame, z_threshold: float = 3.0) -> pd.DataFrame:
    """
    Identify profitable trades beyond 3-sigma (Z-score > threshold).
    
    Args:
        trades_df: DataFrame with trades
        z_threshold: Z-score threshold for outliers
    
    Returns:
        DataFrame with outlier trades
    """
    if len(trades_df) == 0:
        return pd.DataFrame()
    
    # Calculate Z-scores for PnL
    mean_pnl = trades_df['pnl'].mean()
    std_pnl = trades_df['pnl'].std()
    
    if std_pnl == 0:
        trades_df['z_score'] = 0
    else:
        trades_df['z_score'] = (trades_df['pnl'] - mean_pnl) / std_pnl
    
    # Identify positive outliers (exceptional profits)
    outliers = trades_df[(trades_df['z_score'] > z_threshold) & (trades_df['pnl'] > 0)]
    
    return outliers


def analyze_outlier_features(
    df: pd.DataFrame, 
    outlier_trades: pd.DataFrame,
    all_trades: pd.DataFrame
) -> Dict:
    """
    Analyze features of outlier trades vs normal profitable trades.
    
    Features to analyze:
    - Regime
    - IV
    - ATR
    - Time of day
    - Greeks
    - Trade duration
    - EMA gap
    - PCR
    """
    if len(outlier_trades) == 0 or len(all_trades) == 0:
        return {}
    
    # Get profitable but non-outlier trades
    normal_profitable = all_trades[
        (all_trades['pnl'] > 0) & 
        (~all_trades.index.isin(outlier_trades.index))
    ]
    
    analysis = {}
    
    # Match trades to candle data
    def get_trade_features(trades: pd.DataFrame, df: pd.DataFrame):
        features_list = []
        
        for _, trade in trades.iterrows():
            entry_time = trade['entry_time']
            mask = df['datetime'] == entry_time
            
            if mask.any():
                row = df[mask].iloc[0]
                features = {
                    'pnl': trade['pnl'],
                    'regime': trade.get('regime', row.get('regime', 0)),
                    'avg_iv': row.get('avg_iv', 0),
                    'atr': row.get('atr_14', 0),
                    'hour': pd.to_datetime(entry_time).hour,
                    'atm_call_delta': row.get('atm_call_delta', 0),
                    'atm_call_gamma': row.get('atm_call_gamma', 0),
                    'duration': trade.get('duration', 0),
                    'ema_gap': abs(row.get('ema_diff', 0)),
                    'pcr_oi': row.get('pcr_oi', 1)
                }
                features_list.append(features)
        
        return pd.DataFrame(features_list)
    
    outlier_features = get_trade_features(outlier_trades, df)
    normal_features = get_trade_features(normal_profitable, df)
    
    if len(outlier_features) == 0 or len(normal_features) == 0:
        return {}
    
    # Statistical comparison
    feature_cols = ['avg_iv', 'atr', 'hour', 'atm_call_delta', 'atm_call_gamma',
                   'duration', 'ema_gap', 'pcr_oi']
    
    for col in feature_cols:
        if col in outlier_features.columns and col in normal_features.columns:
            outlier_vals = outlier_features[col].dropna()
            normal_vals = normal_features[col].dropna()
            
            if len(outlier_vals) >= 2 and len(normal_vals) >= 2:
                # T-test
                t_stat, t_pval = stats.ttest_ind(outlier_vals, normal_vals)
                
                # Mann-Whitney U test
                u_stat, u_pval = stats.mannwhitneyu(outlier_vals, normal_vals, 
                                                     alternative='two-sided')
                
                analysis[col] = {
                    'outlier_mean': outlier_vals.mean(),
                    'outlier_std': outlier_vals.std(),
                    'normal_mean': normal_vals.mean(),
                    'normal_std': normal_vals.std(),
                    't_statistic': t_stat,
                    't_pvalue': t_pval,
                    'u_statistic': u_stat,
                    'u_pvalue': u_pval,
                    'significant': t_pval < 0.05
                }
    
    # Regime distribution
    if 'regime' in outlier_features.columns:
        outlier_regime_dist = outlier_features['regime'].value_counts(normalize=True)
        normal_regime_dist = normal_features['regime'].value_counts(normalize=True)
        analysis['regime_distribution'] = {
            'outlier': outlier_regime_dist.to_dict(),
            'normal': normal_regime_dist.to_dict()
        }
    
    return analysis, outlier_features, normal_features


def plot_pnl_vs_duration(trades_df: pd.DataFrame, outlier_mask: pd.Series = None,
                         save_path=None):
    """
    Create scatter plot of PnL vs duration.
    """
    if len(trades_df) == 0:
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if outlier_mask is not None and outlier_mask.any():
        # Plot normal trades
        normal = trades_df[~outlier_mask]
        ax.scatter(normal['duration'], normal['pnl'], alpha=0.5, 
                   c='blue', label='Normal trades', s=50)
        
        # Plot outliers
        outliers = trades_df[outlier_mask]
        ax.scatter(outliers['duration'], outliers['pnl'], alpha=0.8,
                   c='red', label='Outliers (Z>3)', s=100, marker='*')
    else:
        ax.scatter(trades_df['duration'], trades_df['pnl'], alpha=0.5, s=50)
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Duration (candles)')
    ax.set_ylabel('PnL (points)')
    ax.set_title('Trade PnL vs Duration')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path is None:
        save_path = PLOTS_DIR / "pnl_vs_duration.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"âœ“ Saved PnL vs Duration plot to {save_path}")


def plot_feature_distributions(
    outlier_features: pd.DataFrame,
    normal_features: pd.DataFrame,
    save_path=None
):
    """
    Create box plots comparing feature distributions.
    """
    if len(outlier_features) == 0 or len(normal_features) == 0:
        return
    
    features_to_plot = ['avg_iv', 'atr', 'ema_gap', 'pcr_oi', 'duration']
    features_to_plot = [f for f in features_to_plot if f in outlier_features.columns]
    
    if len(features_to_plot) == 0:
        return
    
    fig, axes = plt.subplots(1, len(features_to_plot), figsize=(4 * len(features_to_plot), 5))
    if len(features_to_plot) == 1:
        axes = [axes]
    
    for idx, feature in enumerate(features_to_plot):
        ax = axes[idx]
        
        data = [
            outlier_features[feature].dropna(),
            normal_features[feature].dropna()
        ]
        
        bp = ax.boxplot(data, labels=['Outliers', 'Normal'], patch_artist=True)
        bp['boxes'][0].set_facecolor('gold')
        bp['boxes'][1].set_facecolor('lightblue')
        
        ax.set_title(feature.replace('_', ' ').title())
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Feature Distribution: Outliers vs Normal Profitable Trades', fontsize=12)
    plt.tight_layout()
    
    if save_path is None:
        save_path = PLOTS_DIR / "feature_distributions.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"âœ“ Saved feature distributions to {save_path}")


def plot_correlation_heatmap(df: pd.DataFrame, trades_df: pd.DataFrame, save_path=None):
    """
    Create correlation heatmap of features for profitable trades.
    """
    if len(trades_df) == 0:
        return
    
    # Get features at entry times
    features_list = []
    feature_cols = ['avg_iv', 'iv_spread', 'pcr_oi', 'atr_14', 'ema_diff',
                    'atm_call_delta', 'atm_call_gamma', 'futures_basis', 'spot_returns']
    feature_cols = [c for c in feature_cols if c in df.columns]
    
    for _, trade in trades_df.iterrows():
        entry_time = trade['entry_time']
        mask = df['datetime'] == entry_time
        
        if mask.any():
            row = df[mask].iloc[0]
            features = {col: row[col] for col in feature_cols}
            features['pnl'] = trade['pnl']
            features_list.append(features)
    
    if len(features_list) == 0:
        return
    
    features_df = pd.DataFrame(features_list)
    
    # Calculate correlation matrix
    corr = features_df.corr()
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    mask = np.triu(np.ones_like(corr), k=1)
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdYlBu_r',
                center=0, ax=ax, square=True)
    
    ax.set_title('Feature Correlation Heatmap')
    plt.tight_layout()
    
    if save_path is None:
        save_path = PLOTS_DIR / "correlation_heatmap.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"âœ“ Saved correlation heatmap to {save_path}")


def plot_time_distribution(trades_df: pd.DataFrame, outlier_mask: pd.Series = None,
                          save_path=None):
    """
    Create histogram of trade entry times.
    """
    if len(trades_df) == 0:
        return
    
    trades_df = trades_df.copy()
    trades_df['hour'] = pd.to_datetime(trades_df['entry_time']).dt.hour
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    hours = range(9, 16)  # 9 AM to 3 PM
    
    normal_hours = trades_df[~outlier_mask]['hour'] if outlier_mask is not None else trades_df['hour']
    ax.hist(normal_hours, bins=range(9, 17), alpha=0.6, 
            color='blue', label='Normal trades', edgecolor='black')
    
    if outlier_mask is not None and outlier_mask.any():
        outlier_hours = trades_df[outlier_mask]['hour']
        ax.hist(outlier_hours, bins=range(9, 17), alpha=0.8,
                color='red', label='Outliers', edgecolor='black')
    
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Number of Trades')
    ax.set_title('Trade Time Distribution')
    ax.set_xticks(range(9, 16))
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path is None:
        save_path = PLOTS_DIR / "time_distribution.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"âœ“ Saved time distribution to {save_path}")


def generate_trade_analysis_report(
    trades_df: pd.DataFrame,
    outlier_trades: pd.DataFrame,
    analysis: Dict,
    save_path=None
) -> str:
    """
    Generate comprehensive trade analysis report.
    """
    report_lines = [
        "=" * 60,
        "HIGH-PERFORMANCE TRADE ANALYSIS REPORT",
        "=" * 60,
        "",
        "OUTLIER DETECTION SUMMARY",
        "-" * 40,
        f"  Total trades analyzed:     {len(trades_df)}",
        f"  Outlier trades (Z>3):      {len(outlier_trades)}",
        f"  Outlier percentage:        {len(outlier_trades)/len(trades_df)*100:.2f}%",
    ]
    
    if len(outlier_trades) > 0:
        report_lines.extend([
            "",
            f"  Outlier total PnL:         {outlier_trades['pnl'].sum():.2f} pts",
            f"  Outlier mean PnL:          {outlier_trades['pnl'].mean():.2f} pts",
            f"  Best trade:                {outlier_trades['pnl'].max():.2f} pts",
        ])
    
    if analysis:
        report_lines.extend([
            "",
            "FEATURE ANALYSIS (Outliers vs Normal Profitable)",
            "-" * 40
        ])
        
        for feature, stats_dict in analysis.items():
            if feature == 'regime_distribution':
                continue
            
            report_lines.append(f"\n  {feature.upper()}:")
            report_lines.append(f"    Outlier mean:   {stats_dict['outlier_mean']:.4f}")
            report_lines.append(f"    Normal mean:    {stats_dict['normal_mean']:.4f}")
            report_lines.append(f"    T-test p-value: {stats_dict['t_pvalue']:.4f}")
            
            if stats_dict['significant']:
                report_lines.append(f"    *** STATISTICALLY SIGNIFICANT ***")
        
        if 'regime_distribution' in analysis:
            report_lines.extend([
                "",
                "REGIME DISTRIBUTION",
                "-" * 40
            ])
            regime_dist = analysis['regime_distribution']
            for regime_type in ['outlier', 'normal']:
                report_lines.append(f"\n  {regime_type.title()} trades:")
                for regime, pct in regime_dist[regime_type].items():
                    regime_name = {1: 'Uptrend', -1: 'Downtrend', 0: 'Sideways'}.get(regime, str(regime))
                    report_lines.append(f"    {regime_name}: {pct*100:.1f}%")
    
    report_lines.extend(["", "=" * 60])
    
    report = "\n".join(report_lines)
    
    if save_path is None:
        save_path = RESULTS_DIR / "trade_analysis_report.txt"
    
    with open(save_path, 'w') as f:
        f.write(report)
    
    print(report)
    return report


def run_full_trade_analysis(df: pd.DataFrame, trades_df: pd.DataFrame):
    """
    Run full trade analysis pipeline.
    """
    print("=" * 60)
    print("TRADE ANALYSIS PIPELINE")
    print("=" * 60)
    
    if len(trades_df) == 0:
        print("No trades to analyze.")
        return
    
    # 1. Identify outliers
    print("\n--- Identifying Outlier Trades ---")
    trades_df = trades_df.copy()
    outlier_trades = identify_outlier_trades(trades_df, z_threshold=3.0)
    print(f"Found {len(outlier_trades)} outlier trades out of {len(trades_df)}")
    
    # 2. Analyze features
    print("\n--- Analyzing Features ---")
    if len(outlier_trades) > 0:
        analysis, outlier_features, normal_features = analyze_outlier_features(
            df, outlier_trades, trades_df
        )
        
        # Print significant findings
        for feature, stats_dict in analysis.items():
            if feature != 'regime_distribution' and stats_dict.get('significant', False):
                print(f"  Significant difference in {feature}: "
                      f"outlier={stats_dict['outlier_mean']:.3f}, "
                      f"normal={stats_dict['normal_mean']:.3f}")
    else:
        analysis = {}
        outlier_features = pd.DataFrame()
        normal_features = pd.DataFrame()
    
    # 3. Create visualizations
    print("\n--- Creating Visualizations ---")
    outlier_mask = trades_df.index.isin(outlier_trades.index)
    
    plot_pnl_vs_duration(trades_df, outlier_mask)
    
    if len(outlier_features) > 0 and len(normal_features) > 0:
        plot_feature_distributions(outlier_features, normal_features)
    
    plot_correlation_heatmap(df, trades_df)
    plot_time_distribution(trades_df, outlier_mask)
    
    # 4. Generate report
    print("\n--- Generating Report ---")
    generate_trade_analysis_report(trades_df, outlier_trades, analysis)
    
    # Save outlier trades
    if len(outlier_trades) > 0:
        outlier_trades.to_csv(RESULTS_DIR / "outlier_trades.csv", index=False)
        print(f"âœ“ Saved outlier trades to {RESULTS_DIR / 'outlier_trades.csv'}")
    
    print("\n" + "=" * 60)
    print("TRADE ANALYSIS COMPLETE")
    print("=" * 60)
    
    return outlier_trades, analysis
