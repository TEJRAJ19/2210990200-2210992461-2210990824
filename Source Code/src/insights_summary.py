"""
Insights Summary Module
Generates comprehensive insights summary for Task 6.3
"""

import pandas as pd
import numpy as np
from datetime import datetime

from config import FEATURES_DATA_PATH, RESULTS_DIR, PLOTS_DIR


def generate_insights_summary(df: pd.DataFrame, trades_df: pd.DataFrame) -> dict:
    """
    Generate comprehensive insights summary answering:
    1. What percentage are outliers?
    2. Average PnL comparison
    3. Regime patterns
    4. Time-of-day patterns
    5. IV characteristics
    6. Distinguishing features
    """
    insights = {}
    
    if len(trades_df) == 0:
        return insights
    
    trades_df = trades_df.copy()
    
    # ============================================================
    # 1. OUTLIER PERCENTAGE
    # ============================================================
    mean_pnl = trades_df['pnl'].mean()
    std_pnl = trades_df['pnl'].std()
    
    if std_pnl > 0:
        trades_df['z_score'] = (trades_df['pnl'] - mean_pnl) / std_pnl
    else:
        trades_df['z_score'] = 0
    
    outliers = trades_df[trades_df['z_score'] > 3]
    profitable_non_outliers = trades_df[(trades_df['pnl'] > 0) & (trades_df['z_score'] <= 3)]
    losing_trades = trades_df[trades_df['pnl'] <= 0]
    
    insights['outlier_analysis'] = {
        'total_trades': len(trades_df),
        'outlier_count': len(outliers),
        'outlier_percentage': round(len(outliers) / len(trades_df) * 100, 2),
        'outlier_pnl_contribution': round(outliers['pnl'].sum(), 2),
        'outlier_pnl_pct_of_total': round(outliers['pnl'].sum() / trades_df['pnl'].sum() * 100, 2) if trades_df['pnl'].sum() != 0 else 0
    }
    
    # ============================================================
    # 2. AVERAGE PNL COMPARISON
    # ============================================================
    insights['pnl_comparison'] = {
        'outlier_avg_pnl': round(outliers['pnl'].mean(), 2) if len(outliers) > 0 else 0,
        'normal_profitable_avg_pnl': round(profitable_non_outliers['pnl'].mean(), 2) if len(profitable_non_outliers) > 0 else 0,
        'losing_avg_pnl': round(losing_trades['pnl'].mean(), 2) if len(losing_trades) > 0 else 0,
        'overall_avg_pnl': round(trades_df['pnl'].mean(), 2),
        'outlier_median_pnl': round(outliers['pnl'].median(), 2) if len(outliers) > 0 else 0,
        'normal_profitable_median_pnl': round(profitable_non_outliers['pnl'].median(), 2) if len(profitable_non_outliers) > 0 else 0,
        'pnl_ratio_outlier_to_normal': round(outliers['pnl'].mean() / profitable_non_outliers['pnl'].mean(), 2) if len(outliers) > 0 and len(profitable_non_outliers) > 0 and profitable_non_outliers['pnl'].mean() != 0 else 0
    }
    
    # ============================================================
    # 3. REGIME PATTERNS
    # ============================================================
    # Match trades to candle data for regime info
    trade_features = []
    for _, trade in trades_df.iterrows():
        entry_time = trade['entry_time']
        mask = df['datetime'] == entry_time
        if mask.any():
            row = df[mask].iloc[0]
            trade_features.append({
                'pnl': trade['pnl'],
                'is_outlier': trade['z_score'] > 3,
                'regime': trade.get('regime', row.get('regime', 0)),
                'hour': pd.to_datetime(entry_time).hour,
                'avg_iv': row.get('avg_iv', 0),
                'iv_spread': row.get('iv_spread', 0),
                'atr': row.get('atr_14', 0),
                'ema_gap': abs(row.get('ema_diff', 0)),
                'pcr_oi': row.get('pcr_oi', 1),
                'atm_call_delta': row.get('atm_call_delta', 0.5),
                'volatility': row.get('volatility_20', 0)
            })
    
    features_df = pd.DataFrame(trade_features)
    
    if len(features_df) > 0:
        # Regime distribution for all trades
        regime_names = {1: 'Uptrend', -1: 'Downtrend', 0: 'Sideways'}
        regime_stats = {}
        
        for regime in features_df['regime'].unique():
            regime_trades = features_df[features_df['regime'] == regime]
            regime_outliers = regime_trades[regime_trades['is_outlier']]
            
            regime_stats[regime_names.get(regime, str(regime))] = {
                'trade_count': len(regime_trades),
                'trade_pct': round(len(regime_trades) / len(features_df) * 100, 2),
                'avg_pnl': round(regime_trades['pnl'].mean(), 2),
                'total_pnl': round(regime_trades['pnl'].sum(), 2),
                'outlier_count': len(regime_outliers),
                'outlier_pct_in_regime': round(len(regime_outliers) / len(regime_trades) * 100, 2) if len(regime_trades) > 0 else 0
            }
        
        insights['regime_patterns'] = regime_stats
        
        # ============================================================
        # 4. TIME-OF-DAY PATTERNS
        # ============================================================
        time_stats = {}
        time_periods = {
            'First Hour (9:15-10:15)': (9, 10),
            'Mid-Morning (10:15-12:00)': (10, 12),
            'Afternoon (12:00-14:00)': (12, 14),
            'Last Hour (14:00-15:30)': (14, 16)
        }
        
        for period_name, (start_hour, end_hour) in time_periods.items():
            period_trades = features_df[(features_df['hour'] >= start_hour) & (features_df['hour'] < end_hour)]
            period_outliers = period_trades[period_trades['is_outlier']]
            
            time_stats[period_name] = {
                'trade_count': len(period_trades),
                'trade_pct': round(len(period_trades) / len(features_df) * 100, 2) if len(features_df) > 0 else 0,
                'avg_pnl': round(period_trades['pnl'].mean(), 2) if len(period_trades) > 0 else 0,
                'outlier_count': len(period_outliers),
                'win_rate': round((period_trades['pnl'] > 0).sum() / len(period_trades) * 100, 2) if len(period_trades) > 0 else 0
            }
        
        insights['time_of_day_patterns'] = time_stats
        
        # ============================================================
        # 5. IV CHARACTERISTICS
        # ============================================================
        outlier_features = features_df[features_df['is_outlier']]
        normal_features = features_df[~features_df['is_outlier'] & (features_df['pnl'] > 0)]
        
        iv_stats = {
            'outlier_avg_iv': round(outlier_features['avg_iv'].mean(), 4) if len(outlier_features) > 0 else 0,
            'normal_avg_iv': round(normal_features['avg_iv'].mean(), 4) if len(normal_features) > 0 else 0,
            'overall_avg_iv': round(features_df['avg_iv'].mean(), 4),
            'outlier_iv_spread': round(outlier_features['iv_spread'].mean(), 4) if len(outlier_features) > 0 else 0,
            'normal_iv_spread': round(normal_features['iv_spread'].mean(), 4) if len(normal_features) > 0 else 0,
            'high_iv_trades': len(features_df[features_df['avg_iv'] > features_df['avg_iv'].quantile(0.75)]),
            'high_iv_win_rate': round((features_df[features_df['avg_iv'] > features_df['avg_iv'].quantile(0.75)]['pnl'] > 0).mean() * 100, 2)
        }
        
        insights['iv_characteristics'] = iv_stats
        
        # ============================================================
        # 6. DISTINGUISHING FEATURES
        # ============================================================
        # Calculate statistical differences between outliers and normal trades
        from scipy import stats
        
        distinguishing = {}
        feature_cols = ['atr', 'ema_gap', 'pcr_oi', 'avg_iv', 'volatility', 'atm_call_delta']
        
        for col in feature_cols:
            if col in outlier_features.columns and col in normal_features.columns:
                outlier_vals = outlier_features[col].dropna()
                normal_vals = normal_features[col].dropna()
                
                if len(outlier_vals) >= 2 and len(normal_vals) >= 2:
                    t_stat, p_val = stats.ttest_ind(outlier_vals, normal_vals)
                    
                    distinguishing[col] = {
                        'outlier_mean': round(outlier_vals.mean(), 4),
                        'normal_mean': round(normal_vals.mean(), 4),
                        'difference_pct': round((outlier_vals.mean() - normal_vals.mean()) / normal_vals.mean() * 100, 2) if normal_vals.mean() != 0 else 0,
                        'p_value': round(p_val, 4),
                        'significant': p_val < 0.05
                    }
        
        # Rank by significance
        significant_features = {k: v for k, v in distinguishing.items() if v['significant']}
        insights['distinguishing_features'] = {
            'all_features': distinguishing,
            'significant_features': list(significant_features.keys()),
            'most_distinguishing': max(distinguishing.items(), key=lambda x: abs(x[1]['difference_pct']))[0] if distinguishing else None
        }
    
    return insights


def format_insights_report(insights: dict) -> str:
    """Format insights as a readable text report."""
    lines = [
        "=" * 70,
        "TASK 6.3: INSIGHTS SUMMARY",
        "=" * 70,
        "",
    ]
    
    # 1. Outlier Percentage
    if 'outlier_analysis' in insights:
        oa = insights['outlier_analysis']
        lines.extend([
            "1. OUTLIER ANALYSIS",
            "-" * 50,
            f"   Total Trades:           {oa['total_trades']}",
            f"   Outlier Trades:         {oa['outlier_count']}",
            f"   Outlier Percentage:     {oa['outlier_percentage']}%",
            f"   Outlier PnL:            {oa['outlier_pnl_contribution']} pts",
            f"   % of Total PnL:         {oa['outlier_pnl_pct_of_total']}%",
            ""
        ])
    
    # 2. Average PnL Comparison
    if 'pnl_comparison' in insights:
        pc = insights['pnl_comparison']
        lines.extend([
            "2. AVERAGE PNL COMPARISON",
            "-" * 50,
            f"   Outlier Avg PnL:        {pc['outlier_avg_pnl']} pts",
            f"   Normal Profitable Avg:  {pc['normal_profitable_avg_pnl']} pts",
            f"   Losing Trade Avg:       {pc['losing_avg_pnl']} pts",
            f"   Overall Avg:            {pc['overall_avg_pnl']} pts",
            f"   Outlier/Normal Ratio:   {pc['pnl_ratio_outlier_to_normal']}x",
            ""
        ])
    
    # 3. Regime Patterns
    if 'regime_patterns' in insights:
        lines.extend([
            "3. REGIME PATTERNS",
            "-" * 50
        ])
        for regime, stats in insights['regime_patterns'].items():
            lines.extend([
                f"   {regime}:",
                f"      Trades: {stats['trade_count']} ({stats['trade_pct']}%)",
                f"      Avg PnL: {stats['avg_pnl']} pts",
                f"      Total PnL: {stats['total_pnl']} pts",
                f"      Outliers in Regime: {stats['outlier_count']} ({stats['outlier_pct_in_regime']}%)",
            ])
        lines.append("")
    
    # 4. Time-of-Day Patterns
    if 'time_of_day_patterns' in insights:
        lines.extend([
            "4. TIME-OF-DAY PATTERNS",
            "-" * 50
        ])
        for period, stats in insights['time_of_day_patterns'].items():
            lines.extend([
                f"   {period}:",
                f"      Trades: {stats['trade_count']} ({stats['trade_pct']}%)",
                f"      Avg PnL: {stats['avg_pnl']} pts",
                f"      Win Rate: {stats['win_rate']}%",
                f"      Outliers: {stats['outlier_count']}",
            ])
        lines.append("")
    
    # 5. IV Characteristics
    if 'iv_characteristics' in insights:
        iv = insights['iv_characteristics']
        lines.extend([
            "5. IV CHARACTERISTICS",
            "-" * 50,
            f"   Outlier Avg IV:         {iv['outlier_avg_iv']}",
            f"   Normal Avg IV:          {iv['normal_avg_iv']}",
            f"   Overall Avg IV:         {iv['overall_avg_iv']}",
            f"   Outlier IV Spread:      {iv['outlier_iv_spread']}",
            f"   Normal IV Spread:       {iv['normal_iv_spread']}",
            f"   High IV Trades:         {iv['high_iv_trades']}",
            f"   High IV Win Rate:       {iv['high_iv_win_rate']}%",
            ""
        ])
    
    # 6. Distinguishing Features
    if 'distinguishing_features' in insights:
        df_stats = insights['distinguishing_features']
        lines.extend([
            "6. DISTINGUISHING FEATURES",
            "-" * 50,
            f"   Most Distinguishing:    {df_stats['most_distinguishing']}",
            f"   Significant Features:   {', '.join(df_stats['significant_features']) if df_stats['significant_features'] else 'None'}",
            ""
        ])
        
        if df_stats['all_features']:
            lines.append("   Feature Details:")
            for feature, stats in df_stats['all_features'].items():
                sig = "***" if stats['significant'] else ""
                lines.append(
                    f"      {feature}: Outlier={stats['outlier_mean']}, "
                    f"Normal={stats['normal_mean']}, "
                    f"Diff={stats['difference_pct']}%, "
                    f"p={stats['p_value']} {sig}"
                )
        lines.append("")
    
    # Key Takeaways
    lines.extend([
        "=" * 70,
        "KEY TAKEAWAYS",
        "=" * 70,
    ])
    
    if 'outlier_analysis' in insights and 'pnl_comparison' in insights:
        oa = insights['outlier_analysis']
        pc = insights['pnl_comparison']
        lines.append(f"1. Only {oa['outlier_percentage']}% of trades are outliers, but they contribute {oa['outlier_pnl_pct_of_total']}% of total PnL")
    
    if 'pnl_comparison' in insights:
        pc = insights['pnl_comparison']
        lines.append(f"2. Outlier trades earn {pc['pnl_ratio_outlier_to_normal']}x more than normal profitable trades")
    
    if 'distinguishing_features' in insights:
        df_stats = insights['distinguishing_features']
        if df_stats['significant_features']:
            lines.append(f"3. Key distinguishing features: {', '.join(df_stats['significant_features'])}")
    
    if 'regime_patterns' in insights:
        best_regime = max(insights['regime_patterns'].items(), key=lambda x: x[1]['avg_pnl'])
        lines.append(f"4. Best regime for trading: {best_regime[0]} (Avg PnL: {best_regime[1]['avg_pnl']} pts)")
    
    if 'time_of_day_patterns' in insights:
        best_time = max(insights['time_of_day_patterns'].items(), key=lambda x: x[1]['avg_pnl'])
        lines.append(f"5. Best time to trade: {best_time[0]} (Avg PnL: {best_time[1]['avg_pnl']} pts)")
    
    lines.extend(["", "=" * 70])
    
    return "\n".join(lines)


def run_insights_summary(df: pd.DataFrame = None, trades_df: pd.DataFrame = None):
    """
    Run the complete insights summary pipeline.
    """
    print("=" * 60)
    print("TASK 6.3: GENERATING INSIGHTS SUMMARY")
    print("=" * 60)
    
    # Load data if not provided
    if df is None:
        df = pd.read_csv(FEATURES_DATA_PATH)
        df['datetime'] = pd.to_datetime(df['datetime'])
    
    if trades_df is None:
        trades_path = RESULTS_DIR / "all_trades.csv"
        if trades_path.exists():
            trades_df = pd.read_csv(trades_path)
            trades_df['entry_time'] = pd.to_datetime(trades_df['entry_time'])
            trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])
        else:
            print("No trades file found. Run the strategy first.")
            return None, None
    
    # Generate insights
    insights = generate_insights_summary(df, trades_df)
    
    # Format and print report
    report = format_insights_report(insights)
    print(report)
    
    # Save report
    report_path = RESULTS_DIR / "insights_summary.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\n[OK] Saved insights summary to {report_path}")
    
    return insights, report


if __name__ == "__main__":
    run_insights_summary()
