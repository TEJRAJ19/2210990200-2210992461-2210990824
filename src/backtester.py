"""
Backtesting Module
Calculates performance metrics and generates reports.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any

from config import (
    FEATURES_DATA_PATH, PLOTS_DIR, RESULTS_DIR,
    TRAIN_RATIO, TEST_RATIO, TRADING_DAYS_PER_YEAR
)


def calculate_metrics(trades_df: pd.DataFrame, equity_curve: pd.Series,
                      initial_capital: float = 100000) -> Dict[str, Any]:
    """
    Calculate all required backtesting metrics.
    
    Metrics:
    - Total Return
    - Sharpe Ratio
    - Sortino Ratio
    - Calmar Ratio
    - Max Drawdown
    - Win Rate
    - Profit Factor
    - Average Trade Duration
    - Total Trades
    """
    metrics = {}
    
    if len(trades_df) == 0:
        return {
            'total_return': 0,
            'sharpe_ratio': 0,
            'sortino_ratio': 0,
            'calmar_ratio': 0,
            'max_drawdown': 0,
            'max_drawdown_pct': 0,
            'win_rate': 0,
            'profit_factor': 0,
            'avg_trade_duration': 0,
            'total_trades': 0,
            'avg_pnl': 0,
            'avg_win': 0,
            'avg_loss': 0
        }
    
    # Total Return
    total_pnl = trades_df['pnl'].sum()
    metrics['total_return'] = total_pnl
    metrics['total_return_pct'] = (total_pnl / initial_capital) * 100
    
    # Total Trades
    metrics['total_trades'] = len(trades_df)
    
    # Win Rate
    winning_trades = trades_df[trades_df['pnl'] > 0]
    losing_trades = trades_df[trades_df['pnl'] <= 0]
    metrics['win_rate'] = len(winning_trades) / len(trades_df) * 100 if len(trades_df) > 0 else 0
    
    # Average PnL
    metrics['avg_pnl'] = trades_df['pnl'].mean()
    metrics['avg_win'] = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
    metrics['avg_loss'] = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0
    
    # Profit Factor
    gross_profit = winning_trades['pnl'].sum() if len(winning_trades) > 0 else 0
    gross_loss = abs(losing_trades['pnl'].sum()) if len(losing_trades) > 0 else 1
    metrics['profit_factor'] = gross_profit / gross_loss if gross_loss != 0 else float('inf')
    
    # Average Trade Duration
    if 'duration' in trades_df.columns:
        metrics['avg_trade_duration'] = trades_df['duration'].mean()
    else:
        metrics['avg_trade_duration'] = 0
    
    # Calculate drawdown from equity curve
    cumulative_max = equity_curve.cummax()
    drawdown = equity_curve - cumulative_max
    metrics['max_drawdown'] = drawdown.min()
    metrics['max_drawdown_pct'] = (metrics['max_drawdown'] / cumulative_max.max()) * 100 if cumulative_max.max() != 0 else 0
    
    # Calculate returns for ratio calculations
    # Assuming equity curve is in points, calculate daily returns
    returns = equity_curve.pct_change().dropna()
    returns = returns.replace([np.inf, -np.inf], 0)
    
    # Annualization factor (assuming 5-min candles, 75 per day)
    candles_per_year = TRADING_DAYS_PER_YEAR * 75
    annualization_factor = np.sqrt(candles_per_year)
    
    # Sharpe Ratio (assuming risk-free rate is negligible for short-term)
    if returns.std() != 0:
        metrics['sharpe_ratio'] = (returns.mean() / returns.std()) * annualization_factor
    else:
        metrics['sharpe_ratio'] = 0
    
    # Sortino Ratio (uses only downside deviation)
    downside_returns = returns[returns < 0]
    if len(downside_returns) > 0 and downside_returns.std() != 0:
        metrics['sortino_ratio'] = (returns.mean() / downside_returns.std()) * annualization_factor
    else:
        metrics['sortino_ratio'] = metrics['sharpe_ratio']
    
    # Calmar Ratio (Annual Return / Max Drawdown)
    annual_return = returns.mean() * candles_per_year
    if metrics['max_drawdown_pct'] != 0:
        metrics['calmar_ratio'] = abs(annual_return / metrics['max_drawdown_pct'])
    else:
        metrics['calmar_ratio'] = 0
    
    return metrics


def split_train_test(df: pd.DataFrame, train_ratio: float = TRAIN_RATIO) -> tuple:
    """
    Split data into training and testing sets.
    
    Args:
        df: Full dataset
        train_ratio: Ratio of data for training
    
    Returns:
        Tuple of (train_df, test_df)
    """
    split_idx = int(len(df) * train_ratio)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    
    return train_df, test_df


def plot_equity_curve(df: pd.DataFrame, title: str = "Equity Curve",
                      save_path=None):
    """Plot equity curve with drawdown."""
    fig, axes = plt.subplots(2, 1, figsize=(15, 10), height_ratios=[3, 1])
    
    # Equity curve
    ax1 = axes[0]
    equity = df['cumulative_pnl']
    ax1.plot(df['datetime'], equity, 'b-', linewidth=1.5, label='Equity')
    ax1.fill_between(df['datetime'], 0, equity, where=(equity > 0), 
                     alpha=0.3, color='green', label='Profit')
    ax1.fill_between(df['datetime'], 0, equity, where=(equity < 0), 
                     alpha=0.3, color='red', label='Loss')
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax1.set_xlabel('')
    ax1.set_ylabel('Cumulative PnL (points)')
    ax1.set_title(title)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Drawdown
    ax2 = axes[1]
    cummax = equity.cummax()
    drawdown = equity - cummax
    ax2.fill_between(df['datetime'], drawdown, 0, color='red', alpha=0.5)
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Drawdown')
    ax2.set_title('Drawdown')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path is None:
        save_path = PLOTS_DIR / "equity_curve.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"âœ“ Saved equity curve to {save_path}")


def plot_trade_distribution(trades_df: pd.DataFrame, save_path=None):
    """Plot trade PnL distribution."""
    if len(trades_df) == 0:
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # PnL distribution
    ax1 = axes[0]
    trades_df['pnl'].hist(ax=ax1, bins=30, edgecolor='black', alpha=0.7)
    ax1.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax1.axvline(x=trades_df['pnl'].mean(), color='blue', linestyle='--', 
                linewidth=2, label=f"Mean: {trades_df['pnl'].mean():.2f}")
    ax1.set_xlabel('PnL (points)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Trade PnL Distribution')
    ax1.legend()
    
    # Win/Loss pie chart
    ax2 = axes[1]
    wins = (trades_df['pnl'] > 0).sum()
    losses = (trades_df['pnl'] <= 0).sum()
    ax2.pie([wins, losses], labels=['Wins', 'Losses'], autopct='%1.1f%%',
            colors=['green', 'red'], startangle=90)
    ax2.set_title('Win/Loss Ratio')
    
    # Trade duration distribution
    ax3 = axes[2]
    if 'duration' in trades_df.columns:
        trades_df['duration'].hist(ax=ax3, bins=20, edgecolor='black', alpha=0.7)
        ax3.set_xlabel('Duration (candles)')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Trade Duration Distribution')
    
    plt.tight_layout()
    
    if save_path is None:
        save_path = PLOTS_DIR / "trade_distribution.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"âœ“ Saved trade distribution to {save_path}")


def plot_monthly_returns(df: pd.DataFrame, trades_df: pd.DataFrame, save_path=None):
    """Plot monthly returns heatmap."""
    if len(trades_df) == 0:
        return
    
    # Calculate monthly PnL
    trades_df = trades_df.copy()
    trades_df['month'] = pd.to_datetime(trades_df['exit_time']).dt.to_period('M')
    monthly_pnl = trades_df.groupby('month')['pnl'].sum()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 4))
    
    colors = ['red' if x < 0 else 'green' for x in monthly_pnl.values]
    bars = ax.bar(range(len(monthly_pnl)), monthly_pnl.values, color=colors, alpha=0.7)
    ax.set_xticks(range(len(monthly_pnl)))
    ax.set_xticklabels([str(m) for m in monthly_pnl.index], rotation=45)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Month')
    ax.set_ylabel('PnL (points)')
    ax.set_title('Monthly Returns')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path is None:
        save_path = PLOTS_DIR / "monthly_returns.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"âœ“ Saved monthly returns to {save_path}")


def generate_backtest_report(metrics: Dict[str, Any], trades_df: pd.DataFrame,
                            train_or_test: str = "Full", save_path=None) -> str:
    """Generate text report of backtest results."""
    report_lines = [
        "=" * 60,
        f"BACKTEST REPORT - {train_or_test.upper()} PERIOD",
        "=" * 60,
        "",
        "PERFORMANCE METRICS",
        "-" * 40,
        f"  Total Return:        {metrics['total_return']:.2f} pts ({metrics.get('total_return_pct', 0):.2f}%)",
        f"  Sharpe Ratio:        {metrics['sharpe_ratio']:.3f}",
        f"  Sortino Ratio:       {metrics['sortino_ratio']:.3f}",
        f"  Calmar Ratio:        {metrics['calmar_ratio']:.3f}",
        f"  Max Drawdown:        {metrics['max_drawdown']:.2f} pts ({abs(metrics['max_drawdown_pct']):.2f}%)",
        "",
        "TRADE STATISTICS",
        "-" * 40,
        f"  Total Trades:        {metrics['total_trades']}",
        f"  Win Rate:            {metrics['win_rate']:.2f}%",
        f"  Profit Factor:       {metrics['profit_factor']:.2f}",
        f"  Avg Trade PnL:       {metrics['avg_pnl']:.2f} pts",
        f"  Avg Win:             {metrics['avg_win']:.2f} pts",
        f"  Avg Loss:            {metrics['avg_loss']:.2f} pts",
        f"  Avg Duration:        {metrics['avg_trade_duration']:.1f} candles",
        "",
    ]
    
    if len(trades_df) > 0:
        # Trade breakdown by position type
        long_trades = trades_df[trades_df['position'] == 'LONG']
        short_trades = trades_df[trades_df['position'] == 'SHORT']
        
        report_lines.extend([
            "BREAKDOWN BY POSITION",
            "-" * 40,
            f"  Long Trades:         {len(long_trades)} (PnL: {long_trades['pnl'].sum():.2f} pts)",
            f"  Short Trades:        {len(short_trades)} (PnL: {short_trades['pnl'].sum():.2f} pts)",
        ])
        
        # Trade breakdown by regime
        if 'regime' in trades_df.columns:
            report_lines.extend([
                "",
                "BREAKDOWN BY REGIME",
                "-" * 40
            ])
            for regime in trades_df['regime'].unique():
                regime_trades = trades_df[trades_df['regime'] == regime]
                regime_name = {1: 'Uptrend', -1: 'Downtrend', 0: 'Sideways'}.get(regime, str(regime))
                report_lines.append(
                    f"  {regime_name}:          {len(regime_trades)} trades (PnL: {regime_trades['pnl'].sum():.2f} pts)"
                )
    
    report_lines.extend(["", "=" * 60])
    
    report = "\n".join(report_lines)
    
    if save_path is None:
        save_path = RESULTS_DIR / f"backtest_report_{train_or_test.lower()}.txt"
    
    with open(save_path, 'w') as f:
        f.write(report)
    
    print(report)
    return report


def run_full_backtest(df: pd.DataFrame, strategy) -> Dict[str, Any]:
    """
    Run full backtest with train/test split.
    
    Returns:
        Dictionary with results, metrics, and visualizations paths
    """
    print("=" * 60)
    print("BACKTESTING PIPELINE")
    print("=" * 60)
    
    from strategy import run_strategy
    
    # Split data
    train_df, test_df = split_train_test(df)
    print(f"\nData split:")
    print(f"  Training: {len(train_df)} rows ({TRAIN_RATIO*100:.0f}%)")
    print(f"  Testing:  {len(test_df)} rows ({TEST_RATIO*100:.0f}%)")
    
    results = {}
    
    # Backtest on training data
    print("\n--- Training Period Backtest ---")
    train_results, train_trades, _ = run_strategy(train_df, use_regime_filter=True)
    train_equity = train_results['cumulative_pnl'] + 100000  # Add initial capital
    train_metrics = calculate_metrics(train_trades, train_equity)
    generate_backtest_report(train_metrics, train_trades, "Training",
                            RESULTS_DIR / "backtest_report_train.txt")
    results['train'] = {
        'results': train_results,
        'trades': train_trades,
        'metrics': train_metrics
    }
    
    # Backtest on testing data
    print("\n--- Testing Period Backtest ---")
    test_results, test_trades, _ = run_strategy(test_df, use_regime_filter=True)
    test_equity = test_results['cumulative_pnl'] + 100000
    test_metrics = calculate_metrics(test_trades, test_equity)
    generate_backtest_report(test_metrics, test_trades, "Testing",
                            RESULTS_DIR / "backtest_report_test.txt")
    results['test'] = {
        'results': test_results,
        'trades': test_trades,
        'metrics': test_metrics
    }
    
    # Backtest on full data
    print("\n--- Full Period Backtest ---")
    full_results, full_trades, _ = run_strategy(df, use_regime_filter=True)
    full_equity = full_results['cumulative_pnl'] + 100000
    full_metrics = calculate_metrics(full_trades, full_equity)
    generate_backtest_report(full_metrics, full_trades, "Full",
                            RESULTS_DIR / "backtest_report_full.txt")
    results['full'] = {
        'results': full_results,
        'trades': full_trades,
        'metrics': full_metrics
    }
    
    # Generate visualizations
    print("\n--- Generating Visualizations ---")
    plot_equity_curve(full_results, "Equity Curve - Full Period",
                     PLOTS_DIR / "equity_curve_full.png")
    plot_trade_distribution(full_trades, PLOTS_DIR / "trade_distribution.png")
    plot_monthly_returns(full_results, full_trades, PLOTS_DIR / "monthly_returns.png")
    
    # Save trades to CSV
    if len(full_trades) > 0:
        full_trades.to_csv(RESULTS_DIR / "all_trades.csv", index=False)
        print(f"âœ“ Saved trades to {RESULTS_DIR / 'all_trades.csv'}")
    
    print("\n" + "=" * 60)
    print("BACKTESTING COMPLETE")
    print("=" * 60)
    
    return results
