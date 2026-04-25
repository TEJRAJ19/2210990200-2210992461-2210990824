"""
Trading Strategy Module
Implements 5/15 EMA crossover strategy with regime filter.
"""

import pandas as pd
import numpy as np
from enum import Enum
from dataclasses import dataclass
from typing import List, Optional

from config import (
    REGIME_UPTREND, REGIME_DOWNTREND, REGIME_SIDEWAYS,
    EMA_FAST, EMA_SLOW
)


class Position(Enum):
    """Position type enum."""
    LONG = 1
    SHORT = -1
    FLAT = 0


@dataclass
class Trade:
    """Represents a single trade."""
    entry_time: pd.Timestamp
    exit_time: Optional[pd.Timestamp]
    entry_price: float
    exit_price: Optional[float]
    position: Position
    regime: int
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None
    duration: Optional[int] = None  # in candles


class EMAStrategy:
    """
    5/15 EMA Crossover Strategy with Regime Filter.
    
    Rules:
    - LONG Entry: 5 EMA crosses above 15 EMA AND Regime == +1 (Uptrend)
    - LONG Exit: 5 EMA crosses below 15 EMA
    - SHORT Entry: 5 EMA crosses below 15 EMA AND Regime == -1 (Downtrend)
    - SHORT Exit: 5 EMA crosses above 15 EMA
    - No trades when Regime == 0 (Sideways)
    - Entry/Exit at next candle open
    """
    
    def __init__(self, use_regime_filter: bool = True):
        self.use_regime_filter = use_regime_filter
        self.trades: List[Trade] = []
        self.current_position = Position.FLAT
        self.current_trade: Optional[Trade] = None
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on EMA crossover and regime.
        
        Args:
            df: DataFrame with ema_5, ema_15, and regime columns
        
        Returns:
            DataFrame with added signal columns
        """
        df = df.copy()
        
        # Ensure required columns exist
        required_cols = ['ema_5', 'ema_15', 'spot_close', 'spot_open']
        if self.use_regime_filter:
            required_cols.append('regime')
        
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing column: {col}")
        
        # EMA difference and crossover detection
        df['ema_diff'] = df['ema_5'] - df['ema_15']
        df['ema_diff_prev'] = df['ema_diff'].shift(1)
        
        # Crossover signals
        df['bullish_cross'] = (df['ema_diff'] > 0) & (df['ema_diff_prev'] <= 0)
        df['bearish_cross'] = (df['ema_diff'] < 0) & (df['ema_diff_prev'] >= 0)
        
        # Raw signals (1 = long, -1 = short, 0 = no action)
        df['raw_signal'] = 0
        df.loc[df['bullish_cross'], 'raw_signal'] = 1
        df.loc[df['bearish_cross'], 'raw_signal'] = -1
        
        # Apply regime filter
        if self.use_regime_filter:
            df['signal'] = 0
            # Long only in uptrend
            df.loc[(df['raw_signal'] == 1) & (df['regime'] == REGIME_UPTREND), 'signal'] = 1
            # Short only in downtrend
            df.loc[(df['raw_signal'] == -1) & (df['regime'] == REGIME_DOWNTREND), 'signal'] = -1
            # Still need exit signals regardless of regime
            df['exit_long'] = df['bearish_cross']
            df['exit_short'] = df['bullish_cross']
        else:
            df['signal'] = df['raw_signal']
            df['exit_long'] = df['bearish_cross']
            df['exit_short'] = df['bullish_cross']
        
        return df
    
    def run_backtest(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run backtest on the data with generated signals.
        
        Args:
            df: DataFrame with signals
        
        Returns:
            DataFrame with position and equity columns
        """
        df = self.generate_signals(df)
        df = df.copy()
        
        # Initialize columns
        df['position'] = 0  # Current position
        df['entry_price'] = np.nan
        df['trade_pnl'] = 0.0
        df['cumulative_pnl'] = 0.0
        
        self.trades = []
        self.current_position = Position.FLAT
        self.current_trade = None
        
        for i in range(1, len(df)):
            prev_idx = df.index[i - 1]
            curr_idx = df.index[i]
            
            # Get current state
            signal = df.loc[prev_idx, 'signal']  # Signal from previous candle
            entry_price = df.loc[curr_idx, 'spot_open']  # Enter at next candle open
            current_regime = df.loc[prev_idx, 'regime'] if 'regime' in df.columns else 0
            
            # Handle exits first (exits also at next candle open)
            if self.current_position == Position.LONG and df.loc[prev_idx, 'exit_long']:
                # Exit long position
                self._close_trade(df.loc[curr_idx, 'datetime'], entry_price, i)
                self.current_position = Position.FLAT
                
            elif self.current_position == Position.SHORT and df.loc[prev_idx, 'exit_short']:
                # Exit short position
                self._close_trade(df.loc[curr_idx, 'datetime'], entry_price, i)
                self.current_position = Position.FLAT
            
            # Handle entries (only if flat)
            if self.current_position == Position.FLAT:
                if signal == 1:  # Long signal
                    self.current_position = Position.LONG
                    self._open_trade(
                        df.loc[curr_idx, 'datetime'],
                        entry_price,
                        Position.LONG,
                        current_regime
                    )
                elif signal == -1:  # Short signal
                    self.current_position = Position.SHORT
                    self._open_trade(
                        df.loc[curr_idx, 'datetime'],
                        entry_price,
                        Position.SHORT,
                        current_regime
                    )
            
            # Record position
            df.loc[curr_idx, 'position'] = self.current_position.value
            if self.current_trade:
                df.loc[curr_idx, 'entry_price'] = self.current_trade.entry_price
        
        # Close any remaining open trade at the end
        if self.current_trade is not None:
            last_idx = df.index[-1]
            self._close_trade(
                df.loc[last_idx, 'datetime'],
                df.loc[last_idx, 'spot_close'],
                len(df) - 1
            )
        
        # Calculate cumulative PnL
        df = self._calculate_equity_curve(df)
        
        return df
    
    def _open_trade(self, time: pd.Timestamp, price: float, 
                    position: Position, regime: int):
        """Open a new trade."""
        self.current_trade = Trade(
            entry_time=time,
            exit_time=None,
            entry_price=price,
            exit_price=None,
            position=position,
            regime=regime
        )
    
    def _close_trade(self, time: pd.Timestamp, price: float, candle_idx: int):
        """Close the current trade."""
        if self.current_trade is None:
            return
        
        self.current_trade.exit_time = time
        self.current_trade.exit_price = price
        
        # Calculate PnL
        if self.current_trade.position == Position.LONG:
            self.current_trade.pnl = price - self.current_trade.entry_price
        else:  # SHORT
            self.current_trade.pnl = self.current_trade.entry_price - price
        
        self.current_trade.pnl_pct = (
            self.current_trade.pnl / self.current_trade.entry_price * 100
        )
        
        # Calculate duration
        entry_idx = None
        for i, t in enumerate(self.trades):
            pass  # Just counting trades
        self.current_trade.duration = candle_idx  # Simplified
        
        self.trades.append(self.current_trade)
        self.current_trade = None
    
    def _calculate_equity_curve(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate equity curve from trades."""
        df = df.copy()
        df['trade_pnl'] = 0.0
        
        for trade in self.trades:
            if trade.exit_time and trade.pnl:
                # Find the exit candle and record PnL
                mask = df['datetime'] == trade.exit_time
                if mask.any():
                    df.loc[mask, 'trade_pnl'] = trade.pnl
        
        df['cumulative_pnl'] = df['trade_pnl'].cumsum()
        df['cumulative_pnl_pct'] = df['cumulative_pnl'] / df['spot_close'].iloc[0] * 100
        
        return df
    
    def get_trades_df(self) -> pd.DataFrame:
        """Convert trades to DataFrame."""
        if not self.trades:
            return pd.DataFrame()
        
        trades_data = []
        for trade in self.trades:
            trades_data.append({
                'entry_time': trade.entry_time,
                'exit_time': trade.exit_time,
                'entry_price': trade.entry_price,
                'exit_price': trade.exit_price,
                'position': trade.position.name,
                'regime': trade.regime,
                'pnl': trade.pnl,
                'pnl_pct': trade.pnl_pct,
                'duration': trade.duration
            })
        
        return pd.DataFrame(trades_data)


def run_strategy(df: pd.DataFrame, use_regime_filter: bool = True) -> tuple:
    """
    Convenience function to run the EMA strategy.
    
    Args:
        df: DataFrame with required columns
        use_regime_filter: Whether to filter by regime
    
    Returns:
        Tuple of (results_df, trades_df, strategy)
    """
    strategy = EMAStrategy(use_regime_filter=use_regime_filter)
    results = strategy.run_backtest(df)
    trades = strategy.get_trades_df()
    
    return results, trades, strategy
