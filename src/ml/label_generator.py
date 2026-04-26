# -*- coding: utf-8 -*-
"""
===================================
Label Generator — 主升浪标签生成器
===================================

Forward-looking label construction for the "main wave" (主升浪) classifier.

Definition of a positive label (主升浪):
  Within the next `horizon` trading days, the stock must satisfy ALL of:
    1. Max return >= `min_return_pct`   (default 30 %)
    2. Max drawdown <= `max_drawdown_pct` (default 15 %)
  AND at least `min_extra_conditions` of the following:
    a) Max single-day volume >= `volume_surge_ratio` × 30-day avg volume
    b) Price breaks above 60-day rolling high
    c) MACD DIF > 0 for `macd_above_zero_days` consecutive days

Noise filters (rows excluded from labelling):
  - Stock price < `min_price` (default 3 CNY)
  - ST / *ST stocks (detected by name prefix)
  - First `resume_trading_window` sessions after a trading halt / resume
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class LabelConfig:
    """Configuration for label generation."""

    # Forward-looking window (trading days)
    horizon_short: int = 20
    horizon_mid: int = 40
    horizon_long: int = 60

    # Core condition: minimum price appreciation over the horizon
    min_return_pct: float = 30.0

    # Core condition: maximum intra-period drawdown allowed
    max_drawdown_pct: float = 15.0

    # Extra condition (a): volume surge threshold
    volume_surge_ratio: float = 2.0
    volume_lookback: int = 30

    # Extra condition (b): 60-day high breakout lookback
    high_breakout_lookback: int = 60

    # Extra condition (c): MACD above zero-axis
    macd_above_zero_days: int = 10

    # Minimum number of extra conditions that must be satisfied
    min_extra_conditions: int = 2

    # Noise filter: minimum stock price
    min_price: float = 3.0

    # Noise filter: sessions to skip after resume trading
    resume_trading_window: int = 5

    # Gap ratio to detect trading halt / resume (>= this % daily change)
    resume_gap_pct: float = 9.5


class LabelGenerator:
    """
    Generates forward-looking binary labels for the main-wave classifier.

    Usage::

        gen = LabelGenerator()
        df_with_labels = gen.generate(df_ohlcv, stock_name="平安银行")

    The input ``df_ohlcv`` must contain at minimum:
        date, open, high, low, close, volume

    Returns the same DataFrame with three additional boolean columns:
        label_short, label_mid, label_long
    and a boolean column ``is_excluded`` indicating rows filtered out.
    """

    # MACD parameters (standard 12/26/9)
    _MACD_FAST = 12
    _MACD_SLOW = 26
    _MACD_SIGNAL = 9

    def __init__(self, config: Optional[LabelConfig] = None):
        self.cfg = config or LabelConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        df: pd.DataFrame,
        stock_name: str = "",
    ) -> pd.DataFrame:
        """
        Attach forward-looking labels to a sorted daily OHLCV DataFrame.

        Args:
            df:          DataFrame with columns [date, open, high, low, close, volume].
                         Must be sorted ascending by date.
            stock_name:  Optional stock name string used to detect ST stocks.

        Returns:
            A copy of ``df`` with columns appended:
              - ``label_short``  : bool, positive label for short horizon
              - ``label_mid``    : bool, positive label for mid horizon
              - ``label_long``   : bool, positive label for long horizon
              - ``is_excluded``  : bool, True when the row should be excluded
        """
        if df is None or df.empty:
            raise ValueError("Input DataFrame is empty")

        required = {"date", "open", "high", "low", "close", "volume"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        df = df.copy()
        df = df.sort_values("date").reset_index(drop=True)
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")

        # Compute helpers
        df = self._compute_macd(df)
        df = self._mark_resume_sessions(df)

        is_st = self._is_st_stock(stock_name)

        # Generate labels for each horizon
        for col, horizon in [
            ("label_short", self.cfg.horizon_short),
            ("label_mid", self.cfg.horizon_mid),
            ("label_long", self.cfg.horizon_long),
        ]:
            df[col] = self._compute_labels(df, horizon)

        # Mark excluded rows
        df["is_excluded"] = self._mark_excluded(df, is_st)

        # Clear labels on excluded rows to avoid training on noisy samples
        for col in ("label_short", "label_mid", "label_long"):
            df.loc[df["is_excluded"], col] = np.nan

        # Drop helper columns added internally
        df = df.drop(
            columns=[c for c in ("_macd_dif", "_resume") if c in df.columns],
            errors="ignore",
        )

        return df

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_labels(self, df: pd.DataFrame, horizon: int) -> pd.Series:
        """Vectorised forward-looking label computation."""
        cfg = self.cfg
        n = len(df)
        close = df["close"].to_numpy(dtype=float)
        high = df["high"].to_numpy(dtype=float)
        volume = df["volume"].to_numpy(dtype=float)
        dif = df["_macd_dif"].to_numpy(dtype=float)

        labels = np.zeros(n, dtype=float)
        labels[:] = np.nan

        for i in range(n - 1):
            end = min(i + horizon + 1, n)
            if end - i - 1 < 1:
                continue

            future_close = close[i + 1: end]
            future_high = high[i + 1: end]
            entry_price = close[i]

            if entry_price <= 0 or np.isnan(entry_price):
                continue

            # --- Core condition 1: max return ---
            period_returns = (future_close - entry_price) / entry_price * 100
            max_return = float(np.nanmax(period_returns)) if len(period_returns) else -999

            if max_return < cfg.min_return_pct:
                labels[i] = 0
                continue

            # --- Core condition 2: max drawdown ---
            peak_idx = int(np.argmax(future_close))
            peak_price = future_close[peak_idx]
            subsequent = future_close[peak_idx:]
            trough = float(np.nanmin(subsequent)) if len(subsequent) else peak_price
            drawdown = (peak_price - trough) / peak_price * 100 if peak_price > 0 else 999

            if drawdown > cfg.max_drawdown_pct:
                labels[i] = 0
                continue

            # --- Extra conditions ---
            extra = 0

            # (a) Volume surge
            vol_start = max(0, i - cfg.volume_lookback + 1)
            vol_avg = np.nanmean(volume[vol_start: i + 1])
            future_vol = volume[i + 1: end]
            if vol_avg > 0 and len(future_vol) > 0:
                if float(np.nanmax(future_vol)) >= cfg.volume_surge_ratio * vol_avg:
                    extra += 1

            # (b) 60-day high breakout
            high_start = max(0, i - cfg.high_breakout_lookback + 1)
            rolling_high = float(np.nanmax(high[high_start: i + 1])) if i >= high_start else np.nan
            if not np.isnan(rolling_high) and len(future_high) > 0:
                if float(np.nanmax(future_high)) > rolling_high:
                    extra += 1

            # (c) MACD DIF above zero for N consecutive days
            dif_future = dif[i + 1: end]
            if len(dif_future) >= cfg.macd_above_zero_days:
                for start_j in range(len(dif_future) - cfg.macd_above_zero_days + 1):
                    window = dif_future[start_j: start_j + cfg.macd_above_zero_days]
                    if np.all(window > 0):
                        extra += 1
                        break

            labels[i] = 1 if extra >= cfg.min_extra_conditions else 0

        return pd.Series(labels, index=df.index, dtype=float)

    def _compute_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute MACD DIF column used for label generation."""
        close = df["close"]
        ema_fast = close.ewm(span=self._MACD_FAST, adjust=False).mean()
        ema_slow = close.ewm(span=self._MACD_SLOW, adjust=False).mean()
        df = df.copy()
        df["_macd_dif"] = ema_fast - ema_slow
        return df

    def _mark_resume_sessions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Mark sessions immediately after a trading halt/resume."""
        df = df.copy()
        pct_chg = df["close"].pct_change().abs() * 100
        is_resume = pct_chg >= self.cfg.resume_gap_pct

        # Expand the exclusion window forward
        resume_mask = np.zeros(len(df), dtype=bool)
        indices = df.index[is_resume].tolist()
        for idx in indices:
            pos = df.index.get_loc(idx)
            end_pos = min(pos + self.cfg.resume_trading_window, len(df))
            resume_mask[pos:end_pos] = True

        df["_resume"] = resume_mask
        return df

    @staticmethod
    def _is_st_stock(name: str) -> bool:
        """Detect ST / *ST stocks from their name.

        Python's str.startswith accepts a tuple of prefixes and returns True if
        the string starts with ANY of them.  '*ST广田'.startswith(('ST', '*ST'))
        returns True because the second pattern '*ST' matches.
        """
        return name.strip().upper().startswith(("ST", "*ST"))

    def _mark_excluded(self, df: pd.DataFrame, is_st: bool) -> pd.Series:
        """Return a boolean Series marking rows that should be excluded."""
        excluded = pd.Series(False, index=df.index)

        # Price below minimum
        excluded |= df["close"] < self.cfg.min_price

        # ST stocks — exclude all rows
        if is_st:
            excluded[:] = True

        # Resume trading window
        if "_resume" in df.columns:
            excluded |= df["_resume"]

        return excluded
