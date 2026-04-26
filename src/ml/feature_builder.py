# -*- coding: utf-8 -*-
"""
===================================
Feature Builder — 主升浪特征工程
===================================

Builds a flat feature vector from a sorted daily OHLCV DataFrame.

Feature groups
--------------
1. Price momentum     : N-day returns, relative strength vs index, ATR
2. Moving-average     : MA5/10/20/60 alignment score, bias, slope, compression
3. Volume-price       : volume ratio, surge streak, shrink-then-surge signal
4. Technical          : MACD, KDJ, RSI(6/12/24), Bollinger-band width, turnover
5. Market context     : index MA20 position, breadth indicator (if supplied)

All feature values are floats; NaN is used when insufficient history exists.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Minimum rows of history required before any feature is valid
_MIN_HISTORY = 60


class FeatureBuilder:
    """
    Compute a feature dict for a single row of a daily OHLCV time series.

    Usage::

        fb = FeatureBuilder()
        # df must be sorted ascending and contain at minimum:
        #   date, open, high, low, close, volume
        features = fb.build_features(df, index_df=index_df)
        # features is a flat dict[str, float] for one date (the last row)

    To build a full feature matrix (one row per trading day)::

        feature_matrix = fb.build_feature_matrix(df, index_df=index_df)
        # Returns a DataFrame aligned to df's index

    ``index_df`` is optional; it should contain daily OHLCV for the benchmark
    (e.g. SSE 000001) with the same column schema.
    """

    # MA windows
    MA_WINDOWS = (5, 10, 20, 60)

    # MACD parameters (standard 12/26/9)
    MACD_FAST = 12
    MACD_SLOW = 26
    MACD_SIGNAL = 9

    # RSI periods
    RSI_PERIODS = (6, 12, 24)

    # KDJ parameter
    KDJ_N = 9

    # Bollinger band
    BB_WINDOW = 20
    BB_STD = 2

    # Volume lookback windows
    VOL_SHORT = 5
    VOL_MID = 20
    VOL_LONG = 60

    def __init__(self):
        pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_feature_matrix(
        self,
        df: pd.DataFrame,
        index_df: Optional[pd.DataFrame] = None,
        turnover_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Build a feature DataFrame aligned to ``df``.

        Args:
            df:           Daily OHLCV, sorted ascending.
            index_df:     Optional benchmark index OHLCV (same schema).
            turnover_df:  Optional column ``turnover`` (换手率, %) aligned to df.

        Returns:
            DataFrame with one feature row per input row.
            Rows with insufficient history are filled with NaN.
        """
        df = self._prepare(df)
        index_df = self._prepare(index_df) if index_df is not None else None

        # Merge turnover into df if provided
        if turnover_df is not None and "turnover" in turnover_df.columns:
            df = df.copy()
            df["turnover"] = turnover_df["turnover"].values[: len(df)]

        df = self._enrich(df, index_df)
        return self._extract_all_rows(df)

    def build_features(
        self,
        df: pd.DataFrame,
        index_df: Optional[pd.DataFrame] = None,
        turnover_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, float]:
        """
        Build feature dict for the LAST row of ``df``.

        Convenience method for real-time scoring.
        """
        feature_matrix = self.build_feature_matrix(df, index_df, turnover_df)
        if feature_matrix.empty:
            return {}
        last = feature_matrix.iloc[-1]
        return last.to_dict()

    # ------------------------------------------------------------------
    # Step 1: Prepare / validate
    # ------------------------------------------------------------------

    @staticmethod
    def _prepare(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
        if df is None or df.empty:
            return df
        df = df.copy()
        df = df.sort_values("date").reset_index(drop=True)
        for col in ("open", "high", "low", "close", "volume"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    # ------------------------------------------------------------------
    # Step 2: Enrich DataFrame with intermediate indicators
    # ------------------------------------------------------------------

    def _enrich(
        self, df: pd.DataFrame, index_df: Optional[pd.DataFrame]
    ) -> pd.DataFrame:
        df = df.copy()
        df = self._add_mas(df)
        df = self._add_macd(df)
        df = self._add_rsi(df)
        df = self._add_kdj(df)
        df = self._add_bollinger(df)
        df = self._add_volume_features(df)
        if index_df is not None:
            df = self._add_market_context(df, index_df)
        return df

    def _add_mas(self, df: pd.DataFrame) -> pd.DataFrame:
        for w in self.MA_WINDOWS:
            df[f"MA{w}"] = df["close"].rolling(w).mean()
        return df

    def _add_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        ema_fast = df["close"].ewm(span=self.MACD_FAST, adjust=False).mean()
        ema_slow = df["close"].ewm(span=self.MACD_SLOW, adjust=False).mean()
        df["MACD_DIF"] = ema_fast - ema_slow
        df["MACD_DEA"] = df["MACD_DIF"].ewm(span=self.MACD_SIGNAL, adjust=False).mean()
        df["MACD_BAR"] = (df["MACD_DIF"] - df["MACD_DEA"]) * 2
        return df

    def _add_rsi(self, df: pd.DataFrame) -> pd.DataFrame:
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        for p in self.RSI_PERIODS:
            avg_gain = gain.rolling(p).mean()
            avg_loss = loss.rolling(p).mean()
            rs = avg_gain / avg_loss.replace(0, np.nan)
            df[f"RSI_{p}"] = (100 - (100 / (1 + rs))).fillna(50)
        return df

    def _add_kdj(self, df: pd.DataFrame) -> pd.DataFrame:
        n = self.KDJ_N
        low_min = df["low"].rolling(n).min()
        high_max = df["high"].rolling(n).max()
        denom = (high_max - low_min).replace(0, np.nan)
        rsv = (df["close"] - low_min) / denom * 100
        df["KDJ_K"] = rsv.ewm(com=2, adjust=False).mean()
        df["KDJ_D"] = df["KDJ_K"].ewm(com=2, adjust=False).mean()
        df["KDJ_J"] = 3 * df["KDJ_K"] - 2 * df["KDJ_D"]
        return df

    def _add_bollinger(self, df: pd.DataFrame) -> pd.DataFrame:
        w = self.BB_WINDOW
        mid = df["close"].rolling(w).mean()
        std = df["close"].rolling(w).std()
        upper = mid + self.BB_STD * std
        lower = mid - self.BB_STD * std
        df["BB_MID"] = mid
        df["BB_UPPER"] = upper
        df["BB_LOWER"] = lower
        df["BB_WIDTH"] = (upper - lower) / mid.replace(0, np.nan)
        df["BB_POSITION"] = (df["close"] - lower) / (upper - lower).replace(0, np.nan)
        return df

    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        vol = df["volume"]
        df["VOL_MA5"] = vol.rolling(self.VOL_SHORT).mean()
        df["VOL_MA20"] = vol.rolling(self.VOL_MID).mean()
        df["VOL_MA60"] = vol.rolling(self.VOL_LONG).mean()

        # Volume ratios
        df["VOL_RATIO_5"] = vol / df["VOL_MA5"].replace(0, np.nan)
        df["VOL_RATIO_20"] = vol / df["VOL_MA20"].replace(0, np.nan)
        df["VOL_RATIO_60"] = vol / df["VOL_MA60"].replace(0, np.nan)

        # Price direction
        df["_price_up"] = (df["close"] > df["close"].shift(1)).astype(int)

        # Consecutive days of price-up + volume-up (量价齐升天数)
        vol_up = (vol > vol.shift(1)).astype(int)
        price_vol_up = df["_price_up"] & vol_up
        streak = price_vol_up.groupby(
            (price_vol_up != price_vol_up.shift()).cumsum()
        ).cumcount() + 1
        df["VOL_PRICE_UP_STREAK"] = streak.where(price_vol_up, 0)

        # Shrink-then-surge: previous N days volume below MA20, current surge > 1.5x MA20
        shrink_window = 3
        vol_ratio = df["VOL_RATIO_20"]
        shrink_prev = vol_ratio.shift(1).rolling(shrink_window).max() < 0.85
        surge_now = vol_ratio >= 1.5
        df["VOL_SHRINK_SURGE"] = (shrink_prev & surge_now).astype(float)

        return df

    def _add_market_context(
        self, df: pd.DataFrame, index_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Merge benchmark index features into df on matching dates."""
        idx = index_df.copy()
        idx["IDX_MA20"] = idx["close"].rolling(20).mean()
        idx["IDX_ABOVE_MA20"] = (idx["close"] > idx["IDX_MA20"]).astype(float)
        idx["IDX_RET20"] = idx["close"].pct_change(20) * 100

        merged = df.merge(
            idx[["date", "IDX_ABOVE_MA20", "IDX_RET20"]],
            on="date",
            how="left",
        )
        return merged

    # ------------------------------------------------------------------
    # Step 3: Extract feature row(s)
    # ------------------------------------------------------------------

    def _extract_all_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build one feature dict per row and concatenate."""
        rows = []
        for i in range(len(df)):
            rows.append(self._extract_row(df, i))
        result = pd.DataFrame(rows, index=df.index)
        return result

    def _extract_row(self, df: pd.DataFrame, i: int) -> Dict[str, float]:  # noqa: C901
        """Extract flat feature dict for row i."""
        feat: Dict[str, float] = {}
        row = df.iloc[i]

        close = float(row["close"]) if not pd.isna(row.get("close", np.nan)) else np.nan
        if close is None or np.isnan(close) or close <= 0:
            return feat

        # ---- 1. Price momentum features ----
        for days in (5, 10, 20, 60):
            if i >= days:
                prev = float(df.iloc[i - days]["close"])
                feat[f"RET_{days}D"] = (close - prev) / prev * 100 if prev > 0 else np.nan
            else:
                feat[f"RET_{days}D"] = np.nan

        # ATR (14-day average true range)
        if i >= 14:
            atr_window = df.iloc[i - 13: i + 1]
            highs = atr_window["high"].to_numpy(dtype=float)
            lows = atr_window["low"].to_numpy(dtype=float)
            closes = atr_window["close"].shift(1).ffill().to_numpy(dtype=float)
            tr = np.maximum(highs - lows, np.maximum(np.abs(highs - closes), np.abs(lows - closes)))
            feat["ATR_14"] = float(np.nanmean(tr)) / close * 100
        else:
            feat["ATR_14"] = np.nan

        # New N-day high pct (how close to rolling high)
        for days in (20, 60):
            if i >= days - 1:
                period_high = float(df["high"].iloc[i - days + 1: i + 1].max())
                feat[f"HIGH_PCT_{days}D"] = (close - period_high) / period_high * 100 if period_high > 0 else np.nan
            else:
                feat[f"HIGH_PCT_{days}D"] = np.nan

        # Relative strength vs index
        if "IDX_RET20" in df.columns:
            idx_ret = row.get("IDX_RET20", np.nan)
            stock_ret = feat.get("RET_20D", np.nan)
            feat["REL_STRENGTH_20"] = (
                float(stock_ret) - float(idx_ret)
                if not (np.isnan(stock_ret) or pd.isna(idx_ret))
                else np.nan
            )
        else:
            feat["REL_STRENGTH_20"] = np.nan

        # ---- 2. Moving-average features ----
        for w in self.MA_WINDOWS:
            ma_val = row.get(f"MA{w}", np.nan)
            if not pd.isna(ma_val) and float(ma_val) > 0:
                ma_val = float(ma_val)
                feat[f"BIAS_MA{w}"] = (close - ma_val) / ma_val * 100
            else:
                feat[f"BIAS_MA{w}"] = np.nan

        # MA alignment score: +1 for each "higher MA above lower MA" pair
        ma5 = row.get("MA5", np.nan)
        ma10 = row.get("MA10", np.nan)
        ma20 = row.get("MA20", np.nan)
        ma60 = row.get("MA60", np.nan)
        ma_score = 0
        pairs = [(ma5, ma10), (ma10, ma20), (ma20, ma60), (ma5, ma20), (ma5, ma60), (ma10, ma60)]
        valid_pairs = 0
        for a, b in pairs:
            if not (pd.isna(a) or pd.isna(b)):
                valid_pairs += 1
                if float(a) > float(b):
                    ma_score += 1
        feat["MA_ALIGN_SCORE"] = ma_score / valid_pairs if valid_pairs > 0 else np.nan

        # MA20 slope (normalised)
        if i >= 5 and not pd.isna(ma20):
            ma20_prev = df.iloc[i - 5].get("MA20", np.nan)
            if not pd.isna(ma20_prev) and float(ma20_prev) > 0:
                feat["MA20_SLOPE"] = (float(ma20) - float(ma20_prev)) / float(ma20_prev) * 100
            else:
                feat["MA20_SLOPE"] = np.nan
        else:
            feat["MA20_SLOPE"] = np.nan

        # MA compression: std of [MA5, MA10, MA20, MA60] / mean (normalised spread)
        ma_vals = [v for v in [ma5, ma10, ma20, ma60] if not pd.isna(v)]
        if len(ma_vals) >= 3:
            arr = np.array([float(v) for v in ma_vals])
            feat["MA_COMPRESSION"] = float(np.std(arr) / np.mean(arr) * 100)
        else:
            feat["MA_COMPRESSION"] = np.nan

        # ---- 3. Volume-price features ----
        for key in ("VOL_RATIO_5", "VOL_RATIO_20", "VOL_RATIO_60",
                    "VOL_PRICE_UP_STREAK", "VOL_SHRINK_SURGE"):
            val = row.get(key, np.nan)
            feat[key] = float(val) if not pd.isna(val) else np.nan

        # ---- 4. Technical indicator features ----
        # MACD
        for key in ("MACD_DIF", "MACD_DEA", "MACD_BAR"):
            val = row.get(key, np.nan)
            feat[key] = float(val) if not pd.isna(val) else np.nan

        # MACD golden/death cross
        if i >= 1:
            prev_dif = df.iloc[i - 1].get("MACD_DIF", np.nan)
            prev_dea = df.iloc[i - 1].get("MACD_DEA", np.nan)
            cur_dif = row.get("MACD_DIF", np.nan)
            cur_dea = row.get("MACD_DEA", np.nan)
            if not any(pd.isna(v) for v in [prev_dif, prev_dea, cur_dif, cur_dea]):
                prev_gap = float(prev_dif) - float(prev_dea)
                cur_gap = float(cur_dif) - float(cur_dea)
                feat["MACD_GOLDEN_CROSS"] = float(prev_gap <= 0 and cur_gap > 0)
                feat["MACD_DEATH_CROSS"] = float(prev_gap >= 0 and cur_gap < 0)
            else:
                feat["MACD_GOLDEN_CROSS"] = np.nan
                feat["MACD_DEATH_CROSS"] = np.nan
        else:
            feat["MACD_GOLDEN_CROSS"] = np.nan
            feat["MACD_DEATH_CROSS"] = np.nan

        # RSI
        for p in self.RSI_PERIODS:
            val = row.get(f"RSI_{p}", np.nan)
            feat[f"RSI_{p}"] = float(val) if not pd.isna(val) else np.nan

        # KDJ
        for key in ("KDJ_K", "KDJ_D", "KDJ_J"):
            val = row.get(key, np.nan)
            feat[key] = float(val) if not pd.isna(val) else np.nan

        # Bollinger band
        for key in ("BB_WIDTH", "BB_POSITION"):
            val = row.get(key, np.nan)
            feat[key] = float(val) if not pd.isna(val) else np.nan

        # Turnover rate
        if "turnover" in df.columns:
            val = row.get("turnover", np.nan)
            feat["TURNOVER"] = float(val) if not pd.isna(val) else np.nan
        else:
            feat["TURNOVER"] = np.nan

        # Turnover trend (current vs 5-day avg)
        if "turnover" in df.columns and i >= 5:
            prev_turn = df["turnover"].iloc[i - 5: i].mean()
            cur_turn = feat.get("TURNOVER", np.nan)
            if not (pd.isna(prev_turn) or np.isnan(cur_turn)) and prev_turn > 0:
                feat["TURNOVER_TREND"] = (cur_turn - float(prev_turn)) / float(prev_turn) * 100
            else:
                feat["TURNOVER_TREND"] = np.nan
        else:
            feat["TURNOVER_TREND"] = np.nan

        # ---- 5. Market context features ----
        for key in ("IDX_ABOVE_MA20", "IDX_RET20"):
            val = row.get(key, np.nan)
            feat[key] = float(val) if not pd.isna(val) else np.nan

        return feat
