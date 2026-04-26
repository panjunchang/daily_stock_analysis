# -*- coding: utf-8 -*-
"""
===================================
Predictor — 主升浪 ML 推理接口
===================================

Loads a persisted XGBoost model and scores one or many stocks.

Usage — single stock::

    from src.ml.predictor import MainWavePredictor
    predictor = MainWavePredictor()                   # loads from model_store/
    score = predictor.score_stock(df_ohlcv)
    # score is a float in [0, 1]; higher = more likely main wave

Usage — batch scoring::

    scores = predictor.score_batch(
        stock_dict,         # {code: df_ohlcv, ...}
        index_df=index_df,  # optional benchmark
    )
    # Returns sorted list of (code, score) tuples, highest first

Integration with the strategy layer:
    The score is exposed as ``ml_main_wave_score`` in the context that
    the ``main_wave_ml.yaml`` strategy prompt can reference.
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .feature_builder import FeatureBuilder

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_DIR = Path(__file__).parent / "model_store"

# Sentinel used to replace NaN feature values at inference time.
# Must match the value used during training (see trainer._MISSING_SENTINEL).
_MISSING_SENTINEL: float = -999.0


class MainWavePredictor:
    """
    Inference wrapper for the trained main-wave XGBoost model.

    Args:
        label_col:  Which label variant to load (default ``label_mid``).
        model_dir:  Directory containing the ``.pkl`` and ``_meta.json`` files.
    """

    def __init__(
        self,
        label_col: str = "label_mid",
        model_dir: Optional[str] = None,
    ):
        self.label_col = label_col
        self.model_dir = Path(model_dir) if model_dir else _DEFAULT_MODEL_DIR
        self._model = None
        self._feature_cols: List[str] = []
        self._meta: Dict = {}
        self._fb = FeatureBuilder()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> "MainWavePredictor":
        """
        Load model and metadata from disk.

        Raises:
            FileNotFoundError: if model file does not exist.
        """
        model_path = self.model_dir / f"xgb_{self.label_col}.pkl"
        meta_path = self.model_dir / f"xgb_{self.label_col}_meta.json"

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {model_path}. "
                "Run scripts/train_main_wave.py to train the model first."
            )

        with open(model_path, "rb") as f:
            self._model = pickle.load(f)

        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                self._meta = json.load(f)
            self._feature_cols = self._meta.get("features", [])

        logger.info(
            "Loaded main-wave model (%s) with %d features, mean AUC=%.4f",
            self.label_col,
            len(self._feature_cols),
            self._meta.get("mean_auc", 0.0),
        )
        return self

    def is_loaded(self) -> bool:
        """Return True if a model has been loaded."""
        return self._model is not None

    def score_stock(
        self,
        df: pd.DataFrame,
        index_df: Optional[pd.DataFrame] = None,
        turnover_df: Optional[pd.DataFrame] = None,
    ) -> float:
        """
        Score a single stock based on its latest daily data.

        Args:
            df:           Daily OHLCV DataFrame (sorted ascending).
            index_df:     Optional benchmark index DataFrame.
            turnover_df:  Optional turnover rate DataFrame.

        Returns:
            Probability in [0, 1] that the stock is in a main-wave setup.
            Returns ``-1.0`` if scoring fails.
        """
        if not self.is_loaded():
            self.load()

        try:
            features = self._fb.build_features(df, index_df, turnover_df)
            return self._predict_from_dict(features)
        except Exception as exc:
            logger.warning("Scoring failed: %s", exc)
            return -1.0

    def score_batch(
        self,
        stock_dict: Dict[str, pd.DataFrame],
        index_df: Optional[pd.DataFrame] = None,
        min_score: float = 0.0,
        top_k: int = 50,
    ) -> List[Tuple[str, float]]:
        """
        Score multiple stocks and return the top candidates.

        Args:
            stock_dict:  Mapping of stock code → daily OHLCV DataFrame.
            index_df:    Optional shared benchmark index DataFrame.
            min_score:   Minimum score threshold to include in results.
            top_k:       Return only the top-K results.

        Returns:
            List of ``(code, score)`` tuples sorted by score descending.
        """
        if not self.is_loaded():
            self.load()

        results = []
        for code, df in stock_dict.items():
            score = self.score_stock(df, index_df=index_df)
            if score >= min_score:
                results.append((code, round(score, 4)))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def get_model_info(self) -> Dict:
        """Return metadata about the loaded model."""
        if not self._meta:
            return {"loaded": False}
        return {
            "loaded": True,
            "label_col": self.label_col,
            "n_features": len(self._feature_cols),
            "mean_auc": self._meta.get("mean_auc", None),
            "fold_metrics": self._meta.get("fold_metrics", []),
        }

    def get_feature_importance(self, top_n: int = 20) -> List[Tuple[str, float]]:
        """
        Return top-N feature importances from the loaded model.

        Returns list of (feature_name, importance) tuples.
        """
        if not self.is_loaded():
            raise RuntimeError("Model is not loaded. Call load() first.")

        importances = self._model.feature_importances_
        pairs = sorted(
            zip(self._feature_cols, importances),
            key=lambda x: x[1],
            reverse=True,
        )
        return pairs[:top_n]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _predict_from_dict(self, features: Dict[str, float]) -> float:
        """Convert a feature dict to a model score."""
        if not features:
            return -1.0

        if self._feature_cols:
            row = [features.get(col, _MISSING_SENTINEL) for col in self._feature_cols]
        else:
            row = list(features.values())

        # Replace NaN with sentinel (must match training-time sentinel)
        row = [_MISSING_SENTINEL if (v is None or (isinstance(v, float) and np.isnan(v))) else v for v in row]

        X = np.array(row, dtype=float).reshape(1, -1)
        proba = self._model.predict_proba(X)[0][1]
        return float(proba)
