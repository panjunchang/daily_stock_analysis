# -*- coding: utf-8 -*-
"""
===================================
Trainer — 主升浪 XGBoost 训练器
===================================

Implements Walk-Forward Validation to avoid look-ahead bias:

  Train [2015-2020] → Val [2021]
  Train [2015-2021] → Val [2022]
  ...

Pipeline:
  1. Load feature matrix + labels from a Parquet/CSV file
  2. Drop rows where label is NaN (excluded)
  3. Remove highly correlated features (>0.9)
  4. Walk-forward train / evaluate loop
  5. Retrain final model on ALL data
  6. Persist model + feature list to model_store/

Metrics reported per fold:
  - AUC-ROC
  - Precision@20 (top-20 scored stocks hit-rate)
  - Average Precision (AP)

Usage (CLI)::

    python scripts/train_main_wave.py \
        --data path/to/features.parquet \
        --label label_mid \
        --model-dir src/ml/model_store

Or programmatically::

    from src.ml.trainer import MainWaveTrainer
    trainer = MainWaveTrainer()
    trainer.run(data_path="...", label_col="label_mid")
"""

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Default model store directory (relative to this file's location)
_DEFAULT_MODEL_DIR = Path(__file__).parent / "model_store"

# Minimum number of positive samples required in a training fold
_MIN_POSITIVE_SAMPLES = 50


@dataclass
class TrainConfig:
    """Hyper-parameters and split settings for training."""

    # Walk-forward: training window in years (-1 = use all history up to split)
    train_years: int = -1

    # Walk-forward: validation window in years
    val_years: int = 1

    # First validation year (data before this = initial train set)
    first_val_year: int = 2021

    # Last validation year (inclusive)
    last_val_year: int = 2025

    # Maximum Pearson correlation allowed between features
    max_feature_corr: float = 0.90

    # Class weight strategy: "balanced" or a float (positive class weight)
    class_weight: str = "balanced"

    # XGBoost params (overrides accepted via constructor)
    xgb_params: Dict = field(default_factory=lambda: {
        "n_estimators": 500,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "eval_metric": "auc",
        "early_stopping_rounds": 30,
        "random_state": 42,
        "n_jobs": -1,
    })

    # Precision@K evaluation parameter
    precision_at_k: int = 20

    # Model store path
    model_dir: str = str(_DEFAULT_MODEL_DIR)

    # Label column to use
    label_col: str = "label_mid"


class MainWaveTrainer:
    """
    Walk-Forward trainer for the main-wave XGBoost classifier.

    The model is trained to predict a forward-looking binary label
    indicating whether the stock will exhibit a "主升浪" move.
    """

    def __init__(self, config: Optional[TrainConfig] = None):
        self.cfg = config or TrainConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        data_path: str,
        label_col: Optional[str] = None,
        save: bool = True,
    ) -> Dict:
        """
        Full training pipeline.

        Args:
            data_path:  Path to a Parquet or CSV file containing the feature
                        matrix.  Must include columns:
                        ``date``, ``code``, ``<label_col>``, and feature columns.
            label_col:  Override the label column name from config.
            save:       Whether to persist the final model to ``model_dir``.

        Returns:
            Dict with keys:
              ``fold_metrics``  : list of per-fold metric dicts
              ``mean_auc``      : average AUC across folds
              ``mean_ap``       : average AP across folds
              ``features``      : list of feature column names used
        """
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise ImportError(
                "xgboost is required for training. Install it with: pip install xgboost"
            ) from exc

        label_col = label_col or self.cfg.label_col
        logger.info("Loading data from %s", data_path)

        df = self._load_data(data_path)
        feature_cols = self._get_feature_cols(df, label_col)

        logger.info("Loaded %d rows, %d features", len(df), len(feature_cols))

        # Remove highly correlated features
        feature_cols = self._remove_correlated_features(df, feature_cols)
        logger.info("%d features remaining after correlation filter", len(feature_cols))

        # Walk-forward cross-validation
        fold_metrics = []
        for val_year in range(self.cfg.first_val_year, self.cfg.last_val_year + 1):
            metrics = self._train_fold(df, feature_cols, label_col, val_year, XGBClassifier)
            if metrics is not None:
                fold_metrics.append(metrics)
                logger.info(
                    "Val %d | AUC=%.4f  AP=%.4f  P@%d=%.4f",
                    val_year,
                    metrics["auc"],
                    metrics["avg_precision"],
                    self.cfg.precision_at_k,
                    metrics[f"precision_at_{self.cfg.precision_at_k}"],
                )

        mean_auc = float(np.mean([m["auc"] for m in fold_metrics])) if fold_metrics else 0.0
        mean_ap = float(np.mean([m["avg_precision"] for m in fold_metrics])) if fold_metrics else 0.0

        logger.info("Walk-forward mean AUC=%.4f, mean AP=%.4f", mean_auc, mean_ap)

        # Retrain on ALL data
        final_model = self._retrain_final(df, feature_cols, label_col, XGBClassifier)

        if save and final_model is not None:
            self._save_model(final_model, feature_cols, label_col, fold_metrics)

        return {
            "fold_metrics": fold_metrics,
            "mean_auc": mean_auc,
            "mean_ap": mean_ap,
            "features": feature_cols,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_data(path: str) -> pd.DataFrame:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Data file not found: {path}")
        if p.suffix.lower() in (".parquet", ".pq"):
            return pd.read_parquet(path)
        return pd.read_csv(path, parse_dates=["date"])

    @staticmethod
    def _get_feature_cols(df: pd.DataFrame, label_col: str) -> List[str]:
        """Return all numeric columns that are not metadata or labels."""
        exclude = {
            "date", "code", "stock_name",
            "label_short", "label_mid", "label_long",
            "is_excluded", "_macd_dif", "_resume",
        }
        return [
            c for c in df.columns
            if c not in exclude
            and pd.api.types.is_numeric_dtype(df[c])
            and c != label_col
        ]

    def _remove_correlated_features(
        self, df: pd.DataFrame, feature_cols: List[str]
    ) -> List[str]:
        """Drop features with pairwise Pearson |r| > threshold (greedy)."""
        subset = df[feature_cols].dropna(how="all")
        if subset.empty:
            return feature_cols

        corr = subset.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = [col for col in upper.columns if any(upper[col] > self.cfg.max_feature_corr)]
        kept = [c for c in feature_cols if c not in to_drop]
        if to_drop:
            logger.info("Dropped %d correlated features: %s", len(to_drop), to_drop[:10])
        return kept

    def _train_fold(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        label_col: str,
        val_year: int,
        XGBClassifier,
    ) -> Optional[Dict]:
        """Train one fold and return evaluation metrics."""
        from sklearn.metrics import roc_auc_score, average_precision_score

        df["date"] = pd.to_datetime(df["date"])

        # Split
        train_mask = df["date"].dt.year < val_year
        val_mask = df["date"].dt.year == val_year

        train_df = df[train_mask].dropna(subset=[label_col])
        val_df = df[val_mask].dropna(subset=[label_col])

        if len(train_df) == 0 or len(val_df) == 0:
            logger.warning("Skipping fold %d: empty split", val_year)
            return None

        pos_count = int(train_df[label_col].sum())
        if pos_count < _MIN_POSITIVE_SAMPLES:
            logger.warning(
                "Skipping fold %d: only %d positive samples in train set",
                val_year,
                pos_count,
            )
            return None

        X_train = train_df[feature_cols].fillna(-999)
        y_train = train_df[label_col].astype(int)
        X_val = val_df[feature_cols].fillna(-999)
        y_val = val_df[label_col].astype(int)

        # Compute scale_pos_weight for imbalanced classes
        neg = int((y_train == 0).sum())
        pos = int((y_train == 1).sum())
        scale_pos_weight = neg / pos if pos > 0 else 1.0

        params = {**self.cfg.xgb_params}
        early_rounds = params.pop("early_stopping_rounds", 30)
        params.pop("eval_metric", None)

        model = XGBClassifier(
            **params,
            scale_pos_weight=scale_pos_weight,
            use_label_encoder=False,
        )

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=early_rounds,
            verbose=False,
        )

        proba = model.predict_proba(X_val)[:, 1]

        auc = float(roc_auc_score(y_val, proba)) if len(np.unique(y_val)) > 1 else 0.5
        ap = float(average_precision_score(y_val, proba)) if len(np.unique(y_val)) > 1 else 0.0
        p_at_k = self._precision_at_k(y_val.to_numpy(), proba, self.cfg.precision_at_k)

        return {
            "val_year": val_year,
            "train_size": len(train_df),
            "val_size": len(val_df),
            "positive_rate": float(y_val.mean()),
            "auc": auc,
            "avg_precision": ap,
            f"precision_at_{self.cfg.precision_at_k}": p_at_k,
        }

    def _retrain_final(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        label_col: str,
        XGBClassifier,
    ):
        """Retrain on the full dataset (no validation split)."""
        clean = df.dropna(subset=[label_col])
        if clean.empty:
            logger.warning("No labelled data for final retraining")
            return None

        X = clean[feature_cols].fillna(-999)
        y = clean[label_col].astype(int)

        neg = int((y == 0).sum())
        pos = int((y == 1).sum())
        scale_pos_weight = neg / pos if pos > 0 else 1.0

        params = {k: v for k, v in self.cfg.xgb_params.items()
                  if k not in ("early_stopping_rounds", "eval_metric")}

        model = XGBClassifier(
            **params,
            scale_pos_weight=scale_pos_weight,
            use_label_encoder=False,
        )
        model.fit(X, y, verbose=False)
        return model

    def _save_model(
        self,
        model,
        feature_cols: List[str],
        label_col: str,
        fold_metrics: List[Dict],
    ) -> None:
        """Persist the trained model + metadata to model_store."""
        model_dir = Path(self.cfg.model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

        model_path = model_dir / f"xgb_{label_col}.pkl"
        meta_path = model_dir / f"xgb_{label_col}_meta.json"

        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        mean_auc = float(np.mean([m["auc"] for m in fold_metrics])) if fold_metrics else 0.0

        meta = {
            "label_col": label_col,
            "features": feature_cols,
            "n_features": len(feature_cols),
            "fold_metrics": fold_metrics,
            "mean_auc": mean_auc,
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        logger.info("Model saved to %s (AUC=%.4f)", model_path, mean_auc)

    @staticmethod
    def _precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
        """Compute Precision@K: fraction of top-K predictions that are positive."""
        if k <= 0 or len(y_true) == 0:
            return 0.0
        k = min(k, len(y_true))
        top_k_idx = np.argsort(y_score)[::-1][:k]
        return float(y_true[top_k_idx].mean())
