#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
===================================
train_main_wave.py — 主升浪模型训练脚本
===================================

Builds the feature matrix + labels from historical A-share data,
then trains an XGBoost classifier using Walk-Forward validation.

Quick start
-----------
Step 1: Generate features & labels for a list of stocks (using akshare):

    python scripts/train_main_wave.py build-dataset \
        --stocks 600519,000001,300750 \
        --start 2015-01-01 \
        --output data/main_wave_features.parquet

Step 2: Train the model:

    python scripts/train_main_wave.py train \
        --data data/main_wave_features.parquet \
        --label label_mid \
        --model-dir src/ml/model_store

Step 3: Score today's watchlist:

    python scripts/train_main_wave.py score \
        --stocks 600519,000001,300750 \
        --top-k 20

Note
----
This script requires the repository to be on PYTHONPATH.
Run from the project root: ``python scripts/train_main_wave.py ...``
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Ensure the project root is on sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sub-command: build-dataset
# ---------------------------------------------------------------------------

def cmd_build_dataset(args: argparse.Namespace) -> None:
    """Fetch historical OHLCV data and generate the feature + label matrix."""
    import pandas as pd
    from src.ml.feature_builder import FeatureBuilder
    from src.ml.label_generator import LabelGenerator, LabelConfig

    stock_list = [s.strip() for s in args.stocks.split(",") if s.strip()]
    if not stock_list:
        logger.error("No stocks provided.")
        sys.exit(1)

    logger.info("Building dataset for %d stocks from %s to %s", len(stock_list), args.start, args.end)

    fb = FeatureBuilder()
    gen = LabelGenerator(LabelConfig())

    all_rows = []

    for code in stock_list:
        df = _fetch_history(code, args.start, args.end)
        if df is None or df.empty or len(df) < 60:
            logger.warning("Skipping %s: insufficient data", code)
            continue

        try:
            # Generate labels
            df_labelled = gen.generate(df)

            # Build feature matrix
            feature_matrix = fb.build_feature_matrix(df_labelled)

            # Merge labels and metadata
            feature_matrix["date"] = df_labelled["date"].values[: len(feature_matrix)]
            feature_matrix["code"] = code
            for lbl in ("label_short", "label_mid", "label_long", "is_excluded"):
                if lbl in df_labelled.columns:
                    feature_matrix[lbl] = df_labelled[lbl].values[: len(feature_matrix)]

            all_rows.append(feature_matrix)
            logger.info("  %s: %d rows", code, len(feature_matrix))

        except Exception as exc:
            logger.warning("Error processing %s: %s", code, exc)
            continue

    if not all_rows:
        logger.error("No data collected.")
        sys.exit(1)

    combined = pd.concat(all_rows, ignore_index=True)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix.lower() in (".parquet", ".pq"):
        combined.to_parquet(output_path, index=False)
    else:
        combined.to_csv(output_path, index=False)

    pos_rate = combined["label_mid"].dropna().mean() if "label_mid" in combined.columns else None
    logger.info(
        "Dataset saved to %s | rows=%d, stocks=%d, label_mid_positive_rate=%.2f%%",
        output_path,
        len(combined),
        len(stock_list),
        (pos_rate * 100) if pos_rate is not None else 0,
    )


# ---------------------------------------------------------------------------
# Sub-command: train
# ---------------------------------------------------------------------------

def cmd_train(args: argparse.Namespace) -> None:
    """Run the full training pipeline."""
    from src.ml.trainer import MainWaveTrainer, TrainConfig

    cfg = TrainConfig(
        label_col=args.label,
        model_dir=args.model_dir,
        first_val_year=args.first_val_year,
        last_val_year=args.last_val_year,
    )
    trainer = MainWaveTrainer(cfg)
    result = trainer.run(data_path=args.data, label_col=args.label, save=True)

    logger.info("Training complete.")
    logger.info("  Mean AUC : %.4f", result["mean_auc"])
    logger.info("  Mean AP  : %.4f", result["mean_ap"])
    logger.info("  Features : %d", len(result["features"]))

    for m in result["fold_metrics"]:
        logger.info(
            "  [%d] AUC=%.4f  AP=%.4f  P@20=%.4f  pos_rate=%.2f%%",
            m["val_year"],
            m["auc"],
            m["avg_precision"],
            m.get("precision_at_20", 0.0),
            m.get("positive_rate", 0.0) * 100,
        )


# ---------------------------------------------------------------------------
# Sub-command: score
# ---------------------------------------------------------------------------

def cmd_score(args: argparse.Namespace) -> None:
    """Score today's watchlist and print top candidates."""
    from src.ml.predictor import MainWavePredictor

    stock_list = [s.strip() for s in args.stocks.split(",") if s.strip()]
    if not stock_list:
        logger.error("No stocks provided.")
        sys.exit(1)

    predictor = MainWavePredictor(label_col=args.label, model_dir=args.model_dir)
    predictor.load()

    end_date = args.date or None
    start_date = args.start or "2024-01-01"

    stock_dict = {}
    for code in stock_list:
        df = _fetch_history(code, start_date, end_date)
        if df is not None and len(df) >= 60:
            stock_dict[code] = df
        else:
            logger.warning("Skipping %s: insufficient data", code)

    if not stock_dict:
        logger.error("No valid stocks to score.")
        sys.exit(1)

    results = predictor.score_batch(stock_dict, top_k=args.top_k, min_score=args.min_score)

    print("\n=== 主升浪 ML 评分 Top Results ===")
    print(f"{'Rank':<5} {'Code':<10} {'Score':>8}")
    print("-" * 30)
    for rank, (code, score) in enumerate(results, 1):
        signal = "🔥 HIGH" if score >= 0.70 else ("⚡ MID" if score >= 0.50 else "  LOW")
        print(f"{rank:<5} {code:<10} {score:>8.4f}  {signal}")

    info = predictor.get_model_info()
    print(f"\nModel: {info.get('label_col')} | AUC={info.get('mean_auc', 'N/A'):.4f} | Features={info.get('n_features')}")


# ---------------------------------------------------------------------------
# Data fetching helper (wraps existing AkshareFetcher)
# ---------------------------------------------------------------------------

def _fetch_history(code: str, start: str, end: "str | None") -> "object":
    """Fetch daily OHLCV using the project's data provider stack.

    Returns a pandas DataFrame or None.

    Uses ``adjust='hfq'`` (后复权, backward/ex-post price adjustment) so that
    historical prices are scaled to the current unadjusted price level.  This
    ensures that percent-change features computed over long horizons remain
    comparable and are not distorted by stock splits or dividend payouts.
    """
    try:
        import akshare as ak
        import pandas as pd
        from datetime import datetime

        end_str = (end or datetime.today().strftime("%Y%m%d")).replace("-", "")
        start_str = start.replace("-", "")

        df = ak.stock_zh_a_hist(
            symbol=code,
            period="daily",
            start_date=start_str,
            end_date=end_str,
            adjust="hfq",
        )
        if df is None or df.empty:
            return None

        # Normalise column names to the project standard
        col_map = {
            "日期": "date",
            "开盘": "open",
            "最高": "high",
            "最低": "low",
            "收盘": "close",
            "成交量": "volume",
            "成交额": "amount",
            "涨跌幅": "pct_chg",
        }
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
        df["date"] = pd.to_datetime(df["date"])
        return df.sort_values("date").reset_index(drop=True)

    except Exception as exc:
        logger.warning("Failed to fetch %s: %s", code, exc)
        return None


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="train_main_wave",
        description="主升浪 ML 模型：数据集构建 / 训练 / 评分",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- build-dataset ---
    p_build = sub.add_parser("build-dataset", help="Fetch history and generate feature+label matrix")
    p_build.add_argument("--stocks", required=True, help="Comma-separated A-share codes, e.g. 600519,000001")
    p_build.add_argument("--start", default="2015-01-01", help="Start date YYYY-MM-DD")
    p_build.add_argument("--end", default=None, help="End date YYYY-MM-DD (default: today)")
    p_build.add_argument("--output", default="data/main_wave_features.parquet", help="Output file path (.parquet or .csv)")

    # --- train ---
    p_train = sub.add_parser("train", help="Train XGBoost model with Walk-Forward validation")
    p_train.add_argument("--data", required=True, help="Path to feature matrix (.parquet or .csv)")
    p_train.add_argument("--label", default="label_mid", choices=["label_short", "label_mid", "label_long"])
    p_train.add_argument("--model-dir", default="src/ml/model_store", help="Directory to save model files")
    p_train.add_argument("--first-val-year", type=int, default=2021)
    p_train.add_argument("--last-val-year", type=int, default=2025)

    # --- score ---
    p_score = sub.add_parser("score", help="Score today's watchlist")
    p_score.add_argument("--stocks", required=True, help="Comma-separated A-share codes")
    p_score.add_argument("--label", default="label_mid", choices=["label_short", "label_mid", "label_long"])
    p_score.add_argument("--model-dir", default="src/ml/model_store")
    p_score.add_argument("--top-k", type=int, default=20)
    p_score.add_argument("--min-score", type=float, default=0.0)
    p_score.add_argument("--start", default="2024-01-01", help="History start date for scoring features")
    p_score.add_argument("--date", default=None, help="Score as of this date (default: today)")

    args = parser.parse_args()

    dispatch = {
        "build-dataset": cmd_build_dataset,
        "train": cmd_train,
        "score": cmd_score,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
