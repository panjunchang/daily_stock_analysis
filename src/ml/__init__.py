# -*- coding: utf-8 -*-
"""
===================================
ML Module - Main Wave (主升浪) Predictor
===================================

Sub-modules:
- label_generator : forward-looking label construction
- feature_builder : technical feature engineering
- trainer         : XGBoost + Walk-Forward validation
- predictor       : inference interface
"""

from .label_generator import LabelGenerator, LabelConfig
from .feature_builder import FeatureBuilder
from .predictor import MainWavePredictor

__all__ = [
    "LabelGenerator",
    "LabelConfig",
    "FeatureBuilder",
    "MainWavePredictor",
]
