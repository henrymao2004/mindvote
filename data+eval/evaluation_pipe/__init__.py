"""
MindVote Poll Prediction Evaluation Pipeline

A comprehensive evaluation pipeline for assessing Large Language Models (LLMs) 
on opinion prediction within naturalistic social discourse using the MindVote dataset.
"""

from .config import MODELS, EVAL_CONFIG
from .evaluator import PollEvaluator
from .data_processing import DataProcessor, EvaluationSample
from .metrics import PollMetrics
from .utils import create_evaluation_dashboard





__all__ = [
    "MODELS",
    "EVAL_CONFIG", 
    "PollEvaluator",
    "DataProcessor",
    "EvaluationSample",
    "PollMetrics",
    "create_evaluation_dashboard"
] 