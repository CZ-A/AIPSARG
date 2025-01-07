# aipsarg/model/metrics.py
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import Optional

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, average: str = 'weighted') -> Optional[dict]:
    """
    Calculates and returns evaluation metrics.

    Args:
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted labels.
        average (str) : This parameter is required for multiclass/multilabel targets.
          If None, the scores for each class are returned.
        
    Returns:
        Optional[dict]: Dictionary containing calculated metrics, or None if error occurs.
    """
    try:
        y_pred_binary = (y_pred > 0.5).astype(int)
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred_binary),
            "precision": precision_score(y_true, y_pred_binary, average=average, zero_division=0),
            "recall": recall_score(y_true, y_pred_binary, average=average, zero_division=0),
            "f1_score": f1_score(y_true, y_pred_binary, average=average, zero_division=0),
        }
        return metrics
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        return None
