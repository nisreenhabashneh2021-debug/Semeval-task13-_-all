"""
Common metric helpers.
"""

from typing import Dict
from sklearn.metrics import accuracy_score, f1_score, classification_report


def compute_basic_metrics(y_true, y_pred) -> Dict[str, float]:
    """
    Compute accuracy, macro-F1 and micro-F1.
    """
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    micro_f1 = f1_score(y_true, y_pred, average="micro")
    return {"accuracy": acc, "macro_f1": macro_f1, "micro_f1": micro_f1}


def print_classification_report(y_true, y_pred, target_names=None) -> None:
    """
    Print a classification report, optionally with label names.
    """
    print(classification_report(y_true, y_pred, target_names=target_names))
