"""Visualization tools for model evaluation and explainability."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (ConfusionMatrixDisplay, PrecisionRecallDisplay,
                             RocCurveDisplay)


def plot_confusion_matrix(y_test, y_pred, output_path: Optional[Path] = None):
    """Plot and optionally save a confusion matrix."""
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap="Blues", ax=ax)
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=200)
    return fig


def plot_roc_curve(y_test, y_prob, output_path: Optional[Path] = None):
    """Plot and optionally save the ROC curve."""
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_test, y_prob, ax=ax)
    ax.set_title("ROC Curve")
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=200)
    return fig


def plot_precision_recall_curve(y_test, y_prob, output_path: Optional[Path] = None):
    """Plot and optionally save the precision-recall curve."""
    fig, ax = plt.subplots(figsize=(6, 5))
    PrecisionRecallDisplay.from_predictions(y_test, y_prob, ax=ax)
    ax.set_title("Precision-Recall Curve")
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=200)
    return fig
