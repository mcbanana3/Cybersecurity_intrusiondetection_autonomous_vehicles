"""
Evaluation plots (matplotlib / seaborn).

Generates the figures used in the report/paper, all from real model
outputs. Uses a non-interactive backend so it runs headless and saves
PNGs to results/figures/.
"""

from __future__ import annotations

import os
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")  # headless backend (no display needed)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
)

from src.utils.logger import get_logger

logger = get_logger(__name__)

FIG_DIR = os.path.join("results", "figures")


def _ensure_dir() -> None:
    os.makedirs(FIG_DIR, exist_ok=True)


def plot_confusion_matrix(cm: List[List[int]], class_names: List[str]) -> str:
    """Save a confusion-matrix heatmap. Returns the file path."""
    _ensure_dir()
    arr = np.array(cm)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(arr, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Multi-class Confusion Matrix (test)")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "confusion_matrix.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", path)
    return path


def plot_per_class_f1(per_class_f1: Dict[str, float]) -> str:
    """Save a per-class F1 bar chart. Returns the file path."""
    _ensure_dir()
    names = list(per_class_f1.keys())
    vals = list(per_class_f1.values())
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=names, y=vals, ax=ax, color="#3b6ea5")
    ax.set_ylim(0, 1)
    ax.set_ylabel("F1 score")
    ax.set_title("Per-class F1 (test)")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "per_class_f1.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", path)
    return path


def plot_roc(y_true: np.ndarray, y_score: np.ndarray) -> tuple[str, float]:
    """Save a binary ROC curve. Returns (path, auc)."""
    _ensure_dir()
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#c0392b", label=f"ROC (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Binary Detector ROC Curve")
    ax.legend(loc="lower right")
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "roc_curve.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s (AUC=%.3f)", path, roc_auc)
    return path, float(roc_auc)


def plot_pr(y_true: np.ndarray, y_score: np.ndarray) -> tuple[str, float]:
    """Save a binary Precision-Recall curve. Returns (path, average_precision)."""
    _ensure_dir()
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, color="#2c7fb8", label=f"PR (AP = {ap:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Binary Detector Precision-Recall Curve")
    ax.legend(loc="lower left")
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "pr_curve.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s (AP=%.3f)", path, ap)
    return path, float(ap)


def plot_normal_vs_attack(attacked_can: pd.DataFrame) -> str:
    """Save a frame-rate-over-time plot highlighting attack windows."""
    _ensure_dir()
    df = attacked_can.copy()
    df["sec"] = df["timestamp"].astype(int)
    total = df.groupby("sec").size()
    attack = df[df["label"] == "attack"].groupby("sec").size()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(total.index, total.values, label="all frames/s", color="#2c3e50")
    ax.bar(attack.index, attack.values, label="attack frames/s",
           color="#c0392b", alpha=0.7, width=1.0)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("frames per second")
    ax.set_title("CAN Frame Rate: Normal vs Attack Windows")
    ax.legend(loc="upper right")
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "normal_vs_attack.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", path)
    return path