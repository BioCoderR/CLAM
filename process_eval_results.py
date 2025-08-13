"""Process evaluation results produced by :mod:`eval.py`.

Given a directory that contains CSV files for each cross-validation fold
(```fold_*.csv```), this script aggregates the results and generates:

    1. ROC and Precision–Recall curves for every fold.
    2. Per-class ROC and Precision–Recall curves across all folds.
    3. Classification metrics for each fold and overall.
    4. Confusion matrix plots for each fold.

Each fold CSV is the output of ``eval.py`` and must contain the columns:

    - ``Y``: ground truth labels
    - ``Y_hat``: predicted labels
    - ``p_0``...``p_n``: class probabilities

Outputs are saved in the same directory.
"""
import argparse
import glob
import os
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Process eval.py results and generate plots/metrics"
    )
    parser.add_argument(
        "--eval_dir",
        type=str,
        default="./eval_results",
        help="Directory containing fold_*.csv files from eval.py",
    )
    return parser.parse_args()


def _plot_curves(
    curves: List[Dict[str, Any]],
    title: str,
    xlabel: str,
    ylabel: str,
    save_path: str,
) -> None:
    import matplotlib.pyplot as plt
    plt.figure()
    for c in curves:
        label = c.get("label", None)
        plt.plot(c["x"], c["y"], label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if any(c.get("label") for c in curves):
        plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def _plot_confusion(cm: Any, classes: List[str], save_path: str) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    plt.figure()
    im = plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.0
    for i, j in np.ndindex(cm.shape):
        plt.text(
            j,
            i,
            format(cm[i, j], "d"),
            ha="center",
            va="center",
            color="white" if cm[i, j] > thresh else "black",
        )
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def process_eval_dir(eval_dir: str) -> None:
    import numpy as np
    import pandas as pd
    from sklearn.metrics import (
        classification_report,
        confusion_matrix,
        precision_recall_curve,
        roc_curve,
    )
    from sklearn.preprocessing import label_binarize

    fold_paths = sorted(glob.glob(os.path.join(eval_dir, "fold_*.csv")))
    if not fold_paths:
        print(f"No fold csv files found in {eval_dir}, skipping")
        return

    all_y_true: List[np.ndarray] = []
    all_y_pred: List[np.ndarray] = []
    all_probs: List[np.ndarray] = []
    roc_curves: List[Dict[str, np.ndarray]] = []
    pr_curves: List[Dict[str, np.ndarray]] = []

    for fold_idx, fp in enumerate(fold_paths):
        df = pd.read_csv(fp)
        prob_cols = [c for c in df.columns if c.startswith("p_")]
        n_classes = len(prob_cols)
        classes = [str(i) for i in range(n_classes)]

        y_true = df["Y"].values
        y_pred = df["Y_hat"].values
        probs = df[prob_cols].values

        all_y_true.append(y_true)
        all_y_pred.append(y_pred)
        all_probs.append(probs)

        # Classification report per fold
        report = classification_report(y_true, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        report_df.to_csv(
            os.path.join(eval_dir, f"metrics_fold_{fold_idx}.csv"), index=True
        )

        # Confusion matrix per fold
        cm = confusion_matrix(y_true, y_pred)
        _plot_confusion(
            cm,
            classes,
            os.path.join(eval_dir, f"confusion_matrix_fold_{fold_idx}.png"),
        )

        # ROC/PR curves per fold (micro average for multi-class)
        if n_classes == 2:
            fpr, tpr, _ = roc_curve(y_true, probs[:, 1])
            precision, recall, _ = precision_recall_curve(y_true, probs[:, 1])
        else:
            y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))
            fpr, tpr, _ = roc_curve(y_true_bin.ravel(), probs.ravel())
            precision, recall, _ = precision_recall_curve(
                y_true_bin.ravel(), probs.ravel()
            )
        roc_curves.append({"x": fpr, "y": tpr, "label": f"fold {fold_idx}"})
        pr_curves.append(
            {"x": recall, "y": precision, "label": f"fold {fold_idx}"}
        )

    # Plot all fold curves
    _plot_curves(
        roc_curves,
        "ROC curves for all folds",
        "False Positive Rate",
        "True Positive Rate",
        os.path.join(eval_dir, "roc_all_folds.png"),
    )
    _plot_curves(
        pr_curves,
        "Precision-Recall curves for all folds",
        "Recall",
        "Precision",
        os.path.join(eval_dir, "pr_all_folds.png"),
    )

    # Aggregate predictions across folds
    y_true_all = np.concatenate(all_y_true)
    y_pred_all = np.concatenate(all_y_pred)
    probs_all = np.concatenate(all_probs)

    n_classes = probs_all.shape[1]
    classes = [str(i) for i in range(n_classes)]

    # Overall classification report
    report_all = classification_report(y_true_all, y_pred_all, output_dict=True)
    pd.DataFrame(report_all).transpose().to_csv(
        os.path.join(eval_dir, "metrics_overall.csv"), index=True
    )

    # Per-class ROC and PR curves
    y_true_bin_all = label_binarize(y_true_all, classes=list(range(n_classes)))
    roc_cls_curves: List[Dict[str, np.ndarray]] = []
    pr_cls_curves: List[Dict[str, np.ndarray]] = []
    for c in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin_all[:, c], probs_all[:, c])
        precision, recall, _ = precision_recall_curve(
            y_true_bin_all[:, c], probs_all[:, c]
        )
        roc_cls_curves.append({"x": fpr, "y": tpr, "label": f"class {c}"})
        pr_cls_curves.append({"x": recall, "y": precision, "label": f"class {c}"})

    _plot_curves(
        roc_cls_curves,
        "Per-class ROC curves",
        "False Positive Rate",
        "True Positive Rate",
        os.path.join(eval_dir, "roc_per_class.png"),
    )
    _plot_curves(
        pr_cls_curves,
        "Per-class Precision-Recall curves",
        "Recall",
        "Precision",
        os.path.join(eval_dir, "pr_per_class.png"),
    )


if __name__ == "__main__":
    args = parse_args()
    if os.path.isdir(args.eval_dir):
        print(f"Processing directory: {args.eval_dir}")
        process_eval_dir(args.eval_dir)
    else:
        print(f"Evaluation directory {args.eval_dir} not found, skipping")
