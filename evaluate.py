import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
)

from tickets_data import LABELS


def calculate_scores(y_true, y_pred):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def save_confusion_matrix(y_true, y_pred, path):
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        labels=LABELS,
        cmap="Blues",
    )

    plt.title("Test Confusion Matrix")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def save_learning_curve(results, path):
    summary = (
        results.groupby(["method", "n"])["accuracy"]
        .agg(["mean", "min", "max"])
        .reset_index()
    )

    plt.figure(figsize=(9, 6))

    for method, group in summary.groupby("method"):
        group = group.sort_values("n")

        plt.plot(
            group["n"],
            group["mean"],
            marker="o",
            label=method,
        )

        plt.fill_between(
            group["n"],
            group["min"],
            group["max"],
            alpha=0.15,
        )

    plt.xlabel("Number of Training Examples")
    plt.ylabel("Validation Accuracy")
    plt.title("Learning Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
