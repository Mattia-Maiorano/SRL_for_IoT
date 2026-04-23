import json
import os
import sys
from typing import Dict, List, Optional

import matplotlib.pyplot as plt

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


def find_metric_files(root_dir: str) -> List[str]:
    metric_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename == "eval_metrics.json":
                metric_files.append(os.path.join(dirpath, filename))
    return sorted(metric_files)


def load_metrics(file_path: str) -> Optional[Dict]:
    with open(file_path, "r", encoding="utf-8") as reader:
        return json.load(reader)


def label_from_path(file_path: str, root_dir: str) -> str:
    rel = os.path.relpath(os.path.dirname(file_path), root_dir)
    parts = rel.split(os.sep)
    if "baseline" in parts:
        return "Baseline"
    if "augmented" in parts:
        return "Augmented"
    if "best_student" in parts:
        return "Best Student"
    if "sweep_trials" in parts or "trial_" in parts:
        return "Student Trial"
    return parts[-1] if parts else "unknown"


def gather_points(root_dir: str):
    points = []
    for metric_path in find_metric_files(root_dir):
        metrics = load_metrics(metric_path)
        if not metrics:
            continue

        name = label_from_path(metric_path, root_dir)
        f1 = metrics.get("eval_f1", metrics.get("f1"))
        speed = metrics.get("eval_samples_per_second")
        if speed is None:
            runtime = metrics.get("eval_runtime")
            if runtime:
                speed = 1.0 / float(runtime)
        if f1 is None or speed is None:
            continue

        points.append({"name": name, "f1": float(f1), "speed": float(speed), "path": metric_path})
    return points


def plot_tradeoffs(root_dir: str = "outputs", output_path: str = "accuracy_vs_latency_tradeoff.png"):
    points = gather_points(root_dir)
    if not points:
        raise RuntimeError(f"No eval_metrics.json files found under {root_dir}")

    categories = {
        "Baseline": "Baseline",
        "Augmented": "Augmented",
        "Best Student": "Student",
        "Student Trial": "Student Trial",
    }

    colors = {
        "Baseline": "tab:blue",
        "Augmented": "tab:orange",
        "Best Student": "tab:green",
        "Student Trial": "tab:purple",
        "unknown": "tab:gray",
    }

    fig, ax = plt.subplots(figsize=(10, 6))
    for point in points:
        label = categories.get(point["name"], point["name"])
        ax.scatter(point["speed"], point["f1"], color=colors.get(point["name"], "tab:gray"), label=label, alpha=0.8, s=80)
        ax.annotate(point["name"], (point["speed"], point["f1"]), textcoords="offset points", xytext=(5, 5), ha="left", fontsize=8)

    ax.set_xlabel("Inference Speed (samples/sec)")
    ax.set_ylabel("F1 Score")
    ax.set_title("Accuracy vs Latency Tradeoff")
    ax.grid(True, linestyle="--", alpha=0.3)

    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), loc="lower right")

    fig.tight_layout()
    fig.savefig(output_path)
    print(f"Saved tradeoff plot to {output_path}")


if __name__ == "__main__":
    plot_tradeoffs()
