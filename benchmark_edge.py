import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_from_disk
from seqeval.metrics import classification_report
from transformers import AutoModelForTokenClassification, AutoTokenizer

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "src"))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.data_loader.alignment_utils import align_bio_labels, align_bio_labels_pair


def load_model_and_tokenizer(path: str):
    tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(path)
    return tokenizer, model


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def select_test_subset(dataset, num_examples: int = 32):
    subset = dataset.train_test_split(test_size=0.15, seed=42)["test"]
    if len(subset) > num_examples:
        return subset.select(range(num_examples))
    return subset


def align_labels(tokens, labels, tokenizer, definition_tokens: Optional[List[str]] = None):
    if definition_tokens is None:
        aligned = align_bio_labels(tokens, labels, tokenizer, label_all_tokens=False, special_token_label="IGN")
    else:
        aligned = align_bio_labels_pair(tokens, labels, definition_tokens, tokenizer, label_all_tokens=False, special_token_label="IGN")
    return aligned


def tokenize_for_inference(
    tokenizer,
    tokens,
    labels,
    label_to_id,
    definition_tokens: Optional[List[str]] = None,
    max_length: int = 128,
):
    if definition_tokens is not None:
        encoded = tokenizer(
            tokens,
            definition_tokens,
            is_split_into_words=True,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
    else:
        encoded = tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )

    aligned_labels = align_labels(tokens, labels, tokenizer, definition_tokens=definition_tokens)
    label_ids = [label_to_id[label] if label != "IGN" else -100 for label in aligned_labels]
    label_ids = label_ids[:max_length] + [-100] * max(0, max_length - len(label_ids))
    encoded["labels"] = torch.tensor([label_ids], dtype=torch.long)
    return encoded, aligned_labels


def compute_model_metrics(
    model,
    tokenizer,
    dataset,
    label_to_id,
    id_to_label,
    is_augmented: bool,
    max_length: int = 128,
    num_examples: int = 32,
) -> float:
    test_subset = select_test_subset(dataset, num_examples=num_examples)
    all_preds = []
    all_labels = []
    device = next(model.parameters()).device

    model.eval()
    with torch.no_grad():
        for example in test_subset:
            definition_tokens = example.get("definition_tokens") if is_augmented else None
            encoded, aligned_labels = tokenize_for_inference(
                tokenizer,
                example["tokens"],
                example["labels"],
                label_to_id,
                definition_tokens=definition_tokens,
                max_length=max_length,
            )
            inputs = {k: v.to(device) for k, v in encoded.items() if k in {"input_ids", "attention_mask", "token_type_ids"}}
            outputs = model(**inputs)
            pred_ids = outputs.logits.argmax(dim=-1).cpu().numpy()[0]

            filtered_preds = []
            filtered_labels = []
            for pred_id, label in zip(pred_ids, encoded["labels"].cpu().numpy()[0]):
                if label == -100:
                    continue
                filtered_preds.append(id_to_label[int(pred_id)])
                filtered_labels.append(id_to_label[int(label)])

            all_preds.append(filtered_preds)
            all_labels.append(filtered_labels)

    report = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
    return float(report.get("macro avg", {}).get("f1-score", 0.0))


def measure_latency(
    model,
    tokenizer,
    dataset,
    is_augmented: bool,
    max_length: int = 128,
    num_runs: int = 20,
    num_examples: int = 16,
) -> float:
    test_subset = select_test_subset(dataset, num_examples=num_examples)
    device = get_device()
    model.to(device)

    encoded_examples = []
    for example in test_subset:
        definition_tokens = example.get("definition_tokens") if is_augmented else None
        encoded, _ = tokenize_for_inference(
            tokenizer,
            example["tokens"],
            example["labels"],
            label_to_id,
            definition_tokens=definition_tokens,
            max_length=max_length,
        )
        encoded_examples.append({k: v.to(device) for k, v in encoded.items() if k in {"input_ids", "attention_mask", "token_type_ids"}})

    model.eval()
    with torch.no_grad():
        for _ in range(5):
            for inputs in encoded_examples[:5]:
                model(**inputs)

    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_runs):
            for inputs in encoded_examples:
                model(**inputs)
    end = time.perf_counter()

    total_calls = num_runs * len(encoded_examples)
    return (end - start) / total_calls * 1000


def benchmark_model(
    name: str,
    model_path: str,
    dataset,
    label_to_id,
    id_to_label,
    is_augmented: bool,
) -> Tuple[float, float]:
    tokenizer, model = load_model_and_tokenizer(model_path)
    f1 = compute_model_metrics(model, tokenizer, dataset, label_to_id, id_to_label, is_augmented)
    latency = measure_latency(model, tokenizer, dataset, is_augmented)
    print(f"{name}: F1={f1:.4f}, latency={latency:.2f}ms")
    return latency, f1


def plot_tradeoff(results: Dict[str, Tuple[float, float]], output_path: str = "accuracy_latency_tradeoff.png"):
    labels = list(results.keys())
    latencies = [results[k][0] for k in labels]
    f1s = [results[k][1] for k in labels]

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.bar(labels, latencies, color="tab:red", alpha=0.6)
    ax1.set_ylabel("Latency (ms)", color="tab:red")
    ax1.tick_params(axis="y", labelcolor="tab:red")

    ax2 = ax1.twinx()
    ax2.plot(labels, f1s, color="tab:blue", marker="o", linewidth=2)
    ax2.set_ylabel("F1 score", color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue")

    ax1.set_title("Accuracy vs Latency Tradeoff")
    fig.tight_layout()
    fig.savefig(output_path)
    print(f"Saved benchmark tradeoff plot to {output_path}")


def build_label_maps(dataset):
    labels = sorted({label for example in dataset for label in example["labels"]})
    label_to_id = {label: idx for idx, label in enumerate(labels)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    return label_to_id, id_to_label


def main():
    baseline_dataset = load_from_disk("data/processed/huric")
    augmented_dataset = load_from_disk("data/processed/huric_augmented")

    baseline_label_to_id, baseline_id_to_label = build_label_maps(baseline_dataset)
    augmented_label_to_id, augmented_id_to_label = build_label_maps(augmented_dataset)

    results = {}
    results["Baseline"] = benchmark_model(
        "Baseline",
        "outputs/baseline",
        baseline_dataset,
        baseline_label_to_id,
        baseline_id_to_label,
        is_augmented=False,
    )
    results["Augmented"] = benchmark_model(
        "Augmented",
        "outputs/augmented",
        augmented_dataset,
        augmented_label_to_id,
        augmented_id_to_label,
        is_augmented=True,
    )
    results["Distilled"] = benchmark_model(
        "Distilled",
        "outputs/distilled",
        augmented_dataset,
        augmented_label_to_id,
        augmented_id_to_label,
        is_augmented=True,
    )
    results["Quantized"] = benchmark_model(
        "Quantized",
        "outputs/iot_model",
        augmented_dataset,
        augmented_label_to_id,
        augmented_id_to_label,
        is_augmented=True,
    )

    plot_tradeoff(results)


if __name__ == "__main__":
    main()
