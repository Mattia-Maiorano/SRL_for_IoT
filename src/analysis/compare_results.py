import os
import sys
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from datasets import load_from_disk
from seqeval.metrics import classification_report
from transformers import AutoModelForTokenClassification, AutoTokenizer, Trainer

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.data_loader.alignment_utils import align_bio_labels, align_bio_labels_pair


def load_checkpoint(checkpoint_path: str):
    tokenizer = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, use_fast=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)

    model = AutoModelForTokenClassification.from_pretrained(checkpoint_path)
    return tokenizer, model


def build_label_maps(dataset):
    labels = sorted({label for example in dataset for label in example["labels"]})
    label_to_id = {label: idx for idx, label in enumerate(labels)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    return label_to_id, id_to_label


def _align_labels_for_example(tokens, labels, definition_tokens, tokenizer, label_to_id, max_length: int):
    if definition_tokens is None:
        encoded = tokenizer(tokens, is_split_into_words=True, add_special_tokens=True)
        aligned_labels = align_bio_labels(tokens, labels, tokenizer, label_all_tokens=False, special_token_label="IGN")
    else:
        encoded = tokenizer(
            tokens,
            definition_tokens,
            is_split_into_words=True,
            add_special_tokens=True,
        )
        aligned_labels = align_bio_labels_pair(tokens, labels, definition_tokens, tokenizer, label_all_tokens=False, special_token_label="IGN")

    label_ids = [label_to_id[label] if label != "IGN" else -100 for label in aligned_labels]
    label_ids = label_ids[:max_length]
    label_ids += [-100] * (max_length - len(label_ids))
    return label_ids


def tokenize_dataset(dataset, tokenizer, label_to_id, max_length: int = 128):
    is_augmented = "definition_tokens" in dataset.column_names

    def _tokenize(examples):
        if is_augmented:
            tokenized = tokenizer(
                examples["tokens"],
                examples["definition_tokens"],
                is_split_into_words=True,
                truncation=True,
                padding="max_length",
                max_length=max_length,
            )
        else:
            tokenized = tokenizer(
                examples["tokens"],
                is_split_into_words=True,
                truncation=True,
                padding="max_length",
                max_length=max_length,
            )

        all_labels = []
        if is_augmented:
            for tokens, labels, definition_tokens in zip(examples["tokens"], examples["labels"], examples["definition_tokens"]):
                all_labels.append(_align_labels_for_example(tokens, labels, definition_tokens, tokenizer, label_to_id, max_length))
        else:
            for tokens, labels in zip(examples["tokens"], examples["labels"]):
                all_labels.append(_align_labels_for_example(tokens, labels, None, tokenizer, label_to_id, max_length))

        tokenized["labels"] = all_labels
        return tokenized

    return dataset.map(
        _tokenize,
        batched=True,
        remove_columns=dataset.column_names,
    )


def classification_report_dict(predictions, label_ids, id_to_label):
    preds = np.argmax(predictions, axis=2)
    true_predictions = []
    true_labels = []

    for prediction, label in zip(preds, label_ids):
        example_predictions = []
        example_labels = []
        for pred_id, label_id in zip(prediction, label):
            if label_id == -100:
                continue
            example_predictions.append(id_to_label[pred_id])
            example_labels.append(id_to_label[label_id])
        true_predictions.append(example_predictions)
        true_labels.append(example_labels)

    return classification_report(true_labels, true_predictions, output_dict=True, zero_division=0)


def evaluate_checkpoint(checkpoint_path: str, dataset):
    tokenizer, model = load_checkpoint(checkpoint_path)
    label_to_id, id_to_label = build_label_maps(dataset)

    test_dataset = dataset.train_test_split(test_size=0.15, seed=42)["test"]
    tokenized_test = tokenize_dataset(test_dataset, tokenizer, label_to_id)
    trainer = Trainer(model=model)
    predictions, label_ids, _ = trainer.predict(tokenized_test)
    return classification_report_dict(predictions, label_ids, id_to_label)


def top_n_labels(report: Dict, n: int = 10) -> List[str]:
    class_keys = [k for k, v in report.items() if isinstance(v, dict) and k not in {"macro avg", "weighted avg", "micro avg", "accuracy"}]
    ranked = sorted(class_keys, key=lambda label: report[label]["f1-score"], reverse=True)
    return ranked[:n]


def compare_and_plot(baseline_report: Dict, augmented_report: Dict, output_path: str = "comparison_f1_top10.png"):
    labels = top_n_labels(baseline_report, n=10)
    augmented_labels = top_n_labels(augmented_report, n=10)
    labels = list(dict.fromkeys(labels + augmented_labels))[:10]

    baseline_scores = [baseline_report.get(label, {}).get("f1-score", 0.0) for label in labels]
    augmented_scores = [augmented_report.get(label, {}).get("f1-score", 0.0) for label in labels]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - width / 2, baseline_scores, width, label="Baseline")
    ax.bar(x + width / 2, augmented_scores, width, label="Augmented")

    ax.set_xlabel("Argument class")
    ax.set_ylabel("F1 score")
    ax.set_title("Top 10 SRL argument F1 scores: baseline vs augmented")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylim(0, 1)
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_path)
    print(f"Saved comparison chart to {output_path}")


def main():
    dataset = load_from_disk("data/processed/huric")
    baseline_report = evaluate_checkpoint("outputs/baseline", dataset)

    try:
        augmented_dataset = load_from_disk("data/processed/huric_augmented")
    except FileNotFoundError:
        raise FileNotFoundError("Augmented dataset path data/processed/huric_augmented not found. Run prepare_dataset.prepare_huric_augmented_dataset first.")

    augmented_report = evaluate_checkpoint("outputs/augmented", augmented_dataset)

    compare_and_plot(baseline_report, augmented_report)


if __name__ == "__main__":
    main()
