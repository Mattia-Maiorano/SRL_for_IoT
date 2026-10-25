import os
import sys
from typing import Dict, List

import numpy as np
from datasets import load_from_disk
from seqeval.metrics import classification_report
from transformers import AutoModelForTokenClassification, AutoTokenizer, Trainer

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.data_loader.alignment_utils import align_bio_labels


def load_huric_dataset(dataset_path: str = "data/processed/huric"):
    if not os.path.isdir(dataset_path):
        raise FileNotFoundError(f"Processed HuRIC dataset not found: {dataset_path}")
    return load_from_disk(dataset_path)


def build_label_maps(dataset):
    labels = sorted({label for example in dataset for label in example["labels"]})
    label_to_id = {label: idx for idx, label in enumerate(labels)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    return labels, label_to_id, id_to_label


def tokenize_and_align_labels(examples, tokenizer, label_to_id, max_length: int = 128):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        is_split_into_words=True,
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )

    all_labels = []
    for tokens, labels in zip(examples["tokens"], examples["labels"]):
        aligned_labels = align_bio_labels(
            tokens,
            labels,
            tokenizer,
            label_all_tokens=False,
            special_token_label="IGN",
        )
        label_ids = [label_to_id[label] if label != "IGN" else -100 for label in aligned_labels]
        label_ids = label_ids[: max_length]
        padding_length = max_length - len(label_ids)
        if padding_length > 0:
            label_ids += [-100] * padding_length
        all_labels.append(label_ids)

    tokenized_inputs["labels"] = all_labels
    return tokenized_inputs


def compute_report(predictions: np.ndarray, labels: np.ndarray, id_to_label: Dict[int, str]) -> str:
    preds = np.argmax(predictions, axis=2)
    true_predictions = []
    true_labels = []

    for prediction, label in zip(preds, labels):
        example_predictions = []
        example_labels = []
        for pred_id, label_id in zip(prediction, label):
            if label_id == -100:
                continue
            example_predictions.append(id_to_label[pred_id])
            example_labels.append(id_to_label[label_id])
        true_predictions.append(example_predictions)
        true_labels.append(example_labels)

    return classification_report(true_labels, true_predictions, digits=4)


def evaluate_checkpoint(checkpoint_path: str = "outputs/baseline"):
    if not os.path.isdir(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_path}")

    dataset = load_huric_dataset("data/processed/huric")
    splits = dataset.train_test_split(test_size=0.15, seed=42)
    test_dataset = splits["test"]

    labels, label_to_id, id_to_label = build_label_maps(dataset)

    try:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, use_fast=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)

    model = AutoModelForTokenClassification.from_pretrained(checkpoint_path)

    max_length = 128
    tokenized_test = test_dataset.map(
        lambda examples: tokenize_and_align_labels(examples, tokenizer, label_to_id, max_length=max_length),
        batched=True,
        remove_columns=test_dataset.column_names,
    )

    trainer = Trainer(model=model)
    predictions, label_ids, _ = trainer.predict(tokenized_test)
    print("Baseline SRL evaluation report:\n")
    print(compute_report(predictions, label_ids, id_to_label))


if __name__ == "__main__":
    evaluate_checkpoint()
