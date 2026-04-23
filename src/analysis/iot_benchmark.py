import json
import os
import sys
import time
from typing import Dict, Optional

import torch
from datasets import load_from_disk
from seqeval.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
)

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.data_loader.alignment_utils import align_bio_labels, align_bio_labels_pair


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_folder_size_mb(path: str) -> float:
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            if os.path.isfile(fp):
                total += os.path.getsize(fp)
    return total / (1024.0 ** 2)


def load_json_metrics(path: str) -> Optional[Dict]:
    metrics_path = os.path.join(path, "eval_metrics.json")
    if not os.path.isfile(metrics_path):
        return None
    with open(metrics_path, "r", encoding="utf-8") as reader:
        return json.load(reader)


def build_label_maps(dataset):
    labels = sorted({label for example in dataset for label in example["labels"]})
    label_to_id = {label: idx for idx, label in enumerate(labels)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    return label_to_id, id_to_label


def tokenize_examples(examples, tokenizer, label_to_id, max_length: int = 128):
    if "definition_tokens" in examples:
        tokenized = tokenizer(
            examples["tokens"],
            examples["definition_tokens"],
            is_split_into_words=True,
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        label_batches = []
        for tokens, labels, definition_tokens in zip(examples["tokens"], examples["labels"], examples["definition_tokens"]):
            aligned = align_bio_labels_pair(
                tokens,
                labels,
                definition_tokens,
                tokenizer,
                label_all_tokens=False,
                special_token_label="IGN",
            )
            label_batches.append([label_to_id[label] if label != "IGN" else -100 for label in aligned])
    else:
        tokenized = tokenizer(
            examples["tokens"],
            is_split_into_words=True,
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        label_batches = []
        for tokens, labels in zip(examples["tokens"], examples["labels"]):
            aligned = align_bio_labels(
                tokens,
                labels,
                tokenizer,
                label_all_tokens=False,
                special_token_label="IGN",
            )
            label_batches.append([label_to_id[label] if label != "IGN" else -100 for label in aligned])

    seq_len = len(tokenized["input_ids"][0]) if "input_ids" in tokenized else max_length
    for lbl in label_batches:
        if len(lbl) < seq_len:
            lbl.extend([-100] * (seq_len - len(lbl)))
        else:
            del lbl[seq_len:]

    tokenized["labels"] = label_batches
    return tokenized


def compute_metrics(pred, id_to_label: Dict[int, str]):
    preds = pred.predictions
    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().numpy()
    predictions = preds.argmax(axis=2)

    labels = pred.label_ids
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    true_predictions = []
    true_labels = []
    for prediction, label in zip(predictions, labels):
        pred_labels = []
        true_label_seq = []
        for pred_id, label_id in zip(prediction, label):
            if label_id == -100:
                continue
            if int(label_id) not in id_to_label or int(pred_id) not in id_to_label:
                continue
            pred_labels.append(id_to_label[int(pred_id)])
            true_label_seq.append(id_to_label[int(label_id)])
        true_predictions.append(pred_labels)
        true_labels.append(true_label_seq)

    return {
        "eval_precision": precision_score(true_labels, true_predictions),
        "eval_recall": recall_score(true_labels, true_predictions),
        "eval_f1": f1_score(true_labels, true_predictions),
    }


def load_model(folder: str, quantized: bool, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(folder, use_fast=True)
    if quantized:
        if not torch.cuda.is_available() and not torch.backends.mps.is_available():
            raise RuntimeError("Quantized model requires CUDA or MPS support.")
        qconfig = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        model = AutoModelForTokenClassification.from_pretrained(folder, quantization_config=qconfig, device_map="auto")
    else:
        model = AutoModelForTokenClassification.from_pretrained(folder)
        model.to(device)
    model.eval()
    return tokenizer, model


def get_test_subset(model_name: str):
    if model_name == "Baseline":
        dataset = load_from_disk("data/processed/huric")
    else:
        dataset = load_from_disk("data/processed/huric_augmented")
    test = dataset.train_test_split(test_size=0.15, seed=42)["test"]
    return test.select(range(min(50, len(test))))


def evaluate_f1(model_name: str, path: str, tokenizer, model, device: torch.device) -> float:
    metrics = load_json_metrics(path)
    if metrics and "eval_f1" in metrics:
        return float(metrics["eval_f1"])

    dataset = get_test_subset(model_name)
    label_to_id, id_to_label = build_label_maps(dataset)
    if hasattr(model.config, "id2label") and isinstance(model.config.id2label, dict):
        id_to_label = {int(k): v for k, v in model.config.id2label.items()}

    tokenized = dataset.map(
        lambda examples: tokenize_examples(examples, tokenizer, label_to_id),
        batched=True,
        remove_columns=dataset.column_names,
    )
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    dataloader = DataLoader(tokenized, batch_size=8)

    all_true_predictions = []
    all_true_labels = []
    model.eval()

    for batch in dataloader:
        labels = batch.pop("labels")
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        logits = outputs.logits.cpu().numpy()
        preds = logits.argmax(axis=2)
        labels = labels.numpy()

        for prediction, label in zip(preds, labels):
            pred_labels = []
            true_label_seq = []
            for pred_id, label_id in zip(prediction, label):
                if label_id == -100:
                    continue
                if int(label_id) not in id_to_label or int(pred_id) not in id_to_label:
                    continue
                pred_labels.append(id_to_label[int(pred_id)])
                true_label_seq.append(id_to_label[int(label_id)])
            all_true_predictions.append(pred_labels)
            all_true_labels.append(true_label_seq)

    return float(f1_score(all_true_labels, all_true_predictions))


def measure_latency(model, tokenizer, device: torch.device, model_name: str) -> float:
    example = get_test_subset(model_name)[0]
    if "definition_tokens" in example:
        inputs = tokenizer(
            example["tokens"],
            example["definition_tokens"],
            is_split_into_words=True,
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt",
        )
    else:
        inputs = tokenizer(
            example["tokens"],
            is_split_into_words=True,
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt",
        )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        for _ in range(100):
            model(**inputs)

    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(100):
            model(**inputs)
    end = time.perf_counter()

    return (end - start) / 100 * 1000


def markdown_table(rows):
    header = "| Model Name | F1 Score | Disk Size (MB) | Single-Inference Latency (ms) |"
    separator = "|---|---|---|---|"
    lines = [header, separator]
    for row in rows:
        lines.append(
            f"| {row['name']} | {row['f1']:.4f} | {row['size_mb']:.1f} | {row['latency_ms']:.2f} |"
        )
    return "\n".join(lines)


def main():
    device = get_device()
    print(f"Using device: {device}")

    models = [
        {"name": "Baseline", "path": "outputs/baseline", "quantized": False},
        {"name": "Best Student", "path": "outputs/sweep_trials/best_student", "quantized": False},
        {"name": "4-bit Quantized", "path": "outputs/iot_ready_4bit", "quantized": True},
    ]

    rows = []
    for item in models:
        if not os.path.isdir(item["path"]):
            raise FileNotFoundError(f"Model directory not found: {item['path']}")

        tokenizer, model = load_model(item["path"], item["quantized"], device)
        f1 = evaluate_f1(item["name"], item["path"], tokenizer, model, device)
        size_mb = get_folder_size_mb(item["path"])
        latency_ms = measure_latency(model, tokenizer, device, item["name"])

        rows.append({"name": item["name"], "f1": f1, "size_mb": size_mb, "latency_ms": latency_ms})

    table = markdown_table(rows)
    output_path = "iot_benchmark_results.md"
    with open(output_path, "w", encoding="utf-8") as writer:
        writer.write(table)

    print("Benchmark results saved to", output_path)
    print(table)


if __name__ == "__main__":
    main()
