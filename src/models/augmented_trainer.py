import os
import sys
from typing import Dict, List

import numpy as np
import yaml
from datasets import load_from_disk
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.data_loader.alignment_utils import align_bio_labels_pair


def load_config(config_path: str) -> Dict:
    with open(config_path, "r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def load_augmented_huric_dataset(dataset_path: str = "data/processed/huric_augmented"):
    if not os.path.isdir(dataset_path):
        raise FileNotFoundError(f"Processed HuRIC augmented dataset not found: {dataset_path}")
    return load_from_disk(dataset_path)


def build_label_maps(dataset):
    labels = sorted({label for example in dataset for label in example["labels"]})
    label_to_id = {label: idx for idx, label in enumerate(labels)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    return labels, label_to_id, id_to_label


def tokenize_and_align_labels(examples, tokenizer, label_to_id, max_length: int = 128):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        examples["definition_tokens"],
        is_split_into_words=True,
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )

    all_labels = []
    for tokens, labels, definition_tokens in zip(examples["tokens"], examples["labels"], examples["definition_tokens"]):
        aligned_labels = align_bio_labels_pair(
            tokens,
            labels,
            definition_tokens,
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


def compute_metrics(pred):
    predictions = np.argmax(pred.predictions, axis=2)
    labels = pred.label_ids

    true_predictions = []
    true_labels = []

    for prediction, label in zip(predictions, labels):
        example_predictions = []
        example_labels = []
        for pred_id, label_id in zip(prediction, label):
            if label_id == -100:
                continue
            example_predictions.append(id_to_label[pred_id])
            example_labels.append(id_to_label[label_id])
        true_predictions.append(example_predictions)
        true_labels.append(example_labels)

    return {
        "precision": precision_score(true_labels, true_predictions),
        "recall": recall_score(true_labels, true_predictions),
        "f1": f1_score(true_labels, true_predictions),
    }


def train_augmented(config_path: str = "configs/baseline_config.yaml"):
    config = load_config(config_path)
    dataset = load_augmented_huric_dataset("data/processed/huric_augmented")
    train_test = dataset.train_test_split(test_size=0.15, seed=42)
    train_dataset = train_test["train"]
    eval_dataset = train_test["test"]

    labels, label_to_id, label_id_map = build_label_maps(dataset)
    global id_to_label
    id_to_label = label_id_map

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)
    model_config = AutoConfig.from_pretrained(
        "bert-base-uncased",
        num_labels=len(labels),
        id2label=id_to_label,
        label2id=label_to_id,
    )
    model = AutoModelForTokenClassification.from_pretrained(
        "bert-base-uncased",
        config=model_config,
    )

    max_length = int(config.get("max_length", 128))
    tokenized_train = train_dataset.map(
        lambda examples: tokenize_and_align_labels(examples, tokenizer, label_to_id, max_length=max_length),
        batched=True,
        remove_columns=train_dataset.column_names,
    )
    tokenized_eval = eval_dataset.map(
        lambda examples: tokenize_and_align_labels(examples, tokenizer, label_to_id, max_length=max_length),
        batched=True,
        remove_columns=eval_dataset.column_names,
    )

    output_dir = config.get("augmented_output_dir", "outputs/augmented")
    os.makedirs(output_dir, exist_ok=True)
    os.environ["TENSORBOARD_LOGGING_DIR"] = os.path.join(output_dir, "logs")

    learning_rate = float(config["learning_rate"])
    per_device_train_batch_size = int(config["per_device_train_batch_size"])
    per_device_eval_batch_size = int(config.get("per_device_eval_batch_size", per_device_train_batch_size))
    num_train_epochs = float(config["num_train_epochs"])
    weight_decay = float(config["weight_decay"])

    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        num_train_epochs=num_train_epochs,
        weight_decay=weight_decay,
        save_total_limit=2,
        logging_dir=os.path.join(output_dir, "logs"),
        load_best_model_at_end=True,
        metric_for_best_model="f1",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    model.config.save_pretrained(output_dir)
    metrics = trainer.evaluate()

    metrics_path = os.path.join(output_dir, "eval_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as writer:
        import json

        json.dump(metrics, writer, indent=2)

    print("Evaluation results:")
    print(metrics)
    print(f"Model, tokenizer, and checkpoint saved to {output_dir}")


if __name__ == "__main__":
    train_augmented()
