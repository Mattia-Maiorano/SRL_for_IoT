import json
import os
import shutil
import sys
from typing import Dict

import numpy as np
import optuna
import yaml
from datasets import load_from_disk
from seqeval.metrics import f1_score, precision_score, recall_score
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    TrainingArguments,
)

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.models.distill_srl import DistillationTrainer
from src.models.augmented_trainer import build_label_maps, load_augmented_huric_dataset, tokenize_and_align_labels


def load_config(config_path: str = "configs/baseline_config.yaml") -> Dict:
    with open(config_path, "r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def model_init(num_labels: int, id2label: Dict[str, int], label2id: Dict[str, int]):
    config = AutoConfig.from_pretrained(
        "huawei-noah/TinyBERT_General_4L_312D",
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )
    return AutoModelForTokenClassification.from_pretrained(
        "huawei-noah/TinyBERT_General_4L_312D",
        config=config,
    )


def compute_metrics(pred, id_to_label: Dict[int, str]):
    preds = pred.predictions
    if isinstance(preds, list):
        preds = np.array(preds)
    if isinstance(preds, np.ndarray):
        predictions = np.argmax(preds, axis=2)
    else:
        predictions = pred.predictions.argmax(axis=2).cpu().numpy()

    labels = pred.label_ids
    if isinstance(labels, np.ndarray):
        label_ids = labels
    else:
        label_ids = labels.cpu().numpy()

    true_predictions = []
    true_labels = []
    for prediction, label in zip(predictions, label_ids):
        example_predictions = []
        example_labels = []
        for pred_id, label_id in zip(prediction, label):
            if label_id == -100:
                continue
            example_predictions.append(id_to_label[int(pred_id)])
            example_labels.append(id_to_label[int(label_id)])
        true_predictions.append(example_predictions)
        true_labels.append(example_labels)

    return {
        "precision": precision_score(true_labels, true_predictions),
        "recall": recall_score(true_labels, true_predictions),
        "f1": f1_score(true_labels, true_predictions),
    }


def hp_space(trial: optuna.trial.Trial):
    return {
        "learning_rate": trial.suggest_loguniform("learning_rate", 1e-5, 1e-4),
        "num_train_epochs": trial.suggest_categorical("num_train_epochs", [5, 8, 12]),
        "distillation_temperature": trial.suggest_categorical("distillation_temperature", [1.0, 2.0, 4.0]),
    }


def run_sweep(config_path: str = "configs/baseline_config.yaml"):
    config = load_config(config_path)
    dataset = load_augmented_huric_dataset("data/processed/huric_augmented")
    train_test = dataset.train_test_split(test_size=0.15, seed=42)
    train_dataset = train_test["train"]
    eval_dataset = train_test["test"]

    labels, label_to_id, id_to_label = build_label_maps(dataset)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)
    teacher_model = AutoModelForTokenClassification.from_pretrained("outputs/augmented")
    teacher_model.eval()

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

    output_dir = config.get("sweep_output_dir", "outputs/sweep_trials")
    os.makedirs(output_dir, exist_ok=True)

    def objective(trial: optuna.trial.Trial):
        params = hp_space(trial)
        trial_output_dir = os.path.join(output_dir, f"trial_{trial.number}")
        os.makedirs(trial_output_dir, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=trial_output_dir,
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=float(params["learning_rate"]),
            per_device_train_batch_size=int(config["per_device_train_batch_size"]),
            per_device_eval_batch_size=int(config.get("per_device_eval_batch_size", config["per_device_train_batch_size"])),
            num_train_epochs=int(params["num_train_epochs"]),
            weight_decay=float(config["weight_decay"]),
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            dataloader_pin_memory=False,
        )

        trainer = DistillationTrainer(
            model_init=lambda: model_init(num_labels=len(labels), id2label=id_to_label, label2id=label_to_id),
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_eval,
            teacher_model=teacher_model,
            compute_metrics=lambda pred: compute_metrics(pred, id_to_label),
            alpha=float(config.get("distill_alpha", 0.5)),
            temperature=float(params["distillation_temperature"]),
        )

        trainer.train()
        metrics = trainer.evaluate()
        metrics_path = os.path.join(trial_output_dir, "eval_metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as writer:
            json.dump(metrics, writer, indent=2)

        return float(metrics.get("eval_f1", metrics.get("f1", 0.0)))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=int(config.get("sweep_trials", 10)))

    best_params = study.best_trial.params
    best_output_dir = os.path.join(output_dir, "best_student")
    os.makedirs(best_output_dir, exist_ok=True)

    best_training_args = TrainingArguments(
        output_dir=best_output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=float(best_params["learning_rate"]),
        per_device_train_batch_size=int(config["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(config.get("per_device_eval_batch_size", config["per_device_train_batch_size"])),
        num_train_epochs=int(best_params["num_train_epochs"]),
        weight_decay=float(config["weight_decay"]),
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        dataloader_pin_memory=False,
    )

    best_trainer = DistillationTrainer(
        model_init=lambda: model_init(num_labels=len(labels), id2label=id_to_label, label2id=label_to_id),
        args=best_training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        teacher_model=teacher_model,
        compute_metrics=lambda pred: compute_metrics(pred, id_to_label),
        alpha=float(config.get("distill_alpha", 0.5)),
        temperature=float(best_params["distillation_temperature"]),
    )

    best_trainer.train()
    best_trainer.save_model(best_output_dir)
    tokenizer.save_pretrained(best_output_dir)
    with open(os.path.join(best_output_dir, "best_params.yaml"), "w", encoding="utf-8") as writer:
        yaml.safe_dump(best_params, writer)

    public_best_path = "outputs/best_student"
    if os.path.isdir(public_best_path):
        import shutil

        shutil.rmtree(public_best_path)
    shutil.copytree(best_output_dir, public_best_path)

    print("Best trial saved to", best_output_dir)
    print("Also copied best student to", public_best_path)
    print("Best F1:", study.best_value)


if __name__ == "__main__":
    run_sweep()
