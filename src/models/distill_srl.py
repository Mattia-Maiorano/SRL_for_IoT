import os
import sys
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from seqeval.metrics import f1_score, precision_score, recall_score
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

from src.models.augmented_trainer import (
    build_label_maps,
    load_augmented_huric_dataset,
    tokenize_and_align_labels,
)

id_to_label = {}


def load_config(config_path: str) -> Dict:
    with open(config_path, "r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def compute_metrics(pred):
    preds = pred.predictions
    if isinstance(preds, np.ndarray):
        predictions = np.argmax(preds, axis=2)
    else:
        predictions = torch.argmax(preds, dim=2).cpu().numpy()

    labels = pred.label_ids
    if torch.is_tensor(labels):
        labels = labels.cpu().numpy()

    true_predictions = []
    true_labels = []

    for prediction, label in zip(predictions, labels):
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


class DistillationTrainer(Trainer):
    def __init__(self, teacher_model=None, alpha: float = 0.5, temperature: float = 2.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_model = teacher_model
        self.alpha = alpha
        self.temperature = temperature
        if self.teacher_model is not None:
            self.teacher_model.eval()

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        student_logits = outputs.logits

        with torch.no_grad():
            if self.teacher_model is None:
                raise ValueError("Teacher model is required for distillation.")
            self.teacher_model.to(student_logits.device)
            teacher_outputs = self.teacher_model(**inputs)
            teacher_logits = teacher_outputs.logits

        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        task_loss = loss_fct(student_logits.view(-1, student_logits.size(-1)), labels.view(-1))

        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=-1)

        mask = labels != -100
        mask = mask.unsqueeze(-1).expand_as(student_logits)
        student_log_probs = student_log_probs[mask].view(-1, student_logits.size(-1))
        teacher_probs = teacher_probs[mask].view(-1, student_logits.size(-1))

        kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (self.temperature ** 2)
        loss = self.alpha * task_loss + (1.0 - self.alpha) * kl_loss

        if return_outputs:
            return loss, outputs
        return loss


def train_distilled(config_path: str = "configs/baseline_config.yaml"):
    config = load_config(config_path)
    dataset = load_augmented_huric_dataset("data/processed/huric_augmented")
    train_test = dataset.train_test_split(test_size=0.15, seed=42)
    train_dataset = train_test["train"]
    eval_dataset = train_test["test"]

    labels, label_to_id, label_id_map = build_label_maps(dataset)
    global id_to_label
    id_to_label = label_id_map

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)
    teacher = AutoModelForTokenClassification.from_pretrained("outputs/augmented")

    student_config = AutoConfig.from_pretrained(
        "huawei-noah/TinyBERT_General_4L_312D",
        num_labels=len(labels),
        id2label=id_to_label,
        label2id=label_to_id,
    )
    student = AutoModelForTokenClassification.from_pretrained(
        "huawei-noah/TinyBERT_General_4L_312D",
        config=student_config,
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

    output_dir = config.get("distilled_output_dir", "outputs/distilled")
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
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        dataloader_pin_memory=False,
    )

    trainer = DistillationTrainer(
        model=student,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        teacher_model=teacher,
        compute_metrics=compute_metrics,
        alpha=float(config.get("distill_alpha", 0.5)),
        temperature=float(config.get("distill_temperature", 2.0)),
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    student.config.save_pretrained(output_dir)

    metrics = trainer.evaluate()
    with open(os.path.join(output_dir, "eval_metrics.json"), "w", encoding="utf-8") as writer:
        import json

        json.dump(metrics, writer, indent=2)

    print("Distillation completed.")
    print(metrics)
    print(f"Student model saved to {output_dir}")


if __name__ == "__main__":
    train_distilled()
