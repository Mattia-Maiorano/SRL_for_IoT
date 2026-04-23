import os
import shutil
import sys

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer, BitsAndBytesConfig

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


def _get_device_map():
    if torch.cuda.is_available():
        return "auto"
    if torch.backends.mps.is_available():
        return "auto"
    raise RuntimeError(
        "4-bit quantization requires CUDA or supported MPS hardware. "
        "Run this on a GPU/MPS machine or use a non-quantized path."
    )


def quantize_best_student(
    best_student_path: str = "outputs/best_student",
    output_path: str = "outputs/iot_ready_4bit",
    quant_type: str = "nf4",
    compute_dtype: str = "float16",
):
    fallback_path = "outputs/sweep_trials/best_student"
    if not os.path.isdir(best_student_path):
        if os.path.isdir(fallback_path):
            print(f"outputs/best_student not found, using fallback {fallback_path}")
            best_student_path = fallback_path
        else:
            raise FileNotFoundError(f"Best student model not found: {best_student_path} or {fallback_path}")

    tokenizer = AutoTokenizer.from_pretrained(best_student_path, use_fast=True)
    qconfig = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=quant_type,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=getattr(torch, compute_dtype),
    )

    device_map = _get_device_map()
    print(f"Quantizing best student model from {best_student_path} to 4-bit using device_map={device_map}...")

    model = AutoModelForTokenClassification.from_pretrained(
        best_student_path,
        quantization_config=qconfig,
        device_map=device_map,
    )

    os.makedirs(output_path, exist_ok=True)
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    source_metrics = os.path.join(best_student_path, "eval_metrics.json")
    if os.path.isfile(source_metrics):
        shutil.copy2(source_metrics, os.path.join(output_path, "eval_metrics.json"))

    print(f"Quantized model saved to {output_path}")


if __name__ == "__main__":
    quantize_best_student()
