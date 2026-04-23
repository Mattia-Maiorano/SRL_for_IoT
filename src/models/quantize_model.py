import os
import sys

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer, BitsAndBytesConfig

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


def quantize_student(
    student_path: str = "outputs/distilled",
    output_path: str = "outputs/iot_model",
    quant_type: str = "nf4",
    compute_dtype: str = "float16",
):
    if not os.path.isdir(student_path):
        raise FileNotFoundError(f"Distilled student path not found: {student_path}")

    tokenizer = AutoTokenizer.from_pretrained(student_path, use_fast=True)
    qconfig = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=quant_type,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=getattr(torch, compute_dtype),
    )

    print("Loading student model for 4-bit quantization...")
    model = AutoModelForTokenClassification.from_pretrained(
        student_path,
        quantization_config=qconfig,
        device_map="auto",
    )

    os.makedirs(output_path, exist_ok=True)
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print(f"Quantized model saved to {output_path}")


if __name__ == "__main__":
    quantize_student()
