import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import torch
from datasets import Dataset as HFDataset
from torch.utils.data import Dataset
from nltk.corpus import wordnet as wn

from src.data_loader.alignment_utils import align_bio_labels
from src.data_loader.huric_parser import clone_huric_repo, parse_huric_directory


def prepare_huric_dataset(raw_data_dir: str = "data/raw", processed_data_dir: str = "data/processed", preferred_language: str = "en") -> HFDataset:
    os.makedirs(processed_data_dir, exist_ok=True)
    examples = parse_huric_directory(raw_data_dir, preferred_language=preferred_language)
    if not examples:
        raise RuntimeError("No HuRIC examples were parsed from %s" % raw_data_dir)

    hf_dataset = HFDataset.from_list(examples)
    dataset_dir = os.path.join(processed_data_dir, "huric")
    os.makedirs(dataset_dir, exist_ok=True)
    hf_dataset.save_to_disk(dataset_dir)
    hf_dataset.to_parquet(os.path.join(dataset_dir, "huric.parquet"))
    return hf_dataset


def _ensure_wordnet():
    try:
        wn.synsets("run")
    except LookupError:
        import nltk

        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)


def _predicate_definition(synset_name: Optional[str]) -> str:
    if not synset_name:
        return ""

    _ensure_wordnet()
    try:
        return wn.synset(synset_name).definition()
    except Exception:
        return ""


def prepare_huric_augmented_dataset(
    raw_data_dir: str = "data/raw",
    processed_data_dir: str = "data/processed",
    preferred_language: str = "en",
) -> HFDataset:
    os.makedirs(processed_data_dir, exist_ok=True)
    examples = parse_huric_directory(raw_data_dir, preferred_language=preferred_language)
    if not examples:
        raise RuntimeError("No HuRIC examples were parsed from %s" % raw_data_dir)

    augmented_examples = []
    for example in examples:
        predicate_definition = _predicate_definition(example.get("predicate_synset"))
        definition_tokens = predicate_definition.split() if predicate_definition else []
        text_b = (
            f"{example['predicate']} : {predicate_definition}"
            if predicate_definition
            else example["predicate"]
        )

        augmented_examples.append(
            {
                **example,
                "predicate_definition": predicate_definition,
                "definition_tokens": definition_tokens,
                "text_b": text_b,
            }
        )

    hf_dataset = HFDataset.from_list(augmented_examples)
    dataset_dir = os.path.join(processed_data_dir, "huric_augmented")
    os.makedirs(dataset_dir, exist_ok=True)
    hf_dataset.save_to_disk(dataset_dir)
    hf_dataset.to_parquet(os.path.join(dataset_dir, "huric_augmented.parquet"))
    return hf_dataset


@dataclass
class HuricDataset(Dataset):
    examples: List[Dict]
    tokenizer: Optional[object] = None
    label_all_tokens: bool = True
    label_to_id: Optional[Dict[str, int]] = None
    max_length: Optional[int] = None

    def __post_init__(self):
        if self.label_to_id is None:
            unique_labels = sorted({label for example in self.examples for label in example["labels"]})
            self.label_to_id = {label: idx for idx, label in enumerate(unique_labels)}

        if self.tokenizer is None:
            raise ValueError("A Hugging Face tokenizer must be provided for the PyTorch dataset.")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.examples[idx]
        tokens = example["tokens"]
        labels = example["labels"]

        tokenized = self.tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            padding="max_length" if self.max_length else False,
            max_length=self.max_length,
            return_tensors="pt",
        )

        aligned_labels = align_bio_labels(tokens, labels, self.tokenizer, label_all_tokens=self.label_all_tokens)
        aligned_labels = aligned_labels[: tokenized.input_ids.size(1)]
        if self.max_length:
            aligned_labels += ["O"] * (self.max_length - len(aligned_labels))

        tokenized["labels"] = torch.tensor([self.label_to_id[label] for label in aligned_labels], dtype=torch.long)
        return {key: value.squeeze(0) for key, value in tokenized.items()}


def load_huric_dataset(dataset_path: str) -> HFDataset:
    return HFDataset.load_from_disk(dataset_path)


if __name__ == "__main__":
    print("Preparing HuRIC dataset...")
    dataset = prepare_huric_dataset()
    print(f"Saved HuRIC dataset with {len(dataset)} examples to data/processed/huric")
