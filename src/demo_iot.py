import json
import os
import re
import sys
import time
from typing import Dict, List, Optional, Tuple

import nltk
import torch
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from transformers import AutoModelForTokenClassification, AutoTokenizer

try:
    import spacy
    _SPACY_NLP = spacy.load("en_core_web_sm")
except Exception:
    _SPACY_NLP = None

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

MODEL_PATH = os.environ.get("SRL_MODEL_PATH", "outputs/iot_ready_4bit")
FALLBACK_PATH = os.environ.get("SRL_FALLBACK_PATH", "outputs/sweep_trials/best_student")
AUX_VERBS = {
    "be",
    "am",
    "is",
    "are",
    "was",
    "were",
    "been",
    "being",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "can",
    "could",
    "will",
    "would",
    "shall",
    "should",
    "may",
    "might",
    "must",
}
DISCOURSE_MARKERS = {
    "please",
    "kindly",
    "hey",
    "hi",
    "hello",
    "ok",
    "okay",
    "thanks",
    "thank",
    "sure",
}

nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)


def simple_tokenize(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9']+|[.,!?;:-]", text)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model() -> Tuple[AutoTokenizer, AutoModelForTokenClassification, torch.device, str]:
    device = get_device()
    resolved_model_path = os.path.join(ROOT_DIR, MODEL_PATH)
    resolved_fallback_path = os.path.join(ROOT_DIR, FALLBACK_PATH)

    if os.path.isdir(resolved_model_path):
        tokenizer = AutoTokenizer.from_pretrained(resolved_model_path, use_fast=True)
        model = AutoModelForTokenClassification.from_pretrained(resolved_model_path)
        model.to(device)
        model.eval()
        return tokenizer, model, device, resolved_model_path

    if os.path.isdir(resolved_fallback_path):
        print(f"Teacher model not found at {resolved_model_path}, falling back to {resolved_fallback_path}.")
        tokenizer = AutoTokenizer.from_pretrained(resolved_fallback_path, use_fast=True)
        model = AutoModelForTokenClassification.from_pretrained(resolved_fallback_path)
        model.to(device)
        model.eval()
        return tokenizer, model, device, resolved_fallback_path

    raise FileNotFoundError(
        "No model directory found. "
        f"Checked: {resolved_model_path} and {resolved_fallback_path}. "
        "If you extracted only src/, you must also provide a model directory or set SRL_MODEL_PATH. "
        "Example: export SRL_MODEL_PATH=/content/SRL_for_IoT/outputs/baseline"
    )


def find_predicate(text_command: str) -> str:
    lemmatizer = WordNetLemmatizer()
    text = text_command.strip()

    if _SPACY_NLP is not None:
        doc = _SPACY_NLP(text)
        root = next((token for token in doc if token.dep_ == "ROOT"), None)
        if root is not None:
            if root.lower_ not in AUX_VERBS and root.lower_ not in DISCOURSE_MARKERS and root.pos_ == "VERB":
                return root.lemma_.lower()
            if root.pos_ == "AUX":
                for child in root.children:
                    if child.dep_ in {"xcomp", "ccomp", "conj"} and child.pos_ == "VERB":
                        return child.lemma_.lower()
        for token in doc:
            if token.lower_ in DISCOURSE_MARKERS or token.lower_ in AUX_VERBS:
                continue
            if token.pos_ == "VERB":
                return token.lemma_.lower()

    tokens = [t for t in simple_tokenize(text) if t.isalpha()]
    for token in tokens:
        word = token.lower()
        if word in DISCOURSE_MARKERS or word in AUX_VERBS:
            continue
        lemma = lemmatizer.lemmatize(word, wordnet.VERB)
        if wordnet.synsets(lemma, pos=wordnet.VERB):
            return lemma

    for token in tokens:
        if token.isalpha():
            return lemmatizer.lemmatize(token.lower(), wordnet.VERB)

    return "run"


def get_primary_definition(predicate: str) -> str:
    synsets = wordnet.synsets(predicate, pos=wordnet.VERB)
    if synsets:
        return synsets[0].definition()
    return "execute the action described by the predicate"


def collapse_predictions(
    tokenized_input,
    prediction_ids: List[int],
    id_to_label: Dict[int, str],
) -> List[str]:
    word_ids = tokenized_input.word_ids(batch_index=0)
    sequence_ids = tokenized_input.sequence_ids(batch_index=0)
    collapsed_labels: List[str] = []
    previous_word_id = None

    for idx, word_id in enumerate(word_ids):
        if word_id is None or sequence_ids[idx] != 0:
            continue
        if word_id != previous_word_id:
            label_id = prediction_ids[idx]
            label = id_to_label.get(int(label_id), "O")
            collapsed_labels.append(label)
            previous_word_id = word_id

    return collapsed_labels


def extract_arguments(tokens: List[str], labels: List[str]) -> Dict[str, str]:
    arguments: Dict[str, str] = {}
    current_type: Optional[str] = None
    current_tokens: List[str] = []

    def flush() -> None:
        nonlocal current_type, current_tokens
        if current_type and current_tokens:
            joined = " ".join(current_tokens)
            if current_type in arguments:
                arguments[current_type] = f"{arguments[current_type]}; {joined}"
            else:
                arguments[current_type] = joined
        current_type = None
        current_tokens = []

    for token, label in zip(tokens, labels):
        if label == "O":
            flush()
            continue

        prefix, _, entity = label.partition("-")
        if entity == "V":
            continue

        if prefix == "B":
            flush()
            current_type = entity
            current_tokens = [token]
        elif prefix == "I" and current_type == entity:
            current_tokens.append(token)
        else:
            flush()
            if label.startswith("B-"):
                current_type = entity
                current_tokens = [token]
            else:
                current_type = None
                current_tokens = []

    flush()
    return arguments


def parse_command(text_command: str, tokenizer, model, device: torch.device) -> Dict[str, object]:
    raw_command = text_command.strip()
    predicate = find_predicate(raw_command)
    definition = get_primary_definition(predicate)
    formatted_input = f"{predicate} : {definition}"

    raw_tokens = [t for t in simple_tokenize(raw_command) if re.match(r"[A-Za-z0-9']+", t)]
    formatted_tokens = [t for t in simple_tokenize(formatted_input) if re.match(r"[A-Za-z0-9']+", t)]

    encoded = tokenizer(
        raw_tokens,
        formatted_tokens,
        is_split_into_words=True,
        truncation=True,
        padding=True,
        return_tensors="pt",
    )

    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    token_type_ids = encoded.get("token_type_ids")
    if token_type_ids is not None:
        token_type_ids = token_type_ids.to(device)

    start = time.perf_counter()
    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
    latency_ms = (time.perf_counter() - start) * 1000

    logits = outputs.logits[0].cpu().numpy()
    prediction_ids = logits.argmax(axis=-1).tolist()

    id_to_label = {
        int(k): v for k, v in model.config.id2label.items()
    } if hasattr(model.config, "id2label") else {}

    word_ids = encoded.word_ids(batch_index=0)
    sequence_ids = encoded.sequence_ids(batch_index=0)
    raw_token_labels: List[Tuple[str, str]] = []
    previous_word_id = None

    for idx, (word_id, seq_id) in enumerate(zip(word_ids, sequence_ids)):
        if seq_id != 0 or word_id is None:
            continue
        if word_id != previous_word_id:
            label_id = prediction_ids[idx]
            label = id_to_label.get(int(label_id), "O")
            raw_token_labels.append((raw_tokens[word_id], label))
            previous_word_id = word_id

    if len(raw_token_labels) != len(raw_tokens):
        # Ensure no raw words are dropped; fill missing words with O labels if needed.
        aligned_labels = [label for _, label in raw_token_labels]
        aligned_tokens = [token for token, _ in raw_token_labels]
        if len(aligned_tokens) < len(raw_tokens):
            for missing in raw_tokens[len(aligned_tokens):]:
                raw_token_labels.append((missing, "O"))

    print("DEBUG TOKENS -> LABELS:", raw_token_labels)

    word_tokens = [token for token, _ in raw_token_labels]
    collapsed_labels = [label for _, label in raw_token_labels]
    arguments = extract_arguments(word_tokens, collapsed_labels)

    return {
        "raw_command": raw_command,
        "predicate": predicate,
        "arguments": arguments,
        "latency_ms": latency_ms,
    }


def main() -> None:
    tokenizer, model, device, model_path = load_model()
    print(f"Loaded model from {model_path} on {device}.")
    if _SPACY_NLP is None:
        print("Warning: spaCy model unavailable, using fallback predicate extraction.")
    print("Enter a command, or type 'quit' to exit.")

    while True:
        try:
            command = input("command> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting demo.")
            break

        if not command:
            continue
        if command.lower() in {"quit", "exit", "q"}:
            print("Exiting demo.")
            break

        payload = parse_command(command, tokenizer, model, device)
        print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
