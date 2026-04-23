from typing import List


def _token_label(label: str, is_subword: bool) -> str:
    if label == "O":
        return "O"

    if not is_subword:
        return label

    prefix, _, span = label.partition("-")
    if prefix in {"B", "I"} and span:
        return f"I-{span}"

    return label


def align_bio_labels(tokens: List[str], labels: List[str], tokenizer, label_all_tokens: bool = True, special_token_label: str = "O") -> List[str]:
    if len(tokens) != len(labels):
        raise ValueError("Number of tokens and labels must match.")

    encoded = tokenizer(tokens, is_split_into_words=True, add_special_tokens=True)
    word_ids = encoded.word_ids()
    aligned_labels: List[str] = []
    previous_word_id = None

    for word_id in word_ids:
        if word_id is None:
            aligned_labels.append(special_token_label)
        else:
            label = labels[word_id]
            if word_id != previous_word_id:
                aligned_labels.append(label)
            else:
                if label_all_tokens:
                    aligned_labels.append(_token_label(label, is_subword=True))
                else:
                    aligned_labels.append(special_token_label)
        previous_word_id = word_id

    return aligned_labels


def align_bio_labels_pair(
    tokens,
    labels,
    definition_tokens,
    tokenizer,
    label_all_tokens: bool = True,
    special_token_label: str = "O",
) -> List[str]:
    encoded = tokenizer(
        tokens,
        definition_tokens,
        is_split_into_words=True,
        add_special_tokens=True,
    )

    word_ids = encoded.word_ids()
    sequence_ids = encoded.sequence_ids()
    aligned_labels: List[str] = []
    previous_word_id = None

    for word_id, sequence_id in zip(word_ids, sequence_ids):
        if sequence_id != 0:
            aligned_labels.append(special_token_label)
            continue

        if word_id is None:
            aligned_labels.append(special_token_label)
            continue

        if word_id != previous_word_id:
            aligned_labels.append(labels[word_id])
        else:
            if label_all_tokens:
                aligned_labels.append(_token_label(labels[word_id], is_subword=True))
            else:
                aligned_labels.append(special_token_label)

        previous_word_id = word_id

    return aligned_labels
