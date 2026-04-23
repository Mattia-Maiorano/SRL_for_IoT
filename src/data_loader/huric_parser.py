import os
import subprocess
import xml.etree.ElementTree as ET
from typing import Dict, Iterable, List, Optional

import nltk
from nltk.corpus import wordnet as wn


def clone_huric_repo(raw_data_dir: str) -> str:
    huric_dir = os.path.join(raw_data_dir, "huric")
    if os.path.isdir(huric_dir) and os.listdir(huric_dir):
        return huric_dir

    os.makedirs(raw_data_dir, exist_ok=True)
    git_url = "https://github.com/crux82/huric.git"
    subprocess.run(["git", "clone", "--depth", "1", git_url, huric_dir], check=True)
    return huric_dir


def _get_language_folder(huric_dir: str, preferred_language: str = "en") -> str:
    candidate = os.path.join(huric_dir, preferred_language)
    if os.path.isdir(candidate):
        return candidate

    for name in os.listdir(huric_dir):
        candidate = os.path.join(huric_dir, name)
        if os.path.isdir(candidate):
            return candidate

    raise FileNotFoundError("No HuRIC language folder found in: %s" % huric_dir)


def _get_primary_verb_synset(verb: str) -> Optional[str]:
    try:
        synsets = wn.synsets(verb, pos=wn.VERB)
    except LookupError:
        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)
        synsets = wn.synsets(verb, pos=wn.VERB)

    if not synsets:
        return None
    return synsets[0].name()


def _convert_to_bio(token_ids: List[int], arg_spans: Dict[str, List[int]]) -> List[str]:
    labels = ["O"] * len(token_ids)
    id_to_index = {token_id: idx for idx, token_id in enumerate(token_ids)}

    for arg_type, ids in arg_spans.items():
        sorted_ids = sorted(ids)
        previous_index = None
        for token_id in sorted_ids:
            index = id_to_index[token_id]
            if previous_index is None or index != previous_index + 1:
                labels[index] = f"B-{arg_type}"
            else:
                labels[index] = f"I-{arg_type}"
            previous_index = index

    return labels


def _parse_huric_file(file_path: str) -> Iterable[Dict]:
    tree = ET.parse(file_path)
    root = tree.getroot()

    for command in root.findall(".//command"):
        sentence_text = command.findtext("sentence", default="").strip()
        token_elements = command.findall(".//tokens/token")
        tokens = []
        token_id_to_token = {}

        for token_el in token_elements:
            token_id = int(token_el.attrib["id"])
            surface = token_el.attrib.get("surface", "")
            lemma = token_el.attrib.get("lemma", surface)
            pos = token_el.attrib.get("pos", "")
            token_info = {"id": token_id, "surface": surface, "lemma": lemma, "pos": pos}
            tokens.append(token_info)
            token_id_to_token[token_id] = token_info

        tokens.sort(key=lambda x: x["id"])
        token_ids = [token["id"] for token in tokens]
        token_surfaces = [token["surface"] for token in tokens]

        for frame in command.findall(".//semantics/frames/frame"):
            lexical_unit = frame.find("./lexicalUnit/token")
            if lexical_unit is None:
                continue

            predicate_id = int(lexical_unit.attrib["id"])
            predicate_token = token_id_to_token.get(predicate_id)
            if predicate_token is None:
                continue

            spans: Dict[str, List[int]] = {}
            for element in frame.findall(".//frameElement"):
                arg_type = element.attrib.get("type", "ARG")
                arg_token_ids = [int(token.attrib["id"]) for token in element.findall("token")]
                if arg_token_ids:
                    spans.setdefault(arg_type, []).extend(arg_token_ids)

            labels = _convert_to_bio(token_ids, spans)
            predicate_index = token_ids.index(predicate_id)
            labels[predicate_index] = "B-V"

            yield {
                "sentence": sentence_text,
                "tokens": token_surfaces,
                "labels": labels,
                "predicate": predicate_token["surface"],
                "predicate_lemma": predicate_token["lemma"],
                "predicate_synset": _get_primary_verb_synset(predicate_token["lemma"] or predicate_token["surface"]),
                "predicate_index": predicate_index,
                "frame_name": frame.attrib.get("name", ""),
            }


def parse_huric_directory(raw_data_dir: str, preferred_language: str = "en") -> List[Dict]:
    huric_root = clone_huric_repo(raw_data_dir)
    language_folder = _get_language_folder(huric_root, preferred_language)
    examples = []

    for root_dir, _, filenames in os.walk(language_folder):
        for file_name in sorted(filenames):
            if not file_name.lower().endswith(".hrc"):
                continue
            file_path = os.path.join(root_dir, file_name)
            examples.extend(_parse_huric_file(file_path))

    return examples
