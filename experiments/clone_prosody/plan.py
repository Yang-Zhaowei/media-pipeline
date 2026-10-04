"""Portable experiment definition. No model imports, paths or generation guesses."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

CORPUS_PATH = Path(__file__).with_name("corpus.json")
CONDITIONS = ("S3-default", "B3-default", "L1-default", "S3-greedy", "L1-greedy")


def load_corpus() -> dict:
    corpus = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    if corpus["schema_version"] != 1 or [u["id"] for u in corpus["units"]] != [
        "opening", "discussion", "closing"
    ]:
        raise ValueError("unsupported corpus definition")
    if any(type(u["text"]) is not str or not u["text"].strip() for u in corpus["units"]):
        raise ValueError("empty corpus text")
    return corpus


def corpus_hash(corpus: dict) -> str:
    return hashlib.sha256(json.dumps(
        corpus, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()


def joined_text(corpus: dict) -> tuple[str, list[dict]]:
    text = ""
    regions = []
    for unit in corpus["units"]:
        if regions:
            text += corpus["join_separator"]
        start = len(text)
        text += unit["text"]
        regions.append({"unit_id": unit["id"], "start_codepoint": start,
                        "end_codepoint": len(text)})
    return text, regions


def build_matrix(*, conditions=None, batch_supported=True, greedy_supported=True,
                 schedule_seed=8108) -> dict:
    selected = list(CONDITIONS if conditions is None else conditions)
    if not selected or len(set(selected)) != len(selected) or any(c not in CONDITIONS for c in selected):
        raise ValueError("select unique known conditions")
    corpus = load_corpus()
    continuous, regions = joined_text(corpus)
    runs, omitted = [], []
    for condition in CONDITIONS:
        if condition not in selected:
            continue
        if (condition.startswith("B3") and not batch_supported) or (
            condition.endswith("greedy") and not greedy_supported
        ):
            omitted.append({"condition": condition, "reason": "audited capability unavailable"})
            continue
        topology, sampling = condition.split("-")
        greedy = sampling == "greedy"
        for repetition in range(1, (2 if greedy else 3) + 1):
            runs.append({
                "id": f"{topology}-{'G' if greedy else 'D'}-{repetition}",
                "condition": condition, "repetition": repetition,
                "rng_seed": 1728 + repetition,
                "rng_mechanism": "process-global Python/NumPy/PyTorch seeding; not a Qwen argument",
                "controls_mode": "overrides" if greedy else "upstream_defaults",
                "generation_controls": {"do_sample": False, "subtalker_dosample": False} if greedy else {},
                "api_topology": {"S3": "three_separate_calls", "B3": "one_list_call",
                                 "L1": "one_continuous_text_call"}[topology],
                "inputs": [{"unit_id": "continuous", "text": continuous}] if topology == "L1" else [
                    {"unit_id": u["id"], "text": u["text"]} for u in corpus["units"]
                ],
                "logical_regions": regions if topology == "L1" else None,
            })
    order = [run["id"] for run in runs]
    random.Random(schedule_seed).shuffle(order)
    return {"schema_version": 1, "experiment": "qwen-base-clone-prosody-v0",
            "corpus_sha256": corpus_hash(corpus), "runs": runs, "omitted": omitted,
            "execution_order": order, "schedule_seed": schedule_seed,
            "listening_assembly": "raw samples concatenated in corpus order; zero inserted gap; no processing"}
