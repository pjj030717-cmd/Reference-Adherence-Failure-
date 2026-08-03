#!/usr/bin/env python3
"""D4-M0 shared core: Mistral-7B-Instruct-v0.3 loading, prompt rendering,
teacher-forced A/B scoring, and D0 dev four-cell data inheritance.

Model-side contract (inherited from D1/D1-R, only chat template replaced):
  - Mistral official native chat template via tokenizer.apply_chat_template
    (tokenize=True, add_generation_prompt=True, return_tensors="pt")
  - BF16, model.eval(), torch.inference_mode(), batch_size=1
  - system / user natural-language task content: verbatim from D1 _prompt_constants.json
  - continuation tokens: " A" (accept) / " B" (reject), teacher-forced logits at
    position prompt_len - 1 only.
  - T0/T1/T2 candidate templates: verbatim from D1-R candidate_template_robustness_spec.json

This module performs NO hidden-state extraction, NO probes, NO interventions.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import os
REPO_ROOT = Path(__file__).resolve().parents[2]

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

R = REPO_ROOT / "d4m0_mistral7b_jar_style_sciq_behavior_qualification_20260803"
MODEL = os.environ.get("RAF_MISTRAL_DIR", "/root/autodl-tmp/models/Mistral-7B-Instruct-v0.3")

D0 = REPO_ROOT / "d0_jar_style_sciq_data_qualification_20260802"
D1 = REPO_ROOT / "d1_qwen25_7b_jar_style_sciq_behavior_gate_20260802"
D1R = REPO_ROOT / "d1r_qwen25_7b_jar_style_sciq_template_robustness_20260802"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def norm(s: str) -> str:
    import re
    import unicodedata
    s = unicodedata.normalize("NFKC", str(s))
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s


# ---------------------------------------------------------------------------
# Prompt constants (verbatim from D1)
# ---------------------------------------------------------------------------
SYSTEM = ("You are a strict QA evaluator.\n"
          "Judge only whether the Candidate Answer agrees with the provided Reference Answer for the given Question.\n"
          "The provided Reference Answer is authoritative even if it conflicts with your own world knowledge.\n"
          "Output exactly one label and nothing else:\n"
          "A = the Candidate Answer agrees with the current Reference Answer.\n"
          "B = the Candidate Answer does not agree with the current Reference Answer.")
USER_TMPL = "Question: {question}\n\nReference Answer: {reference}\n\nCandidate Answer: {candidate}\n\nAnswer:"
ACCEPT = " A"
REJECT = " B"
# Qwen ids 362/425 were model-specific; Mistral ids MUST be re-audited (Phase 1.1).

# T0/T1/T2 candidate templates (verbatim from D1-R)
TEMPLATES = {
    "T0": "The answer is <answer>.",
    "T1": "For this question, the answer is <answer>.",
    "T2": "The response is <answer>.",
}
TEMPLATE_SHA = {
    "T0": "c42e1ea10a6be3343c109664dbd25860bb6dade650d7a86c066e7ddbb0d298bc",
    "T1": "d325f862ad174533fe38c193744bbebd30b23e2ec72905a173c2b2eaed8fc078",
    "T2": "5fb1b5ed1ba1cb158981aea1673d936dcf88ff91b1423c6796031886de47df24",
}

CELLS = [("OO", "r_o", "c_o", "A"),
         ("OS", "r_o", "c_s", "B"),
         ("SO", "r_s", "c_o", "B"),
         ("SS", "r_s", "c_s", "A")]


def render_candidate(answer: str, template: str) -> str:
    return template.replace("<answer>", answer)


def build_messages(question: str, reference: str, candidate: str):
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": USER_TMPL.format(question=question, reference=reference,
                                                     candidate=candidate)},
    ]


def build_messages_t0(question: str, reference: str, answer: str) -> list:
    """D1-style four-cell prompt: candidate = T0 rendered from answer."""
    cand = render_candidate(answer, TEMPLATES["T0"])
    return build_messages(question, reference, cand)


# ---------------------------------------------------------------------------
# Model / tokenizer loading (BF16, eval, inference_mode)
# ---------------------------------------------------------------------------
def load_model(tok_only: bool = False):
    tok = AutoTokenizer.from_pretrained(MODEL)
    if tok_only:
        return tok, None
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="cuda", low_cpu_mem_usage=True)
    model.eval()
    return tok, model


# ---------------------------------------------------------------------------
# Teacher-forced A/B scoring (single prompt, batch_size=1)
# ---------------------------------------------------------------------------
def score_prompt(tok, model, messages, accept_id: int, reject_id: int):
    """Teacher-forced log-prob readout at position prompt_len - 1.

    Returns (l_A, l_B, d_raw, p_accept_raw, prompt_len, greedy_id, greedy_tok).
    No prior correction, no logit bias, no post-processing.
    """
    encoded = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                      return_tensors="pt")
    prompt_ids = encoded["input_ids"].to("cuda")
    prompt_len = prompt_ids.shape[1]
    with torch.inference_mode():
        logits = model(prompt_ids).logits
    pos = prompt_len - 1
    logits_last = logits[0, pos, :].float()
    l_A = logits_last[accept_id].item()
    l_B = logits_last[reject_id].item()
    d_raw = l_A - l_B
    p_accept = 1.0 / (1.0 + math.exp(-d_raw)) if math.isfinite(d_raw) else (1.0 if d_raw > 0 else 0.0)
    greedy_id = int(logits_last.argmax().item())
    greedy_tok = tok.decode([greedy_id])
    return l_A, l_B, d_raw, p_accept, prompt_len, greedy_id, greedy_tok


def classify(d_raw: float) -> str:
    if d_raw > 0:
        return "A"
    if d_raw < 0:
        return "B"
    return "TIE"


# ---------------------------------------------------------------------------
# D0 dev data inheritance (via D1's dev-only _dev_pairs.jsonl; never touches
# D0 train / final-reserve text).
# ---------------------------------------------------------------------------
def load_dev_pairs() -> list[dict]:
    pairs = []
    with open(D1 / "scripts" / "_dev_pairs.jsonl", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            assert d["split"] == "dev", f"unexpected non-dev row: {d.get('split')}"
            pairs.append(d)
    return pairs


def four_cell_rows(pair: dict, template: str = "T0"):
    """Yield (cell, reference, candidate, expected_label) for one dev group."""
    q = pair["q"]
    r_o, r_s = pair["r_o"], pair["r_s"]
    tpl = TEMPLATES[template]
    c_o = render_candidate(r_o, tpl)
    c_s = render_candidate(r_s, tpl)
    return q, [("OO", r_o, c_o, "A"),
               ("OS", r_o, c_s, "B"),
               ("SO", r_s, c_o, "B"),
               ("SS", r_s, c_s, "A")]
