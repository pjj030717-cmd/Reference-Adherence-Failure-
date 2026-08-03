#!/usr/bin/env python3
"""E01-D1 synthetic semantic readout regression (decision gate).

Loads Qwen2.5-7B-Instruct locally (BF16, eval, inference_mode, batch_size=1).
Runs correct teacher-forced continuation scoring on 24 hand-written synthetic pairs,
then greedy diagnostics.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[2]

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = os.environ.get("RAF_MODEL_DIR", "/root/autodl-tmp/models/Qwen2.5-7B-Instruct")
D1 = REPO_ROOT / "d1_qwen25_7b_jar_style_sciq_behavior_gate_20260802"
CONST = json.loads((D1 / "scripts" / "_prompt_constants.json").read_text(encoding="utf-8"))
SYSTEM, USER_TMPL, ACCEPT, REJECT = CONST["system"], CONST["user_template"], CONST["accept"], CONST["reject"]
ACCEPT_ID, REJECT_ID = CONST["accept_id"], CONST["reject_id"]


def fail(label: str, why: str):
    print("STOP:", label, "-", why)
    (D1 / "artifacts").mkdir(parents=True, exist_ok=True)
    (D1 / "artifacts" / "decision.json").write_text(
        json.dumps({"final_label": label, "reason": why,
                    "hidden_states_read": False, "probe_trained": False,
                    "activation_intervention_run": False, "final_reserve_model_scored": False,
                    "mistral_loaded": False, "prompt_variants_run": False}, indent=2),
        encoding="utf-8")
    sys.exit(1)


# ---- 24 synthetic pairs: 12 agree (A) + 12 disagree (B) ----
# fields: id, question, reference, candidate, expected_label
SYNTHETIC = [
    # --- agree (A): reference == candidate in plain factual terms ---
    ("S01", "What is the capital city of France?", "Paris", "Paris", "A"),
    ("S02", "Which planet is known as the Red Planet?", "Mars", "Mars", "A"),
    ("S03", "What gas do plants absorb during photosynthesis?", "Carbon dioxide", "Carbon dioxide", "A"),
    ("S04", "Who wrote the novel Moby-Dick?", "Herman Melville", "Herman Melville", "A"),
    ("S05", "What is the largest ocean on Earth?", "Pacific Ocean", "Pacific Ocean", "A"),
    ("S06", "How many sides does a hexagon have?", "Six", "Six", "A"),
    ("S07", "What instrument measures atmospheric pressure?", "Barometer", "Barometer", "A"),
    ("S08", "Which metal is liquid at room temperature?", "Mercury", "Mercury", "A"),
    ("S09", "What is the freezing point of water in Celsius?", "Zero", "Zero", "A"),
    ("S10", "Who painted the Mona Lisa?", "Leonardo da Vinci", "Leonardo da Vinci", "A"),
    ("S11", "What is the speed of light in a vacuum?", "About 300,000 km per second", "About 300,000 km per second", "A"),
    ("S12", "Which organ pumps blood through the body?", "Heart", "Heart", "A"),
    # --- disagree (B): candidate clearly differs from reference ---
    ("S13", "What is the capital city of France?", "Paris", "Berlin", "B"),
    ("S14", "Which planet is known as the Red Planet?", "Mars", "Venus", "B"),
    ("S15", "What gas do plants absorb during photosynthesis?", "Carbon dioxide", "Oxygen", "B"),
    ("S16", "Who wrote the novel Moby-Dick?", "Herman Melville", "Mark Twain", "B"),
    ("S17", "What is the largest ocean on Earth?", "Pacific Ocean", "Atlantic Ocean", "B"),
    ("S18", "How many sides does a hexagon have?", "Six", "Eight", "B"),
    ("S19", "What instrument measures atmospheric pressure?", "Barometer", "Thermometer", "B"),
    ("S20", "Which metal is liquid at room temperature?", "Mercury", "Copper", "B"),
    ("S21", "What is the freezing point of water in Celsius?", "Zero", "One hundred", "B"),
    ("S22", "Who painted the Mona Lisa?", "Leonardo da Vinci", "Pablo Picasso", "B"),
    ("S23", "What is the speed of light in a vacuum?", "About 300,000 km per second", "About 300,000 km per day", "B"),
    ("S24", "Which organ pumps blood through the body?", "Heart", "Liver", "B"),
]

# unique manifest (with question/reference/candidate; no D0/SciQ text)
(D1 / "synthetic_pair_manifest.json").write_text(json.dumps(SYNTHETIC, indent=2), encoding="utf-8")

# ---- model access audit hashes (computed before loading) ----
import os
def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

model_dir = Path(MODEL)
revision = (model_dir / "REVISION.txt").read_text(encoding="utf-8").strip()
hash_targets = ["config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
                "model.safetensors.index.json"]
model_hashes = {f: sha256_file(model_dir / f) for f in hash_targets}
print("model file hashes computed; revision =", revision)

# ---- load model ----
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
                                             device_map="cuda", low_cpu_mem_usage=True)
model.eval()
print("model loaded:", type(model).__name__, "params:", sum(p.numel() for p in model.parameters()) / 1e9, "B")


def score_prompt(question: str, reference: str, candidate: str):
    """Correct teacher-forced: logits at pos prompt_len-1."""
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": USER_TMPL.format(question=question, reference=reference, candidate=candidate)},
    ]
    encoded = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                      return_tensors="pt")
    prompt_ids = encoded["input_ids"].to("cuda")
    prompt_len = prompt_ids.shape[1]
    with torch.inference_mode():
        logits = model(prompt_ids).logits
    # teacher-forced position: pos = prompt_len - 1
    pos = prompt_len - 1
    logits_last = logits[0, pos, :]
    l_A = logits_last[ACCEPT_ID].item()
    l_B = logits_last[REJECT_ID].item()
    d_raw = l_A - l_B
    p_accept = 1.0 / (1.0 + math.exp(-d_raw)) if d_raw != float("inf") else 1.0
    # greedy first continuation token (true argmax)
    greedy_id = int(logits_last.argmax().item())
    greedy_tok = tok.decode([greedy_id])
    return l_A, l_B, d_raw, p_accept, greedy_id, greedy_tok


def sigmoid(x):
    return 1.0 / (1.0 + (-x))


results = []
for sid, q, ref, cand, exp in SYNTHETIC:
    l_A, l_B, d_raw, p_accept, greedy_id, greedy_tok = score_prompt(q, ref, cand)
    pred = "A" if d_raw > 0 else ("B" if d_raw < 0 else "TIE")
    correct = pred == exp
    # greedy direction: decode argmax token, strip whitespace -> letter A/B
    g = greedy_tok.strip()
    if g == "A":
        greedy_pred = "A"
    elif g == "B":
        greedy_pred = "B"
    else:
        greedy_pred = f"OTHER({g[:12]!r})"
    greedy_agree = (greedy_pred == pred)
    results.append({"id": sid, "question": q, "reference": ref, "candidate": cand,
                    "expected_label": exp, "l_A": l_A, "l_B": l_B, "d_raw": d_raw,
                    "p_accept_raw": p_accept, "predicted_label": pred, "correct": correct,
                    "greedy_id": greedy_id, "greedy_token": greedy_tok,
                    "greedy_pred": greedy_pred, "greedy_agrees": greedy_agree})
    print(f"{sid} exp={exp} pred={pred} d_raw={d_raw:+.3f} p={p_accept:.4f} greedy={greedy_tok!r}->{greedy_pred} ok={correct} ga={greedy_agree}")

# ---- gate evaluation ----
nA = sum(1 for r in results if r["expected_label"] == "A")
nB = sum(1 for r in results if r["expected_label"] == "B")
accA = sum(1 for r in results if r["expected_label"] == "A" and r["correct"]) / nA
accB = sum(1 for r in results if r["expected_label"] == "B" and r["correct"]) / nB
acc = (sum(1 for r in results if r["correct"])) / len(results)
ties = sum(1 for r in results if r["predicted_label"] == "TIE")
medA = sorted([r["d_raw"] for r in results if r["expected_label"] == "A"])[nA // 2]
medB = sorted([r["d_raw"] for r in results if r["expected_label"] == "B"])[nB // 2]
greedy_agree_n = sum(1 for r in results if r["greedy_agrees"])

print(f"\nsynthetic accuracy: {acc:.3f} ({sum(1 for r in results if r['correct'])}/{len(results)})")
print(f"A-class acc: {accA:.3f}  B-class acc: {accB:.3f}  ties: {ties}")
print(f"median d_raw A: {medA:+.4f}  B: {medB:+.4f}")
print(f"greedy agreement: {greedy_agree_n}/{len(results)}")

# write synthetic_readout_audit.csv
with open(D1 / "synthetic_readout_audit.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
    w.writeheader()
    w.writerows(results)

# gate
if acc < 23 / 24:
    fail("judge_readout_invalid", f"synthetic accuracy {acc:.3f} < 23/24")
if accA < 11 / 12 or accB < 11 / 12:
    fail("judge_readout_invalid", f"per-class accuracy A={accA:.3f}, B={accB:.3f}")
if ties > 0:
    fail("judge_readout_invalid", f"{ties} ties")
if medA <= 0 or medB >= 0:
    fail("judge_readout_invalid", f"median d_raw A={medA}, B={medB} wrong sign")
if greedy_agree_n != len(results):
    fail("judge_readout_invalid", f"greedy agreement {greedy_agree_n}/{len(results)} != 24/24")

print("\nSYNTHETIC READOUT GATE PASSED (judge_readout valid)")
print(f"final label so far: pass (awaiting dev behavior gate)")
