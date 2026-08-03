#!/usr/bin/env python3
"""E01-D2: collect R_end/C_end/D_pos hidden states (all layers) for train + dev.

For each group x 4 cells (T0): run forward with output_hidden_states=True,
recompute l_A/l_B/d_raw/pred, locate R_end/C_end/D_pos via offset mapping,
save hidden states (float16) per position per layer.

Also performs the dev behavior-reproduction audit against D1 (780 rows).
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path
import os
REPO_ROOT = Path(__file__).resolve().parents[2]

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = os.environ.get("RAF_MODEL_DIR", "/root/autodl-tmp/models/Qwen2.5-7B-Instruct")
D1 = REPO_ROOT / "d1_qwen25_7b_jar_style_sciq_behavior_gate_20260802"
D2 = REPO_ROOT / "d2_qwen25_7b_predecision_reference_state_localization_20260802"
CONST = json.loads((D1 / "scripts" / "_prompt_constants.json").read_text(encoding="utf-8"))
SYSTEM, USER_TMPL, ACCEPT, REJECT = CONST["system"], CONST["user_template"], CONST["accept"], CONST["reject"]
ACCEPT_ID, REJECT_ID = CONST["accept_id"], CONST["reject_id"]
T0 = "The answer is <answer>."
HID_DIR = D2 / "hidden_states"


def fail(label: str, why: str):
    print("STOP:", label, "-", why)
    (D2 / "artifacts").mkdir(parents=True, exist_ok=True)
    (D2 / "artifacts" / "decision.json").write_text(
        json.dumps({"final_label": label, "reason": why,
                    "final_reserve_model_scored": False, "final_reserve_hidden_states_read": False,
                    "probe_trained": True, "activation_intervention_run": False,
                    "prompt_baselines_run": False, "mistral_loaded": False}, indent=2), encoding="utf-8")
    sys.exit(1)


tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
                                             device_map="cuda", low_cpu_mem_usage=True)
model.eval()
print("model loaded")


def locate_positions(rendered: str, ref_text: str, cand_text: str, offsets):
    """Return (r_tok, c_tok, d_pos) token indices."""
    ref_marker = "Reference Answer: "
    cand_marker = "Candidate Answer: "
    i0 = rendered.find(ref_marker)
    i1 = rendered.find(cand_marker)
    if i0 < 0 or i1 < 0:
        return None
    ref_start = i0 + len(ref_marker)
    ref_end = ref_start + len(ref_text)
    cand_start = i1 + len(cand_marker)
    cand_end = cand_start + len(cand_text)

    def token_for_char(char_pos):
        for ti, (s, e) in enumerate(offsets):
            if s <= char_pos < e:
                return ti
        return None

    r_tok = token_for_char(ref_end - 1)
    c_tok = token_for_char(cand_end - 1)
    d_pos = len(offsets) - 1
    if r_tok is None or c_tok is None:
        return None
    return r_tok, c_tok, d_pos, (ref_start, ref_end, cand_start, cand_end)


def process(question: str, ref_text: str, cand_text: str):
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": USER_TMPL.format(question=question, reference=ref_text, candidate=cand_text)},
    ]
    rendered = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tok(rendered, return_offsets_mapping=True, add_special_tokens=False)
    ids = enc["input_ids"]
    offsets = enc["offset_mapping"]

    # verify consistency with apply_chat_template tokenize=True
    ids_ct = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    if hasattr(ids_ct, "input_ids"):
        ids_ct = ids_ct["input_ids"]
    if isinstance(ids_ct, list) and len(ids_ct) >= 1 and isinstance(ids_ct[0], list):
        ids_ct = ids_ct[0]
    ids_ct = list(ids_ct)
    if ids_ct != ids:
        return None  # mismatch; caller handles

    loc = locate_positions(rendered, ref_text, cand_text, offsets)
    if loc is None:
        return None
    r_tok, c_tok, d_pos, spans = loc

    prompt_ids = torch.tensor([ids], device="cuda")
    with torch.inference_mode():
        out = model(prompt_ids, output_hidden_states=True)
        logits = out.logits
        hs = out.hidden_states  # tuple length n_layers+1 (index0=embedding)
    pos = prompt_ids.shape[1] - 1
    ll = logits[0, pos, :]
    l_A = ll[ACCEPT_ID].item()
    l_B = ll[REJECT_ID].item()
    d_raw = l_A - l_B

    # collect hidden states at 3 positions for all layers
    n_layers = len(hs) - 1
    h_r = np.stack([hs[l][0, r_tok, :].cpu().float().numpy() for l in range(1, n_layers + 1)])  # (n_layers, d)
    h_c = np.stack([hs[l][0, c_tok, :].cpu().float().numpy() for l in range(1, n_layers + 1)])
    h_d = np.stack([hs[l][0, d_pos, :].cpu().float().numpy() for l in range(1, n_layers + 1)])
    return {"ids": ids, "r_tok": r_tok, "c_tok": c_tok, "d_pos": d_pos,
            "h_r": h_r, "h_c": h_c, "h_d": h_d,
            "l_A": l_A, "l_B": l_B, "d_raw": d_raw, "spans": spans}


def run_split(pairs, split, expected_rows=None, d1_rows=None):
    HID_DIR.mkdir(parents=True, exist_ok=True)
    rows_out = []
    t0 = time.time()
    for gi, p in enumerate(pairs):
        gid = p["original_group_id"]
        specs = [
            ("OO", p["r_o"], T0.replace("<answer>", p["r_o"])),
            ("OS", p["r_o"], T0.replace("<answer>", p["r_s"])),
            ("SO", p["r_s"], T0.replace("<answer>", p["r_o"])),
            ("SS", p["r_s"], T0.replace("<answer>", p["r_s"])),
        ]
        cell_data = {}
        for cell, ref, cand in specs:
            res = process(p["q"], ref, cand)
            if res is None:
                fail("token_span_mapping_invalid", f"mapping failed for {gid} {cell}")
            exp = "A" if cell in ("OO", "SS") else "B"
            pred = "A" if res["d_raw"] > 0 else ("B" if res["d_raw"] < 0 else "TIE")
            rows_out.append({"source_group_id": gid, "cell": cell, "question": p["q"],
                             "reference": ref, "candidate": cand, "expected_label": exp,
                             "l_A": res["l_A"], "l_B": res["l_B"], "d_raw": res["d_raw"],
                             "p_accept_raw": 1.0 / (1.0 + np.exp(-res["d_raw"])),
                             "predicted_label": pred, "correct": pred == exp})
            cell_data[cell] = {"h_r": res["h_r"].astype(np.float16), "h_c": res["h_c"].astype(np.float16),
                               "h_d": res["h_d"].astype(np.float16), "r_tok": res["r_tok"],
                               "c_tok": res["c_tok"], "d_pos": res["d_pos"]}
        np.savez_compressed(HID_DIR / f"{split}_{gid}.npz",
                            OO_h_r=cell_data["OO"]["h_r"], OO_h_c=cell_data["OO"]["h_c"], OO_h_d=cell_data["OO"]["h_d"],
                            OS_h_r=cell_data["OS"]["h_r"], OS_h_c=cell_data["OS"]["h_c"], OS_h_d=cell_data["OS"]["h_d"],
                            SO_h_r=cell_data["SO"]["h_r"], SO_h_c=cell_data["SO"]["h_c"], SO_h_d=cell_data["SO"]["h_d"],
                            SS_h_r=cell_data["SS"]["h_r"], SS_h_c=cell_data["SS"]["h_c"], SS_h_d=cell_data["SS"]["h_d"])
        if (gi + 1) % 50 == 0:
            print(f"  {split} {gi+1}/{len(pairs)} ({time.time()-t0:.0f}s)")
    print(f"{split} done: {len(rows_out)} rows in {time.time()-t0:.0f}s")
    return rows_out


# ---- dev: reproduction audit + save ----
dev_pairs = []
with open(D1 / "scripts" / "_dev_pairs.jsonl", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            dev_pairs.append(json.loads(line))
print("dev pairs:", len(dev_pairs))

d1_rows = list(csv.DictReader(open(D1 / "four_cell_scores_dev.csv", encoding="utf-8")))
d1_map = {(r["source_group_id"], r["cell"]): r for r in d1_rows}

print("=== running dev ===")
dev_rows = run_split(dev_pairs, "dev")

# reproduction audit
mismatches = []
for r in dev_rows:
    d = d1_map.get((r["source_group_id"], r["cell"]))
    if d is None:
        mismatches.append((r["source_group_id"], r["cell"], "missing"))
        continue
    if r["predicted_label"] != d["predicted_label"]:
        mismatches.append((r["source_group_id"], r["cell"], f"pred {r['predicted_label']} vs {d['predicted_label']}"))
    for fld in ("l_A", "l_B", "d_raw"):
        if abs(float(r[fld]) - float(d[fld])) > 1e-3:
            mismatches.append((r["source_group_id"], r["cell"], f"{fld} diff"))

if mismatches:
    print("dev mismatches:", len(mismatches))
    for m in mismatches[:10]:
        print("  ", m)
    fail("behavior_reproduction_invalid", f"{len(mismatches)} dev mismatches vs D1")

# aggregate check
def agg(rows):
    out = {}
    for c in ["OO", "OS", "SO", "SS"]:
        sub = [r for r in rows if r["cell"] == c]
        out[c] = sum(1 for r in sub if r["correct"]) / len(sub)
    return out

agg_dev = agg(dev_rows)
print("dev aggregate:", agg_dev)
exp_agg = {"OO": 1.0, "OS": 1.0, "SO": 0.9282051282051282, "SS": 0.24102564102564103}
for c in ["OO", "OS", "SO", "SS"]:
    if abs(agg_dev[c] - exp_agg[c]) > 1e-6:
        fail("behavior_reproduction_invalid", f"dev aggregate {c}={agg_dev[c]}")

with open(D2 / "scripts" / "_dev_rows.json", "w", encoding="utf-8") as f:
    json.dump(dev_rows, f)
print("dev reproduction audit PASSED")

# ---- train ----
train_pairs = []
with open(D2 / "scripts" / "_train_pairs.jsonl", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            train_pairs.append(json.loads(line))
print("train pairs:", len(train_pairs))
print("=== running train ===")
train_rows = run_split(train_pairs, "train")
with open(D2 / "scripts" / "_train_rows.json", "w", encoding="utf-8") as f:
    json.dump(train_rows, f)
print("ALL DONE")
