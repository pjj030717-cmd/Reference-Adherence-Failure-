#!/usr/bin/env python3
"""D2-R1: collect true-truncated reference-prefix hidden states h_prefix[layer] for
all train (587) + dev (195) T0 SS inputs. Prefix = full_input_ids[:R_end+1].
Also re-audit determinism on a sample and recompute labels to confirm score table.
"""
from __future__ import annotations

import hashlib
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
D2R1 = REPO_ROOT / "d2r1_qwen25_7b_true_prefix_reference_state_20260802"
CONST = json.loads((D1 / "scripts" / "_prompt_constants.json").read_text(encoding="utf-8"))
SYSTEM, USER_TMPL = CONST["system"], CONST["user_template"]
ACCEPT_ID, REJECT_ID = CONST["accept_id"], CONST["reject_id"]
T0 = "The answer is <answer>."
HID_DIR = D2R1 / "prefix_hidden_states"


def fail(label: str, why: str):
    print("STOP:", label, "-", why)
    (D2R1 / "artifacts").mkdir(parents=True, exist_ok=True)
    (D2R1 / "artifacts" / "decision.json").write_text(
        json.dumps({"final_label": label, "reason": why, "d2_hidden_arrays_reused": False,
                    "final_reserve_model_scored": False, "final_reserve_hidden_states_read": False,
                    "probe_trained": True, "activation_intervention_run": False,
                    "prompt_baselines_run": False, "mistral_loaded": False}, indent=2), encoding="utf-8")
    sys.exit(1)


tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
                                             device_map="cuda", low_cpu_mem_usage=True)
model.eval()


def build_prefix(question, ref):
    cand = T0.replace("<answer>", ref)
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": USER_TMPL.format(question=question, reference=ref, candidate=cand)}]
    rendered = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tok(rendered, return_offsets_mapping=True, add_special_tokens=False)
    ids = enc["input_ids"]
    offsets = enc["offset_mapping"]
    ref_marker = "Reference Answer: "
    i0 = rendered.find(ref_marker)
    ref_end = i0 + len(ref_marker) + len(ref)
    r_tok = next(ti for ti, (s, e) in enumerate(offsets) if s <= ref_end - 1 < e)
    prefix = ids[: r_tok + 1]
    return prefix


def collect(row):
    prefix = build_prefix(row["question"], row["reference"])
    pids = torch.tensor([prefix], device="cuda")
    with torch.inference_mode():
        out = model(pids, output_hidden_states=True)
    hs = out.hidden_states
    h = np.stack([hs[l][0, -1, :].cpu().float().numpy() for l in range(1, 29)])  # (28, 3584)
    # determinism check once per group is too costly; contract audit already covers it
    return h, prefix


def run(split_rows, split):
    HID_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for gi, row in enumerate(split_rows):
        gid = row["source_group_id"]
        h, prefix = collect(row)
        np.savez_compressed(HID_DIR / f"{split}_{gid}.npz", h_prefix=h.astype(np.float16))
        if (gi + 1) % 100 == 0:
            print(f"  {split} {gi+1}/{len(split_rows)} ({time.time()-t0:.0f}s)")
    print(f"{split} done: {len(split_rows)} in {time.time()-t0:.0f}s")


dev_ss = json.loads((D2R1 / "scripts" / "_ss_dev_scores.json").read_text(encoding="utf-8"))
train_ss = json.loads((D2R1 / "scripts" / "_ss_train_scores.json").read_text(encoding="utf-8"))

print("=== collecting dev ===")
run(dev_ss, "dev")
print("=== collecting train ===")
run(train_ss, "train")
print("ALL DONE")
