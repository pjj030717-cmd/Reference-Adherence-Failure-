#!/usr/bin/env python3
"""D2-R1: true-prefix contract audit (dev all 195 + train 30 + dev 30 sampled).

For each SS group, build full T0/T1/T2 prompts, locate R_end, truncate to R_end+1,
verify: (a) T0/T1/T2 prefix ids identical up to R_end, R_end token id & position
identical; (b) prefix_input_ids SHA256 identical; (c) repeated forward on the same
prefix gives identical all-layer h_prefix.
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
import sys
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
T0 = "The answer is <answer>."
T1 = "For this question, the answer is <answer>."
T2 = "The response is <answer>."

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
                                             device_map="cuda", low_cpu_mem_usage=True)
model.eval()

dev_ss = json.loads((D2R1 / "scripts" / "_ss_dev_scores.json").read_text(encoding="utf-8"))
train_ss = json.loads((D2R1 / "scripts" / "_ss_train_scores.json").read_text(encoding="utf-8"))


def full_ids(question, ref, template):
    cand = template.replace("<answer>", ref)
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": USER_TMPL.format(question=question, reference=ref, candidate=cand)}]
    rendered = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tok(rendered, return_offsets_mapping=True, add_special_tokens=False)
    ids = enc["input_ids"]
    offsets = enc["offset_mapping"]
    ref_marker = "Reference Answer: "
    i0 = rendered.find(ref_marker)
    ref_start = i0 + len(ref_marker)
    ref_end = ref_start + len(ref)
    r_tok = next(ti for ti, (s, e) in enumerate(offsets) if s <= ref_end - 1 < e)
    return ids, r_tok


def prefix_sha(prefix_ids):
    return hashlib.sha256(np.asarray(prefix_ids, dtype=np.int32).tobytes()).hexdigest()


def get_h_prefix(prefix_ids):
    prompt_ids = torch.tensor([list(prefix_ids)], device="cuda")
    with torch.inference_mode():
        out = model(prompt_ids, output_hidden_states=True)
    hs = out.hidden_states
    return np.stack([hs[l][0, -1, :].cpu().float().numpy() for l in range(1, 29)])


def audit_group(row, do_repeat=True):
    q, ref = row["question"], row["reference"]
    ids_by_t, r_by_t = {}, {}
    for name, t in (("T0", T0), ("T1", T1), ("T2", T2)):
        ids, r_tok = full_ids(q, ref, t)
        ids_by_t[name] = ids
        r_by_t[name] = r_tok
    # checks
    ok = True
    msgs = []
    r0 = r_by_t["T0"]
    for name in ("T1", "T2"):
        if ids_by_t[name][: r0 + 1] != ids_by_t["T0"][: r0 + 1]:
            ok = False
            msgs.append(f"{name} prefix ids differ at/before R_end")
        if r_by_t[name] != r0:
            ok = False
            msgs.append(f"{name} R_end position differs ({r_by_t[name]} vs {r0})")
    sha0 = prefix_sha(ids_by_t["T0"][: r0 + 1])
    shas = {"T0": sha0}
    for name in ("T1", "T2"):
        shas[name] = prefix_sha(ids_by_t[name][: r0 + 1])
        if shas[name] != sha0:
            ok = False
            msgs.append(f"{name} prefix sha differs")
    # determinism check on T0 prefix
    h1 = get_h_prefix(ids_by_t["T0"][: r0 + 1])
    max_diff = np.nan
    if do_repeat:
        h2 = get_h_prefix(ids_by_t["T0"][: r0 + 1])
        max_diff = float(np.abs(h1 - h2).max())
        if max_diff > 0.0:
            ok = False
            msgs.append(f"repeat forward inconsistent (max diff {max_diff})")
    return {
        "gid": row["source_group_id"], "full_len": len(ids_by_t["T0"]),
        "r_end_pos": r0, "prefix_len": r0 + 1,
        "r_end_token_id": ids_by_t["T0"][r0],
        "prefix_sha": sha0, "T1_sha": shas["T1"], "T2_sha": shas["T2"],
        "ok": ok, "msgs": msgs, "max_abs_hidden_diff_repeat": max_diff,
    }


# dev: all 195
dev_rows_out = []
dev_ok = True
for i, row in enumerate(dev_ss):
    res = audit_group(row, do_repeat=True)
    dev_rows_out.append(res)
    if not res["ok"]:
        dev_ok = False
        print("DEV FAIL:", res["gid"], res["msgs"])
    if (i + 1) % 50 == 0:
        print(f"  dev audit {i+1}/195")
print("dev audit all ok:", dev_ok)

# sampled: train 30 + dev 30 (already all 195 dev audited; keep separate sample list for report)
rng = random.Random(20260802)
samp_train = rng.sample(train_ss, 30)
samp_dev = rng.sample(dev_ss, 30)

samples = []
for row in samp_train:
    samples.append(audit_group(row, do_repeat=True))
for row in samp_dev:
    samples.append(audit_group(row, do_repeat=True))

# write contract audit csv: rows for dev-all (prefix details) + sampled repeat-max-diff
with open(D2R1 / "true_prefix_contract_audit.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["group_id", "split", "full_prompt_len", "r_end_pos", "prefix_len",
                "r_end_token_id", "T0_prefix_sha", "T1_prefix_sha", "T2_prefix_sha",
                "prefix_ids_identical", "repeat_max_abs_hidden_diff"])
    seen = set()
    for res in dev_rows_out:
        w.writerow([res["gid"], "dev", res["full_len"], res["r_end_pos"], res["prefix_len"],
                    res["r_end_token_id"], res["prefix_sha"], res["T1_sha"], res["T2_sha"],
                    1 if res["ok"] else 0, res["max_abs_hidden_diff_repeat"]])
        seen.add(res["gid"])
    for res in samples:
        split = "dev" if res["gid"] in {r["source_group_id"] for r in dev_ss} else "train"
        w.writerow([res["gid"], split, res["full_len"], res["r_end_pos"], res["prefix_len"],
                    res["r_end_token_id"], res["prefix_sha"], res["T1_sha"], res["T2_sha"],
                    1 if res["ok"] else 0, res["max_abs_hidden_diff_repeat"]])
print("wrote true_prefix_contract_audit.csv")

# global check: T1/T2 prefix identical in all 195 dev groups
all_ok = dev_ok
for res in samples:
    if not res["ok"]:
        all_ok = False

if not all_ok:
    print("STOP: true_prefix_contract_invalid")
    (D2R1 / "artifacts" / "decision.json").write_text(json.dumps(
        {"final_label": "true_prefix_contract_invalid",
         "d2_hidden_arrays_reused": False, "final_reserve_model_scored": False,
         "final_reserve_hidden_states_read": False, "probe_trained": True,
         "activation_intervention_run": False, "prompt_baselines_run": False,
         "mistral_loaded": False}, indent=2), encoding="utf-8")
    sys.exit(1)

print("TRUE PREFIX CONTRACT PASSED")
