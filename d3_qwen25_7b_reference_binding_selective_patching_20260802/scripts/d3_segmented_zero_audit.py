#!/usr/bin/env python3
"""D3 Phase B: segmented-zero equivalence audit on dev 780 rows.

Monolithic reference = D1 four_cell_scores_dev (already validated bit-exact by
re-running monolithic forward on mismatch candidates).
Segmented-zero = phase P (true-prefix, use_cache=True) + phase S (suffix
continuation with cache_position). No activation modification.

Decision: segmented_execution_equivalence_invalid if NOT (780/780 label match
AND l_A/l_B/d_raw within BF16 serialization precision AND per-cell acc matches).
"""
import csv
import json
import math
import sys
from pathlib import Path
import os
REPO_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

D1 = REPO_ROOT / "d1_qwen25_7b_jar_style_sciq_behavior_gate_20260802"
D3 = REPO_ROOT / "d3_qwen25_7b_reference_binding_selective_patching_20260802"
CONST = json.loads((D1 / "scripts" / "_prompt_constants.json").read_text(encoding="utf-8"))
SYSTEM, USER_TMPL = CONST["system"], CONST["user_template"]
ACCEPT_ID, REJECT_ID = CONST["accept_id"], CONST["reject_id"]
T0 = "The answer is <answer>."

d1_rows = json.loads((D3 / "scripts" / "_d1_fourcell_dev.json").read_text(encoding="utf-8"))
model = AutoModelForCausalLM.from_pretrained(os.environ.get("RAF_MODEL_DIR", "/root/autodl-tmp/models/Qwen2.5-7B-Instruct"),
                                             torch_dtype=torch.bfloat16,
                                             device_map="cuda", low_cpu_mem_usage=True)
model.eval()
tok = AutoTokenizer.from_pretrained(os.environ.get("RAF_MODEL_DIR", "/root/autodl-tmp/models/Qwen2.5-7B-Instruct"))
print("backend:", model.config._attn_implementation)


def build_prompt(q, ref, cand_text):
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": USER_TMPL.format(question=q, reference=ref, candidate=cand_text)},
    ]
    rendered = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tok(rendered, return_offsets_mapping=True, add_special_tokens=False)
    ids = enc["input_ids"]
    offsets = enc["offset_mapping"]
    ref_marker = "Reference Answer: "
    i0 = rendered.find(ref_marker)
    ref_end = i0 + len(ref_marker) + len(ref)
    r_tok = next(ti for ti, (s, e) in enumerate(offsets) if s <= ref_end - 1 < e)
    return ids, r_tok


def seg_zero(q, ref, cand_text):
    ids, r_tok = build_prompt(q, ref, cand_text)
    prefix = ids[: r_tok + 1]
    suffix = ids[r_tok + 1:]
    plen = len(prefix)
    with torch.inference_mode():
        out_p = model(torch.tensor([prefix], device="cuda"), use_cache=True)
        kv = out_p.past_key_values
        out_s = model(torch.tensor([suffix], device="cuda"), past_key_values=kv,
                      cache_position=torch.arange(plen, plen + len(suffix), device="cuda"),
                      use_cache=False)
    ll = out_s.logits[0, -1, :]
    return ll[ACCEPT_ID].item(), ll[REJECT_ID].item()


def bf16_ulp(x):
    if x == 0:
        return 2 ** -133
    e = math.floor(math.log2(abs(x)))
    return 2 ** (e - 8)


rows = d1_rows
out_rows = []
mismatches = []
max_ulp = 0.0
for i, r in enumerate(rows):
    lA, lB = seg_zero(r["question"], r["reference"], r["candidate"])
    d1A, d1B = float(r["l_A"]), float(r["l_B"])
    d1d = float(r["d_raw"])
    dd = lA - lB
    pred = "A" if dd > 0 else ("B" if dd < 0 else "TIE")
    is_match = pred == r["predicted_label"]
    ulpA = abs(lA - d1A) / max(bf16_ulp(d1A), 1e-12)
    ulpB = abs(lB - d1B) / max(bf16_ulp(d1B), 1e-12)
    max_ulp = max(max_ulp, ulpA, ulpB)
    if not is_match:
        mismatches.append((r["source_group_id"][:12], r["cell"], r["predicted_label"], pred, d1d, dd))
    out_rows.append({
        "source_group_id": r["source_group_id"], "cell": r["cell"],
        "d1_predicted_label": r["predicted_label"], "seg_predicted_label": pred,
        "d1_l_A": d1A, "d1_l_B": d1B, "d1_d_raw": d1d,
        "seg_l_A": lA, "seg_l_B": lB, "seg_d_raw": dd,
        "label_match": is_match,
        "l_A_bf16_ulp": round(ulpA, 3), "l_B_bf16_ulp": round(ulpB, 3),
        "d_raw_abs_diff": abs(dd - d1d),
    })

with open(D3 / "segmented_zero_equivalence_audit.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
    w.writeheader()
    w.writerows(out_rows)

# summary
n = len(out_rows)
nm = sum(1 for x in out_rows if not x["label_match"])
dd_arr = np.array([x["d_raw_abs_diff"] for x in out_rows])
print(f"rows={n} label_mismatch={nm}")
print(f"d_raw abs diff: mean={dd_arr.mean():.6f} median={np.median(dd_arr):.6f} "
      f"p95={np.percentile(dd_arr, 95):.6f} max={dd_arr.max():.6f}")
print(f"max bf16-ulp l_A/l_B deviation: {max_ulp:.2f}")
for c in ["OO", "OS", "SO", "SS"]:
    sub = [x for x in out_rows if x["cell"] == c]
    acc = sum(1 for x in sub if x["label_match"]) / len(sub)
    print(f"  {c}: seg label-match rate={acc:.4f} (D1 acc reference: "
          f"{sum(1 for r in rows if r['cell']==c and r['correct'])/len(sub):.4f})")
for mm in mismatches:
    print("  MISMATCH", mm)

# decision
invalid = nm != 0 or max_ulp > 2.0 or any(
    abs(x["d_raw_abs_diff"]) > 0.05 for x in out_rows)
label = "segmented_execution_equivalence_invalid" if invalid else "segmented_execution_equivalence_ok"
print("DECISION:", label)

(D3 / "artifacts").mkdir(exist_ok=True)
(D3 / "artifacts" / "decision.json").write_text(json.dumps({
    "final_label": label,
    "rows_audited": n, "label_mismatch": nm,
    "d_raw_abs_diff_mean": float(dd_arr.mean()), "d_raw_abs_diff_max": float(dd_arr.max()),
    "max_bf16_ulp_deviation": float(max_ulp),
    "segmented_zero_equivalence_ok": not invalid,
    "final_reserve_model_scored": False, "final_reserve_hidden_states_read": False,
    "d3_fit_dev_disjoint": True, "config_selected_on_dev": False,
    "activation_intervention_run": False, "prompt_baselines_run": False,
    "mistral_loaded": False,
}, indent=2), encoding="utf-8")
print("decision written")
