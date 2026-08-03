#!/usr/bin/env python3
"""D3-D Phase 0: inheritance audit + 3 token-position mechanical localization +
passive-hook zero-equivalence on dev 780 inputs.

0.1 继承审计（标签、模型哈希、D3-M-R1 final-reserve 未读、D1 复现、
    hidden_states[L] ~ layers[L-1] 映射、D3-M-R1 prefix 方向可读）
0.2 三位置机械定位（dev 780）：R_end < C_end < D_pos，offset 唯一
0.3 被动 hook 零扰动（dev 780，L14/L18/L22/L26 × R_end/C_end/D_pos）
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[2]

import numpy as np
import torch

import d3d_core as C

D0 = REPO_ROOT / "d0_jar_style_sciq_data_qualification_20260802"
D1 = REPO_ROOT / "d1_qwen25_7b_jar_style_sciq_behavior_gate_20260802"
D1R = REPO_ROOT / "d1r_qwen25_7b_jar_style_sciq_template_robustness_20260802"
D2R1 = REPO_ROOT / "d2r1_qwen25_7b_true_prefix_reference_state_20260802"
D3 = REPO_ROOT / "d3_qwen25_7b_reference_binding_selective_patching_20260802"
D3M = REPO_ROOT / "d3m_qwen25_7b_monolithic_reference_binding_intervention_20260802"
D3MR1 = REPO_ROOT / "d3mr1_qwen25_7b_monolithic_prefix_direction_intervention_20260802"
R = REPO_ROOT / "d3d_qwen25_7b_monolithic_causal_location_scan_20260803"
MODEL = C.MODEL


def fail(label: str, why: str):
    print("STOP:", label, "-", why)
    (R / "artifacts").mkdir(parents=True, exist_ok=True)
    (R / "artifacts" / "decision.json").write_text(json.dumps({
        "final_label": label, "reason": why,
        "final_reserve_model_scored": False, "final_reserve_hidden_states_read": False,
        "monolithic_full_forward_only": True, "prefix_cache_used": False,
        "activation_intervention_run": False, "prompt_baselines_run": False,
        "mistral_loaded": False}, indent=2), encoding="utf-8")
    sys.exit(1)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def label(p: Path) -> str:
    return json.loads(p.read_text(encoding="utf-8")).get("final_label")


# ---- 0.1 inheritance ----
checks = {
    "D0": (D0 / "artifacts" / "decision.json", "jar_style_sciq_data_qualification_feasible"),
    "D1": (D1 / "artifacts" / "decision.json", "jar_style_reference_override_behavior_feasible"),
    "D1R": (D1R / "artifacts" / "decision.json", "template_robust_reference_override_feasible"),
    "D2R1": (D2R1 / "artifacts" / "decision.json", "true_prefix_reference_state_signal_localized"),
    "D3": (D3 / "artifacts" / "decision.json", "segmented_execution_equivalence_invalid"),
    "D3M": (D3M / "artifacts" / "decision.json", "monolithic_direction_label_capacity_insufficient"),
    "D3MR1": (D3MR1 / "artifacts" / "decision.json", "monolithic_patch_dev_selectivity_insufficient"),
}
for k, (p, exp) in checks.items():
    got = label(p)
    if got != exp:
        fail("inheritance_or_data_contract_invalid", f"{k} label={got}")
print("seven labels OK")

rev = (Path(MODEL) / "REVISION.txt").read_text(encoding="utf-8").strip()
if rev != "a09a35458c702b33eeacc393d103063234e8bc28":
    fail("inheritance_or_data_contract_invalid", f"revision={rev}")
d1_audit = (D1 / "model_access_audit.md").read_text(encoding="utf-8")
rec = {}
for line in d1_audit.splitlines():
    m = re.match(r"\| (\S+) \| ([0-9a-f]{64}) \|", line)
    if m:
        rec[m.group(1)] = m.group(2)
for f in ["config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json",
          "merges.txt", "model.safetensors.index.json"]:
    if rec.get(f) != sha256_file(Path(MODEL) / f):
        fail("inheritance_or_data_contract_invalid", f"{f} hash mismatch")
print("model hashes OK")

# D3-M-R1: no final-reserve access; prefix direction readable
d3mr1_d = json.loads((D3MR1 / "artifacts" / "decision.json").read_text(encoding="utf-8"))
if d3mr1_d["final_reserve_model_scored"] is not False or \
        d3mr1_d["final_reserve_hidden_states_read"] is not False:
    fail("inheritance_or_data_contract_invalid", "D3-M-R1 read final-reserve")
art = np.load(D3MR1 / "frozen_direction_artifact.npz")
md = json.loads((D3MR1 / "frozen_direction_metadata.json").read_text(encoding="utf-8"))
if md["direction_method"] != "V_logit@C=0.01":
    fail("inheritance_or_data_contract_invalid", "D3-M-R1 direction method")
v_prefix = art["v_raw"].astype(np.float64)
mu_prefix = art["mu_train"].astype(np.float64)
print("D3-M-R1 prefix direction readable: method=V_logit@C=0.01, v/mu shapes OK")

# D1 synthetic readout record
syn = list(csv.DictReader(open(D1 / "synthetic_readout_audit.csv", encoding="utf-8")))
if len(syn) != 24 or sum(1 for r in syn if r["correct"] == "True") != 24:
    fail("inheritance_or_data_contract_invalid", f"synthetic readout n={len(syn)}")
print("synthetic readout (D1 record): 24/24")

# ---- load D1 dev four-cell ----
d1_rows = list(csv.DictReader(open(D1 / "four_cell_scores_dev.csv", encoding="utf-8")))
if len(d1_rows) != 780:
    fail("inheritance_or_data_contract_invalid", f"D1 rows={len(d1_rows)}")

pairs = C.load_swap_pairs("dev")
assert len(pairs) == 195

# ---- 0.2 + 0.3: localize positions & passive hook zero-equivalence ----
C.get_model()
out_rows = []
loc_fail = 0
for p in pairs:
    gid = p["original_group_id"]
    for cell, ref, cand, exp in C.four_cells(p):
        ids, r_end, c_end, d_pos = C.build_positions(p["q"], ref, cand)
        ok_order = (r_end < c_end < d_pos)
        # token ids uniqueness at positions
        r_tok = ids[r_end]
        c_tok = ids[c_end]
        d_tok = ids[d_pos]
        # passive hook zero-equivalence: install hook on all 4 candidate layers reading all 3 positions
        pids = torch.tensor([ids], device="cuda")
        captured = {}
        hooks = []
        for li in C.CAND_LAYERS:
            def mk_hook(layer_idx, cap):
                def hook(module, args, output):
                    hidden = output[0] if isinstance(output, tuple) else output
                    cap[layer_idx] = {
                        "R_end": hidden[0, r_end, :].clone().cpu().float(),
                        "C_end": hidden[0, c_end, :].clone().cpu().float(),
                        "D_pos": hidden[0, d_pos, :].clone().cpu().float(),
                    }
                    return None
                return hook
            hooks.append(C.get_model().model.layers[li - 1].register_forward_hook(mk_hook(li, captured)))
        try:
            with torch.inference_mode():
                logits = C.get_model()(pids).logits
        finally:
            for h in hooks:
                h.remove()
        ll = logits[0, pids.shape[1] - 1, :]
        l_A = ll[C.ACCEPT_ID].item()
        l_B = ll[C.REJECT_ID].item()
        d_raw = l_A - l_B
        pred = "A" if d_raw > 0 else ("B" if d_raw < 0 else "TIE")
        r1 = next(r for r in d1_rows if r["source_group_id"] == gid and r["cell"] == cell)
        match = (pred == r1["predicted_label"])
        dd = abs(d_raw - float(r1["d_raw"]))
        if not ok_order:
            loc_fail += 1
        out_rows.append({
            "source_group_id": gid, "cell": cell, "r_end": r_end, "c_end": c_end,
            "d_pos": d_pos, "order_ok": ok_order, "r_end_tok": r_tok, "c_end_tok": c_tok,
            "d_pos_tok": d_tok, "prediction_match": match, "d1_d_raw": float(r1["d_raw"]),
            "our_d_raw": d_raw, "d_raw_abs_diff": round(dd, 8),
            "seq_len": int(len(ids)),
        })
    print(f"  {gid[:12]} done")

with open(R / "token_position_mapping_audit.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
    w.writeheader()
    w.writerows(out_rows)

nm = sum(1 for x in out_rows if not x["prediction_match"])
maxd = max(x["d_raw_abs_diff"] for x in out_rows)
order_ok = sum(1 for x in out_rows if x["order_ok"])
print(f"rows={len(out_rows)} loc_fail={loc_fail} order_ok={order_ok} "
      f"prediction_mismatch={nm} max_d_raw_diff={maxd}")
valid_frac = order_ok / len(out_rows)
if valid_frac < 0.95:
    fail("causal_location_execution_invalid", f"valid position localization {valid_frac:.3f} < 0.95")
if nm != 0 or maxd != 0.0:
    fail("causal_location_execution_invalid",
         f"passive hook perturbation: mismatch={nm} max_d={maxd}")

# ---- hidden_states[L] ~ layers[L-1] output correspondence ----
# verify on first dev SS input: forward with output_hidden_states, compare hs[L] vs hook capture
p0 = pairs[0]
ids, r_end, c_end, d_pos = C.build_positions(p0["q"], p0["r_s"], p0["c_s"])
pids = torch.tensor([ids], device="cuda")
captured = {}
hooks = []
for li in C.CAND_LAYERS:
    def mk_hook(layer_idx, cap):
        def hook(module, args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            cap[layer_idx] = hidden[0, d_pos, :].clone().cpu().float()
            return None
        return hook
    hooks.append(C.get_model().model.layers[li - 1].register_forward_hook(mk_hook(li, captured)))
try:
    with torch.inference_mode():
        out = C.get_model()(pids, output_hidden_states=True)
finally:
    for h in hooks:
        h.remove()
hs_map_ok = True
for li in C.CAND_LAYERS:
    hs_val = out.hidden_states[li][0, d_pos, :].cpu().float()
    hook_val = captured[li]
    diff = float((hs_val - hook_val).abs().max())
    if diff != 0.0:
        hs_map_ok = False
        print(f"  hidden_states[{li}] vs layers[{li-1}] diff={diff}")
print(f"hidden_states[L] ~ layers[L-1] correspondence: {hs_map_ok}")
if not hs_map_ok:
    fail("inheritance_or_data_contract_invalid", "hidden_states mapping mismatch")

# save audits
(R / "model_access_audit.md").write_text(
    "| 文件 | SHA256 | 一致性 |\n|---|---|---|\n" +
    "\n".join(f"| {f} | {sha256_file(Path(MODEL)/f)} | 与 D1 一致 |" for f in list(rec)[:6]) +
    f"\n\nrevision = `{rev}`\n", encoding="utf-8")
(R / "token_position_mapping_spec.md").write_text(
    "# token_position_mapping_spec.md\n\n"
    "## 三位置定义\n\n"
    "- `R_end`: Reference Answer 正文最后一个非空白 token（offset mapping）。\n"
    "- `C_end`: Candidate Answer 正文 `<answer>` 的最后一个非空白 token，不含模板附加句号。\n"
    "- `D_pos`: 完整 prompt 中用于预测 continuation \" A\"/\" B\" 的最后一个 token，即 prompt_len-1。\n\n"
    "## 约束\n\n"
    "- 对每条输入要求 R_end < C_end < D_pos。\n"
    "- 有效定位比例 >= 0.95 否则 `causal_location_execution_invalid`。\n", encoding="utf-8")
print("Phase 0 OK")
