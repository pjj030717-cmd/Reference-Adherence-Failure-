#!/usr/bin/env python3
"""D3-M Phase 0A: six-label inheritance + model hashes + T0 SHA + token ids +
teacher-forced semantics + D1 synthetic 24-pair readout regression."""
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

import d3m_core as C

D0 = REPO_ROOT / "d0_jar_style_sciq_data_qualification_20260802"
D1 = REPO_ROOT / "d1_qwen25_7b_jar_style_sciq_behavior_gate_20260802"
D1R = REPO_ROOT / "d1r_qwen25_7b_jar_style_sciq_template_robustness_20260802"
D2 = REPO_ROOT / "d2_qwen25_7b_predecision_reference_state_localization_20260802"
D2R1 = REPO_ROOT / "d2r1_qwen25_7b_true_prefix_reference_state_20260802"
D3 = REPO_ROOT / "d3_qwen25_7b_reference_binding_selective_patching_20260802"
M = REPO_ROOT / "d3m_qwen25_7b_monolithic_reference_binding_intervention_20260802"
MODEL = C.MODEL


def fail(label: str, why: str):
    print("STOP:", label, "-", why)
    (M / "artifacts").mkdir(parents=True, exist_ok=True)
    (M / "artifacts" / "decision.json").write_text(json.dumps({
        "final_label": label, "reason": why,
        "final_reserve_model_scored": False, "final_reserve_hidden_states_read": False,
        "monolithic_full_forward_only": True, "segmented_execution_used": False,
        "prefix_cache_used": False, "activation_intervention_run": False,
        "prompt_baselines_run": False, "mistral_loaded": False}, indent=2), encoding="utf-8")
    sys.exit(1)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def label(p: Path) -> str:
    return json.loads(p.read_text(encoding="utf-8")).get("final_label")


# ---- six labels ----
checks = {
    "D0": (D0 / "artifacts" / "decision.json", "jar_style_sciq_data_qualification_feasible"),
    "D1": (D1 / "artifacts" / "decision.json", "jar_style_reference_override_behavior_feasible"),
    "D1R": (D1R / "artifacts" / "decision.json", "template_robust_reference_override_feasible"),
    "D2": (D2 / "artifacts" / "decision.json", "prefix_causality_audit_invalid"),
    "D2R1": (D2R1 / "artifacts" / "decision.json", "true_prefix_reference_state_signal_localized"),
    "D3": (D3 / "artifacts" / "decision.json", "segmented_execution_equivalence_invalid"),
}
for k, (p, exp) in checks.items():
    got = label(p)
    if got != exp:
        fail("monolithic_protocol_inheritance_invalid", f"{k} label={got} expected={exp}")
print("six labels OK")

# ---- model hashes vs D1 ----
import re
rev = (Path(C.MODEL) / "REVISION.txt").read_text(encoding="utf-8").strip()
if rev != "a09a35458c702b33eeacc393d103063234e8bc28":
    fail("monolithic_protocol_inheritance_invalid", f"revision={rev}")
d1_audit = (D1 / "model_access_audit.md").read_text(encoding="utf-8")
rec = {}


def re_search_hash(line):
    mm = re.match(r"\| (\S+) \| ([0-9a-f]{64}) \|", line)
    return (mm.group(1), mm.group(2)) if mm else None


for line in d1_audit.splitlines():
    m = re_search_hash(line)
    if m:
        rec[m[0]] = m[1]

for f in ["config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json",
          "merges.txt", "model.safetensors.index.json"]:
    cur = sha256_file(Path(C.MODEL) / f)
    if rec.get(f) != cur:
        fail("monolithic_protocol_inheritance_invalid", f"{f} hash mismatch")
print("model hashes OK")

# ---- T0 template SHA ----
spec = json.loads((D1R / "candidate_template_robustness_spec.json").read_text(encoding="utf-8"))
t0 = spec["templates"]["T0"]
if t0["template"] != "The answer is <answer>." or \
        t0["utf8_sha256"] != "c42e1ea10a6be3343c109664dbd25860bb6dade650d7a86c066e7ddbb0d298bc":
    fail("monolithic_protocol_inheritance_invalid", f"T0={t0}")
print("T0 SHA OK: c42e1ea1…")

# ---- token ids ----
if C.ACCEPT_ID != 362 or C.REJECT_ID != 425:
    fail("monolithic_protocol_inheritance_invalid", f"token ids {C.ACCEPT_ID}/{C.REJECT_ID}")
tok = C.get_tok()
if tok.decode([C.ACCEPT_ID]) != " A" or tok.decode([C.REJECT_ID]) != " B":
    fail("monolithic_protocol_inheritance_invalid", "continuation token decode mismatch")
print("continuation token ids OK")

# ---- synthetic 24-pair readout ----
SYNTHETIC = [tuple(r) for r in json.loads((D1 / "synthetic_pair_manifest.json").read_text(encoding="utf-8"))]
model = C.get_model()
results = []
for sid, q, ref, cand, exp in SYNTHETIC:
    s = C.score_monolithic(q, ref, cand)
    pred = s["predicted_label"]
    results.append({"id": sid, "expected_label": exp, "predicted_label": pred,
                    "l_A": s["l_A"], "l_B": s["l_B"], "d_raw": s["d_raw"],
                    "correct": pred == exp})
    print(f"  {sid} exp={exp} pred={pred} d={s['d_raw']:+.3f}")

n = len(results)
acc = sum(1 for r in results if r["correct"])
ties = sum(1 for r in results if r["predicted_label"] == "TIE")
order_ok = all(r["correct"] for r in results)
print(f"synthetic: correct={acc}/{n} ties={ties} order_accuracy={acc/n:.3f}")
if not order_ok or ties != 0:
    fail("monolithic_readout_semantics_invalid", f"correct={acc}/{n} ties={ties}")

with open(M / "teacher_forcing_reproduction_audit.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
    w.writeheader()
    w.writerows(results)

# write inheritance audit
(M / "inheritance_audit.md").write_text(f"""# inheritance_audit.md (D3-M)

## 六标签继承

| 实验 | 要求标签 | 实测 |
|---|---|---|
| D0 | `jar_style_sciq_data_qualification_feasible` | ✓ |
| D1 | `jar_style_reference_override_behavior_feasible` | ✓ |
| D1-R | `template_robust_reference_override_feasible` | ✓ |
| D2 | `prefix_causality_audit_invalid`（保持原样） | ✓ |
| D2-R1 | `true_prefix_reference_state_signal_localized` | ✓ |
| D3 | `segmented_execution_equivalence_invalid`（保持原样） | ✓ |

## 模型与语义核验

- revision `a09a35458c70…`；config/tokenizer/safetensors index 哈希与 D1 一致。
- T0 模板 SHA256 = `c42e1ea10a6be…`（`The answer is <answer>.`）。
- `" A"` / `" B"` continuation token ids = 362 / 425。
- teacher-forced pos = prompt_len - 1。
- synthetic 24-pair readout：order_accuracy = {acc}/24，ties = {ties}。

## 本轮执行边界

```text
D3 的 segmented_execution_equivalence_invalid 保持原样；
本轮不复用 D2 或 D2-R1 的 hidden-state 数组；
本轮不用 prefix cache、past_key_values、分段续算、截断后续算；
本轮只做完整原始输入的 monolithic forward。
```
""", encoding="utf-8")
print("Phase 0A OK")
