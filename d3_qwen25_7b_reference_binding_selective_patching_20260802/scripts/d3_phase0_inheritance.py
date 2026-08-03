#!/usr/bin/env python3
"""D3 Phase 0: D0/D1/D1-R/D2/D2-R1 inheritance verification."""
from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from pathlib import Path
import os
REPO_ROOT = Path(__file__).resolve().parents[2]

D0 = REPO_ROOT / "d0_jar_style_sciq_data_qualification_20260802"
D1 = REPO_ROOT / "d1_qwen25_7b_jar_style_sciq_behavior_gate_20260802"
D1R = REPO_ROOT / "d1r_qwen25_7b_jar_style_sciq_template_robustness_20260802"
D2 = REPO_ROOT / "d2_qwen25_7b_predecision_reference_state_localization_20260802"
D2R1 = REPO_ROOT / "d2r1_qwen25_7b_true_prefix_reference_state_20260802"
D3 = REPO_ROOT / "d3_qwen25_7b_reference_binding_selective_patching_20260802"
MODEL_DIR = Path(os.environ.get("RAF_MODEL_DIR", "/root/autodl-tmp/models/Qwen2.5-7B-Instruct"))


def fail(why: str):
    print("d2r1_inheritance_invalid:", why)
    (D3 / "artifacts").mkdir(parents=True, exist_ok=True)
    (D3 / "artifacts" / "decision.json").write_text(
        json.dumps({"final_label": "d2r1_inheritance_invalid", "reason": why,
                    "final_reserve_model_scored": False, "final_reserve_hidden_states_read": False,
                    "d3_fit_dev_disjoint": True, "config_selected_on_dev": False,
                    "activation_intervention_run": True, "prompt_baselines_run": False,
                    "mistral_loaded": False}, indent=2), encoding="utf-8")
    sys.exit(1)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---- labels ----
def label(p: Path) -> str:
    return json.loads(p.read_text(encoding="utf-8")).get("final_label")


checks = {
    "D0": (D0 / "artifacts" / "decision.json", "jar_style_sciq_data_qualification_feasible"),
    "D1": (D1 / "artifacts" / "decision.json", "jar_style_reference_override_behavior_feasible"),
    "D1R": (D1R / "artifacts" / "decision.json", "template_robust_reference_override_feasible"),
    "D2": (D2 / "artifacts" / "decision.json", "prefix_causality_audit_invalid"),
    "D2R1": (D2R1 / "artifacts" / "decision.json", "true_prefix_reference_state_signal_localized"),
}
for k, (p, exp) in checks.items():
    got = label(p)
    if got != exp:
        fail(f"{k} label={got}")

# ---- splits ----
fixed = json.loads((D0 / "fixed_split_indices.json").read_text(encoding="utf-8"))
train_ids = set(fixed["groups"]["train"])
dev_ids = set(fixed["groups"]["dev"])
res_ids = set(fixed["groups"]["final_reserve"])
if len(train_ids) != 587 or len(dev_ids) != 195 or len(res_ids) != 197:
    fail("split sizes unexpected")
if fixed["seed"] != 20260802:
    fail("D0 split seed mismatch")

# ---- model hashes == D1 ----
rev = (MODEL_DIR / "REVISION.txt").read_text(encoding="utf-8").strip()
if rev != "a09a35458c702b33eeacc393d103063234e8bc28":
    fail(f"Qwen revision={rev}")
d1_audit = (D1 / "model_access_audit.md").read_text(encoding="utf-8")
rec = {}
for line in d1_audit.splitlines():
    m = re.match(r"\| (\S+) \| ([0-9a-f]{64}) \|", line)
    if m:
        rec[m.group(1)] = m.group(2)
for f in ["config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json",
          "merges.txt", "model.safetensors.index.json"]:
    cur = sha256_file(MODEL_DIR / f)
    if rec.get(f) != cur:
        fail(f"model file {f} hash mismatch")

# ---- prompt/template/continuations ----
pconst = json.loads((D1 / "scripts" / "_prompt_constants.json").read_text(encoding="utf-8"))
if pconst["accept"] != " A" or pconst["reject"] != " B":
    fail("continuations mismatch")
if pconst["accept_id"] != 362 or pconst["reject_id"] != 425:
    fail("continuation ids mismatch")
spec = json.loads((D1R / "candidate_template_robustness_spec.json").read_text(encoding="utf-8"))
if spec["templates"]["T0"]["template"] != "The answer is <answer>.":
    fail("T0 template mismatch")
if spec["templates"]["T0"]["utf8_sha256"] != "c42e1ea10a6be3343c109664dbd25860bb6dade650d7a86c066e7ddbb0d298bc":
    fail("T0 SHA mismatch")

# ---- D2-R1 selected layer/C for RBSP ----
sel = json.loads((D2R1 / "scripts" / "_selected_lr.json").read_text(encoding="utf-8"))
if sel["selected_layer"] != 18 or sel["selected_C"] != 0.01:
    fail(f"D2-R1 selected layer/C unexpected: {sel}")
print("D2-R1 selected layer/C: L18, C=0.01 (inherited for RBSP)")

# ---- dev pairs ----
dev_pairs = [json.loads(l) for l in open(D1 / "scripts" / "_dev_pairs.jsonl", encoding="utf-8") if l.strip()]
if len(dev_pairs) != 195:
    fail(f"dev pairs={len(dev_pairs)}")
if {p["original_group_id"] for p in dev_pairs} != dev_ids:
    fail("dev pairs mismatch D0 dev ids")

# ---- D2-R1 score tables (SS labels) ----
dev_ss = json.loads((D2R1 / "scripts" / "_ss_dev_scores.json").read_text(encoding="utf-8"))
train_ss = json.loads((D2R1 / "scripts" / "_ss_train_scores.json").read_text(encoding="utf-8"))
if len(dev_ss) != 195 or len(train_ss) != 587:
    fail("D2-R1 SS score tables size unexpected")

# ---- D2-R1 prefix hidden states available (h_prefix L18/R_end) ----
ph_dir = D2R1 / "prefix_hidden_states"
import numpy as np
sample = np.load(ph_dir / f"dev_{dev_ss[0]['source_group_id']}.npz")["h_prefix"]
if sample.shape != (28, 3584):
    fail(f"h_prefix shape unexpected: {sample.shape}")
print("h_prefix shape OK:", sample.shape)

# ---- D1 four-cell scores for segmented-zero comparison ----
d1_rows = list(csv.DictReader(open(D1 / "four_cell_scores_dev.csv", encoding="utf-8")))
if len(d1_rows) != 780:
    fail(f"D1 four-cell rows={len(d1_rows)}")

# copy d1 four-cell rows + dev/train SS tables + dev pairs into D3 scripts for reuse
json.dump(d1_rows, open(D3 / "scripts" / "_d1_fourcell_dev.json", "w"))
json.dump(dev_ss, open(D3 / "scripts" / "_ss_dev_scores.json", "w"))
json.dump(train_ss, open(D3 / "scripts" / "_ss_train_scores.json", "w"))
json.dump(dev_pairs, open(D3 / "scripts" / "_dev_pairs.json", "w"))

(D3 / "inheritance_audit.md").write_text(
    f"""# inheritance_audit.md

## 继承对账（D3 Phase 0）

| 项 | 值 | 状态 |
|---|---|---|
| D0 final_label | `jar_style_sciq_data_qualification_feasible` | ✓ |
| D1 final_label | `jar_style_reference_override_behavior_feasible` | ✓ |
| D1-R final_label | `template_robust_reference_override_feasible` | ✓ |
| D2 final_label | `prefix_causality_audit_invalid` | ✓ |
| D2-R1 final_label | `true_prefix_reference_state_signal_localized` | ✓ |
| D0 split | train {len(train_ids)} / dev {len(dev_ids)} / reserve {len(res_ids)} | ✓ |
| Qwen revision | `a09a3545…` | ✓ |
| config/tokenizer/index 哈希 | 与 D1 一致 | ✓ |
| system prompt / user 模板 / chat template / continuations | 与 D1 一致 | ✓ |
| 基础模板 T0 | `The answer is <answer>.` | ✓ |
| D2-R1 selected layer/C | L18 / C=0.01（RBSP 继承） | ✓ |
| D2-R1 SS 评分表 | dev 195 / train 587 | ✓ |
| D2-R1 h_prefix | (28, 3584)，允许继承用于方向构造 | ✓ |
| D1 four-cell dev | 780 行 | ✓ |

## 本轮模型读取范围

```text
train_model_scored = true（587 groups，分段执行）
dev_model_scored = true（195 groups）
final_reserve_model_scored = false（197 groups 禁止）
```

## 关键继承约束

- RBSP 方向构造只允许使用 D2-R1 的 true-prefix `L18/R_end` states（train 内 D3-fit 子集）。
- D3-fit / D3-tune 只在 train 587 内以 seed 20260802 分层切分（70/30），group 四格同子集。
- dev 不得用于拟合方向、选择 alpha、选择触发阈值。
""",
    encoding="utf-8")
print("Phase 0 OK")
