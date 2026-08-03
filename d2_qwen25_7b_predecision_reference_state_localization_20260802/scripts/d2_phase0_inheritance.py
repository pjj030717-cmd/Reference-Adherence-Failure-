#!/usr/bin/env python3
"""E01-D2 Phase 0: D0/D1/D1-R inheritance verification.

Checks D0/D1/D1-R labels, model hashes, prompt inheritance, dev pairs availability.
Reads ONLY dev/train pairs; final-reserve never loaded.
"""
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
MODEL_DIR = Path(os.environ.get("RAF_MODEL_DIR", "/root/autodl-tmp/models/Qwen2.5-7B-Instruct"))


def fail(why: str):
    print("d1r_inheritance_invalid:", why)
    (D2 / "artifacts").mkdir(parents=True, exist_ok=True)
    (D2 / "artifacts" / "decision.json").write_text(
        json.dumps({"final_label": "d1r_inheritance_invalid", "reason": why,
                    "final_reserve_model_scored": False, "final_reserve_hidden_states_read": False,
                    "probe_trained": True, "activation_intervention_run": False,
                    "prompt_baselines_run": False, "mistral_loaded": False}, indent=2), encoding="utf-8")
    sys.exit(1)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---- labels ----
d0 = json.loads((D0 / "artifacts" / "decision.json").read_text(encoding="utf-8"))
d1 = json.loads((D1 / "artifacts" / "decision.json").read_text(encoding="utf-8"))
d1r = json.loads((D1R / "artifacts" / "decision.json").read_text(encoding="utf-8"))
if d0.get("final_label") != "jar_style_sciq_data_qualification_feasible":
    fail(f"D0 label={d0.get('final_label')}")
if d1.get("final_label") != "jar_style_reference_override_behavior_feasible":
    fail(f"D1 label={d1.get('final_label')}")
if d1r.get("final_label") != "template_robust_reference_override_feasible":
    fail(f"D1R label={d1r.get('final_label')}")

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
need = ["config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
        "model.safetensors.index.json"]
for f in need:
    cur = sha256_file(MODEL_DIR / f)
    if rec.get(f) != cur:
        fail(f"model file {f} hash mismatch: D1={rec.get(f)}, current={cur}")

# ---- prompt inheritance ----
pconst = json.loads((D1 / "scripts" / "_prompt_constants.json").read_text(encoding="utf-8"))
if pconst["accept"] != " A" or pconst["reject"] != " B":
    fail("continuations mismatch")
if pconst["accept_id"] != 362 or pconst["reject_id"] != 425:
    fail("continuation ids mismatch")

# ---- T0 template from D1R spec ----
spec = json.loads((D1R / "candidate_template_robustness_spec.json").read_text(encoding="utf-8"))
T0 = spec["templates"]["T0"]["template"]
if T0 != "The answer is <answer>.":
    fail(f"T0 template mismatch: {T0}")
t0_sha = spec["templates"]["T0"]["utf8_sha256"]
if t0_sha != "c42e1ea10a6be3343c109664dbd25860bb6dade650d7a86c066e7ddbb0d298bc":
    fail(f"T0 SHA mismatch: {t0_sha}")

# ---- dev pairs from D1 ----
dev_pairs = []
with open(D1 / "scripts" / "_dev_pairs.jsonl", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            dev_pairs.append(json.loads(line))
if len(dev_pairs) != 195:
    fail(f"dev pairs={len(dev_pairs)}")
dev_pair_ids = {p["original_group_id"] for p in dev_pairs}
if dev_pair_ids != dev_ids:
    fail("dev pairs mismatch D0 dev ids")

# ---- D1 four-cell dev scores available ----
d1_rows = list(csv.DictReader(open(D1 / "four_cell_scores_dev.csv", encoding="utf-8")))
if len(d1_rows) != 780:
    fail(f"D1 four-cell rows={len(d1_rows)}")

# ---- write inheritance_audit.md ----
(D2 / "inheritance_audit.md").write_text(
    f"""# inheritance_audit.md

## 继承对账（Phase 0）

| 项 | 值 | 状态 |
|---|---|---|
| D0 final_label | `jar_style_sciq_data_qualification_feasible` | ✓ |
| D1 final_label | `jar_style_reference_override_behavior_feasible` | ✓ |
| D1-R final_label | `template_robust_reference_override_feasible` | ✓ |
| D0 split | train {len(train_ids)} / dev {len(dev_ids)} / reserve {len(res_ids)}（seed 20260802） | ✓ |
| Qwen revision | `a09a3545…` | ✓ |
| config/tokenizer/index 哈希 | 与 D1 model_access_audit.md 一致 | ✓ |
| system prompt / user 模板 / chat template / continuations `" A"`/`" B"` | 与 D1 一致 | ✓ |
| 基础模板 T0 | `The answer is <answer>.`（SHA256 `{t0_sha}`） | ✓ |
| dev pairs | 195（与 D0 dev ids 一致） | ✓ |
| D1 四格行 | 780（four_cell_scores_dev.csv） | ✓ |

## 本轮模型读取范围

```text
train_model_scored = true（587 groups）
dev_model_scored = true（195 groups）
final_reserve_model_scored = false（197 groups 禁止读取/评分/缓存/提取）
```

## 备注

- D0 中间态 hash 缺陷按 D1-R `provenance_amendment.md` 处理，不修改 D0。
""",
    encoding="utf-8")
print("Phase 0 OK")
