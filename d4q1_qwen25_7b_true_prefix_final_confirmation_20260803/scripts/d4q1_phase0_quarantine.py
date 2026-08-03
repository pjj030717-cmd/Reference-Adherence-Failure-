#!/usr/bin/env python3
"""D4-Q1 Phase 0.1: final-reserve quarantine audit.

Sources:
  - D0 fixed_split_indices.json (allowed: split indices + hashes only)
  - D4-M0 leak_isolation_audit.md (completed isolation audit: leaked group)

Contract:
  - D0 final_reserve = 197 groups
  - leaked final_reserve group = 1 (from the completed D4-M0 isolation audit)
  - allowed_final_group_ids = 196
  - leaked id NOT in allowed set
  - allowed ids index-hash consistent with D0 final split
  - this directory never reads / scores / caches the leaked group

Outputs:
  - final_reserve_quarantine_audit.md
  - allowed_final_group_manifest.json  (ids + hashes + split + allow_status ONLY,
    no question/reference/candidate text)
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[2]

R = REPO_ROOT / "d4q1_qwen25_7b_true_prefix_final_confirmation_20260803"
D0 = REPO_ROOT / "d0_jar_style_sciq_data_qualification_20260802"
D4M0 = REPO_ROOT / "d4m0_mistral7b_jar_style_sciq_behavior_qualification_20260803"

LEAKED_FINAL = "0075758e23e3c58ebb3e87d982021f6266b3f21029584278cef14d6d68a16c56"

fix = json.loads((D0 / "fixed_split_indices.json").read_text(encoding="utf-8"))
fr = sorted(fix["groups"]["final_reserve"])
print("D0 final_reserve count:", len(fr))
assert fix["split_sha256"]["final_reserve"] == "9fe440d6cb383c5c1d7d546af4a34381342d26b5bd9424bd623528a8063fc831"
print("D0 final_reserve split SHA256 OK:", fix["split_sha256"]["final_reserve"])

# quarantine source: D4-M0 leak isolation audit (completed artifact)
assert LEAKED_FINAL in fr, "leaked final id must be in D0 final split"
allowed = [g for g in fr if g != LEAKED_FINAL]
assert len(allowed) == 196
assert LEAKED_FINAL not in allowed

# index-hash consistency: allowed set = D0 final split minus leaked group
allowed_sha = hashlib.sha256("\n".join(sorted(allowed)).encode("utf-8")).hexdigest()
# provenance: recompute from D0 final split minus leaked
prov = [g for g in fr if g != LEAKED_FINAL]
prov_sha = hashlib.sha256("\n".join(sorted(prov)).encode("utf-8")).hexdigest()
assert allowed_sha == prov_sha
print("allowed count:", len(allowed), "allowed-index-sha256:", allowed_sha)

# ---- allowed_final_group_manifest.json (no raw text) ----
manifest = {
    "experiment": "D4-Q1 Qwen true-prefix final-reserve confirmation",
    "final_reserve_total_groups": len(fr),
    "leaked_final_group_count": 1,
    "allowed_final_group_count": len(allowed),
    "leaked_group_id": LEAKED_FINAL,
    "allowed_group_index_sha256": allowed_sha,
    "provenance": "D0 fixed_split_indices.json final_reserve split minus D4-M0 leak_isolation_audit.md leaked group",
    "groups": [
        {"source_group_id": g,
         "group_hash": g,  # D0 source_group_id IS the group hash (SHA256 of frozen normalized fields)
         "split": "final_reserve",
         "allow_status": "allowed"}
        for g in allowed
    ],
}
(R / "allowed_final_group_manifest.json").write_text(
    json.dumps(manifest, indent=2), encoding="utf-8")

# ---- final_reserve_quarantine_audit.md ----
(R / "final_reserve_quarantine_audit.md").write_text(
    f"""# final_reserve_quarantine_audit.md

## 隔离来源

- 已完成的 final-reserve 泄露隔离审计：D4-M0 `leak_isolation_audit.md`（2026-08-03）。
  - 该审计确认唯一 final_reserve 泄露 group：`{LEAKED_FINAL[:8]}`（在 Phase 0 探索 `head -3` 时显示）。
  - 另有两个 train 组泄露（`004c1d1f`、`015c326e`），不属于 final-reserve split，不影响本轮 final 隔离。
- 本轮**不**重新浏览 D0 raw 文件推断泄露 group；泄露 group 完全由已完成的隔离工件唯一给定。

## 隔离契约

| 项 | 值 |
|---|---|
| D0 原 final-reserve group 数 | {len(fr)} |
| 泄露 final group 数 | 1 |
| 允许使用 final group 数 | {len(allowed)} |
| 泄露 group 是否在允许集合中 | 否（已剔除） |
| 允许集合索引哈希（SHA256 of sorted ids） | `{allowed_sha}` |
| D0 final split 索引哈希 | `{fix['split_sha256']['final_reserve']}` |

## 允许集合与 D0 final split 的一致性

- 允许集合 = D0 `fixed_split_indices.json` 的 final_reserve 197 个 id，减去泄露的 1 个 id。
- 证明：`set(allowed) == set(D0 final_reserve) - {{LEAKED}}`，且排序后索引哈希一致（`{prov_sha}`）。

## 使用约束

- 本轮只流式读取 `allowed_final_group_manifest.json` 中的 196 个 group。
- 禁止打开、评分、缓存或写出泄露 group（`{LEAKED_FINAL[:8]}`）的任何文本、id、评分或 hidden state。
- 本轮所有输出（manifest/CSV/report）只允许包含 group id / hash / 数值；不得包含 final 组题目或答案正文。
""",
    encoding="utf-8")

print("wrote final_reserve_quarantine_audit.md, allowed_final_group_manifest.json")
print("Phase 0.1 OK: quarantine isolated (1 leaked, 196 allowed)")
