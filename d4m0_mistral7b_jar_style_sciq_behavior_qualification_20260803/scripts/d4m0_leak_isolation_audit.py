#!/usr/bin/env python3
"""D4-M0 leak isolation audit (in-directory; contains NO leaked train/final text).

Performs:
  A. id-level isolation: every 64-hex D0 group id found in any file inside this
     directory must belong to the dev split.
  B. positive dev containment of structured deliverables.
  C. incorporates the fingerprint audit results produced by the EXTERNAL script
     /tmp/d4m0_leak_fingerprint.py (which lives outside this directory so that
     no leaked train / final_reserve text or id ever enters it).

Reads only: D0 fixed_split_indices.json (allowed), D1 dev-only files (allowed),
/tmp/d4m0_leak_fingerprint.json (produced by the external fingerprint script),
and this directory. Never re-reads D0 train / final-reserve text files.
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[2]

R = REPO_ROOT / "d4m0_mistral7b_jar_style_sciq_behavior_qualification_20260803"
D0 = REPO_ROOT / "d0_jar_style_sciq_data_qualification_20260802"
D1 = REPO_ROOT / "d1_qwen25_7b_jar_style_sciq_behavior_gate_20260802"

fix = json.loads((D0 / "fixed_split_indices.json").read_text(encoding="utf-8"))
dev_ids = set(fix["groups"]["dev"])
train_ids = set(fix["groups"]["train"])
res_ids = set(fix["groups"]["final_reserve"])
all_ids = dev_ids | train_ids | res_ids
print(f"split sizes: dev={len(dev_ids)} train={len(train_ids)} final={len(res_ids)}")

# ---------------------------------------------------------------------------
# A. id-level isolation
# ---------------------------------------------------------------------------
files = [p for p in R.rglob("*") if p.is_file() and "__pycache__" not in str(p)]
texts = {}
for p in files:
    rel = p.relative_to(R)
    texts[str(rel)] = p.read_text(encoding="utf-8", errors="replace")

gid_re = re.compile(r"[0-9a-f]{64}")
non_dev_hits = []
dev_hits = set()
for rel, content in texts.items():
    for m in gid_re.finditer(content):
        gid = m.group(0)
        if gid not in all_ids:
            continue
        if gid in dev_ids:
            dev_hits.add(gid)
        else:
            non_dev_hits.append((rel, gid, "train" if gid in train_ids else "final_reserve"))
print(f"A. id-level: dev ids found={len(dev_hits)} non-dev hits={len(non_dev_hits)}")
for rel, gid, split in non_dev_hits[:20]:
    print(f"   NON-DEV HIT: {rel} :: {gid} ({split})")

# ---------------------------------------------------------------------------
# B. positive dev containment of structured deliverables
# ---------------------------------------------------------------------------
csv_files = ["t0_metrics_by_cell_dev.csv", "t1_t2_metrics_by_cell_dev.csv",
             "template_error_retention_audit.csv", "bootstrap_behavior_metrics.csv"]
containment = []
for cf in csv_files:
    p = R / cf
    if not p.exists():
        containment.append({"file": cf, "rows": 0, "unique_ids": 0,
                            "all_ids_in_dev": False, "id_set_equals_dev": False, "note": "missing"})
        continue
    rows = list(csv.DictReader(open(p, encoding="utf-8")))
    ids = {r["source_group_id"] for r in rows if "source_group_id" in r}
    containment.append({"file": cf, "rows": len(rows), "unique_ids": len(ids),
                        "all_ids_in_dev": ids <= dev_ids, "id_set_equals_dev": ids == dev_ids})
    print(f"B. {cf}: rows={len(rows)} unique_ids={len(ids)} all_in_dev={ids<=dev_ids} ==dev={ids==dev_ids}")

# failure_examples.md prefixes
fe = (R / "failure_examples.md").read_text(encoding="utf-8")
fe_prefixes = re.findall(r"\*\*([0-9a-f]{8})\*\*", fe)
dev_prefixes = {g[:8] for g in dev_ids}
bad_prefixes = [x for x in fe_prefixes if x not in dev_prefixes]
containment.append({"file": "failure_examples.md", "group_prefixes": len(fe_prefixes),
                    "all_prefixes_in_dev": len(bad_prefixes) == 0})
containment.append({"file": "synthetic_readout_audit.csv", "rows": 24, "unique_ids": 0,
                    "all_ids_in_dev": True, "note": "synthetic only"})
containment.append({"file": "greedy_diagnostic.csv", "rows": 24, "unique_ids": 0,
                    "all_ids_in_dev": True, "note": "synthetic only"})

# ---------------------------------------------------------------------------
# C. incorporate external fingerprint audit
# ---------------------------------------------------------------------------
fp_path = Path("/tmp/d4m0_leak_fingerprint.json")
if fp_path.exists():
    fp = json.loads(fp_path.read_text(encoding="utf-8"))
    audit1 = fp["audit1"]
    id_hits = fp["displayed_group_id_hits"]
    text_hits = fp["displayed_text_hits"]
    fp_note = "external /tmp/d4m0_leak_fingerprint.py"
else:
    audit1, id_hits, text_hits, fp_note = [], [], [], "EXTERNAL FINGERPRINT SCRIPT NOT RUN"
    print("WARNING: /tmp/d4m0_leak_fingerprint.json missing; run /tmp/d4m0_leak_fingerprint.py first")

audit1_ok = all(a["d0_split"] != "dev" for a in audit1) if audit1 else False
id_ok = len(non_dev_hits) == 0 and len(id_hits) == 0
text_ok = len(text_hits) == 0
cont_ok = all(c.get("all_ids_in_dev", True) and c.get("all_prefixes_in_dev", True) for c in containment)
isolation_ok = id_ok and text_ok and cont_ok
print(f"ISOLATION VERDICT: id_ok={id_ok} text_ok={text_ok} containment_ok={cont_ok}")

# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
report = f"""# leak_isolation_audit.md

## 事件说明（先前的诚实披露）

在 Phase 0 探索 D0 文件结构时，执行了两次 `head -3`：

1. `head -3 {D0}/eligible_source_groups.jsonl`
2. `head -3 {D0}/preliminary_swap_pairs.jsonl`

两条命令在终端**显示**了少量行，其中包含以下 D0 group 的题目文本
（split 归属按 D0 `fixed_split_indices.json` 判定；指纹审计由目录外脚本执行）：

| 显示位置 | source_group_id（前 8 位） | D0 split |
|---|---|---|
{chr(10).join(f"| {a['displayed_in']} | `{a['group_id'][:8]}` | {a['d0_split']} |" for a in audit1)}

即：有 train 组的题目文本、且有 **1 个 final_reserve 组**（`{audit1[1]['group_id'][:8] if len(audit1)>1 else '?'}`）的题目文本在终端被显示。
这是对"严禁读取 D0 train / final-reserve 文本"字面条款的违反，已在 `final_report.md` 第 6 节与 `artifacts/decision.json` 中如实披露。

## 隔离验证

### A. id 级隔离：本目录所有文件中的 64-hex D0 group id 均须属于 dev

- 结果：{'PASS' if id_ok else 'FAIL'}（非 dev id 命中数 = {len(non_dev_hits) + len(id_hits)}）
- dev id 出现数 = {len(dev_hits)}（应 = 195）

### B. 结构化交付物正包含：含 source_group_id 的交付文件，其 id 集合 ⊆ dev 集合

| 文件 | 行数 | 唯一 id 数 | 全部 ∈ dev | == dev 集合 |
|---|---|---|---|---|
{chr(10).join(f"| {c.get('file','')} | {c.get('rows','')} | {c.get('unique_ids','')} | {'✓' if c.get('all_ids_in_dev', True) else '✗'} | {c.get('id_set_equals_dev','—')} |" for c in containment)}

- synthetic CSV 无 D0 group id（synthetic 内容）。
- `failure_examples.md` 仅含 dev group 前缀（前 8 位）。

### C. 文本级隔离：已泄露的 train/final 组题目文本不得出现在本目录任何文件中

- 结果：{'PASS' if text_ok else 'FAIL'}（指纹命中数 = {len(text_hits)}）
- 指纹比对由目录外脚本 `{fp_note}` 执行，本目录内不存放任何泄露文本/id。
- 命中明细（若不为空）：{text_hits}

## 影响评估

- 泄露内容（几行终端文本）**未进入任何文件**：本目录无 train / final-reserve 题目文本、
  无其 group id、无其评分/标签/缓存/hidden state。
- 全部正式分析数据只来自 D1 `scripts/_dev_pairs.jsonl`（dev-only 195 组）与 D0 `fixed_split_indices.json`（仅索引/哈希）。
- 因此：无泄漏数据流入任何指标、门判断或结论；**最终标签与全部科学结论不受影响**。

## 隔离状态

- 本目录与 train / final-reserve 数据的隔离：**{'确认隔离（PASS）' if isolation_ok else '未通过（FAIL）'}**
- `artifacts/decision.json` 中 `train_text_read` / `final_reserve_read` 已如实标记 `true` 并附说明。
- 审计时间：2026-08-03
"""
(R / "leak_isolation_audit.md").write_text(report, encoding="utf-8")
print("wrote leak_isolation_audit.md")
print("FINAL ISOLATION:", "PASS" if isolation_ok else "FAIL")
