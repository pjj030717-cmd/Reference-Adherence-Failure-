#!/usr/bin/env python3
"""E01-D2: swap_overlap_disclosure.md — descriptive data-generalization audit.
Reports string overlap of r_o/r_s between train/dev and swap-donor split relations.
No re-splitting; disclosure only.
"""
from __future__ import annotations

import json
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[2]

D0 = REPO_ROOT / "d0_jar_style_sciq_data_qualification_20260802"
D2 = REPO_ROOT / "d2_qwen25_7b_predecision_reference_state_localization_20260802"

fixed = json.loads((D0 / "fixed_split_indices.json").read_text(encoding="utf-8"))
train_ids = set(fixed["groups"]["train"])
dev_ids = set(fixed["groups"]["dev"])
res_ids = set(fixed["groups"]["final_reserve"])

recs = {"train": [], "dev": [], "reserve": []}
with open(D0 / "preliminary_swap_pairs.jsonl", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        p = json.loads(line)
        gid = p["original_group_id"]
        split = "train" if gid in train_ids else ("dev" if gid in dev_ids else ("reserve" if gid in res_ids else None))
        if split is None:
            continue
        recs[split].append({"gid": gid, "r_o": p["r_o"], "r_s": p["r_s"],
                            "donor": p["swap_source_group_id"]})

# normalized string overlap between train and dev
train_ro = {r["r_o"] for r in recs["train"]}
train_rs = {r["r_s"] for r in recs["train"]}
dev_ro = {r["r_o"] for r in recs["dev"]}
dev_rs = {r["r_s"] for r in recs["dev"]}

def jacc(a, b):
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)

lines = ["# swap_overlap_disclosure.md", "",
         "## 数据泛化限制审计（描述性披露，不用于重抽或改变 split）", ""]
lines += ["| 指标 | train | dev | train↔dev 交集/jaccard |", "|---|---|---|---|"]
lines += [f"| 唯一 r_o 数 | {len(train_ro)} | {len(dev_ro)} | 交 {len(train_ro & dev_ro)} / jaccard {jacc(train_ro, dev_ro):.3f} |"]
lines += [f"| 唯一 r_s 数 | {len(train_rs)} | {len(dev_rs)} | 交 {len(train_rs & dev_rs)} / jaccard {jacc(train_rs, dev_rs):.3f} |"]
lines += [f"| 唯一 (r_o,r_s) 对 | {len({(r['r_o'],r['r_s']) for r in recs['train']})} | {len({(r['r_o'],r['r_s']) for r in recs['dev']})} | — |"]

# swap donor split relation
donor_split = {}
for split, lst in recs.items():
    for r in lst:
        donor_split[r["gid"]] = {"donor": r["donor"], "donor_split": ("train" if r["donor"] in train_ids
                                                                      else ("dev" if r["donor"] in dev_ids
                                                                            else ("reserve" if r["donor"] in res_ids else "unknown")))}

lines += ["", "## swap donor 的 split 关系", "", "| 目标 split | donor 来自 train | donor 来自 dev | donor 来自 reserve | unknown |", "|---|---|---|---|---|"]
for split in ("train", "dev"):
    d = donor_split
    ntr = sum(1 for r in recs[split] if d[r["gid"]]["donor_split"] == "train")
    nde = sum(1 for r in recs[split] if d[r["gid"]]["donor_split"] == "dev")
    nrs = sum(1 for r in recs[split] if d[r["gid"]]["donor_split"] == "reserve")
    nuk = sum(1 for r in recs[split] if d[r["gid"]]["donor_split"] == "unknown")
    lines.append(f"| {split} | {ntr} | {nde} | {nrs} | {nuk} |")

lines += ["", "## 结论", "- 本披露仅用于说明 train/dev 在答案字符串与 swap donor 上的重叠程度，评估 R_end Probe 的泛化边界。",
          "- 不据此重抽、不删除 group、不改变既有 split。"]
(D2 / "swap_overlap_disclosure.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("wrote swap_overlap_disclosure.md")
print(lines)
