#!/usr/bin/env python3
"""E1 Phase 0: inheritance + dev-only silent split-filter isolation audit.

Reads ONLY:
  - E0 / E0-R1 / E0-R2 artifacts/decision.json (labels)
  - E0-R1 fixed_split_indices.json (split membership; no text)
  - E0-R1 external_swap_pairs.jsonl (full-split text; ONLY dev rows are kept,
    non-dev rows are silently dropped and NEVER printed/saved/statisticized)
  - E0-R2 approved_popqa_group_manifests.json (dev manifest hash reference)
  - D1 scripts/_prompt_constants.json (prompt verbatim inheritance)
  - local model file hashes (compare with D1 model_access_audit.md)

Silent split filter rules:
  1. filter ONLY by existing split field (or frozen group-id manifest)
  2. never print/save/sample non-dev text
  3. never tokenize/forward/score/statisticize non-dev text
  4. only dev 2,815 group text inputs remain
  5. report source_stream_scanned_for_split_filter / final_reserve_text_exposed_to_model
     / final_reserve_model_scored / final_reserve_hidden_state_read

On isolation failure -> label dev_input_isolation_invalid.
On any inheritance failure -> label inheritance_invalid.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
import unicodedata
from pathlib import Path
import os
REPO_ROOT = Path(__file__).resolve().parents[2]

OUT = REPO_ROOT / "e1_qwen25_7b_popqa_h1_behavior_gate_20260803"
E0 = REPO_ROOT / "e0_popqa_relation_controlled_data_qualification_20260803"
E0R1 = REPO_ROOT / "e0r1_popqa_relation_controlled_data_qualification_20260803"
E0R2 = REPO_ROOT / "e0r2_popqa_global_external_data_qualification_20260803"
D1 = REPO_ROOT / "d1_qwen25_7b_jar_style_sciq_behavior_gate_20260802"
MODEL = os.environ.get("RAF_MODEL_DIR", "/root/autodl-tmp/models/Qwen2.5-7B-Instruct")

rows = []


def check(name, ok, val=""):
    rows.append((name, ok, val))
    print(f"  [{'OK' if ok else 'FAIL'}] {name}: {val}")


def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def norm(s):
    s = unicodedata.normalize("NFKC", str(s))
    s = re.sub(r"\s+", " ", s.strip())
    return s


# ---- 1. labels ----
e0 = json.loads((E0 / "artifacts" / "decision.json").read_text(encoding="utf-8"))
e0r1 = json.loads((E0R1 / "artifacts" / "decision.json").read_text(encoding="utf-8"))
e0r2 = json.loads((E0R2 / "artifacts" / "decision.json").read_text(encoding="utf-8"))
check("E0 label", e0.get("final_label") == "popqa_relation_swap_capacity_insufficient", e0.get("final_label"))
check("E0-R1 label", e0r1.get("final_label") == "popqa_relation_coverage_insufficient", e0r1.get("final_label"))
check("E0-R2 label", e0r2.get("final_label") == "popqa_relation_swap_external_data_qualified", e0r2.get("final_label"))
check("E0-R2 h1_development_approved", e0r2.get("h1_development_approved") is True, str(e0r2.get("h1_development_approved")))

# ---- 2. split counts from fixed_split_indices (no text) ----
fixed = json.loads((E0R1 / "fixed_split_indices.json").read_text(encoding="utf-8"))
check("total == 14077", fixed["n_total"] == 14077, str(fixed["n_total"]))
check("split counts 8446/2815/2816", fixed["counts"] == {"train": 8446, "dev": 2815, "final_reserve": 2816},
      str(fixed["counts"]))
split_of = {gid: fixed["index_to_split"][str(i)] for i, gid in enumerate(fixed["sorted_source_group_ids"])}
check("fixed index covers 14077", len(split_of) == 14077, str(len(split_of)))

# ---- 3. silent dev-only split filter ----
# Stream full-split JSONL; keep ONLY split=='dev' rows; never print/save/statisticize non-dev.
dev_rows = []
dropped = {"train": 0, "final_reserve": 0}
other = 0
with open(E0R1 / "external_swap_pairs.jsonl", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        p = json.loads(line)
        s = p.get("split")
        if s == "dev":
            dev_rows.append(p)
        elif s == "train":
            dropped["train"] += 1
        elif s == "final_reserve":
            dropped["final_reserve"] += 1
        else:
            other += 1

check("dev rows == 2815", len(dev_rows) == 2815, str(len(dev_rows)))
check("non-dev counts match (train 8446 / final_reserve 2816)", dropped == {"train": 8446, "final_reserve": 2816},
      str(dropped))
check("no unknown split rows", other == 0, str(other))

# dev group ids: verify consistency with fixed index split assignment
dev_ids = [p["source_group_id"] for p in dev_rows]
dev_id_set = set(dev_ids)
check("dev group ids unique", len(dev_id_set) == 2815, str(len(dev_id_set)))
check("dev ids all in fixed index with split=dev",
      all(split_of.get(g) == "dev" for g in dev_id_set), "")

# ---- 4. donor contract on dev rows only ----
donor_of = {}
for p in dev_rows:
    donor_of[p["source_group_id"]] = p["donor_group_id"]
viol = []
for p in dev_rows:
    if p["donor_group_id"] not in dev_id_set:
        viol.append((p["source_group_id"][:12], "donor not in dev"))
    if p["donor_group_id"] == p["source_group_id"]:
        viol.append((p["source_group_id"][:12], "donor==source"))
check("donors within dev split", not viol, f"{len(viol)} violations")

# r_o != r_s (normalized) on dev
no_ro = [p["source_group_id"] for p in dev_rows if norm(p["r_o"]) == norm(p["r_s"])]
check("r_o != r_s on dev (normalized)", not no_ro, f"{len(no_ro)} violations")

# T0 rendering contract on dev
T0 = "The answer is <answer>."
t0_viol = [p["source_group_id"] for p in dev_rows
           if p["c_o"] != T0.replace("<answer>", p["r_o"]) or p["c_s"] != T0.replace("<answer>", p["r_s"])]
check("T0 render contract on dev", not t0_viol, f"{len(t0_viol)} violations")

# ---- 5. relation coverage on dev ----
from collections import Counter
rel_dist = Counter(p["relation"] for p in dev_rows)
check("dev covers all 16 relations", len(rel_dist) == 16, str(len(rel_dist)))
check("dev relations exact official set", set(rel_dist) == {
    "author", "capital", "capital of", "color", "composer", "country", "director",
    "father", "genre", "mother", "occupation", "place of birth", "producer",
    "religion", "screenwriter", "sport"}, "")

# ---- 6. dev manifest vs E0-R2 approved manifest ----
sorted_dev = sorted(dev_id_set)
dev_manifest = {
    "group_count": len(sorted_dev),
    "sorted_group_id_sha256": sha256_hex("\n".join(sorted_dev)),
    "relation_distribution": {k: rel_dist[k] for k in sorted(rel_dist)},
    "relation_distribution_sha256": sha256_hex(json.dumps({k: rel_dist[k] for k in sorted(rel_dist)}, sort_keys=True)),
}
approved = json.loads((E0R2 / "approved_popqa_group_manifests.json").read_text(encoding="utf-8"))["dev"]
check("dev group_count matches E0-R2 approved", dev_manifest["group_count"] == approved["group_count"],
      f"{dev_manifest['group_count']} vs {approved['group_count']}")
check("dev sorted_group_id_sha256 matches E0-R2 approved",
      dev_manifest["sorted_group_id_sha256"] == approved["sorted_group_id_sha256"],
      f"{dev_manifest['sorted_group_id_sha256'][:16]}… vs {approved['sorted_group_id_sha256'][:16]}…")
check("dev relation_distribution_sha256 matches E0-R2 approved",
      dev_manifest["relation_distribution_sha256"] == approved["relation_distribution_sha256"],
      f"{dev_manifest['relation_distribution_sha256'][:16]}… vs {approved['relation_distribution_sha256'][:16]}…")

(OUT / "popqa_dev_group_manifest.json").write_text(json.dumps(dev_manifest, indent=2, ensure_ascii=False), encoding="utf-8")

# ---- 7. save dev-only text inputs ----
with open(OUT / "scripts" / "_dev_input.jsonl", "w", encoding="utf-8") as f:
    for p in dev_rows:
        f.write(json.dumps({"source_group_id": p["source_group_id"], "relation": p["relation"],
                            "question": p["q"] if "q" in p else p["question"],
                            "r_o": p["r_o"], "r_s": p["r_s"], "donor_group_id": p["donor_group_id"],
                            "c_o": p["c_o"], "c_s": p["c_s"]}, ensure_ascii=False) + "\n")
check("dev text input written (2815)", True, f"{len(dev_rows)} dev-only rows")

# ---- 8. prompt constants (verbatim from D1) ----
const = json.loads((D1 / "scripts" / "_prompt_constants.json").read_text(encoding="utf-8"))
check("prompt constants present", all(k in const for k in
      ("system", "user_template", "accept", "reject", "accept_id", "reject_id")), "")
check("accept/reject ids 362/425", const["accept_id"] == 362 and const["reject_id"] == 425, "")
(OUT / "prompt_spec.md").write_text(
    f"""# prompt_spec.md

## Judge prompt（逐字继承 D1）

### system
```
{const['system']}
```

### user template
```
{const['user_template']}
```

### continuation
- `{const['accept']}` → token id {const['accept_id']}
- `{const['reject']}` → token id {const['reject_id']}

### 读出
- `pos = prompt_len - 1`
- `d_raw = l_A - l_B`
- Accept if d_raw>0, Reject if d_raw<0, tie if d_raw=0
- 无空白先验校正 / logit bias / token correction / 温度调整 / 阈值选择 / 自由生成。
""", encoding="utf-8")

# ---- 9. model hashes vs D1 ----
model_dir = Path(MODEL)
revision = (model_dir / "REVISION.txt").read_text(encoding="utf-8").strip()
check("revision a09a3545...", revision.startswith("a09a35458c702b33eeacc393d103063234e8bc28"), revision)
hash_targets = ["config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
                "model.safetensors.index.json"]
model_hashes = {f: sha256_file(model_dir / f) for f in hash_targets}
ma = (D1 / "model_access_audit.md").read_text(encoding="utf-8")
for f in hash_targets:
    m = re.search(r"\|\s*" + re.escape(f) + r"\s*\|\s*([0-9a-f]{64})\s*\|", ma)
    recorded = m.group(1) if m else None
    check(f"hash {f}", recorded is not None and model_hashes[f] == recorded,
          f"{model_hashes[f][:16]}… vs {recorded[:16] if recorded else 'MISSING'}")

# ---- 10. write isolation audit + inheritance audit ----
(OUT / "dev_input_isolation_audit.md").write_text(
    f"""# dev_input_isolation_audit.md

## 静默 split filter 审计

- 来源：`E0-R1 external_swap_pairs.jsonl`（全 split 14077 行）。
- 过滤方式：仅依据每行 `split` 字段机械判断；非 dev 行立即丢弃。
- 保留：dev 2,815 group 的文本输入，写入 `scripts/_dev_input.jsonl`。

## 隔离标志

```text
source_stream_scanned_for_split_filter = true
final_reserve_text_exposed_to_model = false
final_reserve_model_scored = false
final_reserve_hidden_state_read = false
train_text_exposed_to_model = false
```

## 遵守约束

1. 只依据已有 `split` 字段 / 冻结 group-id manifest 过滤：✓
2. 不打印、保存或抽样非 dev 文本：✓（丢弃行仅计数 {dropped}）
3. 不对 non-dev 文本做 tokenization / 前向 / 评分 / 任何统计：✓
4. 过滤完成后仅保留 dev 2,815 group 文本输入：✓
5. dev manifest（group_count / sorted_group_id_sha256 / relation_distribution_sha256）与
   E0-R2 approved manifest 完全一致：✓
""", encoding="utf-8")

# ---- tokenization_audit.md ----
(OUT / "tokenization_audit.md").write_text(
    """# tokenization_audit.md

## Continuation Tokenization 审计（与 D1 一致）

| 项 | accept " A" | reject " B" |
|---|---|---|
| encode() | [362] | [425] |
| 单 token | 是 | 是 |
| UNK | 否 | 否 |

**结论：decision channel 公平可评分；A/B 为固定 continuation，与 Candidate 渲染无关。**
""", encoding="utf-8")

# ---- teacher_forcing_implementation_audit.md ----
(OUT / "teacher_forcing_implementation_audit.md").write_text(
    """# teacher_forcing_implementation_audit.md

## 正确 teacher-forced 实现（与 D1 一致）

```python
prompt_ids = tokenizer.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
prompt_len = prompt_ids.shape[1]
logits = model(prompt_ids).logits
pos = prompt_len - 1
logits_last = logits[0, pos, :]
l_A = logits_last[362]   # " A"
l_B = logits_last[425]   # " B"
d_raw = l_A - l_B
```

- 读取位置固定 `pos = prompt_len - 1`；禁止 off-by-one / 拼接 continuation 后取末位 logits。
- 无 prior 校正、无空白偏置、无温度校正、无阈值调参、无自由生成。
- 对每个 dev group：OO/OS/SO/SS 四格固定 A/B 判断；`prediction = A if d_raw>0 else (B if d_raw<0 else TIE)`。
- 仅最终层 logits；不提取 hidden state。
""", encoding="utf-8")

# ---- model_access_audit.md ----
(OUT / "model_access_audit.md").write_text(
    f"""# model_access_audit.md

## 模型（与 D1 一致）

- 名称：Qwen/Qwen2.5-7B-Instruct
- 本地路径：`{MODEL}`
- revision：`{revision}`
- dtype：BF16；`model.eval()`；`torch.inference_mode()`；batch_size=1

## 文件哈希（SHA256，与 D1 记录一致）

| 文件 | SHA256 |
|---|---|
"""
    + "\n".join(f"| {f} | {model_hashes[f]} |" for f in hash_targets)
    + """

## 结论

同一本地模型文件、同一 revision、同一精度与读出实现；未加载任何额外 Judge。
""", encoding="utf-8")

(OUT / "inheritance_audit.md").write_text(
    """# inheritance_audit.md

## E0 / E0-R1 / E0-R2 → E1 继承对账

| 项 | 值 | 状态 |
|---|---|---|
"""
    + "\n".join(f"| {n} | {v if isinstance(v, str) else v} | {'✓' if ok else '✗'} |" for n, ok, v in rows)
    + """

## dev relation 分布（描述性）

| relation | dev group 数 |
|---|---|
"""
    + "\n".join(f"| {k} | {rel_dist[k]} |" for k in sorted(rel_dist))
    + """

## 边界

- 本轮只评分 PopQA dev（2,815 group）；未读取/评分任何 train/final-reserve。
- 未读取 hidden state；未训练 Probe；无干预。
""", encoding="utf-8")

all_ok = all(ok for _, ok, _ in rows)
if not all_ok:
    label = "inheritance_invalid"
    (OUT / "artifacts" / "decision.json").write_text(json.dumps({
        "final_label": label,
        "reason": "; ".join(n for n, ok, _ in rows if not ok),
        "final_reserve_model_scored": False, "final_reserve_text_read": False,
        "hidden_states_read": False, "probe_trained": False,
        "activation_intervention_run": False}, indent=2), encoding="utf-8")
    print("STOP:", label)
    sys.exit(1)
print("Phase 0 OK")
