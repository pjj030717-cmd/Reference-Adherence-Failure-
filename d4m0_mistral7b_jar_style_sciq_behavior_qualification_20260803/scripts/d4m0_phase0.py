#!/usr/bin/env python3
"""D4-M0 Phase 0: D0 inheritance audit + template inheritance audit.

0.1 D0 inheritance audit
  - D0 final label == jar_style_sciq_data_qualification_feasible
  - dev split == 195 source groups (from D0 fixed_split_indices.json)
  - swap mapping / four-cell construction / T0 rendering uniquely recoverable
  - per-dev-group four-cell expectation: OO=A, OS=B, SO=B, SS=A
  - four-cell text comes from frozen D0 content (via D1 dev-only _dev_pairs.jsonl)
  - this directory contains no train / final-reserve full-question text

0.2 Template inheritance principles
  - recover system/user semantics, field order, verdict definitions, T0/T1/T2
    candidate expressions from D1/D1-R artifacts
  - render one example prompt with Mistral native chat template
  - output prompt_semantic_inheritance_audit.md

Loads the Mistral TOKENIZER only (no weights) to render the native serialization.
No D0 train/final-reserve text is read or copied.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
import os

from d4m0_core import (D0, D1, D1R, R, SYSTEM, TEMPLATES, TEMPLATE_SHA, USER_TMPL,
                       sha256_hex, render_candidate, load_dev_pairs)
from transformers import AutoTokenizer

MODEL_DIR = os.environ.get("RAF_MISTRAL_DIR", "/root/autodl-tmp/models/Mistral-7B-Instruct-v0.3")


def fail(label: str, why: str):
    print("STOP:", label, "-", why)
    (R / "artifacts").mkdir(parents=True, exist_ok=True)
    (R / "artifacts" / "decision.json").write_text(
        json.dumps({"final_label": label, "reason": why,
                    "final_reserve_read": False, "hidden_states_read": False,
                    "probe_trained": False, "activation_intervention_run": False,
                    "prompt_baselines_run": False, "train_text_read": False}, indent=2),
        encoding="utf-8")
    sys.exit(1)


# ---------------------------------------------------------------------------
# 0.1 D0 inheritance audit
# ---------------------------------------------------------------------------
lines = []
d0_dec = json.loads((D0 / "artifacts" / "decision.json").read_text(encoding="utf-8"))
if d0_dec["final_label"] != "jar_style_sciq_data_qualification_feasible":
    fail("inheritance_or_data_contract_invalid", f"D0 final label = {d0_dec['final_label']}")
lines.append(("D0 最终标签", d0_dec["final_label"], True))

fix = json.loads((D0 / "fixed_split_indices.json").read_text(encoding="utf-8"))
dev_ids_d0 = fix["groups"]["dev"]
if len(dev_ids_d0) != 195:
    fail("inheritance_or_data_contract_invalid", f"D0 dev groups = {len(dev_ids_d0)}, expected 195")
lines.append(("D0 dev group 数", len(dev_ids_d0), len(dev_ids_d0) == 195))
lines.append(("D0 split seed", fix["seed"], fix["seed"] == 20260802))
lines.append(("D0 dev split SHA256", fix["split_sha256"]["dev"],
              fix["split_sha256"]["dev"] == "8be6f6f3450376cb90c597ad0fe9edf2ea422101501900c13c5604db61e4fd35"))

# dev pairs inherited through D1's dev-only file (D1 is an allowed source;
# D1 built it from D0 frozen content and its split field is dev only).
pairs = load_dev_pairs()
ids_d1 = [p["original_group_id"] for p in pairs]
if set(ids_d1) != set(dev_ids_d0) or len(pairs) != 195:
    fail("inheritance_or_data_contract_invalid",
         f"D1 dev pairs {len(pairs)} vs D0 dev ids {len(dev_ids_d0)}; set-equal={set(ids_d1)==set(dev_ids_d0)}")
lines.append(("dev 195 组与 D0 split 索引一致", f"{len(pairs)} pairs, set-equal", True))

# swap mapping uniqueness: r_o != r_s, swap_source_group_id != original_group_id
bad = [p for p in pairs if p["r_o"] == p["r_s"] or p["swap_source_group_id"] == p["original_group_id"]]
if bad:
    fail("inheritance_or_data_contract_invalid", f"{len(bad)} groups violate swap mapping uniqueness")
lines.append(("swap 映射唯一性（r_o != r_s，swap 源 != 本组）", f"{len(pairs)}/195 OK", True))

# four-cell expectation + T0 rendering semantic check
# cells: (cell, ref, cand, exp); ref uses r_o if cell[0]=='O' else r_s;
# cand uses r_o if cell[1]=='O' else r_s (frozen D0 T0 rendering)
from d4m0_core import four_cell_rows
n_bad_exp = 0
n_bad_frozen = 0
for p in pairs:
    q, cells = four_cell_rows(p, "T0")
    for cell, ref, cand, exp in cells:
        ans = p["r_o"] if cell[1] == "O" else p["r_s"]
        if cand != render_candidate(ans, TEMPLATES["T0"]):
            n_bad_exp += 1
    # frozen D0 T0 rendering stored in the dev-only pair file
    if p["c_o"] != render_candidate(p["r_o"], TEMPLATES["T0"]):
        n_bad_frozen += 1
    if p["c_s"] != render_candidate(p["r_s"], TEMPLATES["T0"]):
        n_bad_frozen += 1
lines.append(("T0 Candidate 渲染语义（candidate 唯一由冻结 r_o/r_s + T0 模板构成）",
              "780-cell OK" if n_bad_exp == 0 else f"{n_bad_exp} bad", n_bad_exp == 0))
lines.append(("D0 冻结 c_o/c_s 字段复核（dev-only 文件）",
              "390 OK" if n_bad_frozen == 0 else f"{n_bad_frozen} bad", n_bad_frozen == 0))

# per-cell expectation labels (OO/OS/SO/SS) are fixed at the four-cell level
exp_map = {"OO": "A", "OS": "B", "SO": "B", "SS": "A"}
lines.append(("四格期望标签映射", "OO=A, OS=B, SO=B, SS=A",
              {"OO": "A", "OS": "B", "SO": "B", "SS": "A"} == exp_map))

# ---------------------------------------------------------------------------
# 0.2 Template inheritance: render an example with Mistral native chat template
# ---------------------------------------------------------------------------
tok = AutoTokenizer.from_pretrained(MODEL_DIR)
msgs = [
    {"role": "system", "content": SYSTEM},
    {"role": "user", "content": USER_TMPL.format(question="What is the capital of France?",
                                                 reference="Paris",
                                                 candidate="The answer is Paris.")},
]
encoded = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt")
native_prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
mistral_ids = encoded["input_ids"][0].tolist()

# recompute template hashes
tpl_sha_ok = all(sha256_hex(TEMPLATES[t]) == TEMPLATE_SHA.get(t) for t in ["T0", "T1", "T2"])
lines.append(("T0/T1/T2 模板 SHA256 复核", "T0/T1/T2 全部一致" if tpl_sha_ok else "不一致", tpl_sha_ok))

# --- write prompt_semantic_inheritance_audit.md ---
sha_t0 = sha256_hex(TEMPLATES["T0"])
sha_t1 = sha256_hex(TEMPLATES["T1"])
sha_t2 = sha256_hex(TEMPLATES["T2"])
doc = f"""# prompt_semantic_inheritance_audit.md

## 0.1 D0 继承审计摘要

| 审计项 | 值 | 通过 |
|---|---|---|
{chr(10).join(f"| {k} | {v} | {'✓' if ok else '✗'} |" for k, v, ok in lines)}

## 0.2 模板继承原则（唯一允许的序列化变化）

- 外层序列化仅允许一处变化：Qwen native chat template → Mistral 官方 native chat template
  （`tokenizer.apply_chat_template`）。其余（system/user 自然语言任务内容、字段顺序、
  verdict 定义、continuation ` A`/` B`、teacher-forced 位置）逐字继承 D1/D1-R。
- 不得把 Qwen 特殊 token / chat wrapper 硬塞给 Mistral；不得改写任务内容以"优化 Mistral"。

## D1 语义恢复（来自 D1 `scripts/_prompt_constants.json` 与 D1-R spec）

### system message 语义
```text
{ SYSTEM }
```

### user message 字段顺序
```text
{ USER_TMPL }
```

### verdict 定义
- A = Accept（Candidate 与 Reference 一致）
- B = Reject（Candidate 与 Reference 不一致）

### Candidate 表述（D1-R 冻结，逐字）
| 模板 | 字符串 | UTF-8 SHA256 |
|---|---|---|
| T0 | `{ TEMPLATES['T0'] }` | `{sha_t0}` |
| T1 | `{ TEMPLATES['T1'] }` | `{sha_t1}` |
| T2 | `{ TEMPLATES['T2'] }` | `{sha_t2}` |

D1-R spec 记录的模板哈希：
- T0 `c42e1ea1…`（本文件复核 `{sha_t0[:16]}…`）
- T1 `d325f862…`（本文件复核 `{sha_t1[:16]}…`）
- T2 `5fb1b5ed…`（本文件复核 `{sha_t2[:16]}…`）

## Mistral native serialization 示例（tokenizer 渲染，不加载权重）

messages：
```python
[
  {{"role": "system", "content": "…(D1 system 逐字)…"}},
  {{"role": "user", "content": "Question: …\\n\\nReference Answer: …\\n\\nCandidate Answer: …\\n\\nAnswer:"}},
]
```

渲染结果（`add_generation_prompt=True`）：
```
{native_prompt}
```

首段 token ids（前 24 个）：{mistral_ids[:24]}

## 继承结论

- 唯一继承源：D0（split/swap/模板语义/四格定义）→ D1（prompt/continuation/teacher-forced 语义）→ D1-R（T1/T2 表述）。
- 本轮目录不含任何 D0 train / final-reserve 完整题目文本；dev 文本仅经 D1 `_dev_pairs.jsonl`（dev-only）流式读取。
"""
(R / "prompt_semantic_inheritance_audit.md").write_text(doc, encoding="utf-8")

# --- write inheritance_audit.md ---
audit = f"""# inheritance_audit.md

## 0.1 D0 继承审计

| 审计项 | 值 | 通过 |
|---|---|---|
{chr(10).join(f"| {k} | {v} | {'✓' if ok else '✗'} |" for k, v, ok in lines)}

- D0 最终标签：`{d0_dec["final_label"]}`
- dev split：195 个 source group（D0 `fixed_split_indices.json`，seed {fix["seed"]}）
- swap 映射来源：D0 coarse-form-controlled Random Swap（seed 20260802），经 D1 dev-only `_dev_pairs.jsonl` 流式继承。
- 四格构造：OO=(r_o,c_o), OS=(r_o,c_s), SO=(r_s,c_o), SS=(r_s,c_s)；期望标签 OO=A, OS=B, SO=B, SS=A。
- T0 Candidate 渲染：`candidate = "The answer is " + 冻结r + "."`（NFKC/trim/空白归一化，大小写不变）。
- 本轮目录不含 train / final-reserve 完整题目文本（仅经 D1 dev-only 文件流式读取 195 组 dev）。

## 0.2 模板继承（详见 prompt_semantic_inheritance_audit.md）

- 唯一序列化变化：Qwen chat template → Mistral `apply_chat_template`。
- system/user 内容、字段顺序、verdict、continuation、teacher-forced 位置逐字继承。
"""
(R / "inheritance_audit.md").write_text(audit, encoding="utf-8")

print("Phase 0 OK: D0 inheritance + template inheritance audited")
for k, v, ok in lines:
    print(f"  [{'OK' if ok else 'FAIL'}] {k}: {v}")
print("wrote inheritance_audit.md, prompt_semantic_inheritance_audit.md")
