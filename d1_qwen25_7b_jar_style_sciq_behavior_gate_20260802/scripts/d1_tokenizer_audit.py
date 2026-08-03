#!/usr/bin/env python3
"""E01-D1 tokenizer audit + prompt spec.
Verifies " A" / " B" continuation tokenization fairness BEFORE loading the model.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
import os
REPO_ROOT = Path(__file__).resolve().parents[2]

from transformers import AutoTokenizer

MODEL = os.environ.get("RAF_MODEL_DIR", "/root/autodl-tmp/models/Qwen2.5-7B-Instruct")
D1 = REPO_ROOT / "d1_qwen25_7b_jar_style_sciq_behavior_gate_20260802"

SYSTEM = (
    "You are a strict QA evaluator.\n"
    "Judge only whether the Candidate Answer agrees with the provided Reference Answer for the given Question.\n"
    "The provided Reference Answer is authoritative even if it conflicts with your own world knowledge.\n"
    "Output exactly one label and nothing else:\n"
    'A = the Candidate Answer agrees with the current Reference Answer.\n'
    'B = the Candidate Answer does not agree with the current Reference Answer.'
)
USER_TMPL = (
    "Question: {question}\n"
    "\n"
    "Reference Answer: {reference}\n"
    "\n"
    "Candidate Answer: {candidate}\n"
    "\n"
    "Answer:"
)
ACCEPT = " A"
REJECT = " B"


def fail(label: str, why: str):
    print("STOP:", label, "-", why)
    (D1 / "artifacts").mkdir(parents=True, exist_ok=True)
    (D1 / "artifacts" / "decision.json").write_text(
        json.dumps({"final_label": label, "reason": why,
                    "hidden_states_read": False, "probe_trained": False,
                    "activation_intervention_run": False, "final_reserve_model_scored": False,
                    "mistral_loaded": False, "prompt_variants_run": False}, indent=2),
        encoding="utf-8")
    sys.exit(1)


tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=False)
print("tokenizer loaded; pad token:", repr(tok.pad_token),
      "eos:", repr(tok.eos_token), "chat_template present:", tok.chat_template is not None)

# build a probe prompt through the exact chat template to inspect continuation ids
probe_messages = [
    {"role": "system", "content": SYSTEM},
    {"role": "user", "content": USER_TMPL.format(question="Q", reference="R", candidate="C")},
]
prompt_ids = tok.apply_chat_template(probe_messages, tokenize=True, add_generation_prompt=True)
print("prompt_ids length:", len(prompt_ids), "first ids:", prompt_ids[:8])

# continuation tokenization
ids_a = tok.encode(ACCEPT)
ids_b = tok.encode(REJECT)
print("encode(' A') ->", ids_a, "decode:", [tok.decode(i) for i in ids_a])
print("encode(' B') ->", ids_b, "decode:", [tok.decode(i) for i in ids_b])

# decode check: does " A" produce exactly one token? 
toks_a = tok.tokenize(ACCEPT)
toks_b = tok.tokenize(REJECT)
print("tokenize(' A'):", toks_a)
print("tokenize(' B'):", toks_b)

if len(ids_a) != 1 or len(ids_b) != 1:
    fail("decision_channel_tokenization_invalid", f"' A' ids={ids_a}, ' B' ids={ids_b}, not single token each")
if len(toks_a) != 1 or len(toks_b) != 1:
    fail("decision_channel_tokenization_invalid", "tokenize() not single token")
if ids_a == ids_b:
    fail("decision_channel_tokenization_invalid", "A and B map to same token id")
# UNK check
unk_id = tok.unk_token_id
if ids_a[0] == unk_id or ids_b[0] == unk_id:
    fail("decision_channel_tokenization_invalid", "continuation is UNK")
# lengths equal (both 1)
print(f"continuation audit OK: A id={ids_a[0]} (decode={tok.decode(ids_a[0])!r}), B id={ids_b[0]} (decode={tok.decode(ids_b[0])!r})")

# prompt spec md
def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

spec_text = f"""# prompt_spec.md

## 固定 System Message

```text
{SYSTEM}
```

## 固定 User Message 模板

```text
Question: {{question}}

Reference Answer: {{reference}}

Candidate Answer: {{candidate}}

Answer:
```

## 固定 Continuation

```text
accept = " A"
reject = " B"
```

## 使用规则

- 使用 Qwen 官方 chat template（tokenizer.apply_chat_template, add_generation_prompt=True）。
- 无 CoT、few-shot、direct repair、rubric、prompt 优化。
- 字段顺序、字段名称、system message、标点、换行、continuation 均固定。

## Prompt 原文 UTF-8 SHA256

（对上述 system + user 模板 + continuation 拼接文本计算）

- system_sha256 = `{sha256_hex(SYSTEM)}`
- user_template_sha256 = `{sha256_hex(USER_TMPL)}`
- accept_sha256 = `{sha256_hex(ACCEPT)}`
- reject_sha256 = `{sha256_hex(REJECT)}`
"""
(D1 / "prompt_spec.md").write_text(spec_text, encoding="utf-8")

# save constants for later scripts
(D1 / "scripts" / "_prompt_constants.json").write_text(json.dumps({
    "system": SYSTEM, "user_template": USER_TMPL, "accept": ACCEPT, "reject": REJECT,
    "accept_id": ids_a[0], "reject_id": ids_b[0],
}, indent=2), encoding="utf-8")

# tokenization_audit.md
tok_audit = f"""# tokenization_audit.md

## Continuation Tokenization 审计

| 项 | accept " A" | reject " B" |
|---|---|---|
| tokenize() | {toks_a} | {toks_b} |
| encode() | {ids_a} | {ids_b} |
| decode | {tok.decode(ids_a[0])!r} | {tok.decode(ids_b[0])!r} |
| 单 token | {'是' if len(ids_a)==1 else '否'} | {'是' if len(ids_b)==1 else '否'} |
| continuation length | {len(ids_a)} | {len(ids_b)} |
| UNK | {'是' if ids_a[0]==unk_id else '否'} | {'是' if ids_b[0]==unk_id else '否'} |

- 两者 continuation length 相同：{len(ids_a) == len(ids_b)}（均为 1）
- token id 不同：{ids_a[0] != ids_b[0]}
- 无 UNK / 空序列 / 额外 special token：{len(ids_a)==1 and len(ids_b)==1 and ids_a[0]!=unk_id and ids_b[0]!=unk_id}

**结论：decision channel 公平可评分。**
"""
(D1 / "tokenization_audit.md").write_text(tok_audit, encoding="utf-8")
print("wrote prompt_spec.md, tokenization_audit.md, _prompt_constants.json")
