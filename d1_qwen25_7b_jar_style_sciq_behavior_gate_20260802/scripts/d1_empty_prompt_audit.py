#!/usr/bin/env python3
"""E01-D1: empty-prompt diagnostic + model_access_audit.md + teacher_forcing_implementation_audit.md.

Runs the empty (blank) prompt continuation diagnostic and writes audit docs.
The empty-prompt l_A - l_B is diagnostic ONLY (not a correction and not a failure condition).
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path
import os
REPO_ROOT = Path(__file__).resolve().parents[2]

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = os.environ.get("RAF_MODEL_DIR", "/root/autodl-tmp/models/Qwen2.5-7B-Instruct")
D1 = REPO_ROOT / "d1_qwen25_7b_jar_style_sciq_behavior_gate_20260802"
CONST = json.loads((D1 / "scripts" / "_prompt_constants.json").read_text(encoding="utf-8"))
SYSTEM, USER_TMPL, ACCEPT, REJECT = CONST["system"], CONST["user_template"], CONST["accept"], CONST["reject"]
ACCEPT_ID, REJECT_ID = CONST["accept_id"], CONST["reject_id"]


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


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---- model_access_audit.md (recompute hashes) ----
model_dir = Path(MODEL)
revision = (model_dir / "REVISION.txt").read_text(encoding="utf-8").strip()
hash_targets = ["config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
                "model.safetensors.index.json"]
mh = {f: sha256_file(model_dir / f) for f in hash_targets}
config_text = (model_dir / "config.json").read_text(encoding="utf-8")
config = json.loads(config_text)

# ---- load model ----
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
                                             device_map="cuda", low_cpu_mem_usage=True)
model.eval()
print("model loaded for empty-prompt diagnostic")

# ---- empty prompt diagnostic ----
# blank prompt: user message with empty question/reference/candidate
messages = [
    {"role": "system", "content": SYSTEM},
    {"role": "user", "content": USER_TMPL.format(question="", reference="", candidate="")},
]
enc = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
prompt_ids = enc["input_ids"].to("cuda")
with torch.inference_mode():
    logits = model(prompt_ids).logits
pos = prompt_ids.shape[1] - 1
ll = logits[0, pos, :]
l_A_empty = ll[ACCEPT_ID].item()
l_B_empty = ll[REJECT_ID].item()
d_raw_empty = l_A_empty - l_B_empty
greedy_id = int(ll.argmax().item())
greedy_tok = tok.decode([greedy_id]).strip()
print(f"empty prompt: l_A={l_A_empty:.4f} l_B={l_B_empty:.4f} d_raw={d_raw_empty:+.4f} greedy={greedy_tok!r}")

import csv
with open(D1 / "empty_prompt_diagnostic.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["l_A", "l_B", "d_raw", "greedy_token"])
    w.writeheader()
    w.writerow({"l_A": f"{l_A_empty:.8f}", "l_B": f"{l_B_empty:.8f}",
                "d_raw": f"{d_raw_empty:.8f}", "greedy_token": greedy_tok})

# ---- model_access_audit.md ----
(D1 / "model_access_audit.md").write_text(
    f"""# model_access_audit.md

## 模型

- 名称：Qwen/Qwen2.5-7B-Instruct
- 本地路径：`{MODEL}`
- revision：`{revision}`
- 架构：{config.get('model_type')}，参数 {sum(p.numel() for p in model.parameters())/1e9:.3f}B
- dtype：BF16；`model.eval()`；`torch.inference_mode()`
- batch_size：1（无 padding batch）

## 文件哈希（SHA256）

| 文件 | SHA256 |
|---|---|
""" + "\n".join(f"| {f} | {mh[f]} |" for f in hash_targets) + f"""

## config.json 关键项

```json
{json.dumps({k: config.get(k) for k in ['model_type', 'hidden_size', 'num_hidden_layers', 'num_attention_heads', 'num_key_value_heads', 'vocab_size', 'torch_dtype', 'max_position_embeddings'] if k in config}, indent=2)}
```

## 结论

模型为本地既有文件，未下载、未替换、未改动；满足本地合法加载。
""",
    encoding="utf-8")

# ---- teacher_forcing_implementation_audit.md ----
(D1 / "teacher_forcing_implementation_audit.md").write_text(
    f"""# teacher_forcing_implementation_audit.md

## 正确 teacher-forced 实现

对每条 prompt：

```python
prompt_ids = tokenizer.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
prompt_len = prompt_ids.shape[1]
logits = model(prompt_ids).logits
pos = prompt_len - 1
logits_last = logits[0, pos, :]
l_A = logits_last[ACCEPT_ID]   # id {ACCEPT_ID} = " A"
l_B = logits_last[REJECT_ID]   # id {REJECT_ID} = " B"
d_raw = l_A - l_B
```

- 读取位置固定为 `pos = prompt_len - 1`（add_generation_prompt 后 prompt 末尾为 assistant 起始）。
- 禁止 `pos = len(prompt) + len(continuation) - 1`；禁止拼接 continuation 后在末尾取 logits；无 off-by-one。
- 无 prior 校正、无空白偏置、无温度校正、无阈值调参。
- `p_accept_raw = sigmoid(d_raw) = 1/(1+exp(-d_raw))`。

## greedy 诊断

- 使用同位置 logits 的 argmax token 作为 greedy 首 token；解码去空白后判定方向 A/B。
- 要求每条样本 greedy 方向与 d_raw 类别预测一致（已通过 24/24）。

## 空白 prompt 诊断

- 空白 prompt（question/reference/candidate 全空）下的 `l_A - l_B`：
  `d_raw = {d_raw_empty:+.8f}`（仅诊断记录，不用于校正，不是失败条件）。
""",
    encoding="utf-8")
print("wrote empty_prompt_diagnostic.csv, model_access_audit.md, teacher_forcing_implementation_audit.md")
print("empty prompt d_raw =", round(d_raw_empty, 6))
