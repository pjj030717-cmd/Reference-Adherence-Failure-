#!/usr/bin/env python3
"""D4-M0 Phase 1: Mistral A/B decision-channel semantic qualification.

1.1 continuation tokenization audit: " A" / " B" single tokens, distinct ids,
    no UNK, equal length.
1.2 teacher-forced implementation audit: logits at prompt_len-1; no prior
    correction / logit bias / post-processing. Records a code-path audit and
    one worked example (l_A, l_B, d_raw from pos=prompt_len-1 vs forbidden
    len(prompt)+len(cont)-1).
1.3 synthetic semantic regression on the 24 pairs inherited from D1 manifest
    (12 MATCH A + 12 MISMATCH B). Plus greedy first-token diagnostic.

Also records model-access audit (config/tokenizer/index hashes, revision,
peak GPU memory) before and after model load.
"""
from __future__ import annotations

import csv
import json
import math
import sys
import time
from pathlib import Path

import torch

from d4m0_core import R, D1, MODEL, SYSTEM, USER_TMPL, ACCEPT, REJECT
from d4m0_core import sha256_file, build_messages, score_prompt, classify, load_model

# ---------------------------------------------------------------------------
# model access audit (hashes computed BEFORE loading)
# ---------------------------------------------------------------------------
model_dir = Path(MODEL)
hash_targets = ["config.json", "tokenizer.json", "tokenizer_config.json",
                "model.safetensors.index.json"]
model_hashes = {}
for f in hash_targets:
    p = model_dir / f
    model_hashes[f] = sha256_file(p) if p.exists() else "MISSING"
revision = "not-available-locally"
rev_txt = model_dir / "REVISION.txt"
if rev_txt.exists():
    revision = rev_txt.read_text(encoding="utf-8").strip()

mem_start = torch.cuda.max_memory_allocated(device="cuda") if torch.cuda.is_available() else 0
print("model file hashes computed; revision =", revision)

# ---------------------------------------------------------------------------
# load model
# ---------------------------------------------------------------------------
t0 = time.time()
tok, model = load_model()
t_load = time.time() - t0
print("model loaded:", type(model).__name__,
      "params:", sum(p.numel() for p in model.parameters()) / 1e9, "B",
      "load_sec:", round(t_load, 1))

# ---------------------------------------------------------------------------
# 1.1 continuation tokenization audit
# ---------------------------------------------------------------------------
tok_a = tok.encode(ACCEPT, add_special_tokens=False)
tok_b = tok.encode(REJECT, add_special_tokens=False)
audit = {
    "accept_str": ACCEPT, "reject_str": REJECT,
    "accept_token_ids": tok_a, "reject_token_ids": tok_b,
    "accept_single": len(tok_a) == 1,
    "reject_single": len(tok_b) == 1,
    "ids_distinct": len(tok_a) == 1 and len(tok_b) == 1 and tok_a[0] != tok_b[0],
    "no_unk": all(i != tok.unk_token_id for i in tok_a + tok_b),
    "equal_length": len(tok_a) == len(tok_b),
}
print("tokenization audit:", json.dumps(audit, indent=2))

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

if not all([audit["accept_single"], audit["reject_single"], audit["ids_distinct"],
            audit["no_unk"], audit["equal_length"]]):
    fail("mistral_decision_channel_invalid", f"continuation tokenization: {json.dumps(audit)}")

# ---------------------------------------------------------------------------
# 1.2 teacher-forced implementation audit (worked example)
# ---------------------------------------------------------------------------
# Example prompt (synthetic, no D0 text)
msgs = build_messages("What is the capital city of France?", "Paris", "The answer is Paris.")
enc = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt")
prompt_len = enc["input_ids"].shape[1]
accept_id = tok_a[0]
reject_id = tok_b[0]
with torch.inference_mode():
    logits = model(enc["input_ids"].to("cuda")).logits
# correct teacher-forced position: prompt_len - 1
l_A_correct = logits[0, prompt_len - 1, accept_id].item()
l_B_correct = logits[0, prompt_len - 1, reject_id].item()
# forbidden positions (audit illustration only, never used for scoring):
#   (a) pos = prompt_len  would be out of bounds -> IndexError
#   (b) pos = prompt_len - 2 is a wrong "prompt-mid" position showing sensitivity
forbidden_out_of_bounds = False
try:
    _ = logits[0, prompt_len, accept_id].item()
except IndexError:
    forbidden_out_of_bounds = True
l_A_wrong = logits[0, prompt_len - 2, accept_id].item()
l_B_wrong = logits[0, prompt_len - 2, reject_id].item()

tf_audit = {
    "prompt_len": prompt_len,
    "accept_id": accept_id,
    "reject_id": reject_id,
    "pos_correct": prompt_len - 1,
    "pos_forbidden_len_plus_cont": prompt_len,  # out of bounds -> IndexError
    "pos_forbidden_out_of_bounds": forbidden_out_of_bounds,
    "pos_forbidden_demo": prompt_len - 2,       # wrong mid-prompt position; illustrative only
    "l_A_at_correct": l_A_correct,
    "l_B_at_correct": l_B_correct,
    "d_raw_correct": l_A_correct - l_B_correct,
    "l_A_at_forbidden": l_A_wrong,
    "l_B_at_forbidden": l_B_wrong,
    "d_raw_forbidden": l_A_wrong - l_B_wrong,
    "uses_prior_correction": False,
    "uses_logit_bias": False,
    "uses_postprocessing": False,
    "scoring_position_rule": "logits[:, prompt_len-1, :]",
}
print("teacher-forced audit:", json.dumps(tf_audit, indent=2))

# ---------------------------------------------------------------------------
# 1.3 synthetic semantic regression (24 pairs from D1 manifest)
# ---------------------------------------------------------------------------
SYN = json.loads((D1 / "synthetic_pair_manifest.json").read_text(encoding="utf-8"))
# manifest rows: [id, question, reference, candidate, expected_label]
assert len(SYN) == 24, f"synthetic manifest size {len(SYN)} != 24"

results = []
for sid, q, ref, cand, exp in SYN:
    msgs = build_messages(q, ref, cand)
    l_A, l_B, d_raw, p_accept, plen, gid, gtok = score_prompt(tok, model, msgs, accept_id, reject_id)
    pred = classify(d_raw)
    correct = pred == exp
    g = gtok.strip()
    if g == "A":
        greedy_pred = "A"
    elif g == "B":
        greedy_pred = "B"
    else:
        greedy_pred = f"OTHER({gtok!r})"
    greedy_agree = (greedy_pred == pred) and correct
    results.append({"id": sid, "question": q, "reference": ref, "candidate": cand,
                    "expected_label": exp, "l_A": l_A, "l_B": l_B, "d_raw": d_raw,
                    "p_accept_raw": p_accept, "prompt_len": plen,
                    "predicted_label": pred, "correct": correct,
                    "greedy_id": gid, "greedy_token": gtok,
                    "greedy_pred": greedy_pred, "greedy_agrees": greedy_agree})

# metrics
nA = sum(1 for r in results if r["expected_label"] == "A")
nB = sum(1 for r in results if r["expected_label"] == "B")
acc = sum(1 for r in results if r["correct"]) / len(results)
accA = sum(1 for r in results if r["expected_label"] == "A" and r["correct"]) / nA
accB = sum(1 for r in results if r["expected_label"] == "B" and r["correct"]) / nB
ties = sum(1 for r in results if r["predicted_label"] == "TIE")
medA = sorted(r["d_raw"] for r in results if r["expected_label"] == "A")[nA // 2]
medB = sorted(r["d_raw"] for r in results if r["expected_label"] == "B")[nB // 2]
greedy_n = sum(1 for r in results if r["greedy_agrees"])

print(f"\nsynthetic: acc={acc:.3f} ({int(acc*24)}/24) A={accA:.3f} B={accB:.3f} ties={ties}")
print(f"median d_raw A={medA:+.4f} B={medB:+.4f} greedy_agree={greedy_n}/24")

# write CSV + manifest (same 24 pairs, re-saved to this directory)
with open(R / "synthetic_readout_audit.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
    w.writeheader()
    w.writerows(results)
(R / "synthetic_pair_manifest.json").write_text(json.dumps(SYN, indent=2), encoding="utf-8")

# greedy_diagnostic.csv: greedy first-token per pair + agreement
with open(R / "greedy_diagnostic.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["id", "expected_label", "predicted_label",
                                      "greedy_id", "greedy_token", "greedy_pred",
                                      "greedy_agrees", "d_raw"])
    w.writeheader()
    for r in results:
        w.writerow({k: r[k] for k in ["id", "expected_label", "predicted_label",
                                      "greedy_id", "greedy_token", "greedy_pred",
                                      "greedy_agrees", "d_raw"]})

# ---- gate ----
if acc != 1.0 or accA != 1.0 or accB != 1.0:
    fail("mistral_decision_channel_invalid",
         f"synthetic acc={acc} A={accA} B={accB}; need 24/24, 12/12, 12/12")
if ties != 0:
    fail("mistral_decision_channel_invalid", f"{ties} ties")
if medA <= 0 or medB >= 0:
    fail("mistral_decision_channel_invalid", f"median d_raw A={medA} B={medB} wrong sign")
if greedy_n != len(results):
    fail("mistral_decision_channel_invalid", f"greedy agreement {greedy_n}/24 != 24/24")

# ---------------------------------------------------------------------------
# model access audit md
# ---------------------------------------------------------------------------
mem_peak = torch.cuda.max_memory_allocated(device="cuda")
mac = f"""# model_access_audit.md

## 模型
- 路径：`{MODEL}`
- 架构：MistralForCausalLM（Mistral-7B-Instruct-v0.3）
- revision（本地）：`{revision}`
- 加载：BF16、`model.eval()`、`torch.inference_mode()`、`batch_size=1`
- 加载耗时：{t_load:.1f}s

## 文件哈希（SHA256，加载前计算）
| 文件 | SHA256 |
|---|---|
{chr(10).join(f"| `{f}` | `{model_hashes[f]}` |" for f in hash_targets)}

## 显存
- 加载前峰值：{mem_start / 1e9:.2f} GB
- 评分后峰值：{mem_peak / 1e9:.2f} GB

## 访问结论
- 模型在本地、BF16 正常加载；无下载、无替换。
"""
(R / "model_access_audit.md").write_text(mac, encoding="utf-8")

# ---------------------------------------------------------------------------
# tokenization_audit.md / teacher_forcing_implementation_audit.md
# ---------------------------------------------------------------------------
tok_doc = f"""# tokenization_audit.md

## 1.1 continuation tokenization（Mistral tokenizer，重新验证，不信旧值）

| 项 | 值 |
|---|---|
| `"{ACCEPT}"` token ids | {audit['accept_token_ids']} |
| `"{REJECT}"` token ids | {audit['reject_token_ids']} |
| 均为单 token | {audit['accept_single']} / {audit['reject_single']} |
| token id 不同 | {audit['ids_distinct']} |
| 无 UNK | {audit['no_unk']} |
| token 长度相等 | {audit['equal_length']} |

accept_id = `{accept_id}`；reject_id = `{reject_id}`。

## 结论
- {"通过" if all([audit['accept_single'], audit['reject_single'], audit['ids_distinct'], audit['no_unk'], audit['equal_length']]) else "失败"}：A/B continuation 是公平、语义正确的决策通道。
"""
(R / "tokenization_audit.md").write_text(tok_doc, encoding="utf-8")

tf_doc = f"""# teacher_forcing_implementation_audit.md

## 1.2 正确 teacher-forced 实现

- 定义：`prompt_len = len(prompt input_ids)`
- A/B continuation 概率取 `logits[:, prompt_len - 1, :]`
- 严禁使用 `len(prompt) + len(continuation) - 1` 或已生成 ` A`/` B` 后的位置

## 实测（合成示例：Q=France 首都，ref=Paris，cand="The answer is Paris."）

| 项 | 值 |
|---|---|
| prompt_len | {tf_audit['prompt_len']} |
| accept_id / reject_id | {accept_id} / {reject_id} |
| 正确位置 pos = prompt_len-1 | {tf_audit['pos_correct']} |
| len(prompt)+len(cont)-1 位置（越界演示） | IndexError（越界={tf_audit['pos_forbidden_out_of_bounds']}） |
| 中间错误位置演示 pos=prompt_len-2 | {tf_audit['pos_forbidden_demo']} |
| l_A（正确位置） | {tf_audit['l_A_at_correct']:.4f} |
| l_B（正确位置） | {tf_audit['l_B_at_correct']:.4f} |
| d_raw（正确位置） | {tf_audit['d_raw_correct']:.4f} |
| l_A（中间错误位置，仅演示） | {tf_audit['l_A_at_forbidden']:.4f} |
| d_raw（中间错误位置，仅演示） | {tf_audit['d_raw_forbidden']:.4f} |

## 无任何额外处理
- prior correction：否（{tf_audit['uses_prior_correction']}）
- logit bias：否
- 后处理阈值：否
- prediction = A if d_raw > 0 else B（d_raw==0 → TIE 单独计数）

## 结论
- 实现符合固定定义，正式评分仅使用 `logits[:, prompt_len-1, :]`。
"""
(R / "teacher_forcing_implementation_audit.md").write_text(tf_doc, encoding="utf-8")

print("\nPhase 1 OK: decision channel qualified")
print("  token ids:", audit['accept_token_ids'], audit['reject_token_ids'])
print("  synthetic acc 24/24, A 12/12, B 12/12, ties 0")
print("wrote model_access_audit.md, tokenization_audit.md, teacher_forcing_implementation_audit.md,"
      " synthetic_readout_audit.csv, synthetic_pair_manifest.json, greedy_diagnostic.csv")
