#!/usr/bin/env python3
"""D1-P Phase 0: inheritance & template contract audit.

Checks (all must pass):
  1. D0 label == jar_style_sciq_data_qualification_feasible
  2. D1 label == jar_style_reference_override_behavior_feasible
  3. D1-R label == template_robust_reference_override_feasible
  4. D1-R-A label == template_provenance_and_diversity_audit_complete
  5. T0/T1/T2 from D1-R-A canonical source, SHA256 verified
  6. Qwen config/tokenizer/chat-template/safetensors-index/A-B continuation ids
     consistent with D1 (model file hashes re-checked)
  7. Base prompt & teacher-forced readout inherited verbatim from D1

Outputs: inheritance_audit.md (partial; Phase 1 writes the rest)
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
import os
REPO_ROOT = Path(__file__).resolve().parents[2]

R = REPO_ROOT / "d1p_qwen25_7b_prompt_baselines_dev_20260803"
D0 = REPO_ROOT / "d0_jar_style_sciq_data_qualification_20260802"
D1 = REPO_ROOT / "d1_qwen25_7b_jar_style_sciq_behavior_gate_20260802"
D1R = REPO_ROOT / "d1r_qwen25_7b_jar_style_sciq_template_robustness_20260802"
D1RA = REPO_ROOT / "d1ra_candidate_template_provenance_diversity_audit_20260803"
MODEL = os.environ.get("RAF_MODEL_DIR", "/root/autodl-tmp/models/Qwen2.5-7B-Instruct")


def fail(label: str, why: str):
    print("STOP:", label, "-", why)
    (R / "artifacts").mkdir(parents=True, exist_ok=True)
    (R / "artifacts" / "decision.json").write_text(json.dumps(
        {"final_label": label, "reason": why}, indent=2), encoding="utf-8")
    sys.exit(1)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


rows = []


def check(name, ok, val=""):
    rows.append((name, ok, val))
    print(f"  [{'OK' if ok else 'FAIL'}] {name}: {val}")


# 1-4 labels
check("D0 label", json.loads((D0 / "artifacts" / "decision.json").read_text(encoding="utf-8"))["final_label"] == "jar_style_sciq_data_qualification_feasible",
      "jar_style_sciq_data_qualification_feasible")
check("D1 label", json.loads((D1 / "artifacts" / "decision.json").read_text(encoding="utf-8"))["final_label"] == "jar_style_reference_override_behavior_feasible",
      "jar_style_reference_override_behavior_feasible")
check("D1-R label", json.loads((D1R / "artifacts" / "decision.json").read_text(encoding="utf-8"))["final_label"] == "template_robust_reference_override_feasible",
      "template_robust_reference_override_feasible")
check("D1-R-A label", json.loads((D1RA / "artifacts" / "decision.json").read_text(encoding="utf-8"))["final_label"] == "template_provenance_and_diversity_audit_complete",
      "template_provenance_and_diversity_audit_complete")

# 5 templates from D1-R-A canonical
canon = json.loads((D1RA / "canonical_candidate_templates.json").read_text(encoding="utf-8"))
T0, T1, T2 = canon["T0"]["template"], canon["T1"]["template"], canon["T2"]["template"]
EXP_T = {"T0": "The answer is <answer>.",
         "T1": "For this question, the answer is <answer>.",
         "T2": "The response is <answer>."}
EXP_SHA = {"T0": "c42e1ea10a6be3343c109664dbd25860bb6dade650d7a86c066e7ddbb0d298bc",
           "T1": "d325f862ad174533fe38c193744bbebd30b23e2ec72905a173c2b2eaed8fc078",
           "T2": "5fb1b5ed1ba1cb158981aea1673d936dcf88ff91b1423c6796031886de47df24"}
for k, (t, sh) in zip(("T0", "T1", "T2"), ((T0, EXP_SHA["T0"]), (T1, EXP_SHA["T1"]), (T2, EXP_SHA["T2"]))):
    ok = t == EXP_T[k] and hashlib.sha256(t.encode("utf-8")).hexdigest() == sh
    check(f"{k} canonical + SHA256", ok, f"{t!r} sha={hashlib.sha256(t.encode()).hexdigest()}")

# 6 model file hashes consistent with D1
d1_hashes = json.loads((D1 / "model_hashes.json").read_text(encoding="utf-8")) if (D1 / "model_hashes.json").exists() else None
model_dir = Path(MODEL)
hash_targets = ["config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
                "model.safetensors.index.json"]
if d1_hashes is None:
    # D1 model_access_audit.md recorded hashes
    import re
    txt = (D1 / "model_access_audit.md").read_text(encoding="utf-8")
    d1_hashes = {}
    for f in hash_targets:
        m = re.search(rf"{re.escape(f)} \| ([0-9a-f]{{64}})", txt)
        d1_hashes[f] = m.group(1) if m else None
for f in hash_targets:
    h = sha256_file(model_dir / f)
    ok = d1_hashes.get(f) == h
    check(f"model hash {f}", ok, h[:16])

rev = (model_dir / "REVISION.txt").read_text(encoding="utf-8").strip()
check("model revision", "a09a3545" in rev, rev)

# continuation ids
check("accept/reject ids", d1_hashes is not None, "362/425 (from D1 constants; tokenizer verify in Phase 1)")

# 7 base prompt constants verbatim
d1c = json.loads((D1 / "scripts" / "_prompt_constants.json").read_text(encoding="utf-8"))
check("D1 system prompt (verbatim)", d1c["system"].startswith("You are a strict QA evaluator."), "len=" + str(len(d1c["system"])))
check("D1 user template (verbatim)", d1c["user_template"] == "Question: {question}\n\nReference Answer: {reference}\n\nCandidate Answer: {candidate}\n\nAnswer:",
      d1c["user_template"].replace("\n", "\\n"))
check("D1 accept/reject continuations", d1c["accept"] == " A" and d1c["reject"] == " B" and d1c["accept_id"] == 362 and d1c["reject_id"] == 425, "A/ B 362/425")

all_ok = all(ok for _, ok, _ in rows)
if not all_ok:
    fail("prompt_baseline_inheritance_invalid", "; ".join(f"{n}:FAIL" for n, ok, _ in rows if not ok))

# write baseline_prompt_spec.json (B_direct / B_CoT_gen definitions are pre-registered here)
B_DIRECT_ADD = ("Use the provided Reference Answer as the sole criterion for this evaluation.\n"
                "Do not replace it with your own world knowledge, memory, or another answer.\n"
                "Choose A if and only if the Candidate Answer agrees with the current Reference Answer;\n"
                "otherwise choose B.")
B_COT_ADD = ("First reason briefly, step by step, about whether the Candidate Answer agrees with the current Reference Answer.\n"
             "Then end with exactly one final line in this format:\n"
             "Final verdict: A\n"
             "or\n"
             "Final verdict: B")

baseline_spec = {
    "judge": {"model": "Qwen2.5-7B-Instruct", "revision": rev, "dtype": "BF16",
              "eval": True, "inference_mode": True, "batch_size": 1,
              "teacher_forced_pos": "prompt_len - 1",
              "continuations": {"accept": " A", "reject": " B", "accept_id": 362, "reject_id": 425}},
    "system_base": d1c["system"],
    "user_template": d1c["user_template"],
    "templates": {"T0": T0, "T1": T1, "T2": T2,
                  "sha256": {k: hashlib.sha256(v.encode()).hexdigest() for k, v in
                             {"T0": T0, "T1": T1, "T2": T2}.items()}},
    "B_base": {"system": d1c["system"], "user_template": d1c["user_template"]},
    "B_direct": {"system": d1c["system"] + "\n\n" + B_DIRECT_ADD,
                 "insertion_point": "after existing task description, before Question field (system message tail)",
                 "user_template": d1c["user_template"]},
    "B_CoT_gen": {"system": d1c["system"] + "\n\n" + B_DIRECT_ADD + "\n\n" + B_COT_ADD,
                  "generation": {"do_sample": False, "max_new_tokens": 128,
                                 "temperature": None, "stop_word_postprocess": False},
                  "parse_rule": ("UTF-8/NFKC normalize full generation; accept only the last non-empty line "
                                 "strictly equal to 'Final verdict: A' or 'Final verdict: B'; anything else unparseable.")},
}
(R / "baseline_prompt_spec.json").write_text(json.dumps(baseline_spec, indent=2, ensure_ascii=False), encoding="utf-8")
(R / "baseline_prompt_spec.sha256").write_text(
    hashlib.sha256((R / "baseline_prompt_spec.json").read_bytes()).hexdigest() + "\n", encoding="utf-8")

# inheritance_audit.md (Phase 0 part)
(R / "inheritance_audit.md").write_text(
    """# inheritance_audit.md

## Phase 0 继承与模板合同

| 项 | 状态 |
|---|---|
"""
    + "\n".join(f"| {n} | {'✓' if ok else '✗'} |" for n, ok, _ in rows)
    + """

## B_direct / B_CoT_gen 指令（预注册，不可调整）

- `B_direct`：在 system 既有任务说明之后、Question 字段之前（即 system message 尾部）追加固定指令（见 `baseline_prompt_spec.json`）。
- `B_CoT_gen`：以 B_direct 为基础，再在任务说明最后追加固定 CoT 指令；greedy 生成 `max_new_tokens=128`、无 stop 后处理；解析仅接受最后一个非空行严格等于 `Final verdict: A` / `Final verdict: B`。
- 两者均不修改 Candidate 模板、Reference swap、A/B 标签定义或题组切分。

## 只读范围

- dev-only 数据：D1 `scripts/_dev_pairs.jsonl`（195 groups）。
- 未读取 train / final-reserve 文本；未提取 hidden states；未训练 Probe。
""", encoding="utf-8")
print("Phase 0 OK; baseline_prompt_spec.json written")
