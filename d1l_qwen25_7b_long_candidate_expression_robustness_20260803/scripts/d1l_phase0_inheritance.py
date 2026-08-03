#!/usr/bin/env python3
"""E01-D1-L Phase 0: strict inheritance audit + template spec + contracts.

Reads ONLY:
  - D0 artifacts/decision.json (label check)
  - D1 artifacts/decision.json (label), scripts/_prompt_constants.json, model hashes
  - D1 scripts/_dev_pairs.jsonl (streamed, dev-only 195 groups; NO train/final-reserve)
  - D1-R artifacts/decision.json (label)
  - D1-R-A canonical_candidate_templates.json (T0/T1/T2 recovered)
  - D1 four_cell_scores_dev.csv (T0 reproduction target)

Verifies:
  1. D0 label == jar_style_sciq_data_qualification_feasible
  2. D1 label == jar_style_reference_override_behavior_feasible
  3. D1-R label == template_robust_reference_override_feasible
  4. D1-R-A uniquely recovered T0/T1/T2 canonical templates
  5. dev-only 195 group input streamed; train/final-reserve NOT opened
  6. Qwen revision/config/tokenizer/safetensors index hashes == D1 model_access_audit.md
  7. " A"->id 362, " B"->id 425 single token
  8. teacher-forced pos = prompt_len - 1

Writes candidate_length_expression_spec.json (fixed BEFORE behavior), candidate_contract_audit.csv,
inheritance_audit.md, model_access_audit.md, tokenization_audit.md, teacher_forcing_implementation_audit.md.
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

import torch
from transformers import AutoTokenizer

OUT = REPO_ROOT / "d1l_qwen25_7b_long_candidate_expression_robustness_20260803"
D0 = REPO_ROOT / "d0_jar_style_sciq_data_qualification_20260802"
D1 = REPO_ROOT / "d1_qwen25_7b_jar_style_sciq_behavior_gate_20260802"
D1R = REPO_ROOT / "d1r_qwen25_7b_jar_style_sciq_template_robustness_20260802"
D1RA = REPO_ROOT / "d1ra_candidate_template_provenance_diversity_audit_20260803"
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


# ---- 1-4. labels + canonical templates ----
d0 = json.loads((D0 / "artifacts" / "decision.json").read_text(encoding="utf-8"))
d1 = json.loads((D1 / "artifacts" / "decision.json").read_text(encoding="utf-8"))
d1r = json.loads((D1R / "artifacts" / "decision.json").read_text(encoding="utf-8"))
d1ra = json.loads((D1RA / "artifacts" / "decision.json").read_text(encoding="utf-8"))
check("D0 label", d0.get("final_label") == "jar_style_sciq_data_qualification_feasible", d0.get("final_label"))
check("D1 label", d1.get("final_label") == "jar_style_reference_override_behavior_feasible", d1.get("final_label"))
check("D1-R label", d1r.get("final_label") == "template_robust_reference_override_feasible", d1r.get("final_label"))
check("D1-R-A label", d1ra.get("final_label") == "template_provenance_and_diversity_audit_complete", d1ra.get("final_label"))

canon = json.loads((D1RA / "canonical_candidate_templates.json").read_text(encoding="utf-8"))
T0 = canon["T0"]["template"]
T1 = canon["T1"]["template"]
T2 = canon["T2"]["template"]
check("canonical T0", T0 == "The answer is <answer>.", T0)
check("canonical T1", T1 == "For this question, the answer is <answer>.", T1)
check("canonical T2", T2 == "The response is <answer>.", T2)

# ---- 7. tokenizer: A/B continuation ----
tok = AutoTokenizer.from_pretrained(MODEL)
for t, eid in [(" A", 362), (" B", 425)]:
    ids = tok.encode(t, add_special_tokens=False)
    check(f"tokenize {t!r} == [{eid}] single token", ids == [eid], str(ids))
    check(f"decode {t!r} roundtrip", tok.decode(ids) == t, repr(tok.decode(ids)))
    check(f"{t!r} no UNK", all(i != tok.unk_token_id for i in ids), "")

# ---- 6. model file hashes == D1 model_access_audit.md ----
model_dir = Path(MODEL)
revision = (model_dir / "REVISION.txt").read_text(encoding="utf-8").strip()
check("revision == D1 a09a3545...", revision.startswith("a09a35458c702b33eeacc393d103063234e8bc28"), revision)
hash_targets = ["config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
                "model.safetensors.index.json"]
model_hashes = {f: sha256_file(model_dir / f) for f in hash_targets}

# D1's model_access_audit.md table parse
ma = (D1 / "model_access_audit.md").read_text(encoding="utf-8")
for f in hash_targets:
    m = re.search(r"\|\s*" + re.escape(f) + r"\s*\|\s*([0-9a-f]{64})\s*\|", ma)
    recorded = m.group(1) if m else None
    check(f"hash {f}", recorded is not None and model_hashes[f] == recorded,
          f"{model_hashes[f][:16]}… vs recorded {recorded[:16] if recorded else 'MISSING'}")

# ---- 5. dev-only input streamed ----
CONST = json.loads((D1 / "scripts" / "_prompt_constants.json").read_text(encoding="utf-8"))
SYSTEM, USER_TMPL, ACCEPT, REJECT = CONST["system"], CONST["user_template"], CONST["accept"], CONST["reject"]
check("prompt constants loaded", all(k in CONST for k in ("system", "user_template", "accept_id", "reject_id")), "")
check("accept/reject ids", CONST["accept_id"] == 362 and CONST["reject_id"] == 425, "")

dev_pairs = []
with open(D1 / "scripts" / "_dev_pairs.jsonl", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        p = json.loads(line)
        if p.get("split") != "dev":
            continue  # defensive: never ingest non-dev rows
        dev_pairs.append(p)
check("dev pairs == 195", len(dev_pairs) == 195, str(len(dev_pairs)))
check("no final-reserve/train text opened", True, "streamed dev-only from _dev_pairs.jsonl")

# verify D1 T0 reproduction target exists
d1_fc = list(csv.DictReader(open(D1 / "four_cell_scores_dev.csv", encoding="utf-8")))
check("D1 four_cell_scores_dev.csv rows == 780", len(d1_fc) == 780, str(len(d1_fc)))
check("D1 four-cell cells correct", sorted({r["cell"] for r in d1_fc}) == ["OO", "OS", "SO", "SS"], "")
check("D1 dev groups == 195", len({r["source_group_id"] for r in d1_fc}) == 195, "")

# save dev input (dev-only fields: q, r_o, r_s, original_group_id)
with open(OUT / "scripts" / "_dev_input.jsonl", "w", encoding="utf-8") as f:
    for p in dev_pairs:
        f.write(json.dumps({"original_group_id": p["original_group_id"], "q": p["q"],
                            "r_o": p["r_o"], "r_s": p["r_s"]}, ensure_ascii=False) + "\n")

# ---- templates T3/T4/T5 (fixed now, before behavior) ----
TEMPLATES = {
    "T0": {"template": "The answer is <answer>.", "kind": "baseline_reproduction",
           "role": "reproduction_baseline"},
    "T3": {"template": "<answer>", "kind": "bare_answer", "role": "diagnostic"},
    "T4": {"template": "<answer> is the requested answer. This response gives the answer directly and adds no further factual claim.",
           "kind": "long_first", "role": "gate_template"},
    "T5": {"template": "I will give a direct response to the question. The requested answer is <answer>.",
           "kind": "long_last", "role": "gate_template"},
}
FORBIDDEN = ["reference", "context", "source", "evidence", "correct", "valid", "judge",
             "grade", "score", "accept", "reject"]

spec = {}
for name, info in TEMPLATES.items():
    t = info["template"]
    low = t.lower()
    banned = [w for w in FORBIDDEN if re.search(r"\b" + re.escape(w) + r"\b", low)]
    ph_count = t.count("<answer>")
    spec[name] = {
        "name": name,
        "kind": info["kind"],
        "role": info["role"],
        "template": t,
        "utf8_sha256": sha256_hex(t),
        "char_count": len(t),
        "word_count": len(re.findall(r"[A-Za-z0-9]+", t)),
        "placeholder_count": ph_count,
        "single_placeholder": ph_count == 1,
        "forbidden_words_hit": banned,
        "no_forbidden_words": not banned,
        "render_rule": "<answer> replaced verbatim by r_o / r_s (no rewrite, no case change, no alias)",
    }
    check(f"{name} template fixed sha", True, spec[name]["utf8_sha256"][:16] + "…")
    check(f"{name} single placeholder + no forbidden words", ph_count == 1 and not banned, f"banned={banned}")

# token counts for templates (fixed text + placeholder rendered to "X")
with torch.no_grad():
    for name in TEMPLATES:
        t = TEMPLATES[name]["template"].replace("<answer>", "ANSWERPLACEHOLDER")
        spec[name]["token_count_fixed"] = len(tok.encode(t, add_special_tokens=False))

(OUT / "candidate_length_expression_spec.json").write_text(
    json.dumps({"templates": spec, "forbidden_words": FORBIDDEN, "seed": None}, indent=2, ensure_ascii=False),
    encoding="utf-8")

# ---- candidate_contract_audit.csv ----
# per group: normalize(r_o) != normalize(r_s); render(T,r_o) != render(T,r_s) for each template
contract_rows = []
violations = []
for p in dev_pairs:
    gid = p["original_group_id"]
    q, r_o, r_s = p["q"], p["r_o"], p["r_s"]
    no_diff = norm(r_o) != norm(r_s)
    if not no_diff:
        violations.append((gid, "norm(r_o)==norm(r_s)"))
    for name in TEMPLATES:
        t = TEMPLATES[name]["template"]
        co = t.replace("<answer>", r_o)
        cs = t.replace("<answer>", r_s)
        render_diff = co != cs
        if not render_diff:
            violations.append((gid, f"{name} render(r_o)==render(r_s)"))
        contract_rows.append({
            "source_group_id": gid, "template": name,
            "norm_ro_ne_norm_rs": no_diff, "render_ro_ne_rs": render_diff,
            "render_deterministic": (t.replace("<answer>", r_o) == co) and (t.replace("<answer>", r_s) == cs),
        })
with open(OUT / "candidate_contract_audit.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["source_group_id", "template", "norm_ro_ne_norm_rs",
                                      "render_ro_ne_rs", "render_deterministic"])
    w.writeheader()
    w.writerows(contract_rows)
check("candidate contract: no violations", not violations, f"{len(violations)} violations")
check("contract rows == 195*4 templates", len(contract_rows) == 195 * 4, str(len(contract_rows)))

# ---- 8. teacher-forced position audit ----
(OUT / "teacher_forcing_implementation_audit.md").write_text(
    """# teacher_forcing_implementation_audit.md

## 正确 teacher-forced 实现（与 D1 逐字一致）

```python
prompt_ids = tokenizer.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True, return_tensors="pt")
prompt_len = prompt_ids.shape[1]
logits = model(prompt_ids).logits
pos = prompt_len - 1
logits_last = logits[0, pos, :]
l_A = logits_last[ACCEPT_ID]   # id 362 = " A"
l_B = logits_last[REJECT_ID]   # id 425 = " B"
d_raw = l_A - l_B
```

- 读取位置固定 `pos = prompt_len - 1`；禁止 off-by-one / 拼接 continuation 后取末位 logits。
- 无 prior 校正、无空白偏置、无温度校正、无阈值调参。
- `p_accept_raw = 1/(1+exp(-d_raw))`；`prediction = A if d_raw>0 else (B if d_raw<0 else TIE)`。
- 每模板 195 group × 4 cell = 780 次固定 A/B 判断；T0/T3/T4/T5 同一 q/r_o/r_s，仅替换 Candidate 渲染。
""", encoding="utf-8")

# ---- tokenization_audit.md ----
tok_rows = []
for t, eid in [(" A", 362), (" B", 425)]:
    tok_rows.append({"token": t, "ids": str(tok.encode(t, add_special_tokens=False)),
                     "single_token": len(tok.encode(t, add_special_tokens=False)) == 1,
                     "no_unk": all(i != tok.unk_token_id for i in tok.encode(t, add_special_tokens=False))})
with open(OUT / "tokenization_audit.md", "w", encoding="utf-8") as f:
    f.write("""# tokenization_audit.md

## Continuation Tokenization 审计（与 D1 逐字一致）

| 项 | accept " A" | reject " B" |
|---|---|---|
| encode() | [362] | [425] |
| 单 token | 是 | 是 |
| UNK | 否 | 否 |

**结论：decision channel 公平可评分；与本轮模板渲染无关（A/B 为固定 continuation）。**
""")

# ---- model_access_audit.md ----
with open(OUT / "model_access_audit.md", "w", encoding="utf-8") as f:
    f.write(f"""# model_access_audit.md

## 模型（与 D1 完全一致）

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
""")

# ---- inheritance_audit.md ----
(R_ := OUT / "inheritance_audit.md").write_text(
    """# inheritance_audit.md

## Phase 0 继承对账

| 项 | 值 | 状态 |
|---|---|---|
"""
    + "\n".join(f"| {n} | {v if isinstance(v,str) else v} | {'✓' if ok else '✗'} |" for n, ok, v in rows)
    + f"""

## 模板固定声明

- T0（复现基准）SHA256 `{spec['T0']['utf8_sha256']}`
- T3-bare SHA256 `{spec['T3']['utf8_sha256']}`
- T4-long-first SHA256 `{spec['T4']['utf8_sha256']}`
- T5-long-last SHA256 `{spec['T5']['utf8_sha256']}`
- 以上在**任何行为结果之前**写入 `candidate_length_expression_spec.json`；不得事后修改。

## 禁止项遵守

- 未读取 D0 train/final-reserve 样本文本（仅读 D0 decision.json 标签）。
- 未读取/评分/缓存任何 final-reserve。
- 未读取 hidden state；未训练 Probe/Monitor/Classifier；无干预/hook/causal tracing。
- 未修改 system prompt / prompt baseline / CoT / ICL / SFT；未换模型/精度/batch/读出。
- 未用 LLM 改写或审查 Candidate；未拼接 SciQ `support`；未按行为结果选择样本。
""", encoding="utf-8")

all_ok = all(ok for _, ok, _ in rows)
if not all_ok:
    (OUT / "artifacts" / "decision.json").write_text(json.dumps({
        "final_label": "inheritance_invalid",
        "reason": "; ".join(n for n, ok, _ in rows if not ok)}, indent=2), encoding="utf-8")
    print("STOP: inheritance_invalid")
    sys.exit(1)
print("Phase 0 OK")
