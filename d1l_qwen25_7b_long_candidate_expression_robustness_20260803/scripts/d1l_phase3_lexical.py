#!/usr/bin/env python3
"""E01-D1-L Phase 3: lexical comparator audit (no model).

Two model-free diagnostics, fixed BEFORE any behavior-result tuning:

1) B_slot_oracle (构造 oracle):
   - Given the known template and rendered candidate, recover the answer placed
     in the single <answer> slot: a = candidate with template prefix/suffix removed.
   - Predict Accept iff normalize(a) == normalize(r); else Reject.

2) B_exact_match (简单 lexical comparator):
   - Does NOT read the placeholder. Tokenize candidate; predict Accept iff
     normalize(r) appears as a complete token/span (contiguous subsequence) in it.

Both evaluated on the T0/T3/T4/T5 four-cell grid:
  - accuracy, SS false-rejection (1 - SS acc), SO false-acceptance (1 - SO acc)
Honest scoping: strong rule comparators do NOT delete or rewrite Judge results;
they delimit how rule-izable the controlled task is.
Writes lexical_comparator_spec.md + lexical_comparator_metrics.csv.
"""
from __future__ import annotations

import csv
import json
import re
import unicodedata
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[2]

OUT = REPO_ROOT / "d1l_qwen25_7b_long_candidate_expression_robustness_20260803"

spec = json.loads((OUT / "candidate_length_expression_spec.json").read_text(encoding="utf-8"))
TPL = {n: spec["templates"][n]["template"] for n in ("T0", "T3", "T4", "T5")}

dev_pairs = []
with open(OUT / "scripts" / "_dev_input.jsonl", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            dev_pairs.append(json.loads(line))


def norm(s):
    s = unicodedata.normalize("NFKC", str(s))
    s = re.sub(r"\s+", " ", s.strip())
    return s.lower()


def render(tpl, ans):
    return tpl.replace("<answer>", ans)


def slot_extract(tpl, cand):
    """Recover answer placed in the single <answer> slot."""
    pre, post = tpl.split("<answer>", 1)
    assert cand.startswith(pre) and cand.endswith(post) and len(cand) >= len(pre) + len(post)
    return cand[len(pre): len(cand) - len(post)]


def tokenize(s):
    return re.findall(r"[a-z0-9']+", norm(s))


def span_match(ref_tokens, cand_tokens):
    if not ref_tokens:
        return False
    L = len(ref_tokens)
    return any(cand_tokens[i:i + L] == ref_tokens for i in range(len(cand_tokens) - L + 1))


def predict_slot(tpl, cand, r):
    return "A" if norm(slot_extract(tpl, cand)) == norm(r) else "B"


def predict_exact(cand, r):
    return "A" if span_match(tokenize(r), tokenize(cand)) else "B"


# write spec FIRST (fixed)
(OUT / "lexical_comparator_spec.md").write_text(
    """# lexical_comparator_spec.md

## 定义（在行为结果之前固定）

### B_slot_oracle —— 构造 oracle，不可当可部署 Judge
- 已知模板与渲染后的 Candidate，从唯一的 `<answer>` 占位符位置恢复答案 a
  （`a = candidate` 去掉模板 prefix/suffix）。
- 预测规则：`normalize(a) == normalize(r)` → Accept，否则 Reject。
- `normalize` = NFKC + trim + 空白折叠 + 小写。

### B_exact_match —— 简单 lexical comparator（非 oracle）
- 不读取占位符；将 Candidate 按 token 化（小写词形）。
- 预测规则：`normalize(r)` 作为完整 token/span（连续子序列）出现在 Candidate 中 → Accept，否则 Reject。

## 诚实定位

- 若规则 comparator 在此受控任务上表现强，说明该任务并不要求我们证明 Probe 优于字符串匹配；
  本轮研究的问题仍是冻结 LLM Judge 为何未遵从已给 Reference，以及该失效是否可由 Candidate 前状态预测。
- 不得因 lexical comparator 很强而删除、重写或否定 Judge 行为结果。
""", encoding="utf-8")

# metrics
rows = []
for name in ("T0", "T3", "T4", "T5"):
    tpl = TPL[name]
    for p in dev_pairs:
        gid = p["original_group_id"]
        q, r_o, r_s = p["q"], p["r_o"], p["r_s"]
        cells = [
            ("OO", r_o, render(tpl, r_o), "A"),
            ("OS", r_o, render(tpl, r_s), "B"),
            ("SO", r_s, render(tpl, r_o), "B"),
            ("SS", r_s, render(tpl, r_s), "A"),
        ]
        for cell, ref, cand, exp in cells:
            ps = predict_slot(tpl, cand, ref)
            pe = predict_exact(cand, ref)
            rows.append({"source_group_id": gid, "template": name, "cell": cell,
                         "expected_label": exp, "slot_oracle_pred": ps, "slot_oracle_correct": ps == exp,
                         "exact_match_pred": pe, "exact_match_correct": pe == exp})

with open(OUT / "lexical_comparator_metrics.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["source_group_id", "template", "cell", "expected_label",
                                      "slot_oracle_pred", "slot_oracle_correct",
                                      "exact_match_pred", "exact_match_correct"])
    w.writeheader()
    w.writerows(rows)

# summary table
summary = []
for name in ("T0", "T3", "T4", "T5"):
    sub = [r for r in rows if r["template"] == name]
    for algo, predk, corrk in [("B_slot_oracle", "slot_oracle_pred", "slot_oracle_correct"),
                               ("B_exact_match", "exact_match_pred", "exact_match_correct")]:
        cells = {}
        for c in ["OO", "OS", "SO", "SS"]:
            sc = [r for r in sub if r["cell"] == c]
            cells[c] = sum(1 for r in sc if r[corrk]) / len(sc)
        acc = sum(1 for r in sub if r[corrk]) / len(sub)
        fr_ss = sum(1 for r in sub if r["cell"] == "SS" and r[predk] == "B") / sum(1 for r in sub if r["cell"] == "SS")
        fa_so = sum(1 for r in sub if r["cell"] == "SO" and r[predk] == "A") / sum(1 for r in sub if r["cell"] == "SO")
        summary.append({"template": name, "comparator": algo, "accuracy": acc,
                        "OO": cells["OO"], "OS": cells["OS"], "SO": cells["SO"], "SS": cells["SS"],
                        "SS_false_rejection": fr_ss, "SO_false_acceptance": fa_so})
        print(f"{name} {algo}: acc={acc:.4f} OO={cells['OO']:.4f} OS={cells['OS']:.4f} "
              f"SO={cells['SO']:.4f} SS={cells['SS']:.4f} FR_SS={fr_ss:.4f} FA_SO={fa_so:.4f}")

with open(OUT / "lexical_comparator_metrics_summary.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["template", "comparator", "accuracy", "OO", "OS", "SO", "SS",
                                      "SS_false_rejection", "SO_false_acceptance"])
    w.writeheader()
    for s in summary:
        w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in s.items()})

print("Phase 3 OK")
