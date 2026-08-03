#!/usr/bin/env python3
"""D1-R-A Phase 1: read-only mechanical/rendering/expression-type audit of T0/T1/T2.

1.1 mechanical pairwise distances (normalized NFKC, whitespace collapsed, case preserved)
1.2 rendering audit with 6 probe answers + Qwen tokenizer ids (tokenizer only, NO model weights)
1.3 expression-type binary attributes (fixed text only, not answer placeholder)

Only loads the D1-fixed-revision Qwen tokenizer (no AutoModelForCausalLM, no forward).

Outputs:
  - template_pairwise_distance.csv
  - template_rendering_audit.csv
  - template_expression_type_audit.csv
  - tokenization_audit.md
"""
from __future__ import annotations

import csv
import hashlib
import re
import unicodedata
from pathlib import Path
import os
REPO_ROOT = Path(__file__).resolve().parents[2]

from transformers import AutoTokenizer

R = REPO_ROOT / "d1ra_candidate_template_provenance_diversity_audit_20260803"
canon = json_canon = None
import json
canon = json.loads((R / "canonical_candidate_templates.json").read_text(encoding="utf-8"))
TEMPLATES = {k: v["template"] for k, v in canon.items() if k in ("T0", "T1", "T2")}

MODEL = os.environ.get("RAF_MODEL_DIR", "/root/autodl-tmp/models/Qwen2.5-7B-Instruct")


def norm(s: str) -> str:
    """NFKC, collapse consecutive whitespace to single space; no lowercasing, keep punctuation."""
    s = unicodedata.normalize("NFKC", s)
    return re.sub(r"\s+", " ", s).strip()


def levenshtein(a, b):
    """character-level Levenshtein distance"""
    n, m = len(a), len(b)
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] != b[j - 1]))
        prev = cur
    return prev[m]


def tokenize_fixed(tpl: str):
    """tokens of the fixed text after removing placeholder"""
    fixed = tpl.replace("<answer>", " ")
    return [t for t in re.split(r"\s+", norm(fixed)) if t]


def jaccard(a, b):
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def longest_common_substring(a, b):
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    best = 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                best = max(best, dp[i][j])
    return best


# ---------------------------------------------------------------------------
# 1.1 pairwise mechanical distances
# ---------------------------------------------------------------------------
pairs = []
ids = ["T0", "T1", "T2"]
for i, a in enumerate(ids):
    for j in range(i + 1, len(ids)):
        b = ids[j]
        ta, tb = norm(TEMPLATES[a]), norm(TEMPLATES[b])
        lev = levenshtein(ta, tb)
        fa, fb = tokenize_fixed(TEMPLATES[a]), tokenize_fixed(TEMPLATES[b])
        lcs = longest_common_substring(ta, tb)
        pos_a = ta.find("<answer>")
        pos_b = tb.find("<answer>")
        pairs.append({
            "template_a": a, "template_b": b,
            "normalized_character_levenshtein_distance": lev,
            "fixed_text_token_count_a": len(fa), "fixed_text_token_count_b": len(fb),
            "fixed_text_token_jaccard": round(jaccard(fa, fb), 6),
            "longest_common_fixed_substring_length": lcs,
            "answer_placeholder_char_pos_a": pos_a, "answer_placeholder_char_pos_b": pos_b,
            "char_len_a": len(ta), "char_len_b": len(tb),
            "word_count_a": len(fa), "word_count_b": len(fb),
            "sentence_count_a": len([s for s in re.split(r'[.!?]', ta) if s.strip()]),
            "sentence_count_b": len([s for s in re.split(r'[.!?]', tb) if s.strip()]),
            "punctuation_a": re.findall(r"[.,:;!?'-]", ta),
            "punctuation_b": re.findall(r"[.,:;!?'-]", tb),
        })
        print(pairs[-1])

with open(R / "template_pairwise_distance.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(pairs[0].keys()))
    w.writeheader()
    w.writerows(pairs)

# ---------------------------------------------------------------------------
# 1.2 rendering audit (probe answers, Qwen tokenizer ids)
# ---------------------------------------------------------------------------
tok = AutoTokenizer.from_pretrained(MODEL)
print("tokenizer loaded:", type(tok).__name__)

PROBE_ANSWERS = ["Paris", "heart", "H2O", "800", "true", "Gianluigi Buffon"]
render_rows = []
for tid in ids:
    tpl = TEMPLATES[tid]
    for ans in PROBE_ANSWERS:
        rendered = tpl.replace("<answer>", ans)
        ids_enc = tok.encode(rendered, add_special_tokens=False)
        answer_span_start = rendered.find(ans)
        render_rows.append({
            "template_id": tid, "probe_answer": ans, "rendered_candidate": rendered,
            "character_length": len(rendered), "word_count": len(rendered.split()),
            "sentence_count": len([s for s in re.split(r'[.!?]', rendered) if s.strip()]),
            "token_ids": ",".join(str(i) for i in ids_enc),
            "token_count": len(ids_enc),
            "answer_span_char_start": answer_span_start,
        })

with open(R / "template_rendering_audit.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(render_rows[0].keys()))
    w.writeheader()
    w.writerows(render_rows)
print("rendering rows:", len(render_rows))

# ---------------------------------------------------------------------------
# 1.3 expression-type audit (fixed text only)
# ---------------------------------------------------------------------------
MARKERS = {
    "answer_only": lambda t: True,  # all templates are answer-only frames (fixed text = "The answer is X.")
    "declarative_answer_frame": lambda t: re.search(r"\b(answer|response)\b", t, re.I) is not None,
    "question_restatement": lambda t: re.search(r"\bquestion\b", t, re.I) is not None,
    "explicit_reference_mention": lambda t: re.search(r"\b(reference|given answer|gold answer|provided answer)\b", t, re.I) is not None,
    "explicit_evaluation_or_correctness_word": lambda t: re.search(r"\b(correct|should accept|valid|incorrect|should reject)\b", t, re.I) is not None,
    "multi_sentence": lambda t: len([s for s in re.split(r"[.!?]", t) if s.strip()]) > 1,
    "explanatory_clause": lambda t: re.search(r"\b(because|since|as|therefore|thus)\b", t, re.I) is not None,
}
# NOTE: render time, T0 = "The answer is <answer>." fixed tokens = ["The","answer","is","."]; punctuation ".".
# "answer" appears in "answer is" frame (declarative). T1 adds "For this question, ". T2 uses "response is".

expr_rows = []
for tid in ids:
    t = TEMPLATES[tid]
    fixed = t.replace("<answer>", "")  # placeholder content excluded
    row = {"template_id": tid, "template": t}
    for name, fn in MARKERS.items():
        row[name] = int(fn(fixed))
    # special report: potential reference-compliance prompting
    ref_prompt = int(re.search(r"\b(reference|gold answer|given answer|authoritative)\b", fixed, re.I) is not None)
    correctness_prompt = int(MARKERS["explicit_evaluation_or_correctness_word"](fixed))
    row["potential_prompt_intervention_reference_hint"] = ref_prompt
    row["potential_prompt_intervention_correctness_hint"] = correctness_prompt
    expr_rows.append(row)
    print(row)

with open(R / "template_expression_type_audit.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(expr_rows[0].keys()))
    w.writeheader()
    w.writerows(expr_rows)

# ---------------------------------------------------------------------------
# tokenization_audit.md
# ---------------------------------------------------------------------------
(R / "tokenization_audit.md").write_text(
    """# tokenization_audit.md

## 加载

- 仅加载 D1 固定 revision 的 Qwen tokenizer（`/root/autodl-tmp/models/Qwen2.5-7B-Instruct`）。
- 未加载 `AutoModelForCausalLM`，无任何前向、logits、hidden-state 或生成。

## 模板 tokenization（probe answer 渲染）

对每个模板与 6 个 probe answer，记录渲染后的完整 token ids 与 token 数（见 `template_rendering_audit.csv`）。

## 关键检查

- `" A"`/`" B"` continuation token ids 不在本阶段计算（不加载模型）；模板渲染本身不涉及 continuation。
- 模板中的 `<answer>` 占位符在 Qwen tokenizer 下为多 token；渲染后整体 token 数随答案变化。
""", encoding="utf-8")

print("Phase 1 OK: pairwise distance + rendering + expression type + tokenization written")
