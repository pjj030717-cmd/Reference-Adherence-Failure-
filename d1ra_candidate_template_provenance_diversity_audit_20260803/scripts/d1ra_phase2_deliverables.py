#!/usr/bin/env python3
"""D1-R-A Phase 2: pre-registered interpretation boundary + deliverables + decision.

Classification is determined SOLELY by Phase 1 template attributes (never by any
behavioral accuracy / FR_SS / other downstream result):

  prompt_intervention_contamination:
      any template fixed text carries explicit reference/gold/correct/accept guidance
      (explicit_reference_mention or explicit_evaluation_or_correctness_word = 1).
  structural_candidate_variation:
      >= 1 template changes answer-presentation structure (question restatement frame,
      multi-sentence, or a non-minimal sentence-shape change).
  minimal_surface_variation:
      differences are only short fixed-phrase / word-order / punctuation tweaks.

Outputs:
  - final_report.md
  - template_provenance_audit.md
  - inheritance_audit.md
  - failure_examples.md
  - artifacts/decision.json
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[2]

R = REPO_ROOT / "d1ra_candidate_template_provenance_diversity_audit_20260803"

canon = json.loads((R / "canonical_candidate_templates.json").read_text(encoding="utf-8"))
TEMPLATES = {k: v["template"] for k, v in canon.items() if k in ("T0", "T1", "T2")}

expr = {}
with open(R / "template_expression_type_audit.csv", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        expr[r["template_id"]] = r

# ---- pre-registered classification (Phase 1 attributes only) ----
ref_hints = [k for k, v in expr.items() if v["explicit_reference_mention"] == "1"]
corr_hints = [k for k, v in expr.items() if v["explicit_evaluation_or_correctness_word"] == "1"]
q_restate = [k for k, v in expr.items() if v["question_restatement"] == "1"]
multi_sent = [k for k, v in expr.items() if v["multi_sentence"] == "1"]

contaminated = bool(ref_hints or corr_hints)
structural = bool(q_restate or multi_sent) and not contaminated

if contaminated:
    category = "prompt_intervention_contamination"
elif structural:
    category = "structural_candidate_variation"
else:
    category = "minimal_surface_variation"

print(f"ref_hints={ref_hints} corr_hints={corr_hints} q_restate={q_restate} multi_sent={multi_sent}")
print("CATEGORY:", category)

# ---------------- template_provenance_audit.md ----------------
(R / "template_provenance_audit.md").write_text(
    f"""# template_provenance_audit.md

## 模板唯一来源恢复（Phase 0）

| 模板 | 原始 UTF-8 模板 | UTF-8 SHA256 | 来源数 | 逐字一致 |
|---|---|---|---|---|
| T0 | `{TEMPLATES['T0']}` | `{canon['T0']['utf8_sha256']}` | 3 | 是 |
| T1 | `{TEMPLATES['T1']}` | `{canon['T1']['utf8_sha256']}` | 3 | 是 |
| T2 | `{TEMPLATES['T2']}` | `{canon['T2']['utf8_sha256']}` | 3 | 是 |

来源（均逐字一致）：
- D1-R `candidate_template_robustness_spec.json` → `templates.<T>.template`
- D1-R `scripts/d1r_template_spec.py` → `TEMPLATES['<T>']`（可执行定义）
- D1-R `scripts/d1r_eval.py` → `TEMPLATES['<T>']`（可执行渲染路径）

## T0 与 D1/D0 基础渲染对齐

| 检查 | 结果 |
|---|---|
| D0 `candidate_rendering_spec.json` template == T0 | 是（`The answer is <answer>.`） |
| D1 `scripts/_dev_pairs.jsonl` c_o/c_s 全部 == T0 渲染 | 是（195 dev groups，0 违例） |
| D1-R `t0_reproduction_audit.csv` 780 行 candidate == T0 渲染 | 是（0 违例） |

注意：D0 `candidate_rendering_spec.json` 内嵌 `sha256_utf8` 字段为已知中间态哈希（`d41ad577…`），
D1-R `provenance_amendment.md` 已透明记录；模板**字符串**的 UTF-8 SHA256 为 `{canon['T0']['utf8_sha256']}`，以此继承。

## 占位符合同（Phase 0.2）

- 每模板恰有 1 个 `<answer>` 占位符，无其他占位符。
- `<answer>` 仅用于答案插入（D0 冻结 r_o/r_s 归一化原文）。
- 模板可确定性渲染。
- T0 ≠ T1 ≠ T2（逐字互异）。

## 机械差异（Phase 1.1，见 `template_pairwise_distance.csv`）

| 对 | 字符 Levenshtein | 固定词 token Jaccard | 最长公共固定子串 | 标点 |
|---|---|---|---|---|
| T0↔T1 | 20 | 0.375 | 22 | `.` vs `,`+`.` |
| T0↔T2 | 7 | 0.600 | 13 | `.` vs `.` |
| T1↔T2 | 21 | 0.222 | 13 | `,`+`.` vs `.` |

## 表达类型（Phase 1.3，见 `template_expression_type_audit.csv`）

| 模板 | answer_only | declarative_answer_frame | question_restatement | explicit_reference | eval/correct | multi_sentence | 污染提示 |
|---|---|---|---|---|---|---|---|
| T0 | 1 | 1 | 0 | 0 | 0 | 0 | 无 |
| T1 | 1 | 1 | 1 | 0 | 0 | 0 | 无 |
| T2 | 1 | 1 | 0 | 0 | 0 | 0 | 无 |
""", encoding="utf-8")

# ---------------- inheritance_audit.md ----------------
(R / "inheritance_audit.md").write_text(
    """# inheritance_audit.md

## 本轮只读范围

- D1-R：`candidate_template_robustness_spec.json`、`scripts/d1r_template_spec.py`、`scripts/d1r_eval.py`、
  `t0_reproduction_audit.csv`、`inheritance_audit.md`、`provenance_amendment.md`、`artifacts/decision.json`。
- D1：`scripts/_prompt_constants.json`、`synthetic_pair_manifest.json`、`scripts/_dev_pairs.jsonl`（dev-only）。
- D0：`candidate_rendering_spec.json`（模板字段）。

## 禁止项确认

- 未加载任何 Judge 模型权重（仅加载 D1 固定 revision 的 Qwen tokenizer）。
- 无 Judge 前向 / A/B likelihood / 自由生成 / hidden-state / Probe / intervention。
- 未读取 final-reserve 或 train 文本（仅 dev-only 的 `_dev_pairs.jsonl` 用于 T0 渲染核对；未复制任何文本）。
- 未构造/评测/建议 T3/T4 模板；未根据既有行为结果挑选模板。
- 未改动 D0/D1/D1-R 或任何既有目录。
""", encoding="utf-8")

# ---------------- failure_examples.md ----------------
# No behavioral failures exist in this audit; record the most divergent renderings
# (probe answers only, no D0 text).
render_rows = []
with open(R / "template_rendering_audit.csv", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        render_rows.append(r)
(R / "failure_examples.md").write_text(
    """# failure_examples.md

本实验为只读模板溯源/差异审计，无行为失败样本。以下列出 6 个 probe answer 下三个模板渲染差异最显著的示例
（仅模板渲染字符串，不含 D0/SciQ 数据文本）。

| probe_answer | T0 | T1 | T2 |
|---|---|---|---|
"""
    + "\n".join(
        f"| {a} | `{next(r['rendered_candidate'] for r in render_rows if r['template_id']=='T0' and r['probe_answer']==a)}` | "
        f"`{next(r['rendered_candidate'] for r in render_rows if r['template_id']=='T1' and r['probe_answer']==a)}` | "
        f"`{next(r['rendered_candidate'] for r in render_rows if r['template_id']=='T2' and r['probe_answer']==a)}` |"
        for a in ["Paris", "heart", "H2O", "800", "true", "Gianluigi Buffon"])
    + "\n"
)
print("wrote failure_examples.md")

# ---------------- artifacts/decision.json ----------------
(R / "artifacts").mkdir(parents=True, exist_ok=True)
decision = {
    "final_label": "template_provenance_and_diversity_audit_complete",
    "interpretation_category": category,
    "templates_recovered": ["T0", "T1", "T2"],
    "templates": {k: {"template": TEMPLATES[k], "utf8_sha256": canon[k]["utf8_sha256"]} for k in ("T0", "T1", "T2")},
    "placeholder_contract": {
        "single_answer_placeholder": True,
        "no_undeclared_placeholders": True,
        "deterministic_render": True,
        "templates_pairwise_distinct": True,
    },
    "t0_alignment": {
        "D0_candidate_rendering_spec": True,
        "D1_dev_pairs_c_o_c_s": True,
        "D1R_t0_reproduction_audit": True,
    },
    "mechanical_summary": {
        "levenshtein_T0_T1": 20, "levenshtein_T0_T2": 7, "levenshtein_T1_T2": 21,
        "jaccard_T0_T1": 0.375, "jaccard_T0_T2": 0.6, "jaccard_T1_T2": 0.222,
    },
    "expression_type": {
        "T0": {"answer_only": 1, "declarative_answer_frame": 1, "question_restatement": 0,
               "explicit_reference_mention": 0, "explicit_evaluation_or_correctness_word": 0,
               "multi_sentence": 0},
        "T1": {"answer_only": 1, "declarative_answer_frame": 1, "question_restatement": 1,
               "explicit_reference_mention": 0, "explicit_evaluation_or_correctness_word": 0,
               "multi_sentence": 0},
        "T2": {"answer_only": 1, "declarative_answer_frame": 1, "question_restatement": 0,
               "explicit_reference_mention": 0, "explicit_evaluation_or_correctness_word": 0,
               "multi_sentence": 0},
    },
    "prompt_intervention_contamination": contaminated,
    "judge_model_loaded": False,
    "judge_forward_run": False,
    "tokenizer_only": True,
    "train_text_read": False,
    "final_reserve_text_read": False,
    "new_templates_constructed": False,
    "behavior_results_used_for_classification": False,
}
(R / "artifacts" / "decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
print("wrote artifacts/decision.json")

# ---------------- final_report.md ----------------
(R / "final_report.md").write_text(
    f"""# D1-R-A：候选模板溯源与差异强度审计

## 问题与结果

| 问题 | 结果 |
|---|---|
| T0/T1/T2 是否可唯一恢复？ | 是（3 来源逐字一致，SHA256 唯一） |
| T0 是否与 D1/D0 基础渲染一致？ | 是（D0 rendering spec / D1 dev pairs / D1-R T0 复现 780 行全部一致） |
| 是否存在显式 Reference/评分引导污染？ | 否（reference/correct/valid 引导词全部为 0） |
| 模板差异属于何种强度？ | **{category}** |
| 是否运行了任何 Judge 推理？ | 否（仅加载 tokenizer） |
| 是否读取了 train/final-reserve 文本？ | 否 |
| 最终标签 | **template_provenance_and_diversity_audit_complete** |

## 模板（canonical，来自 D1-R 可执行来源）

| 模板 | 原始 UTF-8 | SHA256 | 占位符 |
|---|---|---|---|
| T0 | `{TEMPLATES['T0']}` | `{canon['T0']['utf8_sha256']}` | 1×`<answer>` |
| T1 | `{TEMPLATES['T1']}` | `{canon['T1']['utf8_sha256']}` | 1×`<answer>` |
| T2 | `{TEMPLATES['T2']}` | `{canon['T2']['utf8_sha256']}` | 1×`<answer>` |

来源：D1-R `candidate_template_robustness_spec.json` + `scripts/d1r_template_spec.py` + `scripts/d1r_eval.py`（逐字一致）。

## 机械差异

| 对 | 字符 Levenshtein | 固定词 token Jaccard | 最长公共固定子串 | T 词数 | T 标点 |
|---|---|---|---|---|---|
| T0↔T1 | 20 | 0.375 | 22 | 4↔7 | `.` ↔ `,`+`.` |
| T0↔T2 | 7 | 0.600 | 13 | 4↔4 | `.` ↔ `.` |
| T1↔T2 | 21 | 0.222 | 13 | 7↔4 | `,`+`.` ↔ `.` |

- T2 与 T0 差异最小（换词 `answer`→`response`）；T1 与两者差异最大（新增 `For this question, ` 前缀）。

## 渲染审计

6 个 probe answer（`Paris`/`heart`/`H2O`/`800`/`true`/`Gianluigi Buffon`）渲染后记录字符数、词数、句子数、Qwen token ids 与 token 数（见 `template_rendering_audit.csv`、`tokenization_audit.md`）。

## 表达类型与预注册归类

| 模板 | answer_only | declarative | question_restatement | reference | eval/correct | multi_sentence |
|---|---|---|---|---|---|---|
| T0 | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ |
| T1 | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ |
| T2 | ✓ | ✓ | ✗ | ✗ | ✗ | ✗ |

归类（仅基于 Phase 1 模板属性，未参考任何行为结果）：

- **`prompt_intervention_contamination`**：否。所有模板固定文本均无 reference/gold/correct/valid/should accept 等引导词。
- **`structural_candidate_variation`**：是。T1（`For this question, the answer is <answer>.`）把答案放入问题语境框架（question-restatement 结构），不再只是孤立答案陈述；T2 为换词表面变化；T0 为基准。
- **`minimal_surface_variation`**：T0↔T2 符合（仅换词），但 T1 的存在使整体归类升级为 `structural_candidate_variation`。

结论：**{category}**。T1/T2 可支持“跨 Candidate 表达稳健性”的较强证据（T1 改变答案呈现框架），且无 prompt intervention 污染；若仅比较 T0↔T2，则属于最小表面变化。

## 结论边界

- 本审计只回答“已用于 D1-R 的 T0/T1/T2 能否唯一恢复 + 差异强度”，不检验 H1/H2，不构成行为结论。
- 归类严格由模板固定文本属性决定；未使用 D1-R 准确率、SS 错拒率或任何后续结果。
- T1 的 `For this question` 是对问题语境的显式引用，但**不含** reference/correct/评分规则字样，故不构成 prompt intervention。
""", encoding="utf-8")
print("wrote final_report.md")
print("FINAL LABEL: template_provenance_and_diversity_audit_complete | CATEGORY:", category)
