# inheritance_audit.md

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
