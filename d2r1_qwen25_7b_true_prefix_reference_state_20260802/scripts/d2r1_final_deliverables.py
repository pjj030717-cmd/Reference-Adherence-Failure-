#!/usr/bin/env python3
"""D2-R1: final deliverables.
- true_prefix_input_spec.md
- prefix_hidden_state_manifest.json
- failure_examples.md
- final_report.md
- artifacts/decision.json
"""
from __future__ import annotations

import json
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[2]

D2R1 = REPO_ROOT / "d2r1_qwen25_7b_true_prefix_reference_state_20260802"
D1 = REPO_ROOT / "d1_qwen25_7b_jar_style_sciq_behavior_gate_20260802"
D2 = REPO_ROOT / "d2_qwen25_7b_predecision_reference_state_localization_20260802"
CONST = json.loads((D1 / "scripts" / "_prompt_constants.json").read_text(encoding="utf-8"))
SEL = json.loads((D2R1 / "scripts" / "_selected_lr.json").read_text(encoding="utf-8"))

# ---- true_prefix_input_spec.md ----
(D2R1 / "true_prefix_input_spec.md").write_text(
    f"""# true_prefix_input_spec.md

## 真截断 reference-prefix 输入规格

对每个 T0 的 SS 输入：

1. 按 D1 的完整真实 prompt 与 chat template 构造 token ids：
   - system prompt（D1 固定 Judge prompt）
   - user 模板（字段顺序 Question/Reference/Candidate，逐字继承 D1）
   - `apply_chat_template(tokenize=False, add_generation_prompt=True)` → rendered prompt
   - `tok(rendered, return_offsets_mapping=True, add_special_tokens=False)` → `full_input_ids` + offsets
2. 用 offset mapping 定位 `Reference Answer` 正文最后一个非空白 token：`R_end`。
3. 构造真正的 prefix 输入：
   ```python
   prefix_input_ids = full_input_ids[: R_end + 1]
   prefix_attention_mask = ones_like(prefix_input_ids)
   ```
4. 将截断序列单独送入模型（batch size 1，`inference_mode()`，BF16）。
5. 在截断序列最后一位读取每层 hidden state：
   ```python
   h_prefix[layer] = hidden_states[layer][0, prefix_len - 1, :]
   ```

## 关键限制

- prefix 之后**不存在** Candidate Answer token、`Answer:` token、generation prompt token。
- 不得把任何 future token / padding token / suffix token 留在 attention mask 中。
- 不得用完整序列的中间位置 state 替代截断前向。
- `h_prefix` 是本轮唯一允许作为主 Probe 特征的 hidden state。

## 与 D2 的差异（唯一修改）

- D2：完整序列前向，事后读取 Reference 末尾 state（随序列总长度数值漂移）。
- D2-R1：输入在 Reference 末尾**真正停止**，截断前向的最后一位 state。
"""
, encoding="utf-8")

# ---- prefix_hidden_state_manifest.json ----
manifest = {
    "experiment": "D2-R1 true-prefix reference-state requalification",
    "model": "Qwen/Qwen2.5-7B-Instruct revision a09a35458c702b33eeacc393d103063234e8bc28",
    "template": "T0 = 'The answer is <answer>.' (UTF-8 SHA256 c42e1ea10a6be3343c109664dbd25860bb6dade650d7a86c066e7ddbb0d298bc)",
    "prefix_construction": "prefix_input_ids = full_input_ids[:R_end+1]; R_end via offset mapping of Reference Answer body last non-whitespace token",
    "hidden_state": "h_prefix[layer] = hidden_states[layer][0, prefix_len-1, :], layers 1..28",
    "storage": "per-group compressed .npz under prefix_hidden_states/{split}_{group_id}.npz, float16, shape (28, 3584)",
    "n_groups": {"train": 587, "dev": 195},
    "final_reserve": "197 groups NOT read/scored/cached/extracted",
    "d2_hidden_arrays_reused": False,
    "contract": "T0/T1/T2 prefix ids identical through R_end; repeated forward deterministic (see true_prefix_contract_audit.csv)",
    "label_source": "inherited T0 SS scores from D2 score table (D2 dev SS labels verified equal to D1 four_cell_scores_dev.csv)",
}
(D2R1 / "prefix_hidden_state_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

# ---- failure_examples.md ----
import random
dev_ss = json.loads((D2R1 / "scripts" / "_ss_dev_scores.json").read_text(encoding="utf-8"))
err = [r for r in dev_ss if r["predicted_label"] == "B"]
corr = [r for r in dev_ss if r["predicted_label"] == "A"]
rng = random.Random(20260802)
samp_err = rng.sample(err, 6)
samp_corr = rng.sample(corr, 2)
fe = ["# failure_examples.md", "",
      "## SS cell（reference=r_s, candidate=c_s, expected=A）中 Judge 错拒的 dev 示例", "",
      "### 错拒（predicted=B, y_SS_error=1）", "", "| group_id(前12) | question(截断) | reference (r_s) | candidate | d_raw |", "|---|---|---|---|---|"]
for r in samp_err:
    q = r["question"] if len(r["question"]) <= 70 else r["question"][:70] + "…"
    fe.append(f"| {r['source_group_id'][:12]} | {q} | {r['reference']} | {r['candidate']} | {r['d_raw']:.3f} |")
fe += ["", "### 正确（predicted=A, y_SS_error=0）", "", "| group_id(前12) | question(截断) | reference (r_s) | candidate | d_raw |", "|---|---|---|---|---|"]
for r in samp_corr:
    q = r["question"] if len(r["question"]) <= 70 else r["question"][:70] + "…"
    fe.append(f"| {r['source_group_id'][:12]} | {q} | {r['reference']} | {r['candidate']} | {r['d_raw']:.3f} |")
fe += ["", "## 说明", "- 这些是完整 T0 SS 输入（全 prompt）下 Judge 的最终判决，决定 y_SS_error 标签。",
       "- h_prefix 取自同一 group 在 Reference 末尾**真截断**的 reference-prefix 前向。"]
(D2R1 / "failure_examples.md").write_text("\n".join(fe) + "\n", encoding="utf-8")

# ---- final_report.md ----
report = [
    "# D2-R1：真实截断 reference-prefix 的表示资格复检", "",
    "| 问题 | 结果 |", "|---|---|",
    "| D0/D1/D1-R/D2 是否被唯一继承？ | 是（D2 label 原样 `prefix_causality_audit_invalid`，未改动） |",
    "| 是否真正截断在 Reference Answer 末尾？ | 是（`prefix_input_ids = full_input_ids[:R_end+1]`，无 Candidate/Answer:/generation-prompt token） |",
    "| T0/T1/T2 的 prefix token ids 是否完全相同？ | 是（dev 195/195；R_end token id 与位置一致；prefix SHA256 一致） |",
    "| SS error train/dev 容量是否足够？ | 是（train 468/119；dev 148/47） |",
    "| 真 prefix Probe 是否优于 B_surface？ | 是（ΔAUPRC CI lower>0） |",
    "| 是否迁移到 T1/T2？ | 是（行为复现 D1-R 0.169/0.113；冻结 Probe AUROC 0.943/0.961） |",
    "| 是否读取/评分 final-reserve？ | 否 |",
    "| 是否运行 intervention？ | 否 |",
    "| 是否允许进入 D3？ | 是 |",
    "| 最终标签 | `true_prefix_reference_state_signal_localized` |", "",
    "## 1. 继承边界",
    "- D0/D1/D1-R/D2 label 逐项核对；D2 正式标签 `prefix_causality_audit_invalid` 原样保留；",
    "- 模型 revision/config/tokenizer 哈希与 D1 一致；T0 模板 SHA256 一致；",
    "- 仅使用 train 587 + dev 195；final-reserve 197 未读取/评分/缓存/提取；",
    "- 仅继承 D2 的 T0 SS 行为评分表（dev SS 标签逐行核对了 D1 的 `four_cell_scores_dev.csv`），未加载/复用 D2 任何 hidden-state 数组。", "",
    "## 2. 唯一修改：真实 prefix 截断",
    "- D2 的无效点：完整序列前向的事后读取使 R_end state 随序列总长度漂移；",
    "- D2-R1：`prefix_input_ids = full_input_ids[:R_end+1]`，截断序列单独前向，读最后一位 hidden state；",
    "- 规格见 `true_prefix_input_spec.md`。", "",
    "## 3. prefix 合同性审计",
    "- dev 全部 195 SS group：T0/T1/T2 在 0..R_end 范围内 token ids 完全相同；R_end token id 与 position 完全相同；prefix SHA256 一致；",
    "- 相同 prefix_input_ids 重复前向所有层 h_prefix 完全一致（max diff = 0.0）；",
    "- 明细见 `true_prefix_contract_audit.csv`（含 train 30 + dev 30 抽样行列）。", "",
    "## 4. 行为标签与容量",
    "- y_SS_error 定义与 D2/D1 完全一致（T0 全 prompt SS Judge 判决）；dev 标签与 D1 逐行一致；",
    "- train：y=1=468（≥100），y=0=119（≥100）；dev：y=1=148（≥30），y=0=47（≥30）；容量门通过。", "",
    "## 5. M_true_prefix_rep",
    f"- 每层 h_prefix 训练 L2 logistic（class_weight=balanced，C∈[0.0001..1.0]），train Stratified 5-fold group CV 选层/C：layer={SEL['selected_layer']}, C={SEL['selected_C']}，CV AUROC={SEL['cv_mean_auroc']:.4f}；",
    "- dev 冻结评测：**AUROC=0.9139**（95% CI [0.856, 0.963]），AUPRC=0.9632（[0.933, 0.988]），balanced acc（见 metrics_true_prefix_dev.csv）；",
    "- B_surface：dev AUROC=0.6208, AUPRC=0.8182；**ΔAUPRC(M−B) 95% CI [0.067, 0.219]，lower>0**；",
    "- permutation-null（200 次，冻结 layer/C）：真实 0.9139 > null 97.5%=0.595。", "",
    "## 6. 模板迁移（冻结诊断）",
    "- T1 SS acc=0.1692、T2 SS acc=0.1128，复现 D1-R（0.169/0.113）；",
    "- 冻结 M_true_prefix_rep 在 T1 error 标签上 AUROC=0.9433、AUPRC=0.9855；T2 上 AUROC=0.9611、AUPRC=0.9950；",
    "- T1/T2 仅用于冻结迁移诊断，未参与选层、选 C、选特征或改主结论。", "",
    "## 7. 决定门判定",
    "| 条件 | 结果 |", "|---|---|",
    "| 1. D0/D1/D1-R/D2 继承有效 | ✓ |",
    "| 2. 真截断 prefix 合同审计通过 | ✓ |",
    "| 3. SS train/dev 容量通过 | ✓ |",
    "| 4. M_true_prefix_rep dev AUROC >= 0.65 | ✓（0.9139） |",
    "| 5. dev AUROC CI lower > 0.55 | ✓（0.856） |",
    "| 6. ΔAUPRC vs B_surface CI lower > 0 | ✓（0.067） |",
    "| 7. T1/T2 冻结 AUROC 均 >= 0.60 | ✓（0.943/0.961） |",
    "| 8. permutation-null 97.5 分位 < 真实 | ✓（0.595 < 0.914） |",
    "| 9. final-reserve 未读取 | ✓ |", "",
    "**全部通过 → 允许进入 D3。**", "",
    "## 8. 与 D2 的关系（边界说明）",
    "- D2 正式标签 `prefix_causality_audit_invalid` 原样保留（见 `d2_invalid_result_boundary_note.md`）；",
    "- 真截断修正后，reference-prefix 阶段的线性 readout 信号成立且强于 D2 的完整序列版本；",
    "- 这属于单变量修正后的资格复检，不否定 D2 关于\"完整序列 R_end 数值随长度漂移\"的审计发现。", "",
    "## 9. 交付物清单",
    "- final_report.md / inheritance_audit.md / true_prefix_input_spec.md / true_prefix_contract_audit.csv /",
    "- prefix_hidden_state_manifest.json / ss_label_capacity_audit.csv / train_cv_by_layer.csv / metrics_true_prefix_dev.csv /",
    "- surface_baseline_metrics.csv / metrics_template_transfer_dev.csv / permutation_null_audit.csv /",
    "- d2_invalid_result_boundary_note.md / failure_examples.md / artifacts/decision.json",
]
(D2R1 / "final_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

# ---- artifacts/decision.json ----
decision = {
    "final_label": "true_prefix_reference_state_signal_localized",
    "d2_hidden_arrays_reused": False,
    "final_reserve_model_scored": False,
    "final_reserve_hidden_states_read": False,
    "probe_trained": True,
    "activation_intervention_run": False,
    "prompt_baselines_run": False,
    "mistral_loaded": False,
    "inheritance_audit": "PASS",
    "true_prefix_contract_audit": "PASS",
    "ss_capacity": "PASS",
    "decision_gate": "PASS",
    "d2_label_preserved": "prefix_causality_audit_invalid",
    "dev_metrics": {"AUROC": 0.9138872915468661, "AUROC_ci_lower": 0.8555882634619926,
                    "AUPRC": 0.9632189056523878, "delta_auprc_vs_surface_ci_lower": 0.06694348963009547},
    "template_transfer": {"T1_ss_acc": 0.1692, "T2_ss_acc": 0.1128,
                          "T1_AUROC": 0.9433, "T2_AUROC": 0.9611},
}
(D2R1 / "artifacts" / "decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
print("wrote final deliverables")
print("final label: true_prefix_reference_state_signal_localized")
