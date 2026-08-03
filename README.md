# Reference-Adherence Failure: 研究 LLM Judge 对参考答案的依从失败机制

本仓库包含一个系统的机制研究项目，研究 LLM 担任 **Judge（裁判/评分者）** 时对题目提供的 **参考答案（Reference）** 的**依从失败（Reference-Adherence Failure）** 的机制。

- 研究主线：Judge 在收到参考答案后，其决策（打分/选择/判断）在何种条件下偏离参考答案；失败发生在模型内部处理的哪一层、哪个 token 位置。
- 方法主线：数据构建 → 行为门控 → 表征定位（hidden-state probing / prefix intervention / selective patching / monolithic causal location scan）→ 冻结模型复现 → final-reserve 确认，全程遵循严格的 train/dev/final-reserve 数据隔离与继承审计协议。

> 本项目为内部研究仓库；所有实验结论以各实验目录下的 `final_report.md` 为准。

---

## 目录结构

仓库根目录下共 20 个实验目录，前缀表示研究阶段：

| 前缀 | 阶段 | 实验目录 |
|---|---|---|
| `d0` | SCIQ 数据构建与资格确认 | `d0_jar_style_sciq_data_qualification_20260802` |
| `d1` | SCIQ 行为门控 | `d1_qwen25_7b_jar_style_sciq_behavior_gate_20260802` |
| `d1r` | 模板鲁棒性 | `d1r_qwen25_7b_jar_style_sciq_template_robustness_20260802` |
| `d1ra` | 候选模板溯源多样性 | `d1ra_candidate_template_provenance_diversity_audit_20260803` |
| `d1l` | 长候选表达式鲁棒性 | `d1l_qwen25_7b_long_candidate_expression_robustness_20260803` |
| `d1p` | Prompt 基线 | `d1p_qwen25_7b_prompt_baselines_dev_20260803` |
| `d2` | 决策前参考状态定位 | `d2_qwen25_7b_predecision_reference_state_localization_20260802` |
| `d2r1` | True-prefix 参考状态 | `d2r1_qwen25_7b_true_prefix_reference_state_20260802` |
| `d3` | 选择性 patching | `d3_qwen25_7b_reference_binding_selective_patching_20260802` |
| `d3m` | 整体化参考绑定干预 | `d3m_qwen25_7b_monolithic_reference_binding_intervention_20260802` |
| `d3mr1` | 整体化 prefix 方向干预 | `d3mr1_qwen25_7b_monolithic_prefix_direction_intervention_20260802` |
| `d3d` | 整体化因果位置扫描 | `d3d_qwen25_7b_monolithic_causal_location_scan_20260803` |
| `d4m0` | Mistral-7B 行为资格确认（跨模型） | `d4m0_mistral7b_jar_style_sciq_behavior_qualification_20260803` |
| `d4q1` | True-prefix final 确认 | `d4q1_qwen25_7b_true_prefix_final_confirmation_20260803` |
| `a1` | 无参考事实偏好依从审计 | `a1_qwen_no_reference_factual_preference_adherence_audit_20260803` |
| `a2` | SCIQ 复现增量 vs 知识预检 | `a2_qwen_sciq_rep_increment_over_knowledge_precheck_20260803` |
| `e0` / `e0r1` / `e0r2` | PopQA 关系受控数据构建（多轮修订） | `e0_*_popqa_*` |
| `e1` | PopQA H1 行为门控 | `e1_qwen25_7b_popqa_h1_behavior_gate_20260803` |

每个实验目录包含：

- `final_report.md`：结论与全部数值证据；
- `*_audit.md` / `*_audit.csv`：继承审计、隔离审计、模型访问审计等；
- `scripts/`：可复现脚本（详见下文"脚本可移植性"）；
- 各类 `.csv` / `.json` / `.md` 中间产物与最终指标。

---

## 复现环境

- Python 3.10+（开发环境为 3.12）
- 依赖：`torch`、`transformers`、`numpy`、`scikit-learn`
- 建议使用原项目相同的包版本记录（各实验 `scripts/*_phase0_inheritance.py` 内的 `REVISION.txt` 校验）。

### 模型权重

脚本通过环境变量定位本地模型，**不把模型权重提交到本仓库**：

| 环境变量 | 默认值 | 模型 |
|---|---|---|
| `RAF_MODEL_DIR` | `/root/autodl-tmp/models/Qwen2.5-7B-Instruct` | Qwen2.5-7B-Instruct（主模型） |
| `RAF_MISTRAL_DIR` | `/root/autodl-tmp/models/Mistral-7B-Instruct-v0.3` | Mistral-7B-Instruct-v0.3（跨模型对照） |

```bash
# 覆盖默认路径
export RAF_MODEL_DIR=/path/to/Qwen2.5-7B-Instruct
export RAF_MISTRAL_DIR=/path/to/Mistral-7B-Instruct-v0.3
```

---

## 脚本可移植性

本仓库所有脚本路径已**相对化**：

- 仓库根定位：`REPO_ROOT = Path(__file__).resolve().parents[2]`（脚本统一位于 `<exp>/scripts/`，向上两级即仓库根）；
- 所有跨实验目录引用（如 `D0 = REPO_ROOT / "d0_..."`）均基于 `REPO_ROOT`；
- 模型权重路径通过上述环境变量解析；
- 已在 102 个脚本中替换 243 处绝对路径引用，并全部通过 `py_compile` 语法校验。

因此 clone 到任意路径后，只要模型权重路径通过环境变量正确指向，即可按各实验 `scripts/` 顺序复跑。

> 注意：个别脚本 docstring/注释中仍保留原始本地路径文字（如 `/root/autodl-tmp/models/...`）作为模型版本记录，不影响执行。

---

## 数据与隔离协议

本项目严格执行**三层数据隔离**：

1. **train**：模型拟合/特征提取使用的样本；
2. **dev**：选择、调参、冻结配置使用的样本；
3. **final_reserve**：仅在最终确认阶段一次性评估，任何中间阶段不得接触。

各阶段均有继承审计（`inheritance_*_audit.md`）、隔离审计（`*_isolation_audit.md`）与 final-reserve 零访问审计（`final_reserve_zero_access_audit.md`）背书。

---

## 目录内产物说明

每个实验目录下同名 `scripts/` 均按阶段编号执行（`*_phase0_*` → `*_phase1_*` → … → `*_gate.py` / `*_final`），以 `final_report.md` 为终点。冻结的探针/方向产物（如 `scripts/_frozen/probe.npz`）与 hidden states（`prefix_hidden_states/*.npz`）为中间产物，被 `.gitignore` 排除，不随仓库分发。

---

## License

见 [LICENSE](./LICENSE)。
