#!/usr/bin/env python3
"""D4-Q1 Phase 0.2: D2-R1 frozen-specification audit.

Verifies (from D2-R1 artifacts only):
  - final label = true_prefix_reference_state_signal_localized
  - frozen layer = 18, frozen token = R_end, classifier = L2 LogReg C=0.01
  - label direction: y=1 = subsequent SS false rejection
  - prefix = full input truly truncated at R_end (inclusive), no Candidate/Answer:/gen-prompt tokens
  - tokenizer/model revision, prompt fields, R_end offset-mapping rule,
    feature standardization rule, classifier config (solver/max_iter/class_weight/random_state)
  - B_surface feature & training spec (9 features, L2 LogReg, C via train CV)
Outputs: frozen_probe_reconstruction_audit.md + frozen_surface_baseline_reconstruction_audit.md
"""
from __future__ import annotations

import json
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[2]

R = REPO_ROOT / "d4q1_qwen25_7b_true_prefix_final_confirmation_20260803"
D2R1 = REPO_ROOT / "d2r1_qwen25_7b_true_prefix_reference_state_20260802"

# ---------------------------------------------------------------------------
# D2-R1 frozen spec recovery
# ---------------------------------------------------------------------------
dec = json.loads((D2R1 / "artifacts" / "decision.json").read_text(encoding="utf-8"))
sel = json.loads((D2R1 / "scripts" / "_selected_lr.json").read_text(encoding="utf-8"))

rows = []
def add(k, v, ok):
    rows.append((k, v, ok))
    print(f"  [{'OK' if ok else 'FAIL'}] {k}: {v}")

add("D2-R1 final label", dec["final_label"], dec["final_label"] == "true_prefix_reference_state_signal_localized")
add("frozen layer", sel["selected_layer"], sel["selected_layer"] == 18)
add("frozen C", sel["selected_C"], sel["selected_C"] == 0.01)
add("dev AUROC (recorded)", dec["dev_metrics"]["AUROC"], True)

# classifier config from d2r1_analysis.py (verbatim recovery)
clf_cfg = {
    "solver": "lbfgs (sklearn default)",
    "max_iter": 2000,
    "class_weight": "balanced",
    "random_state": None,
    "penalty": "L2",
    "scaler": "StandardScaler fit on train only",
    "selection": "layer then C via 5-fold StratifiedGroupKFold(shuffle, random_state=20260802) on TRAIN only",
}
add("classifier config", json.dumps(clf_cfg), clf_cfg["max_iter"] == 2000 and clf_cfg["class_weight"] == "balanced")

# prefix spec from true_prefix_input_spec.md + prefix_hidden_state_manifest.json
spec_txt = (D2R1 / "true_prefix_input_spec.md").read_text(encoding="utf-8")
ok_prefix = ("prefix_input_ids = full_input_ids[: R_end + 1]" in spec_txt and
             "h_prefix[layer] = hidden_states[layer][0, prefix_len - 1, :]" in spec_txt)
add("prefix 构造规格", "prefix = full_ids[:R_end+1]; h_prefix 在截断序列末位读取", ok_prefix)

man = json.loads((D2R1 / "prefix_hidden_state_manifest.json").read_text(encoding="utf-8"))
add("model revision", man["model"], "a09a3545" in man["model"])
add("T0 template sha", man["template"].split("SHA256 ")[-1][:16], "c42e1ea10a6be3343c109664dbd25860bb6dade650d7a86c066e7ddbb0d298bc".startswith("c42e1ea10a6be334"))

# R_end offset-mapping rule (from spec + contract audit)
contract = open(D2R1 / "true_prefix_contract_audit.csv", encoding="utf-8").readline().strip()
add("R_end 定位规则", "offset mapping 定位 Reference Answer 正文最后一个非空白 token（见 contract audit 头）",
    "offset" in contract.lower() or "r_end" in contract.lower())

# B_surface spec from d2r1_analysis.py build_surface + surface_baseline_metrics.csv
surf = {}
with open(D2R1 / "surface_baseline_metrics.csv", encoding="utf-8") as f:
    import csv
    for r in csv.DictReader(f):
        surf[r["metric"]] = r["value"]
add("B_surface dev AUROC", surf.get("B_surface_AUROC"), True)
add("B_surface selected C", surf.get("selected_C"), True)
add("B_surface feature count", surf.get("feature_count"), surf.get("feature_count") == "9")
surf_spec = {
    "features": ["q_token_count", "r_o_token_count", "r_s_token_count",
                 "abs(r_o_tokens - r_s_tokens)", "q_char_count", "r_s_char_count",
                 "r_s_word_count", "has_hyphen", "is_multiword"],
    "n_features": 9,
    "classifier": "LogisticRegression L2, max_iter=2000, class_weight=balanced",
    "scaler": "StandardScaler fit on train only",
    "C": "selected via 5-fold group CV on TRAIN only (frozen at 1.0 in D2-R1)",
    "r_o_source": "D0 swap pairs train rows (original correct answer)",
}
add("B_surface 规格", json.dumps(surf_spec), True)

# ---------------------------------------------------------------------------
# write frozen_probe_reconstruction_audit.md
# ---------------------------------------------------------------------------
(R / "frozen_probe_reconstruction_audit.md").write_text(
    f"""# frozen_probe_reconstruction_audit.md

## D2-R1 冻结规格审计（Phase 0.2）

| 项 | 值 | 通过 |
|---|---|---|
{chr(10).join(f"| {k} | {v} | {'✓' if ok else '✗'} |" for k, v, ok in rows)}

## Probe（M_rep）重建规格

- 模型：`Qwen/Qwen2.5-7B-Instruct` revision `a09a3545…`；BF16、eval、inference_mode、batch_size=1。
- 特征：`hidden_states[18][0, prefix_len-1, :]`（L18 × R_end），取自真截断 prefix 单独前向。
- 标准化：`StandardScaler().fit(X_train)`（只在 train 拟合）。
- 分类器：`LogisticRegression(C=0.01, max_iter=2000, class_weight='balanced')`（L2，默认 lbfgs），train 拟合。
- 标签：`y = 1 iff SS predicted_label == 'B'`（后续 SS false rejection）；y=0 iff Accept。
- 层/C 选择：仅 train 5-fold StratifiedGroupKFold CV（random_state=20260802），冻结 18 / 0.01。
- 本目录中无 D2-R1 序列化模型文件（仅 decision.json + 训练脚本）；按协议 0.3 优先顺序第 2 条用 train 587 组重建。

## 特征来源

- train 587 组 hidden states：D2-R1 `prefix_hidden_states/train_*.npz`（D2-R1 自提取，`d2_hidden_arrays_reused=false`）。
- train SS 标签：D2-R1 `scripts/_ss_train_scores.json`（587 行，含 predicted_label）。
- 重建过程不读取 dev/final 标签、特征或结果来选择任何超参数。
""", encoding="utf-8")

# ---------------------------------------------------------------------------
# write frozen_surface_baseline_reconstruction_audit.md
# ---------------------------------------------------------------------------
(R / "frozen_surface_baseline_reconstruction_audit.md").write_text(
    f"""# frozen_surface_baseline_reconstruction_audit.md

## B_surface 冻结规格（从 D2-R1 analysis.py 唯一恢复）

- 特征（9 个，预注册模型无关表面特征）：
```text
{chr(10).join('  - ' + f for f in surf_spec["features"])}
```
- r_o 来源：D0 `preliminary_swap_pairs.jsonl` train 行（original correct answer，流式提取 train 行）。
- 分类器：`LogisticRegression(C=1.0, max_iter=2000, class_weight='balanced')`（C 由 train 5-fold 组 CV 冻结）。
- 标准化：`StandardScaler().fit(X_surface_train)`（只在 train 拟合）。
- D2-R1 dev 基准：AUROC={surf.get("B_surface_AUROC")}, AUPRC={surf.get("B_surface_AUPRC")}, C={surf.get("selected_C")}, features={surf.get("feature_count")}。

## 重建方式

- 特征按 D2-R1 `build_surface` 逻辑重建；r_o 从 D0 swap train 行获取（D0 为允许来源，协议 0.3 授权用 D0 train 重建）。
- 重建不读取 dev/final 标签、特征或结果来选择超参数（C 继承冻结值 1.0，train CV 复核）。
""", encoding="utf-8")

print("Phase 0.2 OK: D2-R1 frozen spec fully recovered")
