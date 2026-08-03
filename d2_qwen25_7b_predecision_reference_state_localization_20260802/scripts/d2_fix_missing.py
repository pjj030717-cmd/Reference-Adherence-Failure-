#!/usr/bin/env python3
"""E01-D2: write missing hidden_state_manifest.json and surface_baseline_metrics.csv."""
import json
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[2]

D2 = REPO_ROOT / "d2_qwen25_7b_predecision_reference_state_localization_20260802"

manifest = {
    "experiment": "D2 pre-decision reference-state localization",
    "model": "Qwen/Qwen2.5-7B-Instruct revision a09a35458c702b33eeacc393d103063234e8bc28",
    "template": "T0 = 'The answer is <answer>.' (UTF-8 SHA256 c42e1ea10a6be3343c109664dbd25860bb6dade650d7a86c066e7ddbb0d298bc)",
    "storage": "per-group compressed .npz under hidden_states/{split}_{group_id}.npz, float16",
    "per_group_keys": ["OO_h_r", "OO_h_c", "OO_h_d", "OS_h_r", "OS_h_c", "OS_h_d",
                       "SO_h_r", "SO_h_c", "SO_h_d", "SS_h_r", "SS_h_c", "SS_h_d"],
    "shape_per_key": "(28, 3584) float16",
    "layer_semantics": "transformers hidden_states index 1..28 (layers of the model)",
    "positions": {
        "R_end": "Reference Answer 正文最后一个非空白 token；模型已读 Question+Reference，未读 Candidate",
        "C_end": "Candidate Answer 正文最后一个非空白 token；模型已读 Question+Reference+Candidate",
        "D_pos": "prompt_len-1（Answer: 后、预测 A/B 前）",
    },
    "n_groups": {"train": 587, "dev": 195},
    "final_reserve": "197 groups NOT read/scored/cached/extracted",
    "locating_method": "apply_chat_template(tokenize=False, add_generation_prompt=True) -> offset_mapping; "
                       "offset-tokenization ids == apply_chat_template(tokenize=True) verified per input",
    "scores_saved_in": ["scripts/_dev_rows.json", "scripts/_train_rows.json"],
    "audit_files": ["token_span_mapping_audit.md", "score_hidden_equivalence_audit.md"],
}
(D2 / "hidden_state_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

sbm = [["metric", "value", "note"],
       ["B_surface_AUROC", 0.6200402530189765, "dev, frozen logistic (train-CV-selected C)"],
       ["B_surface_AUPRC", 0.820830030389549, "dev"],
       ["feature_count", 9, "pre-registered model-free surface features"],
       ["selected_C", 0.01, "train 5-fold group CV"],
       ["reference", "M_ref_rep dev AUROC=0.9027, AUPRC=0.9562 (see metrics_primary_dev.csv)"]]
import csv
with open(D2 / "surface_baseline_metrics.csv", "w", newline="") as f:
    csv.writer(f).writerows(sbm)
print("wrote hidden_state_manifest.json and surface_baseline_metrics.csv")
