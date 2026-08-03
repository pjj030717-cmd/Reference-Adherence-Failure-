#!/usr/bin/env python3
"""E01-D2: stream-filter D0 pairs to extract train rows; drop final-reserve rows
without writing their text anywhere in D2."""
from __future__ import annotations

import json
import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[2]

D0 = REPO_ROOT / "d0_jar_style_sciq_data_qualification_20260802"
D2 = REPO_ROOT / "d2_qwen25_7b_predecision_reference_state_localization_20260802"

fixed = json.loads((D0 / "fixed_split_indices.json").read_text(encoding="utf-8"))
train_ids = set(fixed["groups"]["train"])
res_ids = set(fixed["groups"]["final_reserve"])

train_pairs = []
dropped_res = 0
with open(D0 / "preliminary_swap_pairs.jsonl", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        p = json.loads(line)
        gid = p["original_group_id"]
        if gid in train_ids:
            # keep only D1-style fields (no split text needed; verify matches fixed index)
            train_pairs.append({"q": p["q"], "r_o": p["r_o"], "r_s": p["r_s"],
                                "c_o": p["c_o"], "c_s": p["c_s"],
                                "original_group_id": gid,
                                "swap_source_group_id": p["swap_source_group_id"]})
        elif gid in res_ids:
            dropped_res += 1
        # other ids shouldn't exist

if len(train_pairs) != 587:
    print(f"ERROR: train pairs={len(train_pairs)} expected 587")
    sys.exit(1)
print(f"train pairs: {len(train_pairs)}, dropped reserve rows: {dropped_res}")

with open(D2 / "scripts" / "_train_pairs.jsonl", "w", encoding="utf-8") as f:
    for p in train_pairs:
        f.write(json.dumps(p) + "\n")
print("wrote _train_pairs.jsonl")
