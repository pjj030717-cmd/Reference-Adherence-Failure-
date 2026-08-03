#!/usr/bin/env python3
"""A1 Phase 0: inheritance + dev isolation + synthetic factual-choice readout regression.

0A. dev label completeness (read-only):
  - SciQ: D1 four_cell_scores_dev.csv -> 195 groups; y_SS inherited verbatim
  - PopQA: E1 scripts/_dev_input.jsonl + scripts/_dev_fourcell_rows.csv -> 2815 groups
  - each source_group_id unique; r_o != r_s; y_SS uniquely recoverable; no missing fields;
    group set matches each dataset's dev manifest.

0B. Synthetic factual-choice manifest: 24 basic factual pairs (12 correct in A, 12 in B),
  written to synthetic_factual_choice_manifest.json BEFORE any real-dev scoring, then
  synthetic_factual_choice_audit.csv after scoring (done in Phase 1 loader or here? here).

This script writes the manifest + 0A audits. On 0A failure -> behavioral_attribution_input_invalid.
The synthetic readout regression itself is run in phase1 script (model needed) and gates on
factual_preference_readout_invalid. Here we only WRITE the frozen manifest + do 0A.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
import unicodedata
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[2]

OUT = REPO_ROOT / "a1_qwen_no_reference_factual_preference_adherence_audit_20260803"
D1 = REPO_ROOT / "d1_qwen25_7b_jar_style_sciq_behavior_gate_20260802"
E1 = REPO_ROOT / "e1_qwen25_7b_popqa_h1_behavior_gate_20260803"
E0R2 = REPO_ROOT / "e0r2_popqa_global_external_data_qualification_20260803"

rows = []


def check(name, ok, val=""):
    rows.append((name, ok, val))
    print(f"  [{'OK' if ok else 'FAIL'}] {name}: {val}")


def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def norm(s):
    s = unicodedata.normalize("NFKC", str(s))
    s = re.sub(r"\s+", " ", s.strip())
    return s.lower()


# ============ 0A: SciQ ============
d1_fc = list(csv.DictReader(open(D1 / "four_cell_scores_dev.csv", encoding="utf-8")))
sciq_byg = {}
for r in d1_fc:
    sciq_byg.setdefault(r["source_group_id"], {})[r["cell"]] = r
gids_sciq = sorted(sciq_byg.keys())
check("SciQ groups == 195", len(gids_sciq) == 195, str(len(gids_sciq)))
check("SciQ 4 cells each", all(all(c in sciq_byg[g] for c in ("OO", "OS", "SO", "SS")) for g in gids_sciq), "")

sciq_records = []
for g in gids_sciq:
    oo, ss = sciq_byg[g]["OO"], sciq_byg[g]["SS"]
    r_o, r_s = oo["reference"], sciq_byg[g]["SO"]["reference"]
    y_ss = 1 if ss["predicted_label"] == "B" else 0
    sciq_records.append({"source_group_id": g, "dataset": "SciQ", "question": oo["question"],
                         "r_o": r_o, "r_s": r_s, "y_SS": y_ss})
check("SciQ r_o != r_s", all(norm(r["r_o"]) != norm(r["r_s"]) for r in sciq_records),
      f"{sum(1 for r in sciq_records if norm(r['r_o'])==norm(r['r_s']))} violations")
check("SciQ y_SS=1 count == 148", sum(1 for r in sciq_records if r["y_SS"] == 1) == 148,
      str(sum(1 for r in sciq_records if r["y_SS"] == 1)))

# ============ 0A: PopQA ============
e1_in = []
with open(E1 / "scripts" / "_dev_input.jsonl", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            e1_in.append(json.loads(line))
e1_fc = list(csv.DictReader(open(E1 / "scripts" / "_dev_fourcell_rows.csv", encoding="utf-8")))
pop_byg = {}
for r in e1_fc:
    pop_byg.setdefault(r["source_group_id"], {})[r["cell"]] = r
gids_pop = sorted(pop_byg.keys())
check("PopQA groups == 2815", len(gids_pop) == 2815, str(len(gids_pop)))
check("PopQA dev input groups match fourcell groups", set(gids_pop) == {p["source_group_id"] for p in e1_in}, "")

in_by_gid = {p["source_group_id"]: p for p in e1_in}
pop_records = []
for g in gids_pop:
    p = in_by_gid[g]
    ss = pop_byg[g]["SS"]
    y_ss = 1 if ss["predicted_label"] == "B" else 0
    pop_records.append({"source_group_id": g, "dataset": "PopQA", "relation": p["relation"],
                        "question": p["question"], "r_o": p["r_o"], "r_s": p["r_s"], "y_SS": y_ss})
check("PopQA r_o != r_s", all(norm(r["r_o"]) != norm(r["r_s"]) for r in pop_records),
      f"{sum(1 for r in pop_records if norm(r['r_o'])==norm(r['r_s']))} violations")
check("PopQA y_SS=1 count == 144", sum(1 for r in pop_records if r["y_SS"] == 1) == 144,
      str(sum(1 for r in pop_records if r["y_SS"] == 1)))
check("PopQA 16 relations", len({r["relation"] for r in pop_records}) == 16,
      str(len({r["relation"] for r in pop_records})))

# dev manifest alignment
approved = json.loads((E0R2 / "approved_popqa_group_manifests.json").read_text(encoding="utf-8"))["dev"]
man_sha = sha256_hex("\n".join(gids_pop))
check("PopQA dev manifest sha == E0-R2 approved", man_sha == approved["sorted_group_id_sha256"],
      f"{man_sha[:16]}… vs {approved['sorted_group_id_sha256'][:16]}…")

# ---- write dev_group_alignment_audit.csv ----
with open(OUT / "dev_group_alignment_audit.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["dataset", "source_group_id", "relation", "r_o", "r_s", "y_SS"])
    w.writeheader()
    for r in sciq_records:
        w.writerow({"dataset": "SciQ", "source_group_id": r["source_group_id"], "relation": "NA",
                    "r_o": r["r_o"], "r_s": r["r_s"], "y_SS": r["y_SS"]})
    for r in pop_records:
        w.writerow({"dataset": "PopQA", "source_group_id": r["source_group_id"], "relation": r["relation"],
                    "r_o": r["r_o"], "r_s": r["r_s"], "y_SS": r["y_SS"]})

# save inputs for later phases
with open(OUT / "scripts" / "_sciq_dev.jsonl", "w", encoding="utf-8") as f:
    for r in sciq_records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
with open(OUT / "scripts" / "_popqa_dev.jsonl", "w", encoding="utf-8") as f:
    for r in pop_records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

all_ok_0a = all(ok for _, ok, _ in rows)
if not all_ok_0a:
    (OUT / "artifacts" / "decision.json").write_text(json.dumps({
        "final_label": "behavioral_attribution_input_invalid",
        "reason": "; ".join(n for n, ok, _ in rows if not ok),
        "final_reserve_model_scored": False, "final_reserve_text_read": False,
        "hidden_states_read": False, "probe_trained": False,
        "activation_intervention_run": False}, indent=2), encoding="utf-8")
    print("STOP: behavioral_attribution_input_invalid")
    sys.exit(1)

# ============ 0B: frozen synthetic manifest (24 basic factual pairs) ============
# 12 with correct answer in A, 12 with correct answer in B. Basic facts; no SciQ/PopQA text.
SYN = [
    # correct in A (12)
    ("A01", "What is the capital of France?", "Paris", "Rome", "A"),
    ("A02", "Which planet is known as the Red Planet?", "Mars", "Venus", "A"),
    ("A03", "How many sides does a triangle have?", "Three", "Five", "A"),
    ("A04", "What is the freezing point of water in Celsius?", "Zero", "One hundred", "A"),
    ("A05", "Which ocean is the largest?", "Pacific", "Atlantic", "A"),
    ("A06", "Who wrote the novel Moby-Dick?", "Herman Melville", "Mark Twain", "A"),
    ("A07", "What gas do plants absorb during photosynthesis?", "Carbon dioxide", "Oxygen", "A"),
    ("A08", "Which metal is liquid at room temperature?", "Mercury", "Copper", "A"),
    ("A09", "How many continents are there?", "Seven", "Six", "A"),
    ("A10", "What is the largest planet in the solar system?", "Jupiter", "Saturn", "A"),
    ("A11", "Which organ pumps blood through the body?", "Heart", "Liver", "A"),
    ("A12", "What instrument measures atmospheric pressure?", "Barometer", "Thermometer", "A"),
    # correct in B (12)
    ("B01", "What is the capital of Italy?", "Milan", "Rome", "B"),
    ("B02", "Which planet has the most prominent rings?", "Mars", "Saturn", "B"),
    ("B03", "How many legs does a spider have?", "Six", "Eight", "B"),
    ("B04", "What is the boiling point of water in Celsius?", "Zero", "One hundred", "B"),
    ("B05", "Which country is known as the Land of the Rising Sun?", "China", "Japan", "B"),
    ("B06", "Who painted the Mona Lisa?", "Michelangelo", "Leonardo da Vinci", "B"),
    ("B07", "Which gas do humans breathe in to live?", "Helium", "Oxygen", "B"),
    ("B08", "What is the fastest land animal?", "Lion", "Cheetah", "B"),
    ("B09", "How many strings does a standard guitar have?", "Four", "Six", "B"),
    ("B10", "What is the smallest planet in the solar system?", "Earth", "Mercury", "B"),
    ("B11", "Which is the longest river in Africa?", "Congo", "Nile", "B"),
    ("B12", "What is the currency of Japan?", "Yuan", "Yen", "B"),
]
(OUT / "synthetic_factual_choice_manifest.json").write_text(json.dumps(SYN, indent=2, ensure_ascii=False), encoding="utf-8")
check("synthetic manifest 24 pairs (12 A / 12 B)",
      len(SYN) == 24 and sum(1 for s in SYN if s[4] == "A") == 12 and sum(1 for s in SYN if s[4] == "B") == 12, "")

# ---- no_reference_prompt_spec.md + sha256 ----
PROMPT_SPEC = {
    "system": "Choose the option that is factually correct for the question.\nReply with only A or B.",
    "user_template": "Question:\n{q}\n\nOption A:\n{option_a}\n\nOption B:\n{option_b}\n\nAnswer:",
    "note": "no Reference / Candidate / Judge wording; ordinary two-choice factual QA",
    "continuation": {"A": " A", "A_id": 362, "B": " B", "B_id": 425},
    "teacher_forcing_pos": "prompt_len - 1",
    "k_definition": "k = (d_1 + d_2) / 2  with d_1 = l_A - l_B (Order1: A=r_o,B=r_s), d_2 = l_B - l_A (Order2: A=r_s,B=r_o)",
}
spec_text = json.dumps(PROMPT_SPEC, indent=2, ensure_ascii=False)
(OUT / "no_reference_prompt_spec.md").write_text(spec_text, encoding="utf-8")
(OUT / "no_reference_prompt_spec.sha256").write_text(sha256_hex(spec_text) + "  no_reference_prompt_spec.md\n", encoding="utf-8")
check("prompt spec written + sha", True, sha256_hex(spec_text)[:16] + "…")

# ---- inheritance_and_dev_isolation_audit.md ----
(OUT / "inheritance_and_dev_isolation_audit.md").write_text(
    """# inheritance_and_dev_isolation_audit.md

## 既有结论（只读核验）

| 实验 | 标签 | 状态 |
|---|---|---|
"""
    + "\n".join(f"| {n} | {v} | {'✓' if ok else '✗'} |" for n, ok, v in rows)
    + """

## dev 隔离

```text
source_stream_scanned_for_split_filter = false (PopQA dev 直接取自 E1 已隔离的 _dev_input.jsonl)
final_reserve_text_exposed_to_model = false
final_reserve_model_scored = false
train_text_exposed_to_model = false
hidden_states_read = false
probe_trained = false
activation_intervention_run = false
```

- SciQ dev：D1 `four_cell_scores_dev.csv`（195 group，SS 标签逐字继承）。
- PopQA dev：E1 `scripts/_dev_input.jsonl` + `scripts/_dev_fourcell_rows.csv`（2,815 group，SS 标签逐字继承）。
- 未读取任何 train / final-reserve 文本；未读取既有 final-reserve score/prediction 文件。

## 合成 manifest

- 24 条基础事实二选一（12 条正确在 A，12 条正确在 B），在真实 dev 评分前冻结写入
  `synthetic_factual_choice_manifest.json`；不得按模型输出修改。
""", encoding="utf-8")

print("Phase 0A OK (labels+isolation); synthetic manifest frozen")
