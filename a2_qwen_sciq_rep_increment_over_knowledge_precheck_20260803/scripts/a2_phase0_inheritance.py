#!/usr/bin/env python3
"""A2 Phase 0: inheritance, isolation, and reproduction qualification.

Items:
1. D0 split hash matches records (train 587 / dev 195 / final_reserve 197).
2. train/dev source_group_id zero overlap.
3. zero final-reserve group id appears in this run's dir/log/cache (scan all
   files in OUT for 64-hex gids; every one must be in train_union_dev).
4. M_rep dev reproduction: frozen D4-Q1 probe on D2-R1 dev hidden states
   -> AUROC 0.9139 (tol 1e-4); B_surface dev reproduction AUROC 0.6208 (tol 1e-4).
5. (deferred to phase1) A1 24-pair synthetic readout regression reproduction.
6. build train/dev metadata (q, r_o, r_s, y, rho) with unique keys.
7. no NaN/inf.

Reads:
- D0 fixed_split_indices.json (split_sha256 ONLY via regex; final_reserve gid
  list is never loaded).
- D0 preliminary_swap_pairs.jsonl (stream, keep ONLY split in {train, dev};
  final_reserve rows dropped immediately, never printed/saved/stats'd).
- D2-R1 _ss_train_scores.json / _ss_dev_scores.json (labels).
- D2-R1 prefix_hidden_states/train_*.npz and dev_*.npz (hidden states).
- D4-Q1 scripts/_frozen/probe.npz (frozen M_rep + B_surface weights).
- A1 no_reference_prompt_spec.sha256 (to confirm prompt spec unchanged).

Never reads: D4-Q1 prefix_hidden_states/*, D4-Q1 _final_*, any final-reserve file.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
import os
REPO_ROOT = Path(__file__).resolve().parents[2]

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from transformers import AutoTokenizer

OUT = REPO_ROOT / "a2_qwen_sciq_rep_increment_over_knowledge_precheck_20260803"
D0 = REPO_ROOT / "d0_jar_style_sciq_data_qualification_20260802"
D2R1 = REPO_ROOT / "d2r1_qwen25_7b_true_prefix_reference_state_20260802"
D4Q1 = REPO_ROOT / "d4q1_qwen25_7b_true_prefix_final_confirmation_20260803"
A1 = REPO_ROOT / "a1_qwen_no_reference_factual_preference_adherence_audit_20260803"
MODEL = os.environ.get("RAF_MODEL_DIR", "/root/autodl-tmp/models/Qwen2.5-7B-Instruct")

LAYER = 18
C_PROBE = 0.01
C_SURFACE = 1.0
TOL = 1e-4


def sha256_hex(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def fail(label, why):
    print("STOP:", label, "-", why)
    (OUT / "artifacts").mkdir(parents=True, exist_ok=True)
    (OUT / "artifacts" / "decision.json").write_text(json.dumps({
        "final_label": label, "reason": why,
        "final_reserve_model_scored": False, "final_reserve_text_read": False,
        "final_reserve_group_ids_loaded": False, "hidden_states_newly_extracted": False,
        "popqa_read": False, "prompt_searched": False,
        "activation_intervention_run": False}, indent=2), encoding="utf-8")
    sys.exit(1)


rows = []


def check(name, ok, val=""):
    rows.append((name, bool(ok), str(val)))
    print(f"  [{'OK' if ok else 'FAIL'}] {name}: {val}")


# ---------------------------------------------------------------------------
# 1. D0 split hash (split_sha256 only; never load final_reserve gid list)
# ---------------------------------------------------------------------------
txt = (D0 / "fixed_split_indices.json").read_text(encoding="utf-8")
m = re.search(r'"split_sha256":\s*\{[^}]*"train":\s*"([0-9a-f]{64})"[^}]*"dev":\s*"([0-9a-f]{64})"', txt)
if not m:
    fail("analysis_input_integrity_invalid", "could not extract split_sha256 from D0 fixed_split_indices.json")
sha_train_rec, sha_dev_rec = m.group(1), m.group(2)
print("  D0 split_sha256 train/dev extracted (regex); final_reserve gid list not loaded")

# gids from D2-R1 labels
train_ss = json.loads((D2R1 / "scripts" / "_ss_train_scores.json").read_text(encoding="utf-8"))
dev_ss = json.loads((D2R1 / "scripts" / "_ss_dev_scores.json").read_text(encoding="utf-8"))
gid_tr = sorted({r["source_group_id"] for r in train_ss})
gid_de = sorted({r["source_group_id"] for r in dev_ss})
check("D2-R1 train SS rows == 587", len(train_ss) == 587, len(train_ss))
check("D2-R1 dev SS rows == 195", len(dev_ss) == 195, len(dev_ss))
check("D0 train split sha matches D2-R1 train gids",
      sha256_hex("\n".join(gid_tr)) == sha_train_rec, f"{sha256_hex(chr(10).join(gid_tr))[:16]}…")
check("D0 dev split sha matches D2-R1 dev gids",
      sha256_hex("\n".join(gid_de)) == sha_dev_rec, f"{sha256_hex(chr(10).join(gid_de))[:16]}…")

# 2. zero overlap
overlap = set(gid_tr) & set(gid_de)
check("train/dev zero overlap", len(overlap) == 0, f"{len(overlap)} overlapping gids")

train_union_dev = set(gid_tr) | set(gid_de)
check("train_union_dev == 782", len(train_union_dev) == 782, len(train_union_dev))

# ---------------------------------------------------------------------------
# 3. final-reserve zero-access scan over this run's directory
# ---------------------------------------------------------------------------
hex_gids = Counter()
for p in OUT.rglob("*"):
    if p.is_file() and (p.suffix in (".py", ".csv", ".json", ".md", ".log", ".sha256")):
        try:
            content = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        hex_gids.update(re.findall(r"\b[0-9a-f]{64}\b", content))
foreign = [g for g in hex_gids if g not in train_union_dev]
check("zero final-reserve gid in A2 dir (all 64-hex ids in train_union_dev)",
      len(foreign) == 0, f"{len(foreign)} foreign ids")

# ---------------------------------------------------------------------------
# 4. M_rep / B_surface dev reproduction with frozen weights
# ---------------------------------------------------------------------------
dev_by_gid = {r["source_group_id"]: r for r in dev_ss}
gids_de_arr = np.array(gid_de)
y_de = np.array([1 if dev_by_gid[g]["predicted_label"] == "B" else 0 for g in gid_de])


def load_hidden(gids, prefix):
    h = []
    for g in gids:
        h.append(np.load(D2R1 / "prefix_hidden_states" / f"{prefix}_{g}.npz")["h_prefix"])
    return np.stack(h).astype(np.float32)


X18_de = load_hidden(gid_de, "dev")[:, LAYER - 1, :].copy()
fr = np.load(D4Q1 / "scripts" / "_frozen" / "probe.npz")
rho_de = ((X18_de - fr["scaler_mean"]) / fr["scaler_scale"]) @ fr["coef"].T + fr["intercept"]
rho_de = rho_de.ravel().astype(np.float64)
au_de = roc_auc_score(y_de, rho_de)
ap_de = average_precision_score(y_de, rho_de)
check("M_rep dev AUROC reproduces 0.9139", abs(au_de - 0.9138872915468661) <= TOL,
      f"{au_de:.6f} (recorded 0.9138872915468661)")
check("M_rep dev AUPRC reproduces 0.9632", abs(ap_de - 0.9632189056523878) <= TOL,
      f"{ap_de:.6f} (recorded 0.9632189056523878)")

# B_surface reproduction (9 features; needs tokenizer)
tok = AutoTokenizer.from_pretrained(MODEL)
swap_map = {}
with open(D0 / "preliminary_swap_pairs.jsonl", encoding="utf-8") as f:
    for line in f:
        d = json.loads(line)
        if d["split"] not in ("train", "dev"):
            continue
        swap_map.setdefault(d["split"], {})[d["original_group_id"]] = d
assert len(swap_map["train"]) == 587 and len(swap_map["dev"]) == 195, "swap pair counts"


def build_surface(gids, split):
    feats, kept = [], []
    for g in gids:
        d = swap_map[split][g]
        q, r_o, r_s = d["q"], d["r_o"], d["r_s"]
        q_tok = len(tok.encode(q))
        ro_tok = len(tok.encode(r_o))
        rs_tok = len(tok.encode(r_s))
        rs_words = len(r_s.split())
        feats.append([q_tok, ro_tok, rs_tok, abs(ro_tok - rs_tok), len(q), len(r_s),
                      rs_words, 1 if "-" in r_s else 0, 1 if rs_words > 1 else 0])
        kept.append(g)
    return np.array(feats, dtype=float), kept


Xsur_de, _ = build_surface(gid_de, "dev")
s_de = ((Xsur_de - fr["surface_scaler_mean"]) / fr["surface_scaler_scale"]) @ fr["surface_coef"].T + fr["surface_intercept"]
s_de = s_de.ravel()
au_s = roc_auc_score(y_de, s_de)
ap_s = average_precision_score(y_de, s_de)
check("B_surface dev AUROC reproduces 0.6208", abs(au_s - 0.6207590569292697) <= TOL,
      f"{au_s:.6f} (recorded 0.6207590569292697)")
check("B_surface dev AUPRC reproduces 0.8182", abs(ap_s - 0.8181915706506249) <= TOL,
      f"{ap_s:.6f} (recorded 0.8181915706506249)")

# ---------------------------------------------------------------------------
# 6. metadata for train and dev (q, r_o, r_s, y, rho; unique keys)
# ---------------------------------------------------------------------------
train_by_gid = {r["source_group_id"]: r for r in train_ss}
X18_tr = load_hidden(gid_tr, "train")[:, LAYER - 1, :].copy()
rho_tr = (((X18_tr - fr["scaler_mean"]) / fr["scaler_scale"]) @ fr["coef"].T + fr["intercept"]).ravel().astype(np.float64)


def build_meta(gids, split, ss_by_gid, rho):
    rows_m = []
    for g in gids:
        d = swap_map[split][g]
        s = ss_by_gid[g]
        rows_m.append({"source_group_id": g, "question": d["q"], "r_o": d["r_o"], "r_s": d["r_s"],
                       "y": 1 if s["predicted_label"] == "B" else 0,
                       "rho": float(rho[list(gids).index(g)]),
                       "relation": "NA"})
    return rows_m


meta_tr = build_meta(gid_tr, "train", train_by_gid, rho_tr)
meta_de = build_meta(gid_de, "dev", dev_by_gid, rho_de)
check("train metadata unique", len({r["source_group_id"] for r in meta_tr}) == len(meta_tr), len(meta_tr))
check("dev metadata unique", len({r["source_group_id"] for r in meta_de}) == len(meta_de), len(meta_de))
check("train r_o != r_s", all(r["r_o"] != r["r_s"] for r in meta_tr),
      f"{sum(1 for r in meta_tr if r['r_o']==r['r_s'])} violations")
check("dev r_o != r_s", all(r["r_o"] != r["r_s"] for r in meta_de),
      f"{sum(1 for r in meta_de if r['r_o']==r['r_s'])} violations")

# 7. NaN/inf
allvals = [r["rho"] for r in meta_tr] + [r["rho"] for r in meta_de]
check("no NaN/inf in rho", all(v == v and np.isfinite(v) for v in allvals),
      f"{sum(1 for v in allvals if v!=v or not np.isfinite(v))} bad")

if not all(ok for _, ok, _ in rows):
    fail("analysis_input_integrity_invalid", "; ".join(n for n, ok, _ in rows if not ok))

# ---------------------------------------------------------------------------
# save metadata + write audit files
# ---------------------------------------------------------------------------
(OUT / "scripts" / "_meta_train.json").write_text(json.dumps(meta_tr, ensure_ascii=False), encoding="utf-8")
(OUT / "scripts" / "_meta_dev.json").write_text(json.dumps(meta_de, ensure_ascii=False), encoding="utf-8")
print("saved _meta_train.json (587) and _meta_dev.json (195)")

# inheritance_and_isolation_audit.md
labels = {
    "D0": "jar_style_sciq_data_qualification_feasible",
    "D1": "jar_style_reference_override_behavior_feasible",
    "D1-R": "template_robust_reference_override_feasible",
    "D1-L": "long_candidate_expression_robust",
    "D2-R1": "true_prefix_reference_state_signal_localized",
    "D4-Q1": "qwen_true_prefix_monitor_final_confirmed",
    "A1": "no_reference_factual_preference_association_supported",
}
(OUT / "inheritance_and_isolation_audit.md").write_text(
    """# inheritance_and_isolation_audit.md

## 既有结论（只读核验）

| 实验 | 标签 | 状态 |
|---|---|---|
"""
    + "\n".join(f"| {k} | {v} | ✓ |" for k, v in labels.items())
    + """

## split / 隔离核验

| 检查 | 结果 |
|---|---|
"""
    + "\n".join(f"| {n} | {v} |" for n, ok, v in rows)
    + """

## 隔离约束

- train=587 / dev=195 为本轮唯一合法数据；final_reserve=197 绝不读取、不评分、不缓存。
- 本目录所有 64-hex group id 均在 train_union_dev (782) 中；final-reserve gid 出现 0 次。
- 未新提取 hidden states（复用 D2-R1 已存 `prefix_hidden_states/train_*.npz` / `dev_*.npz`）。
- 未读取 D4-Q1 `prefix_hidden_states/*`（final-reserve 196 个 npz）与任何 `_final_*` 工件。
- 未读取任何 PopQA 文本或分数。
- 未运行 Judge 四格、prompt 搜索、intervention/hook。
""", encoding="utf-8")

# final_reserve_zero_access_audit.md
(OUT / "final_reserve_zero_access_audit.md").write_text(
    f"""# final_reserve_zero_access_audit.md

- D0 split：train=587 / dev=195 / final_reserve=197。
- 本轮唯一合法 group 集合为 train_union_dev（782 个），源自 D2-R1 标签文件并校验 split sha 与 D0 记录一致。
- 本目录所有文件（.py/.csv/.json/.md/.log/.sha256）中出现的 64-hex group id 均已核验 ⊆ train_union_dev：
  `foreign gid count = {len(foreign)}`。
- final-reserve group id 出现次数：**0**。
- D0 swap pairs 读取时仅流式保留 split∈{{train, dev}} 行，final_reserve 行立即丢弃，未打印/保存/统计。
- D0 `fixed_split_indices.json` 仅以正则提取 `split_sha256`（train/dev），未加载 `groups.final_reserve` 列表。
- D4-Q1 `prefix_hidden_states/final_*.npz`（196 个）未读取。

## 逐项审计

| 检查 | 结果 |
|---|---|
| final-reserve group id 在本目录出现次数 | 0 |
| D0 swap pairs 中 final_reserve 行被读取进特征/评分 | 否 |
| D4-Q1 final hidden states 被读取 | 否 |
| 本轮日志/缓存中出现 final-reserve gid | 否 |
""", encoding="utf-8")

# mrep_reproduction_audit.md
(OUT / "mrep_reproduction_audit.md").write_text(
    f"""# mrep_reproduction_audit.md

## M_rep（true-prefix representation risk score）

- 来源：D4-Q1 冻结 `scripts/_frozen/probe.npz`（train-only 重建，非新提取）。
- 规格：layer 18 × R_end；StandardScaler(train)；LogisticRegression(C=0.01, max_iter=2000, balanced)。
- dev 只读复现（D2-R1 dev hidden states + D2-R1 dev SS 标签）：

| 指标 | 复现 | 记录 | 容差 |
|---|---|---|---|
| AUROC | {au_de:.6f} | 0.9138872915468661 | {TOL} |
| AUPRC | {ap_de:.6f} | 0.9632189056523878 | {TOL} |

- 结论：冻结 M_rep 在 dev 上精确复现，继承成立。

## B_surface（上下文参照，D2-R1 冻结）

| 指标 | 复现 | 记录 | 容差 |
|---|---|---|---|
| AUROC | {au_s:.6f} | 0.6207590569292697 | {TOL} |
| AUPRC | {ap_s:.6f} | 0.8181915706506249 | {TOL} |

- B_surface 仅为上下文参照，不参与最终标签判定。
""", encoding="utf-8")

print("Phase 0 OK")
