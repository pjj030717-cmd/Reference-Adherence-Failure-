#!/usr/bin/env python3
"""D1-P Phase 3: deliverables.

- failure_examples.md  (SS false-rejection examples across baselines/templates)
- final_report.md
- artifacts/decision.json
- final label: prompt_baselines_dev_complete
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[2]

R = REPO_ROOT / "d1p_qwen25_7b_prompt_baselines_dev_20260803"
D1 = REPO_ROOT / "d1_qwen25_7b_jar_style_sciq_behavior_gate_20260802"

agg = json.loads((R / "scripts" / "_phase2_agg.json").read_text(encoding="utf-8"))
boot = agg["boot"]
tie_cnt = agg["tie_cnt"]
unp_cnt = agg["unp_cnt"]

# load dev pairs (dev-only)
dev_pairs = {}
for line in open(D1 / "scripts" / "_dev_pairs.jsonl", encoding="utf-8"):
    d = json.loads(line)
    dev_pairs[d["original_group_id"]] = d

# group-level verdicts
groups = list(csv.DictReader(open(R / "baseline_group_level_verdicts.csv", encoding="utf-8")))
gmap = {r["source_group_id"]: r for r in groups}

BASELINES = ["B_base", "B_direct", "B_CoT_gen"]
TEMPLATES = ["T0", "T1", "T2"]
CELLS = ["OO", "OS", "SO", "SS"]

# ---------------- failure_examples.md ----------------
def fr_ss_groups(base, tname):
    col = f"{base}_{tname}_SS_verdict"
    out = []
    for r in groups:
        v = r[col]
        if v != "A":  # SS expects Accept (A); anything else is false rejection
            out.append(r["source_group_id"])
    return out

lines = ["# failure_examples.md", "",
         "## 说明", "",
         "SS 格预期 Accept（A）。凡 SS verdict ≠ A 记为 false rejection（含 B 与 unparseable）。",
         "仅列 dev 组（允许读取），不含任何 train / final-reserve 文本。",
         ""]
for base in BASELINES:
    for tname in TEMPLATES:
        ids = fr_ss_groups(base, tname)
        lines.append(f"### {base} / {tname}：SS false-rejection {len(ids)} 组")
        # show up to 8 examples with dev text
        for gid in ids[:8]:
            p = dev_pairs[gid]
            v = gmap[gid][f"{base}_{tname}_SS_verdict"]
            d = gmap[gid].get(f"{base}_{tname}_SS_d_raw", "")
            d = "" if d is None or d == "" else f" d_raw={float(d):+.3f}"
            lines.append(f"- `{gid[:8]}` SS verdict={v}{d}：Q={p['q'][:60]!r} Ref={p['r_s'][:40]!r} Cand={p['c_s'][:40]!r}")
        if len(ids) > 8:
            lines.append(f"- … 共 {len(ids)} 组")
        lines.append("")
(R / "failure_examples.md").write_text("\n".join(lines), encoding="utf-8")
print("failure_examples.md written")

# ---------------- decision.json ----------------
def get_boot(base, tname, metric):
    for r in boot:
        if r["baseline"] == base and r["template"] == tname:
            return r
    return None

dec = {
    "final_label": "prompt_baselines_dev_complete",
    "phase0_inheritance": {
        "d0_label_ok": True, "d1_label_ok": True, "d1r_label_ok": True, "d1ra_label_ok": True,
        "templates_sha256_verified": True,
        "qwen_revision_consistent_with_d1": True,
        "teacher_forced_readout_inherited_from_d1": True},
    "phase1_semantic_qualification": {
        "B_base": {"pairwise_order_accuracy": "24/24", "per_class": "12/12+12/12", "ties": 0, "mean_delta_sign": "OK", "greedy_agrees": "24/24"},
        "B_direct": {"pairwise_order_accuracy": "24/24", "per_class": "12/12+12/12", "ties": 0, "mean_delta_sign": "OK", "greedy_agrees": "24/24"},
        "B_CoT_gen": {"parseable": "24/24", "match_to_A": "12/12", "mismatch_to_B": "12/12"}},
    "phase2_bbase_reproduction": {
        "vs_D1": "780/780", "vs_D1_R": "780/780", "repro_total": 780},
    "phase2_bootstrap": {
        "iterations": 2000, "seed": 20260815, "resample_unit": "source_group_id",
        "per_baseline_template": {
            f"{b}|{t}": {m: get_boot(b, t, m)[f"{m}_obs"] for m in ("SS_FR", "RPAG", "ACC_o", "ACC_s")}
            for b in BASELINES for t in TEMPLATES}},
    "aggregate_by_cell": {
        f"{b}|{t}": {c: agg["agg"][f"{b}|{t}"][c] for c in CELLS}
        for b in BASELINES for t in TEMPLATES},
    "tie_rate": {k: v for k, v in tie_cnt.items()},
    "unparseable_rate": {k: v for k, v in unp_cnt.items()},
    "final_reserve_read": False,
    "hidden_states_read": False,
    "probe_trained": False,
    "activation_intervention_run": False,
    "train_text_read": False,
}
(R / "artifacts").mkdir(parents=True, exist_ok=True)
(R / "artifacts" / "decision.json").write_text(json.dumps(dec, indent=2, ensure_ascii=False), encoding="utf-8")
print("decision.json written")

# ---------------- final_report.md ----------------
def cell_acc(base, tname, cell):
    c = agg["agg"][f"{base}|{tname}"][cell]
    return c[1] / c[0] if c[0] else float("nan")

# identify candidates: low SS false rejection across all three templates AND not degrading OO/OS/SO
def assess():
    report = []
    for base in BASELINES:
        frs = [get_boot(base, t, "SS_FR")["SS_FR_obs"] for t in TEMPLATES]
        oo = [cell_acc(base, t, "OO") for t in TEMPLATES]
        os_ = [cell_acc(base, t, "OS") for t in TEMPLATES]
        so = [cell_acc(base, t, "SO") for t in TEMPLATES]
        stable_low = all(x <= 0.25 for x in frs)   # pre-registered "low" threshold for candidate fix
        degrades = any(x < 0.85 for x in oo) or any(x < 0.85 for x in os_) or any(x < 0.85 for x in so)
        report.append({"baseline": base, "FR_SS_T0_T1_T2": frs, "ACC_o": oo, "ACC_os": os_, "ACC_so": so,
                       "stable_low_FR_SS": stable_low, "degrades_other_cells": degrades,
                       "candidate_prompt_fix": stable_low and not degrades})
    return report

assessments = assess()

lines = ["# D1-P：Prompt Baseline 的 Reference-Adherence 行为预检 — 最终报告", "",
         "## 结果总表", "",
         "| 问题 | 结果 |",
         "|---|---|",
         "| D0/D1/D1-R/D1-R-A 是否可唯一继承？ | 是（四个标签与模板 SHA256 全部通过） |",
         "| B_base 是否逐行复现既有 development 行为？ | 是（T0 四格 780/780 与 D1 及 D1-R 一致） |",
         "| B_direct 的读出语义是否有效？ | 是（24/24，12/12+12/12，ties=0） |",
         "| B_CoT_gen 的生成与解析是否有效？ | 是（24/24 可解析，MATCH→A 12/12，MISMATCH→B 12/12） |",
         "| 哪些 baseline 在三个模板下均保持低 SS false rejection？ | 无（全部 baseline×模板 FR_SS 在 0.74–0.83） |",
         "| 是否读取 final-reserve？ | 否 |",
         "| 是否提取 hidden states / 训练 Probe？ | 否 |",
         "| 最终标签 | prompt_baselines_dev_complete |",
         "",
         "## 方法", "",
         "- Judge：Qwen2.5-7B-Instruct（revision a09a3545…，BF16，eval，inference_mode），batch_size=1。",
         "- B_base：逐字继承 D1 基础 prompt 与 teacher-forced A/B 读出（logits 位置 prompt_len-1，A/B token 362/425）。",
         "- B_direct：在 system 既有任务说明后、Question 前插入固定 Reference 指令。",
         "- B_CoT_gen：以 B_direct 为基础追加 CoT 指令；greedy 生成（max_new_tokens=128，无 stop 后处理），"
         "仅接受最后一个非空行严格等于 `Final verdict: A` / `Final verdict: B`。",
         "- 数据：D1 的 195 个 dev groups × 四格（OO/OS/SO/SS），每个 baseline×模板使用完全相同的四格内容。",
         "- Bootstrap：group 重采样（source_group_id），2000 次，seed=20260815，95% CI（2.5/97.5 分位）。",
         "",
         "## Phase 1 读出语义资格", "",
         "| baseline | 24/24 | MATCH→A | MISMATCH→B | ties | mean_delta 符号 | greedy 一致 |",
         "|---|---|---|---|---|---|---|",
         "| B_base | 24/24 | 12/12 | 12/12 | 0 | 正确（+20.5/−22.9） | 24/24 |",
         "| B_direct | 24/24 | 12/12 | 12/12 | 0 | 正确（+21.5/−24.9） | 24/24 |",
         "| B_CoT_gen | 24/24 可解析 | 12/12 | 12/12 | — | — | — |",
         "",
         "## Phase 2：dev 四格行为（195 groups）", "",
         "| baseline | 模板 | ACC_o | ACC_s | RPAG | SS 错拒率 | SS 错拒组数 | SO 误纳率 | tie | unparseable |",
         "|---|---|---|---|---|---|---|---|---|---|",
         ]
for base in BASELINES:
    for t in TEMPLATES:
        frs = get_boot(base, t, "SS_FR")
        oo = cell_acc(base, t, "OO")
        ss = cell_acc(base, t, "SS")
        so = cell_acc(base, t, "SO")
        os_ = cell_acc(base, t, "OS")
        n_ss = agg["agg"][f"{base}|{t}"]["SS"][0]
        n_fr = n_ss - agg["agg"][f"{base}|{t}"]["SS"][1]
        tie = tie_cnt.get(f"{base}|{t}", 0)
        unp = unp_cnt.get(f"{base}|{t}", 0)
        lines.append(f"| {base} | {t} | {oo:.4f} | {ss:.4f} | {frs['RPAG_obs']:.4f} | {frs['SS_FR_obs']:.4f} | {n_fr} | {1-so:.4f} | {tie} | {unp} |")
lines += ["",
          "### Bootstrap 95% CI（seed 20260815，2000 次）", "",
          "| baseline | 模板 | SS 错拒率 CI | RPAG CI | ACC_o CI | ACC_s CI |",
          "|---|---|---|---|---|---|"]
for r in boot:
    lines.append(f"| {r['baseline']} | {r['template']} | [{r['SS_FR_ci_low']:.4f}, {r['SS_FR_ci_high']:.4f}] | "
                 f"[{r['RPAG_ci_low']:.4f}, {r['RPAG_ci_high']:.4f}] | "
                 f"[{r['ACC_o_ci_low']:.4f}, {r['ACC_o_ci_high']:.4f}] | "
                 f"[{r['ACC_s_ci_low']:.4f}, {r['ACC_s_ci_high']:.4f}] |")
lines += ["",
          "## 解释（仅允许的层级）", "",
          "- 没有任何 baseline 在 T0/T1/T2 三个模板下保持低 SS false rejection（预注册阈值 ≤0.25）；"
          "FR_SS 全部在 0.74–0.83。",
          "- B_direct 相对 B_base 在所有模板下有 1.5–2.6 个百分点的 SS 错拒率下降（T0: 0.759→0.744，T1: 0.826→0.805，T2: 0.826→0.800），"
          "但远未达到低错拒，不足以称为候选 prompt 修复。",
          "- B_CoT_gen 相对 B_base 的 SS 错拒率变化不系统（T0: 0.759→0.754，T1: 0.826→0.810，T2: 0.826→0.759），且 T1 的 SO 误纳率恶化"
          "（0.0205→0.1282），故不构成稳定缓解。",
          "- 结论：对当前 Qwen2.5-7B Judge，仅修改任务指令（B_direct）或追加显式比较 CoT（B_CoT_gen）均不能在既有 "
          "T0/T1/T2 表达下稳定缓解 reference-adherence failure。",
          "- 本轮为 development baseline 预检，不构成 final-reserve 确认；不修改 H2、Probe、层、token、readout 或因果路径设定。",
          ""]
(R / "final_report.md").write_text("\n".join(lines), encoding="utf-8")
print("final_report.md written")

# verify all deliverables present
required = ["final_report.md", "inheritance_audit.md", "baseline_prompt_spec.json", "baseline_prompt_spec.sha256",
            "teacher_forcing_semantic_audit.csv", "cot_generation_semantic_audit.csv",
            "baseline_metrics_by_template_cell.csv", "baseline_group_level_verdicts.csv",
            "bootstrap_baseline_metrics.csv", "bbase_reproduction_audit.csv", "cot_parse_audit.csv",
            "failure_examples.md", "artifacts/decision.json"]
for f in required:
    print(("OK " if (R / f).exists() else "MISSING ") + f)
