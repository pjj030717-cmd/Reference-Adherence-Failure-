#!/usr/bin/env python3
"""D1-R-A Phase 0: template source recovery + hash/placeholder contract.

Sources (allowed, read-only):
  D1R:
    - candidate_template_robustness_spec.json
    - scripts/d1r_template_spec.py  (executable definition)
    - scripts/d1r_eval.py           (executable rendering code path)
  D1:
    - scripts/_prompt_constants.json (user template, continuations)
    - synthetic_pair_manifest.json   (T0-rendered synthetic candidates)
    - scripts/_dev_pairs.jsonl       (dev-only c_o/c_s, T0 render check)
  D0:
    - candidate_rendering_spec.json  (T0 base template)

Outputs (Phase 0): canonical_candidate_templates.json + canonical_candidate_templates.sha256
"""
from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[2]

R = REPO_ROOT / "d1ra_candidate_template_provenance_diversity_audit_20260803"
D1R = REPO_ROOT / "d1r_qwen25_7b_jar_style_sciq_template_robustness_20260802"
D1 = REPO_ROOT / "d1_qwen25_7b_jar_style_sciq_behavior_gate_20260802"
D0 = REPO_ROOT / "d0_jar_style_sciq_data_qualification_20260802"


def sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def fail(label: str, why: str):
    print("STOP:", label, "-", why)
    (R / "artifacts").mkdir(parents=True, exist_ok=True)
    (R / "artifacts" / "decision.json").write_text(json.dumps(
        {"final_label": label, "reason": why,
         "template_provenance_unrecoverable": label == "template_provenance_unrecoverable",
         "template_contract_inconsistent": label == "template_contract_inconsistent"}, indent=2), encoding="utf-8")
    import sys
    sys.exit(1)


# ---------------------------------------------------------------------------
# 1. collect template definitions from every executable/structured source
# ---------------------------------------------------------------------------
def collect_from_json(path, loc):
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    t = d.get("templates")
    if not isinstance(t, dict):
        return {}
    return {k: {"template": v["template"], "source": str(path), "locator": f"templates.{k}.template", "kind": "json"} for k, v in t.items()}


def collect_from_py(path, loc):
    src = Path(path).read_text(encoding="utf-8")
    m = re.search(r"TEMPLATES\s*=\s*\{", src)
    if not m:
        return {}
    # parse dict literal via ast
    tree = ast.parse(src)
    found = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == "TEMPLATES" and isinstance(node.value, ast.Dict):
                    for k, v in zip(node.value.keys, node.value.values):
                        if isinstance(k, ast.Constant) and isinstance(v, ast.Constant):
                            found[k.value] = v.value
    return {k: {"template": v, "source": str(path), "locator": "TEMPLATES dict (executable)", "kind": "python"} for k, v in found.items()}


def collect_from_eval_py(path):
    src = Path(path).read_text(encoding="utf-8")
    tree = ast.parse(src)
    found = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == "TEMPLATES" and isinstance(node.value, ast.Dict):
                    for k, v in zip(node.value.keys, node.value.values):
                        if isinstance(k, ast.Constant) and isinstance(v, ast.Constant):
                            found[k.value] = v.value
    return found

templates = {}  # id -> list of {template, source, locator, kind}


def merge(collected: dict):
    for k, v in collected.items():
        templates.setdefault(k, []).append(v)


merge(collect_from_json(D1R / "candidate_template_robustness_spec.json", "templates"))
merge(collect_from_py(D1R / "scripts" / "d1r_template_spec.py", "TEMPLATES"))
merge(collect_from_py(D1R / "scripts" / "d1r_eval.py", "TEMPLATES"))

print("collected template ids:", sorted(templates.keys()))
for k in sorted(templates):
    print(f"  {k}: {templates[k][0]['template']!r} from {len(templates[k])} sources")

# multiple-source per-template verification
by_id = templates

ok_all = True
detail = []
for tid in ("T0", "T1", "T2"):
    srcs = by_id.get(tid, [])
    if not srcs:
        ok_all = False
        detail.append(f"{tid}: NO source")
        continue
    strs = {s["template"] for s in srcs}
    if len(strs) != 1:
        ok_all = False
        detail.append(f"{tid}: inconsistent strings {strs}")
    else:
        s = next(iter(strs))
        h = sha256(s)
        detail.append(f"{tid}: {s!r} sha256={h} sources={len(srcs)}")
        # placeholder contract
        phs = re.findall(r"<(\w+)>", s)
        if phs != ["answer"]:
            ok_all = False
            detail.append(f"  PLACEHOLDER CONTRACT FAIL: {phs}")
        else:
            detail.append(f"  placeholder: {phs} (single <answer>)")
        # T0/T1/T2 must not be identical
for tid in ("T0", "T1", "T2"):
    for tid2 in ("T0", "T1", "T2"):
        if tid < tid2 and by_id.get(tid) and by_id.get(tid2):
            if by_id[tid][0]["template"] == by_id[tid2][0]["template"]:
                ok_all = False
                detail.append(f"  IDENTICAL: {tid} == {tid2}")
print("\n".join(detail))

if not ok_all:
    fail("template_contract_inconsistent", "; ".join(detail))

# ---------------------------------------------------------------------------
# 2. T0 vs D1/D0 base rendering alignment
# ---------------------------------------------------------------------------
T0 = by_id["T0"][0]["template"]
assert T0 == "The answer is <answer>.", f"T0 unexpected: {T0!r}"

# D0 rendering spec template
d0 = json.loads((D0 / "candidate_rendering_spec.json").read_text(encoding="utf-8"))
d0_tpl = d0["template"]
ok_d0 = d0_tpl == T0

# D1 dev pairs c_o/c_s render check (dev-only)
T0_final = T0
dev_bad = 0
with open(D1 / "scripts" / "_dev_pairs.jsonl", encoding="utf-8") as f:
    for line in f:
        d = json.loads(line)
        if d.get("split") != "dev":
            continue
        if d["c_o"] != T0_final.replace("<answer>", d["r_o"]):
            dev_bad += 1
        if d["c_s"] != T0_final.replace("<answer>", d["r_s"]):
            dev_bad += 1
ok_d1 = dev_bad == 0

# D1-R t0_reproduction_audit.csv candidate check (dev-only)
D1R_T0 = T0
audit_bad = 0
import csv
with open(D1R / "t0_reproduction_audit.csv", encoding="utf-8") as f:
    for r in csv.DictReader(f):
        cell, ref, cand = r["cell"], r["reference"], r["candidate"]
        exp = D1R_T0.replace("<answer>", ref)
        if cell in ("OO", "SS"):
            if cand != exp:
                audit_bad += 1
        else:
            if not (cand.startswith("The answer is ") and cand.endswith(".")):
                audit_bad += 1
ok_d1r_audit = audit_bad == 0

print(f"T0 vs D0 rendering spec: {ok_d0} (spec={d0_tpl!r})")
print(f"T0 vs D1 dev pairs c_o/c_s: {ok_d1} (bad={dev_bad})")
print(f"T0 vs D1-R t0_reproduction_audit: {ok_d1r_audit} (bad={audit_bad})")

if not (ok_d0 and ok_d1 and ok_d1r_audit):
    fail("template_contract_inconsistent",
         f"T0 alignment: D0={ok_d0} D1dev={ok_d1} D1Raudit={ok_d1r_audit}")

# ---------------------------------------------------------------------------
# 3. canonical templates artifact
# ---------------------------------------------------------------------------
canonical = {
    "T0": {"template": T0, "utf8_sha256": sha256(T0),
           "sources": [{"path": str(D1R / "candidate_template_robustness_spec.json"), "locator": "templates.T0.template"},
                       {"path": str(D1R / "scripts" / "d1r_template_spec.py"), "locator": "TEMPLATES['T0']"},
                       {"path": str(D1R / "scripts" / "d1r_eval.py"), "locator": "TEMPLATES['T0']"}]},
    "T1": {"template": by_id["T1"][0]["template"], "utf8_sha256": sha256(by_id["T1"][0]["template"]),
           "sources": [{"path": str(D1R / "candidate_template_robustness_spec.json"), "locator": "templates.T1.template"},
                       {"path": str(D1R / "scripts" / "d1r_template_spec.py"), "locator": "TEMPLATES['T1']"},
                       {"path": str(D1R / "scripts" / "d1r_eval.py"), "locator": "TEMPLATES['T1']"}]},
    "T2": {"template": by_id["T2"][0]["template"], "utf8_sha256": sha256(by_id["T2"][0]["template"]),
           "sources": [{"path": str(D1R / "candidate_template_robustness_spec.json"), "locator": "templates.T2.template"},
                       {"path": str(D1R / "scripts" / "d1r_template_spec.py"), "locator": "TEMPLATES['T2']"},
                       {"path": str(D1R / "scripts" / "d1r_eval.py"), "locator": "TEMPLATES['T2']"}]},
    "placeholder": "<answer>",
    "render_rule": "<answer> is replaced by D0-frozen r_o/r_s normalized text (NFKC+trim+whitespace collapse).",
    "t0_alignments": {"D0_rendering_spec": ok_d0, "D1_dev_pairs": ok_d1, "D1R_t0_reproduction_audit": ok_d1r_audit},
}
(R / "canonical_candidate_templates.json").write_text(json.dumps(canonical, indent=2), encoding="utf-8")
(R / "canonical_candidate_templates.sha256").write_text(
    json.dumps({k: {"utf8_sha256": v["utf8_sha256"], "template": v["template"]} for k, v in canonical.items()
                if k in ("T0", "T1", "T2")}, indent=2) + "\n", encoding="utf-8")
print("wrote canonical_candidate_templates.json + .sha256")
print("Phase 0 OK")
