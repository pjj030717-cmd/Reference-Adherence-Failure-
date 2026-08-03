#!/usr/bin/env python3
"""D1-P Phase 1: readout semantic qualification.

1.1 B_base & B_direct: teacher-forced regression on the 24 synthetic pairs.
    requirements: 24/24, 12/12 MATCH->A, 12/12 MISMATCH->B, ties=0, mean_delta>0,
    greedy first-token direction agrees.
1.2 B_CoT_gen: greedy generation on the same 24 pairs.
    requirements: 24/24 parseable, MATCH->A 12/12, MISMATCH->B 12/12, no extra final-line format.

Outputs: teacher_forcing_semantic_audit.csv, cot_generation_semantic_audit.csv
"""
from __future__ import annotations

import csv
import json
import math
import sys
import unicodedata
from pathlib import Path
import os
REPO_ROOT = Path(__file__).resolve().parents[2]

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

R = REPO_ROOT / "d1p_qwen25_7b_prompt_baselines_dev_20260803"
D1 = REPO_ROOT / "d1_qwen25_7b_jar_style_sciq_behavior_gate_20260802"
MODEL = os.environ.get("RAF_MODEL_DIR", "/root/autodl-tmp/models/Qwen2.5-7B-Instruct")
CONST = json.loads((D1 / "scripts" / "_prompt_constants.json").read_text(encoding="utf-8"))
ACCEPT_ID, REJECT_ID = CONST["accept_id"], CONST["reject_id"]
spec = json.loads((R / "baseline_prompt_spec.json").read_text(encoding="utf-8"))
SYS_BASE = spec["system_base"]
USER_TMPL = spec["user_template"]

SYN = json.loads((D1 / "synthetic_pair_manifest.json").read_text(encoding="utf-8"))
assert len(SYN) == 24


def fail(label: str, why: str):
    print("STOP:", label, "-", why)
    (R / "artifacts").mkdir(parents=True, exist_ok=True)
    (R / "artifacts" / "decision.json").write_text(json.dumps(
        {"final_label": label, "reason": why,
         "B_base_ok": False, "B_direct_ok": False, "B_CoT_gen_ok": False}, indent=2), encoding="utf-8")
    sys.exit(1)


tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
                                             device_map="cuda", low_cpu_mem_usage=True)
model.eval()

# tokenizer continuation ids verify
ta = tok.encode(" A", add_special_tokens=False)
tb = tok.encode(" B", add_special_tokens=False)
print("continuation token ids:", ta, tb)
assert ta == [ACCEPT_ID] and tb == [REJECT_ID], "continuation ids mismatch D1"


def render_messages(system, question, reference, candidate):
    return [{"role": "system", "content": system},
            {"role": "user", "content": USER_TMPL.format(question=question, reference=reference, candidate=candidate)}]


def teacher_forced(system, question, reference, candidate):
    msgs = render_messages(system, question, reference, candidate)
    ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt")["input_ids"].to("cuda")
    plen = ids.shape[1]
    with torch.inference_mode():
        logits = model(ids).logits
    pos = plen - 1
    lA = logits[0, pos, ACCEPT_ID].item()
    lB = logits[0, pos, REJECT_ID].item()
    d_raw = lA - lB
    pred = "A" if d_raw > 0 else ("B" if d_raw < 0 else "TIE")
    gid = int(logits[0, pos].argmax().item())
    gtok = tok.decode([gid])
    g = gtok.strip()
    gpred = "A" if g == "A" else ("B" if g == "B" else f"OTHER({gtok!r})")
    return lA, lB, d_raw, pred, gpred


# ============================================================
# 1.1 B_base and B_direct teacher-forced
# ============================================================
all_tf_rows = []
for sysname, system in (("B_base", SYS_BASE), ("B_direct", spec["B_direct"]["system"])):
    rows = []
    for sid, q, ref, cand, exp in SYN:
        lA, lB, d, pred, gpred = teacher_forced(system, q, ref, cand)
        rows.append({"baseline": sysname, "id": sid, "expected_label": exp, "l_A": lA, "l_B": lB, "d_raw": d,
                     "predicted_label": pred, "correct": pred == exp, "greedy_pred": gpred,
                     "greedy_agrees": gpred == pred})
    acc = sum(1 for r in rows if r["correct"]) / 24
    accA = sum(1 for r in rows if r["expected_label"] == "A" and r["correct"]) / 12
    accB = sum(1 for r in rows if r["expected_label"] == "B" and r["correct"]) / 12
    ties = sum(1 for r in rows if r["predicted_label"] == "TIE")
    mean_A = sum(r["d_raw"] for r in rows if r["expected_label"] == "A") / 12
    mean_B = sum(r["d_raw"] for r in rows if r["expected_label"] == "B") / 12
    greedy_n = sum(1 for r in rows if r["greedy_agrees"])
    print(f"{sysname}: acc={acc} A={accA} B={accB} ties={ties} meanA={mean_A:.3f} meanB={mean_B:.3f} greedy={greedy_n}/24")
    all_tf_rows.extend(rows)
    # "mean_delta > 0" is interpreted per protocol as per-class sign:
    # MATCH-class mean d_raw > 0 AND MISMATCH-class mean d_raw < 0 (consistent with D1 medA/medB gate).
    if sysname == "B_base":
        if not (acc == 1.0 and accA == 1.0 and accB == 1.0 and ties == 0 and greedy_n == 24
                and mean_A > 0 and mean_B < 0):
            fail("prompt_baseline_inheritance_invalid",
                 f"B_base readout: acc={acc} A={accA} B={accB} ties={ties} greedy={greedy_n} meanA={mean_A} meanB={mean_B}")
        print("B_base readout regression PASSED")
    else:
        if not (acc == 1.0 and accA == 1.0 and accB == 1.0 and ties == 0 and greedy_n == 24
                and mean_A > 0 and mean_B < 0):
            fail("prompt_baseline_readout_invalid",
                 f"B_direct readout: acc={acc} A={accA} B={accB} ties={ties} greedy={greedy_n} meanA={mean_A} meanB={mean_B}")
        print("B_direct readout regression PASSED")

with open(R / "teacher_forcing_semantic_audit.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(all_tf_rows[0].keys()))
    w.writeheader()
    w.writerows(all_tf_rows)
for old in (R / "teacher_forcing_semantic_audit_bbase.csv", R / "teacher_forcing_semantic_audit_bdirect.csv"):
    if old.exists():
        old.unlink()

# ============================================================
# 1.2 B_CoT_gen greedy generation
# ============================================================
system_cot = spec["B_CoT_gen"]["system"]
parse_rows = []
ok_parse = 0
for sid, q, ref, cand, exp in SYN:
    msgs = render_messages(system_cot, q, ref, cand)
    enc = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt")
    ids = enc["input_ids"].to("cuda")
    attn = torch.ones_like(ids)
    with torch.inference_mode():
        out = model.generate(ids, attention_mask=attn, do_sample=False, max_new_tokens=128,
                             pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
    gen = out[0, ids.shape[1]:]
    # <|im_end|> is the chat-control end-of-message token (eos_id 151645), not model text;
    # skip special tokens so the final text line is the model's actual verdict line.
    text = tok.decode(gen, skip_special_tokens=True)
    text = unicodedata.normalize("NFKC", text)
    nonempty = [ln.strip() for ln in text.splitlines() if ln.strip()]
    verdict = None
    if nonempty:
        last = nonempty[-1]
        if last == "Final verdict: A":
            verdict = "A"
        elif last == "Final verdict: B":
            verdict = "B"
        else:
            verdict = "UNPARSEABLE"
    else:
        verdict = "UNPARSEABLE"
    parsed_ok = verdict in ("A", "B")
    correct = verdict == exp
    parse_rows.append({"id": sid, "expected_label": exp, "generation": text,
                       "last_nonempty_line": nonempty[-1] if nonempty else "",
                       "verdict": verdict, "parseable": parsed_ok, "correct": correct})
    ok_parse += 1 if parsed_ok else 0
    print(f"{sid} exp={exp} verdict={verdict} ok={correct} last={nonempty[-1][:40]!r}" if nonempty else f"{sid} empty")

n_parse = sum(1 for r in parse_rows if r["parseable"])
nA = sum(1 for r in parse_rows if r["expected_label"] == "A" and r["verdict"] == "A")
nB = sum(1 for r in parse_rows if r["expected_label"] == "B" and r["verdict"] == "B")
with open(R / "cot_generation_semantic_audit.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(parse_rows[0].keys()))
    w.writeheader()
    w.writerows(parse_rows)

print(f"B_CoT_gen: parseable={n_parse}/24 A-match={nA}/12 B-match={nB}/12")
cot_ok = (n_parse == 24 and nA == 12 and nB == 12)
if not cot_ok:
    fail("cot_baseline_execution_invalid", f"parseable={n_parse} A={nA} B={nB}")
print("B_CoT_gen generation + parse PASSED")

# stash for phase2
json.dump({"B_base_ok": True, "B_direct_ok": True, "B_CoT_gen_ok": True},
          open(R / "scripts" / "_phase1_ok.json", "w"), indent=2)
print("Phase 1 OK")
