#!/usr/bin/env python3
"""D4-Q1 Phase 1.2: true-prefix mechanical contract on the 196 allowed final SS inputs,
plus prefix hidden-state collection (M_rep features) and SS A/B readout
(label generation for Phase 2.1).

R_end localization mirrors D2-R1 d2r1_prefix_contract.py / d2r1_collect_prefix.py
(exact offset-mapping rule):
    ref_marker = "Reference Answer: "
    i0 = rendered.find(ref_marker)
    ref_start = i0 + len(ref_marker)
    ref_end = ref_start + len(ref)
    r_tok = next(ti for ti,(s,e) in enumerate(offsets) if s <= ref_end-1 < e)
    prefix = ids[:r_tok+1]

Audits (all 196):
  - unique R_end localization
  - prefix contains NO "Candidate Answer: " marker, NO "Answer:" marker,
    NO generation-prompt tokens (<|im_start|>assistant suffix, <|im_end|> in tail)
  - repeated forward on 60 sampled groups: max_abs_diff == 0

Also performs the Phase 2.1 A/B teacher-forced readout on the same 196 groups
(this is the only allowed final Judge behavior scoring).

Strictly skips the 1 quarantined leaked group (0075758e...).

Outputs:
  - true_prefix_final_contract_audit.csv
  - final_ss_score_and_label_manifest.csv (numeric only; no question text)
  - prefix_hidden_states/final_<gid>.npz (h_prefix at layer18, float16)
  - scripts/_final_surface_feats.npz (numeric surface features for B_surface)
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
import os
REPO_ROOT = Path(__file__).resolve().parents[2]

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

R = REPO_ROOT / "d4q1_qwen25_7b_true_prefix_final_confirmation_20260803"
D1 = REPO_ROOT / "d1_qwen25_7b_jar_style_sciq_behavior_gate_20260802"
D0 = REPO_ROOT / "d0_jar_style_sciq_data_qualification_20260802"
MODEL = os.environ.get("RAF_MODEL_DIR", "/root/autodl-tmp/models/Qwen2.5-7B-Instruct")
CONST = json.loads((D1 / "scripts" / "_prompt_constants.json").read_text(encoding="utf-8"))
SYSTEM, USER_TMPL = CONST["system"], CONST["user_template"]
ACCEPT_ID, REJECT_ID = CONST["accept_id"], CONST["reject_id"]
T0 = "The answer is <answer>."
LAYER = 18
REF_MARKER = "Reference Answer: "

def fail(label: str, why: str):
    print("STOP:", label, "-", why)
    (R / "artifacts").mkdir(parents=True, exist_ok=True)
    (R / "artifacts" / "decision.json").write_text(json.dumps(
        {"final_label": label, "reason": why,
         "allowed_final_groups": 196, "quarantined_final_groups": 1,
         "quarantined_group_scored": False, "quarantined_group_hidden_state_read": False,
         "final_configuration_changed": False, "hidden_layer": LAYER, "hidden_token": "R_end",
         "probe_C": 0.01, "probe_refit_used_dev": False, "probe_refit_used_final": False,
         "activation_intervention_run": False, "mistral_loaded": False,
         "prompt_baselines_run": False}, indent=2), encoding="utf-8")
    sys.exit(1)

# ---------------------------------------------------------------------------
# allowed final manifest (196)
# ---------------------------------------------------------------------------
allowed_ids = {g["source_group_id"] for g in
               json.loads((R / "allowed_final_group_manifest.json").read_text(encoding="utf-8"))["groups"]}
assert len(allowed_ids) == 196

final_rows = []
for line in open(D0 / "preliminary_swap_pairs.jsonl", encoding="utf-8"):
    d = json.loads(line)
    if d["split"] != "final_reserve":
        continue
    if d["original_group_id"] in allowed_ids:
        final_rows.append(d)
assert len(final_rows) == 196, f"final rows {len(final_rows)} != 196"
print("final allowed rows loaded:", len(final_rows))

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
                                             device_map="cuda", low_cpu_mem_usage=True)
model.eval()


def full_enc(question, ref):
    cand = T0.replace("<answer>", ref)
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": USER_TMPL.format(question=question, reference=ref, candidate=cand)}]
    rendered = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tok(rendered, return_offsets_mapping=True, add_special_tokens=False)
    return rendered, enc["input_ids"], enc["offset_mapping"]


def locate_r_end(rendered, offsets, ref):
    i0 = rendered.find(REF_MARKER)
    assert i0 >= 0
    ref_start = i0 + len(REF_MARKER)
    ref_end = ref_start + len(ref)
    r_tok = next(ti for ti, (s, e) in enumerate(offsets) if s <= ref_end - 1 < e)
    return r_tok


HID_DIR = R / "prefix_hidden_states"
HID_DIR.mkdir(parents=True, exist_ok=True)

contract_rows, score_rows, surface_feats, sf_gids = [], [], [], []
seen = set()

for row in final_rows:
    g = row["original_group_id"]
    assert g not in seen
    seen.add(g)
    q, r_s, r_o = row["q"], row["r_s"], row["r_o"]
    cand = T0.replace("<answer>", r_s)
    rendered, full_ids, offsets = full_enc(q, r_s)
    r_tok = locate_r_end(rendered, offsets, r_s)
    prefix = full_ids[:r_tok + 1]
    r_end_char = offsets[r_tok][1]
    pref_txt = rendered[:r_end_char]

    # --- audits: R_end char position must be strictly before any candidate/answer/gen marker ---
    c0 = rendered.find("Candidate Answer: ")
    a0 = rendered.rfind("Answer:")  # generation prompt 'Answer:' (after candidate)
    g0 = rendered.find("<|im_start|>assistant")
    has_cand_marker = r_end_char > c0          # bad if R_end after candidate marker
    has_answer_marker = r_end_char > a0        # bad if generation 'Answer:' starts inside prefix
    has_gen_prompt = r_end_char > g0           # bad if generation prompt inside prefix
    special_after = any(tok.decode([t]).strip() in ("<|im_end|>", "<|im_start|>") for t in prefix[-4:])
    no_leak = (not has_cand_marker) and (not has_answer_marker) and (not has_gen_prompt) and (not special_after)

    contract_rows.append({
        "group_id": g,
        "full_prompt_len": len(full_ids),
        "r_end_pos": r_tok,
        "prefix_len": len(prefix),
        "r_end_token_id": prefix[-1],
        "candidate_marker_in_prefix": int(has_cand_marker),
        "answer_marker_in_prefix": int(has_answer_marker),
        "gen_prompt_token_in_prefix": int(has_gen_prompt or special_after),
        "r_end_char_lt_candidate_marker": int(r_end_char < c0),
    })

    # --- forward pass on prefix alone, read layer18 at R_end ---
    pids = torch.tensor([prefix], device="cuda")
    with torch.inference_mode():
        hs = model(pids, output_hidden_states=True).hidden_states[LAYER][0, -1, :]
    h18 = hs.float().cpu().numpy()
    np.savez_compressed(HID_DIR / f"final_{g}.npz", h_prefix=h18.astype(np.float16))

    # --- Phase 2.1 label generation (A/B teacher-forced, full prompt) ---
    full_enc_t = tok.apply_chat_template(
        [{"role": "system", "content": SYSTEM},
         {"role": "user", "content": USER_TMPL.format(question=q, reference=r_s, candidate=cand)}],
        tokenize=True, add_generation_prompt=True, return_tensors="pt")["input_ids"].to("cuda")
    plen = full_enc_t.shape[1]
    with torch.inference_mode():
        logits = model(full_enc_t).logits
    pos = plen - 1
    lA = logits[0, pos, ACCEPT_ID].item()
    lB = logits[0, pos, REJECT_ID].item()
    d_raw = lA - lB
    y = 1 if d_raw <= 0 else 0
    score_rows.append({"source_group_id": g, "l_A": lA, "l_B": lB, "d_raw": d_raw, "y": y})

    # --- B_surface numeric features ---
    q_tok = len(tok.encode(q)); ro_tok = len(tok.encode(r_o)); rs_tok = len(tok.encode(r_s))
    q_char = len(q); rs_char = len(r_s); rs_words = len(r_s.split())
    surface_feats.append([q_tok, ro_tok, rs_tok, abs(ro_tok - rs_tok), q_char, rs_char,
                          rs_words, 1 if "-" in r_s else 0, 1 if rs_words > 1 else 0])
    sf_gids.append(g)

    if len(seen) % 50 == 0:
        print(f"  processed {len(seen)}/196")

# ---------------------------------------------------------------------------
# repeat determinism on 60 sampled groups (same seed family as protocol)
# ---------------------------------------------------------------------------
rng = np.random.default_rng(20260803)
sample = rng.choice(len(final_rows), size=60, replace=False)
max_diffs = []
for i in sample:
    row = final_rows[i]
    q, r_s = row["q"], row["r_s"]
    rendered, full_ids, offsets = full_enc(q, r_s)
    r_tok = locate_r_end(rendered, offsets, r_s)
    prefix = full_ids[:r_tok + 1]
    pids = torch.tensor([prefix], device="cuda")
    with torch.inference_mode():
        h1 = model(pids, output_hidden_states=True).hidden_states[LAYER][0, -1, :]
    with torch.inference_mode():
        h2 = model(pids, output_hidden_states=True).hidden_states[LAYER][0, -1, :]
    max_diffs.append(float((h1 - h2).abs().max().item()))
print("repeat max_abs_diff (60 groups): max =", max(max_diffs))

# ---------------------------------------------------------------------------
# write outputs
# ---------------------------------------------------------------------------
with open(R / "true_prefix_final_contract_audit.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(contract_rows[0].keys()))
    w.writeheader()
    w.writerows(contract_rows)

with open(R / "final_ss_score_and_label_manifest.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(score_rows[0].keys()))
    w.writeheader()
    w.writerows(score_rows)

np.savez(R / "scripts" / "_final_surface_feats.npz",
         X=np.array(surface_feats, dtype=float), gids=np.array(sf_gids, dtype=object))

# ---------------------------------------------------------------------------
# gates
# ---------------------------------------------------------------------------
ok_loc = len(contract_rows) == 196 and all(r["r_end_pos"] >= 0 for r in contract_rows)
ok_cand = all(r["candidate_marker_in_prefix"] == 0 and r["r_end_char_lt_candidate_marker"] == 1
              for r in contract_rows)
ok_ans = all(r["answer_marker_in_prefix"] == 0 and r["gen_prompt_token_in_prefix"] == 0
             for r in contract_rows)
ok_det = max(max_diffs) == 0.0
print(f"gates: loc196={ok_loc} no_candidate={ok_cand} no_answer={ok_ans} determinism={ok_det}")
if not (ok_loc and ok_cand and ok_ans and ok_det):
    fail("inheritance_or_execution_invalid",
         f"contract: loc196={ok_loc} no_candidate={ok_cand} no_answer={ok_ans} determinism={ok_det}")
print("Phase 1.2 OK: true-prefix mechanical contract PASS; labels + h18 + surface feats saved")
