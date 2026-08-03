#!/usr/bin/env python3
"""D3-M Phase 0C: L18 hook layer mapping uniqueness audit.

Verify hidden_states[18] == model.model.layers[17].forward output at R_end,
by comparing output_hidden_states forward vs passive forward-hook capture.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[2]

import torch

import d3m_core as C

D1 = REPO_ROOT / "d1_qwen25_7b_jar_style_sciq_behavior_gate_20260802"
M = REPO_ROOT / "d3m_qwen25_7b_monolithic_reference_binding_intervention_20260802"


def fail(label: str, why: str):
    print("STOP:", label, "-", why)
    (M / "artifacts").mkdir(parents=True, exist_ok=True)
    (M / "artifacts" / "decision.json").write_text(json.dumps({
        "final_label": label, "reason": why,
        "final_reserve_model_scored": False, "final_reserve_hidden_states_read": False,
        "monolithic_full_forward_only": True, "segmented_execution_used": False,
        "prefix_cache_used": False, "activation_intervention_run": False,
        "prompt_baselines_run": False, "mistral_loaded": False}, indent=2), encoding="utf-8")
    sys.exit(1)


dev_pairs = C.load_swap_pairs("dev")
# sample 5 dev SS inputs + 5 train SS inputs for mapping audit
train_pairs = C.load_swap_pairs("train")
samples = []
for p in dev_pairs[:5]:
    samples.append(("dev", p["original_group_id"], p["q"], p["r_s"], p["c_s"]))
for p in train_pairs[:5]:
    samples.append(("train", p["original_group_id"], p["q"], p["r_s"], p["c_s"]))

model = C.get_model()
rows = []
all_ok = True
for split, gid, q, ref, cand in samples:
    ids, r_end = C.build_prompt_ids(q, ref, cand)
    pids = torch.tensor([ids], device="cuda")

    # forward A: output_hidden_states
    with torch.inference_mode():
        out = model(pids, output_hidden_states=True)
    hs18 = out.hidden_states[C.L_HIDDEN_INDEX][0, r_end].cpu().float()  # (hidden,)

    # forward B: passive hook capturing layers[17].output
    captured = {}
    hook = model.model.layers[C.L_BLOCK_INDEX].register_forward_hook(
        C._hook_factory(r_end, None, captured))
    try:
        with torch.inference_mode():
            model(pids)
    finally:
        hook.remove()
    hook18 = captured["pre"]  # (1, hidden)

    max_diff = (hs18 - hook18).abs().max().item()
    identical = torch.equal(hs18, hook18[0])
    all_ok &= identical
    rows.append({
        "split": split, "source_group_id": gid,
        "hidden_states_index": C.L_HIDDEN_INDEX, "decoder_block_index": C.L_BLOCK_INDEX,
        "module_path": f"model.model.layers[{C.L_BLOCK_INDEX}]",
        "output_is_tuple": False, "seq_len": int(pids.shape[1]),
        "r_end_pos": r_end, "max_abs_diff": float(max_diff),
        "bit_identical": bool(identical),
    })
    print(f"  [{split}] {gid[:12]}: r_end={r_end}/{pids.shape[1]} max_diff={max_diff} identical={identical}")

with open(M / "hook_layer_mapping_audit.md", "w", encoding="utf-8") as f:
    f.write(f"""# hook_layer_mapping_audit.md (D3-M)

## 层映射

- D2-R1 表示层：`hidden-state index = 18`。
- 本模型映射：`hidden_states[18]` = `model.model.layers[17]`（decoder block 17）的 forward 输出。
- decoder layer 输出为纯 tensor（非 tuple）。
- R_end 位置由 offset mapping 定位（prefix 构造与 D2-R1 合同一致，但本轮只读完整前向）。

## 唯一性验证（output_hidden_states vs 被动 hook 捕获）

| split | group | r_end/seq | max_abs_diff | bit_identical |
|---|---|---|---|---|
""")
    for r in rows:
        f.write(f"| {r['split']} | {r['source_group_id'][:12]} | {r['r_end_pos']}/{r['seq_len']} | {r['max_abs_diff']:.3e} | {r['bit_identical']} |\n")
    f.write(f"""
## 结论

{'' if all_ok else '失败：'}hidden_states[18] 与 layers[17] output 在 R_end 处逐位一致 = {all_ok}。
""" + ("hook 干预点 = model.model.layers[17] output（hidden_states index 18），唯一映射有效。\n"
       if all_ok else "无唯一映射 → monolithic_hook_layer_mapping_invalid。\n"))

print("DECISION:", "ok" if all_ok else "monolithic_hook_layer_mapping_invalid")
if not all_ok:
    fail("monolithic_hook_layer_mapping_invalid", "hidden_states[18] != layers[17] output")
print("Phase 0C OK")
