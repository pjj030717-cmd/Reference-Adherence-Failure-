#!/usr/bin/env python3
"""E01-D1-R: candidate template robustness spec (T0/T1/T2) + SHA256."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[2]

D1R = REPO_ROOT / "d1r_qwen25_7b_jar_style_sciq_template_robustness_20260802"

TEMPLATES = {
    "T0": "The answer is <answer>.",
    "T1": "For this question, the answer is <answer>.",
    "T2": "The response is <answer>.",
}


def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


spec = {
    "templates": {k: {"template": v, "utf8_sha256": sha256_hex(v)} for k, v in TEMPLATES.items()},
    "fixed_rules": [
        "<answer> 直接使用 D0 已冻结的 r_o 或 r_s 归一化原文",
        "除候选句模板文字外不改变任何字符",
        "不得增添解释/理由/CoT/reference 字样/correct|incorrect 字样/额外事实/例子/新 prompt",
        "不得重抽 swap、删除 group、按模型分数选择样本",
        "每个 group 的 OO/OS/SO/SS 四格必须使用同一模板",
        "T0 为 D1 原模板；T1/T2 为仅改变表述的新模板",
    ],
    "user_template_unchanged": "Question: {question}\n\nReference Answer: {reference}\n\nCandidate Answer: {candidate}\n\nAnswer:",
    "continuations": {"accept": " A", "reject": " B"},
}
(D1R / "candidate_template_robustness_spec.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
print(json.dumps({k: {"sha256": v["utf8_sha256"]} for k, v in spec["templates"].items()}, indent=2))
