#!/usr/bin/env python3
"""E0-R1: write source_data_contract.md (inherited verbatim from E0).
"""
from __future__ import annotations

from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[2]

R = REPO_ROOT / "e0r1_popqa_relation_controlled_data_qualification_20260803"
E0 = REPO_ROOT / "e0_popqa_relation_controlled_data_qualification_20260803"

src = (E0 / "source_data_contract.md").read_text(encoding="utf-8")
(R / "source_data_contract.md").write_text(
    src.replace("# source_data_contract.md",
                "# source_data_contract.md\n\n> 由 E0 逐字继承（E0-R1 不重新恢复 schema；内容与 E0 完全一致）。"),
    encoding="utf-8")
print("source_data_contract.md written (inherited from E0)")
