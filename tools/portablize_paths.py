#!/usr/bin/env python3
"""Portablize absolute paths in RAF scripts for GitHub upload.

Replacements (in every <exp>/scripts/*.py):
  1. Path("/root/autodl-tmp/inspector/<dir>")  -> REPO_ROOT / "<dir>"
     and inject  REPO_ROOT = Path(__file__).resolve().parents[2]
     (scripts/<exp>/<dir> layout => parents[2] is the repo root)
  2. "/root/autodl-tmp/models/Qwen2.5-7B-Instruct"     -> os.environ.get("RAF_MODEL_DIR", <default>)
     Path("/root/autodl-tmp/models/Qwen2.5-7B-Instruct") -> Path(os.environ.get("RAF_MODEL_DIR", <default>))
     Mistral similarly via RAF_MISTRAL_DIR
     and ensure `import os` present.

Core modules imported as `import <exp>_core as C` also carry MODEL / D1 / D0
definitions; they are converted identically (they live in the same scripts/ dir).

Dry-run: python3 portablize_paths.py --dry
Apply:    python3 portablize_paths.py --apply
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # repo root
MODE = "--apply" in sys.argv

INSPECTOR_RE = re.compile(r'Path\("/root/autodl-tmp/inspector/([a-z0-9_]+)"\)')
QWEN_DEFAULT = "/root/autodl-tmp/models/Qwen2.5-7B-Instruct"
MISTRAL_DEFAULT = "/root/autodl-tmp/models/Mistral-7B-Instruct-v0.3"

# The two _DEFAULT constants above must not be self-matched by the model regexes.
_SELF = Path(__file__).resolve()

# model-path assignment forms -> replacement
MODEL_FORMS = [
    # (regex, replacement)
    # Qwen: string and Path-wrapped string assignments
    (re.compile(r'MODEL_DIR = Path\("' + re.escape(QWEN_DEFAULT) + r'"\)'),
     'MODEL_DIR = Path(os.environ.get("RAF_MODEL_DIR", "%s"))' % QWEN_DEFAULT),
    (re.compile(r'MODEL_TOK = "' + re.escape(QWEN_DEFAULT) + r'"'),
     'MODEL_TOK = os.environ.get("RAF_MODEL_DIR", "%s")' % QWEN_DEFAULT),
    (re.compile(r'MODEL = "' + re.escape(QWEN_DEFAULT) + r'"'),
     'MODEL = os.environ.get("RAF_MODEL_DIR", "%s")' % QWEN_DEFAULT),
    (re.compile(r'MODEL = Path\("' + re.escape(QWEN_DEFAULT) + r'"\)'),
     'MODEL = Path(os.environ.get("RAF_MODEL_DIR", "%s"))' % QWEN_DEFAULT),
    (re.compile(r'MODEL_DIR = "' + re.escape(QWEN_DEFAULT) + r'"'),
     'MODEL_DIR = os.environ.get("RAF_MODEL_DIR", "%s")' % QWEN_DEFAULT),
    # Mistral
    (re.compile(r'MODEL_DIR = "' + re.escape(MISTRAL_DEFAULT) + r'"'),
     'MODEL_DIR = os.environ.get("RAF_MISTRAL_DIR", "%s")' % MISTRAL_DEFAULT),
    (re.compile(r'MODEL = "' + re.escape(MISTRAL_DEFAULT) + r'"'),
     'MODEL = os.environ.get("RAF_MISTRAL_DIR", "%s")' % MISTRAL_DEFAULT),
    # bare occurrences inside expressions (e.g. from_pretrained("..."))
    # The sub callback skips matches that sit in the default-value slot of an
    # already converted os.environ.get(...) call, so re-running is idempotent.
    (re.compile(r'"' + re.escape(QWEN_DEFAULT) + r'"'),
     'os.environ.get("RAF_MODEL_DIR", "%s")' % QWEN_DEFAULT, "RAF_MODEL_DIR"),
    (re.compile(r'"' + re.escape(MISTRAL_DEFAULT) + r'"'),
     'os.environ.get("RAF_MISTRAL_DIR", "%s")' % MISTRAL_DEFAULT, "RAF_MISTRAL_DIR"),
]

LINES_INSPECTOR = re.compile(r'Path\("/root/autodl-tmp/inspector/([a-z0-9_]+)"\)')


def process(path: Path, dry: bool) -> dict:
    src = path.read_text(encoding="utf-8")
    orig = src
    stats = {"inspector": 0, "model": 0, "inject_root": False, "inject_os": False}

    # 1. inspector paths
    def insp_sub(m):
        stats["inspector"] += 1
        return 'REPO_ROOT / "%s"' % m.group(1)

    src = INSPECTOR_RE.sub(insp_sub, src)
    if stats["inspector"] and 'REPO_ROOT = ' not in src:
        stats["inject_root"] = True

    # 2. model paths
    for rx, rep, *_extra in MODEL_FORMS:
        var_name = _extra[0] if _extra else None

        def _sub(m, _rep=rep, _var=var_name):
            # skip if this match is the default-value slot of an already
            # converted os.environ.get("...", "<default>") call
            if _var:
                start = m.start()
                ctx = src[max(0, start - 40):start]
                if ctx.rstrip().endswith(f'os.environ.get("{_var}",'):
                    return m.group(0)
            stats["model"] += 1
            return _rep

        n = len(rx.findall(src))
        if n:
            src = rx.sub(_sub, src)

    # 3. injections
    if stats["inject_root"]:
        anchor = "from pathlib import Path"
        assert anchor in src, f"{path}: no pathlib anchor"
        root_line = f"\nREPO_ROOT = Path(__file__).resolve().parents[2]"
        src = src.replace(anchor, anchor + root_line, 1)
    if stats["model"] and not re.search(r"^\s*import os\b", src, re.M):
        stats["inject_os"] = True
        # put `import os` right after the last `import X` line at column 0 in the import block
        # simplest robust insertion: after `from pathlib import Path`
        anchor = "from pathlib import Path"
        if anchor in src:
            src = src.replace(anchor, anchor + "\nimport os", 1)
        else:
            src = "import os\n" + src

    changed = src != orig
    if dry:
        if changed:
            print(f"DRY {path.relative_to(ROOT)}: {stats}")
        return stats
    if changed:
        path.write_text(src, encoding="utf-8")
        print(f"WROTE {path.relative_to(ROOT)}: {stats}")
    return stats


def main():
    pys = sorted(ROOT.rglob("*.py"))
    total = {"inspector": 0, "model": 0, "files": 0}
    for p in pys:
        if p.resolve() == _SELF:
            continue
        s = process(p, dry=(MODE is False))
        if MODE:
            if s["inspector"] or s["model"]:
                total["files"] += 1
                total["inspector"] += s["inspector"]
                total["model"] += s["model"]
    print(f"\n{'APPLY' if MODE else 'DRY-RUN'}: files changed={total['files']}, "
          f"inspector refs={total['inspector']}, model refs={total['model']}")


if __name__ == "__main__":
    main()
