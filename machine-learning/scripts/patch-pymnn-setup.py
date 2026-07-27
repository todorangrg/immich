#!/usr/bin/env python3
"""Patch upstream PyMNN for an inference-only Immich ML build.

1. setup.py — drop converter/tools extension and train/opencv/audio compile defines
   so we can link against a cmake build that only produces libMNN.
2. MNN/__init__.py — soft-import optional submodules (_data/_optim/cv/audio need
   train/opencv/audio APIs we intentionally omit). Core Interpreter/Tensor still load.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

INFERENCE_INIT = '''\
# Patched by Immich ML for inference-only PyMNN (no train/opencv/audio APIs).
from _mnncengine import *

from . import expr
from . import nn
from . import numpy

try:
    from . import data
except ImportError:
    data = None
try:
    from . import optim
except ImportError:
    optim = None
try:
    from . import cv
except ImportError:
    cv = None
try:
    from . import audio
except ImportError:
    audio = None
try:
    from . import llm
except ImportError:
    llm = None
'''


def patch_setup(text: str) -> str:
    # Inference-only: keep EXPR API, drop optional helper APIs.
    text = text.replace("'-DPYMNN_EXPR_API', '-DPYMNN_OPENCV_API', '-DPYMNN_AUDIO_API'", "'-DPYMNN_EXPR_API'")
    text = text.replace("['-DBUILD_OPTYPE', '-DPYMNN_TRAIN_API']", "['-DBUILD_OPTYPE']")

    # Do not build/link the _tools extension (needs converter + protobuf).
    text = re.sub(
        r"\n\s*tools = Extension\(\"_tools\",.*?extensions\.append\(tools\)\n",
        "\n",
        text,
        count=1,
        flags=re.DOTALL,
    )
    if 'Extension("_tools"' in text or "extensions.append(tools)" in text:
        raise RuntimeError("failed to remove _tools Extension from setup.py")

    # Drop CLI entry points that depend on converter/quant tooling.
    text = re.sub(
        r"entry_points = \{.*?\}",
        "entry_points = {}",
        text,
        count=1,
        flags=re.DOTALL,
    )
    if re.search(r"entry_points = \{[^}]*mnnconvert", text, flags=re.DOTALL):
        raise RuntimeError("failed to clear entry_points in setup.py")

    return text


def main() -> None:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <path-to-pymnn-pip_package-dir-or-setup.py>", file=sys.stderr)
        sys.exit(2)

    target = Path(sys.argv[1])
    if target.is_dir():
        setup_path = target / "setup.py"
        init_path = target / "MNN" / "__init__.py"
    elif target.name == "setup.py":
        setup_path = target
        init_path = target.parent / "MNN" / "__init__.py"
    else:
        raise SystemExit(f"expected setup.py or pip_package dir, got {target}")

    original_setup = setup_path.read_text()
    patched_setup = patch_setup(original_setup)
    if patched_setup == original_setup:
        raise RuntimeError(f"no changes applied to {setup_path}")
    setup_path.write_text(patched_setup)
    print(f"patched {setup_path} for inference-only PyMNN")

    if not init_path.is_file():
        raise RuntimeError(f"missing {init_path}")
    init_path.write_text(INFERENCE_INIT)
    print(f"patched {init_path} for optional submodule imports")


if __name__ == "__main__":
    main()
