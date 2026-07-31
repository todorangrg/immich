#!/usr/bin/env python3
"""Patch MNN OpenCL matmul_params_buf DEAL_BIAS macros for Mali (RK3588).

Why
---
On Rockchip Mali OpenCL (arm_release_ver g24p0 / rk_so_ver 6), building
`matmul_params_buf` (`Xgemm` / `XgemmBatched`) fails with:

    error: implicit declaration of function 'DEAL_BIAS' is invalid in OpenCL
    DEAL_BIAS(result.s0, eval.s0);   // <source>:~843
    err:-11

Those call sites sit under `#if BIAS_TYPE > 0`. DEAL_BIAS is only #define'd in
an `#if / #elif / #elif ...` chain for BIAS_TYPE in {1,2,3,4,5}. Mali's CL
frontend can take the `BIAS_TYPE > 0` branch while failing to apply the matching
`#elif BIAS_TYPE == N` arm, leaving DEAL_BIAS undefined → compile failure.

Immich hits this for:
  - BIAS_TYPE=1 : MatMul with bias (ViT Linear / FFN), large-tile Xgemm path
  - BIAS_TYPE=2 : fused Attention QK gemm with Add-mask (XgemmBatched)

Both of those need the same body: `x = x + a`.

Fix (semantic-preserving)
-------------------------
1. Replace the `#elif` chain with independent `#if BIAS_TYPE == N` blocks so each
   value is gated on its own equality test (no elif fall-through).
2. Collapse 1 and 2 into one `#if` (identical bodies).
3. `#ifndef DEAL_BIAS` → `#define DEAL_BIAS(x,a) x=x+a` as last resort. Harmless
   when BIAS_TYPE==0 (calls are still under `#if BIAS_TYPE > 0`). For Immich's
   type 1/2 paths this matches the real op if equality matching still fails.

Must patch matmul_params_buf_mnn_cl.cpp: OpenCL codegen from .cl is commented out
in MNN's CMakeLists, so the pregenerated string is what gets linked into libMNN.
Also patch the .cl sibling so a future regen stays consistent.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Exact upstream 3.6.1 blocks (must match byte-for-byte or we refuse to patch).
CPP_OLD = (
    '"#if BIAS_TYPE == 1\\n"\n'
    '"#define DEAL_BIAS(x,a) x=x+a\\n"\n'
    '"#elif BIAS_TYPE == 2\\n"\n'
    '"#define DEAL_BIAS(x,a) x=x+a\\n"\n'
    '"#elif BIAS_TYPE == 3\\n"\n'
    '"#define DEAL_BIAS(x,a) x=x-a\\n"\n'
    '"#elif BIAS_TYPE == 4\\n"\n'
    '"#define DEAL_BIAS(x,a) x=a-x\\n"\n'
    '"#elif BIAS_TYPE == 5\\n"\n'
    '"#define DEAL_BIAS(x,a) x=(a == 0 ? (FLOAT)(-FLT_MAX) : x)\\n"\n'
    '"#endif\\n"'
)

CPP_NEW = (
    '"/* Immich: Mali OpenCL - avoid #elif for DEAL_BIAS (see patch-mnn-opencl-deal-bias.py) */\\n"\n'
    '"#if (BIAS_TYPE == 1) || (BIAS_TYPE == 2)\\n"\n'
    '"#define DEAL_BIAS(x,a) x=x+a\\n"\n'
    '"#endif\\n"\n'
    '"#if BIAS_TYPE == 3\\n"\n'
    '"#define DEAL_BIAS(x,a) x=x-a\\n"\n'
    '"#endif\\n"\n'
    '"#if BIAS_TYPE == 4\\n"\n'
    '"#define DEAL_BIAS(x,a) x=a-x\\n"\n'
    '"#endif\\n"\n'
    '"#if BIAS_TYPE == 5\\n"\n'
    '"#define DEAL_BIAS(x,a) x=(a == 0 ? (FLOAT)(-FLT_MAX) : x)\\n"\n'
    '"#endif\\n"\n'
    '"#ifndef DEAL_BIAS\\n"\n'
    '"#define DEAL_BIAS(x,a) x=x+a\\n"\n'
    '"#endif\\n"'
)

CL_OLD = """#if BIAS_TYPE == 1
#define DEAL_BIAS(x, a) x = x + a
#elif BIAS_TYPE == 2
#define DEAL_BIAS(x, a) x = x + a
#elif BIAS_TYPE == 3
#define DEAL_BIAS(x, a) x = x - a
#elif BIAS_TYPE == 4
#define DEAL_BIAS(x, a) x = a - x
#elif BIAS_TYPE == 5
#define DEAL_BIAS(x, a) x = (a == 0 ? (FLOAT)(-FLT_MAX) : x)
#endif"""

CL_NEW = """/* Immich: Mali OpenCL - avoid #elif for DEAL_BIAS (see patch-mnn-opencl-deal-bias.py) */
#if (BIAS_TYPE == 1) || (BIAS_TYPE == 2)
#define DEAL_BIAS(x, a) x = x + a
#endif
#if BIAS_TYPE == 3
#define DEAL_BIAS(x, a) x = x - a
#endif
#if BIAS_TYPE == 4
#define DEAL_BIAS(x, a) x = a - x
#endif
#if BIAS_TYPE == 5
#define DEAL_BIAS(x, a) x = (a == 0 ? (FLOAT)(-FLT_MAX) : x)
#endif
#ifndef DEAL_BIAS
#define DEAL_BIAS(x, a) x = x + a
#endif"""

_PATCH_MARKER = "Immich: Mali OpenCL - avoid #elif for DEAL_BIAS"


def _patch_file(path: Path, old: str, new: str) -> None:
    text = path.read_text()
    if _PATCH_MARKER in text:
        print(f"already patched: {path}")
        return
    if old not in text:
        raise RuntimeError(
            f"DEAL_BIAS block not found in {path} — MNN version may have changed; "
            "update scripts/patch-mnn-opencl-deal-bias.py"
        )
    path.write_text(text.replace(old, new, 1))
    print(f"patched {path}")


def main() -> None:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <path-to-MNN-source-root>", file=sys.stderr)
        sys.exit(2)

    root = Path(sys.argv[1])
    cl_dir = root / "source" / "backend" / "opencl" / "execution" / "cl"
    cpp = cl_dir / "matmul_params_buf_mnn_cl.cpp"
    cl = cl_dir / "matmul_params_buf.cl"
    if not cpp.is_file():
        raise SystemExit(f"missing {cpp}")
    if not cl.is_file():
        raise SystemExit(f"missing {cl}")

    _patch_file(cpp, CPP_OLD, CPP_NEW)
    _patch_file(cl, CL_OLD, CL_NEW)

    # Sanity: call sites must still exist, and DEAL_BIAS must now always be defined.
    cpp_text = cpp.read_text()
    if "DEAL_BIAS(result" not in cpp_text:
        raise RuntimeError(f"{cpp} lost DEAL_BIAS call sites after patch")
    if cpp_text.count("#ifndef DEAL_BIAS") < 1:
        raise RuntimeError(f"{cpp} missing #ifndef DEAL_BIAS fallback after patch")
    if "#elif BIAS_TYPE == 2" in cpp_text:
        raise RuntimeError(f"{cpp} still contains #elif BIAS_TYPE == 2 after patch")


if __name__ == "__main__":
    main()
