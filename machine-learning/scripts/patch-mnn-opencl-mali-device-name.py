#!/usr/bin/env python3
"""Patch MNN OpenCL Mali device-name lookup for Rockchip (RK3588).

Why
---
1. OpenCL reports ``CL_DEVICE_NAME`` like ``Mali-G610 r0p0``, but MNN 3.6.1 looks
   up ``maliArMap`` with an exact match on ``Mali-G610``. Miss → ``GpuLevel::UNDEFINED``.

2. Stock map lists G610 as ``LOW``. With MNN's enum ordering
   ``UNDEFINED=0, TOP=1, MEDIUM=2, LOW=3``, the gemm fast-path
   ``if (getGpuLevel() >= MEDIUM)`` is true for **LOW** (and MEDIUM) but **false for TOP**.
   That early-return uses fixed (conservative) Xgemm tiles and skips Wide/Heavy timed
   tune — measured ~185 ms vs ~120 ms on RK3588 OCR rec @ 48×1000 when we first
   "fixed" the name match to stock LOW.

Fix
---
- Longest-prefix match so ``Mali-G610 r0p0`` hits map key ``Mali-G610``.
- For Immich/RK3588: after match, promote ``Mali-G610`` / ``Mali-G615`` to ``TOP``
  so gemm stays on the timed-tune path (same as UNDEFINED did), not the LOW early-return.
"""

from __future__ import annotations

import sys
from pathlib import Path

_MARKER = "Immich: Mali CL_DEVICE_NAME prefix match"

OLD = """            } else if (deviceName.find("Mali") != std::string::npos) {
                mGpuType = MALI;
                if(maliArMap.find(deviceName) != maliArMap.end()){
                    mMaliAr = maliArMap[deviceName].first;
                    mGpuLevel = maliArMap[deviceName].second;
                }else{
                    mMaliAr = VALHALL;
                    mGpuLevel = UNDEFINED;
                }
                mDeviceInfo = deviceName;
"""

NEW = """            } else if (deviceName.find("Mali") != std::string::npos) {
                mGpuType = MALI;
                /* Immich: Mali CL_DEVICE_NAME prefix match (see patch-mnn-opencl-mali-device-name.py)
                   Rockchip reports "Mali-G610 r0p0"; stock map keys are "Mali-G610".
                   G610/G615 → TOP: stock LOW trips getGemmParams (>= MEDIUM) early-return
                   (enum LOW=3 > MEDIUM=2) and skips Heavy/Wide tune — slower on RK3588. */
                {
                    auto maliIt = maliArMap.end();
                    size_t bestLen = 0;
                    for (auto it = maliArMap.begin(); it != maliArMap.end(); ++it) {
                        const std::string& key = it->first;
                        if (deviceName.compare(0, key.size(), key) == 0 && key.size() > bestLen) {
                            if (deviceName.size() == key.size() ||
                                !std::isalnum(static_cast<unsigned char>(deviceName[key.size()]))) {
                                maliIt = it;
                                bestLen = key.size();
                            }
                        }
                    }
                    if (maliIt != maliArMap.end()) {
                        mMaliAr = maliIt->second.first;
                        mGpuLevel = maliIt->second.second;
                        if (maliIt->first == "Mali-G610" || maliIt->first == "Mali-G615") {
                            mGpuLevel = TOP;
                        }
                    } else {
                        mMaliAr = VALHALL;
                        mGpuLevel = UNDEFINED;
                    }
                }
                mDeviceInfo = deviceName;
"""


def main() -> None:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <path-to-MNN-source-root>", file=sys.stderr)
        sys.exit(2)

    path = Path(sys.argv[1]) / "source/backend/opencl/core/runtime/OpenCLRuntime.cpp"
    if not path.is_file():
        raise SystemExit(f"missing {path}")

    text = path.read_text()
    if _MARKER in text:
        # Allow re-apply after we changed NEW: replace from marker block if old Immich patch present.
        if "G610/G615 → TOP" in text or 'maliIt->first == "Mali-G610"' in text:
            print(f"already patched (G610→TOP): {path}")
            return
        raise RuntimeError(
            f"{path} has an older Immich Mali name patch without G610→TOP. "
            "Revert OpenCLRuntime.cpp to stock 3.6.1 (or rebuild from tarball) then re-run this script."
        )
    if OLD not in text:
        raise RuntimeError(
            f"Mali deviceName block not found in {path} — MNN version may have changed; "
            "update scripts/patch-mnn-opencl-mali-device-name.py"
        )

    if "#include <cctype>" not in text:
        idx = 0
        last_inc = -1
        for line in text.splitlines(keepends=True):
            if line.startswith("#include"):
                last_inc = idx + len(line)
            idx += len(line)
            if last_inc >= 0 and line.strip() == "" and idx > last_inc:
                break
        if last_inc < 0:
            raise RuntimeError(f"could not find #include block in {path}")
        text = text[:last_inc] + "#include <cctype>\n" + text[last_inc:]

    text = text.replace(OLD, NEW, 1)
    path.write_text(text)
    print(f"patched {path}")


if __name__ == "__main__":
    main()
