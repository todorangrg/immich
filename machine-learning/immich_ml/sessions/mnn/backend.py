from __future__ import annotations

from pathlib import Path
from typing import Any

from immich_ml.config import log, settings

_installed = False

try:
    import MNN

    is_available = True
except ImportError:
    MNN = None  # type: ignore[assignment]
    is_available = False


def cache_file_for_model(model_path: Path | str) -> Path:
    """OpenCL/Vulkan kernel cache path next to the Immich-downloaded model (like ArmNN `.anncache`)."""
    return Path(model_path).with_suffix(".mnncache")


def session_config() -> dict[str, Any]:
    return {
        "backend": settings.mnn_backend,
        # For OPENCL, numThread is a kernel-tuning hint (not CPU cores). 68 is a strong
        # default on RK3588 Mali; override via MACHINE_LEARNING_MNN_NUM_THREAD if needed.
        "numThread": settings.mnn_num_thread,
        "precision": settings.mnn_precision,
        "memory": settings.mnn_memory,
        "power": settings.mnn_power,
    }


def _apply_cache_file(interpreter: Any, model_path: Path | str | None) -> None:
    if model_path is None:
        return
    cache_path = cache_file_for_model(model_path)
    interpreter.setCacheFile(str(cache_path))
    log.debug("MNN kernel cache: %s", cache_path)


def _update_cache_file(interpreter: Any, session: Any) -> None:
    try:
        interpreter.updateCacheFile(session, 0)
    except Exception:
        log.debug("MNN updateCacheFile failed", exc_info=True)


def install_mnn_defaults() -> bool:
    """Wrap MNN.Interpreter so RapidOCR's bare createSession() gets Immich GPU settings + cache.

    RapidOCR's MNNInferSession calls createSession() with no config. We inject OpenCL/Vulkan
    defaults, set a per-model kernel cache under the Immich model cache, and persist tuning
    after inference via updateCacheFile (same idea as ArmNN gpu-tuning / .anncache).
    """
    global _installed
    if _installed or not is_available or not settings.mnn or MNN is None:
        return _installed

    real_interpreter = MNN.Interpreter

    class ConfiguredInterpreter:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self._obj = real_interpreter(*args, **kwargs)
            self._model_path: Path | None = Path(args[0]) if args else None

        def createSession(self, *args: Any, **kwargs: Any) -> Any:
            _apply_cache_file(self._obj, self._model_path)
            if not args and not kwargs:
                return self._obj.createSession(session_config())
            return self._obj.createSession(*args, **kwargs)

        def runSession(self, session: Any, *args: Any, **kwargs: Any) -> Any:
            result = self._obj.runSession(session, *args, **kwargs)
            _update_cache_file(self._obj, session)
            return result

        def __getattr__(self, name: str) -> Any:
            return getattr(self._obj, name)

    MNN.Interpreter = ConfiguredInterpreter  # type: ignore[misc, assignment]
    # TEMPORARY (testing only): delete when RapidOCR lands getNumpyData() for MNNInferSession.
    from ._temporary_rapidocr_numpy_patch import apply_temporary_rapidocr_numpy_patch

    apply_temporary_rapidocr_numpy_patch()
    _installed = True
    log.info(
        "Configured MNN session defaults: backend=%s, numThread=%s, precision=%s, memory=%s, power=%s",
        settings.mnn_backend,
        settings.mnn_num_thread,
        settings.mnn_precision,
        settings.mnn_memory,
        settings.mnn_power,
    )
    return True
