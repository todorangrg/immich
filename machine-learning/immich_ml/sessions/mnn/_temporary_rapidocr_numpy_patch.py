"""TEMPORARY — testing only.

Remove this module once RapidOCR ships MNNInferSession using Tensor.getNumpyData()
instead of Tensor.getData() (Python list conversion is extremely slow on recognition).

Do not expand this file with lasting Immich behavior.
"""

from __future__ import annotations

import traceback
from typing import Any

import numpy as np

from immich_ml.config import log

_applied = False


def apply_temporary_rapidocr_numpy_patch() -> bool:
    """Monkeypatch RapidOCR MNN output to getNumpyData() for local OCR performance testing."""
    global _applied
    if _applied:
        return True

    try:
        import MNN
        from rapidocr.inference_engine.mnn.main import MNNError, MNNInferSession
    except ImportError:
        return False

    def __call__(self: Any, img: np.ndarray) -> np.ndarray:
        try:
            self.interpreter.resizeTensor(self.input_tensor, img.shape)
            self.interpreter.resizeSession(self.session)

            tmp = MNN.Tensor(
                img.shape,
                MNN.Halide_Type_Float,
                img,
                MNN.Tensor_DimensionType_Caffe,
            )
            self.input_tensor.copyFromHostTensor(tmp)
            self.interpreter.runSession(self.session)

            output = self.interpreter.getSessionOutput(self.session)
            out_shape = output.getShape()
            out_tensor = MNN.Tensor(out_shape, MNN.Halide_Type_Float, MNN.Tensor_DimensionType_Caffe)
            output.copyToHostTensor(out_tensor)
            return np.array(out_tensor.getNumpyData(), copy=True).reshape(out_shape)
        except Exception as e:
            error_info = traceback.format_exc()
            raise MNNError(f"MNN inference failed: {error_info}") from e

    MNNInferSession.__call__ = __call__  # type: ignore[method-assign]
    _applied = True
    log.warning(
        "TEMPORARY: patched RapidOCR MNNInferSession to use getNumpyData() "
        "(remove immich_ml/sessions/mnn/_temporary_rapidocr_numpy_patch.py when RapidOCR lands this)"
    )
    return True
