from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from immich_ml.schemas import SessionNode

from .backend import install_mnn_defaults, is_available, session_config

if is_available:
    import MNN


class _MnnNode:
    def __init__(self, name: str | None, shape: tuple[int, ...]) -> None:
        self.name = name
        self.shape = shape


class MnnSession:
    """ModelSession adapter around MNN for Immich OCR detection."""

    def __init__(self, model_path: Path | str) -> None:
        if not is_available:
            raise RuntimeError("MNN is not installed")
        install_mnn_defaults()
        self.model_path = Path(model_path)
        # Interpreter is wrapped by install_mnn_defaults: setCacheFile + session defaults.
        self.interpreter = MNN.Interpreter(str(self.model_path))
        self.session = self.interpreter.createSession(session_config())
        self.input_tensor = self.interpreter.getSessionInput(self.session)

    def get_inputs(self) -> list[SessionNode]:
        shape = tuple(self.input_tensor.getShape())
        return [_MnnNode("x", shape)]

    def get_outputs(self) -> list[SessionNode]:
        output = self.interpreter.getSessionOutput(self.session)
        shape = tuple(output.getShape())
        return [_MnnNode(None, shape)]

    def run(
        self,
        output_names: list[str] | None,
        input_feed: dict[str, NDArray[np.float32]] | dict[str, NDArray[np.int32]],
        run_options: Any = None,
    ) -> list[NDArray[np.float32]]:
        if len(input_feed) != 1:
            raise ValueError("MnnSession expects exactly one input tensor")
        img = next(iter(input_feed.values()))
        img = np.ascontiguousarray(img, dtype=np.float32)

        self.interpreter.resizeTensor(self.input_tensor, img.shape)
        self.interpreter.resizeSession(self.session)

        host = MNN.Tensor(
            img.shape,
            MNN.Halide_Type_Float,
            img,
            MNN.Tensor_DimensionType_Caffe,
        )
        self.input_tensor.copyFromHostTensor(host)
        # runSession is wrapped to call updateCacheFile after inference.
        self.interpreter.runSession(self.session)

        output = self.interpreter.getSessionOutput(self.session)
        out_shape = output.getShape()
        out_tensor = MNN.Tensor(out_shape, MNN.Halide_Type_Float, MNN.Tensor_DimensionType_Caffe)
        output.copyToHostTensor(out_tensor)
        # getData() builds a Python list and is extremely slow for OCR outputs.
        return [np.array(out_tensor.getNumpyData(), copy=True).reshape(out_shape).astype(np.float32, copy=False)]
