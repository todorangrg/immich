from typing import Any, Iterable

import numpy as np
import numpy.typing as npt
from rapidocr.utils.typings import EngineType, LangRec
from typing_extensions import TypedDict


class TextDetectionOutput(TypedDict):
    boxes: npt.NDArray[np.float32]
    scores: npt.NDArray[np.float32]


class TextRecognitionOutput(TypedDict):
    box: npt.NDArray[np.float32]
    boxScore: npt.NDArray[np.float32]
    text: Iterable[str]
    textScore: npt.NDArray[np.float32]


# RapidOCR expects `engine_type`, `lang_type`, and `font_path` to be attributes
class OcrOptions(dict[str, Any]):
    def __init__(
        self,
        lang_type: LangRec | None = None,
        engine_type: EngineType = EngineType.ONNXRUNTIME,
        **options: Any,
    ) -> None:
        super().__init__(**options)
        self.engine_type = engine_type
        self.lang_type = lang_type
        self.font_path = None
        # RapidOCR also reads these as attributes for non-ORT engines (e.g. MNN dict download)
        for key in ("ocr_version", "task_type", "model_type"):
            if key in options:
                setattr(self, key, options[key])
