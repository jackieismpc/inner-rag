"""可插拔 OCR 后端。

默认后端为 ``none``（不启用 OCR），此时图片与扫描页会给出明确提示而不是
写入占位文本污染索引。已知实现：

* ``paddle``  —— 本地 PaddleOCR 3.x，需要 ``uv sync --extra ocr-paddle``，
  模型体积较大且首次运行需下载权重。
"""

from __future__ import annotations

import io
import os
from abc import ABC, abstractmethod
from typing import Any

from loguru import logger
from PIL import Image

from inner_rag.core.config import settings

# 必须在任何 paddleocr / paddlex import 之前设置：paddlepaddle 3.3.x 的 oneDNN 后端在
# PIR（新 IR）下对 ArrayAttribute<DoubleAttribute> 未实现，CPU 推理会抛
# "ConvertPirAttribute2RuntimeAttribute not support"（onednn_instruction.cc）。
# paddlex 在 import 期读取该开关（paddlex/utils/flags.py 的 ENABLE_MKLDNN_BYDEFAULT），
# 默认 True → run_mode=mkldnn 触发 bug；设 False 让 run_mode 走纯 paddle 内核。
# 放在模块顶部而非 _get_engine 里，因为 is_available() 就会 import paddleocr，
# 到 _get_engine 再设就太晚了。
os.environ.setdefault("PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT", "False")


class OCRBackend(ABC):
    """OCR 后端接口：输入图片，输出纯文本。"""

    name: str = "base"

    @abstractmethod
    def is_available(self) -> bool:
        """依赖是否就绪（未安装依赖时应返回 False 而不是抛异常）。"""

    @abstractmethod
    def ocr_image(self, image: Image.Image) -> str:
        """识别 PIL 图片，返回按行拼接的文本。"""

    def ocr_bytes(self, data: bytes) -> str:
        try:
            with Image.open(io.BytesIO(data)) as image:
                return self.ocr_image(image.convert("RGB"))
        except Exception as exc:
            logger.error(f"[OCR] 图片解码失败: {exc}")
            return ""


class DisabledOCR(OCRBackend):
    """默认实现：不启用 OCR。"""

    name = "none"

    def is_available(self) -> bool:
        return False

    def ocr_image(self, image: Image.Image) -> str:
        return ""


class PaddleOCRBackend(OCRBackend):
    """本地 PaddleOCR 3.x 后端（可选依赖）。"""

    name = "paddle"

    def __init__(self) -> None:
        self._engine: Any | None = None

    def is_available(self) -> bool:
        try:
            import paddleocr  # noqa: F401
        except ImportError:
            return False
        return True

    def _get_engine(self) -> Any:
        if self._engine is None:
            from paddleocr import PaddleOCR

            self._engine = PaddleOCR(
                lang=settings.OCR_LANG,
                use_textline_orientation=True,
            )
            logger.info(f"[OCR] PaddleOCR 初始化完成 (lang={settings.OCR_LANG})")
        return self._engine

    def ocr_image(self, image: Image.Image) -> str:
        import numpy as np

        result = self._get_engine().predict(np.array(image))
        return "\n".join(_extract_rec_texts(result))


def _extract_rec_texts(result: Any) -> list[str]:
    """兼容 PaddleOCR 3.x 结果对象与旧版嵌套列表结构。"""
    texts: list[str] = []
    for item in result or []:
        rec_texts = item.get("rec_texts") if hasattr(item, "get") else None
        if rec_texts:
            texts.extend(str(text) for text in rec_texts)
            continue
        # 旧版结构：[[[box, (text, score)], ...]]
        for line in item or []:
            if (
                isinstance(line, (list, tuple))
                and len(line) >= 2
                and isinstance(line[1], (list, tuple))
            ):
                texts.append(str(line[1][0]))
    return [text for text in texts if text]


_BACKENDS: dict[str, type[OCRBackend]] = {
    DisabledOCR.name: DisabledOCR,
    PaddleOCRBackend.name: PaddleOCRBackend,
}

_backend: OCRBackend | None = None


def get_ocr_backend() -> OCRBackend:
    """按配置返回 OCR 后端单例。"""
    global _backend
    if _backend is not None:
        return _backend

    configured = settings.OCR_BACKEND.strip().lower()
    backend_cls = _BACKENDS.get(configured, DisabledOCR)
    candidate = backend_cls()
    if not candidate.is_available():
        if configured != DisabledOCR.name:
            logger.warning(
                f"[OCR] 配置的 OCR_BACKEND={configured!r} 不可用（依赖未安装），已降级为不启用 OCR"
            )
        candidate = DisabledOCR()
    _backend = candidate
    logger.info(f"[OCR] 后端: {_backend.name}")
    return _backend
