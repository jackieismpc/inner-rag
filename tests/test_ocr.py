"""OCR 后端测试（离线，不实际调用 PaddleOCR 推理）。

覆盖：
1. ``_extract_rec_texts`` 结果解析对新旧两种 PaddleOCR 结果结构的兼容；
2. 模块导入时已设置 ``PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT=False``（绕过 paddlepaddle
   3.3.x 的 oneDNN + PIR bug，见 ocr.py 模块顶部注释）；
3. ``PaddleOCRBackend.is_available()`` 在未安装 paddleocr 时返回 False（离线 CI）。
"""

from __future__ import annotations

import os

import pytest

from inner_rag.services.ocr import (
    DisabledOCR,
    PaddleOCRBackend,
    _extract_rec_texts,
)


def test_extract_rec_texts_new_structure() -> None:
    """PaddleOCR 3.x 结果对象：list[dict]，每个 dict 带 rec_texts。"""
    result = [
        {"rec_texts": ["路明非", "坐在窗边"], "rec_scores": [0.9, 0.8]},
        {"rec_texts": ["看书"], "rec_scores": [0.95]},
    ]
    assert _extract_rec_texts(result) == ["路明非", "坐在窗边", "看书"]


def test_extract_rec_texts_old_structure() -> None:
    """旧版嵌套结构：[[[box, (text, score)], ...]]。"""
    result = [
        [[[0, 0, 1, 1], ("路明非", 0.9)], [[0, 0, 1, 1], ("看书", 0.8)]],
    ]
    assert _extract_rec_texts(result) == ["路明非", "看书"]


def test_extract_rec_texts_empty() -> None:
    assert _extract_rec_texts([]) == []
    assert _extract_rec_texts(None) == []


def test_extract_rec_texts_filters_empty() -> None:
    """rec_texts 里的空字符串被过滤。"""
    result = [{"rec_texts": ["", "路明非", ""], "rec_scores": [0, 0.9, 0]}]
    assert _extract_rec_texts(result) == ["路明非"]


def test_module_sets_mkldnn_env_flag() -> None:
    """导入 ocr 模块时已禁用 mkldnn 默认值，绕过 oneDNN+PIR 的崩溃 bug。"""
    assert os.environ.get("PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT") == "False"


def test_paddle_backend_unavailable_without_dependency() -> None:
    """离线 CI 没装 paddleocr 时，is_available 返回 False（而非抛异常）。

    本用例在装有 paddleocr 的环境里会跳过（此时 is_available 为 True）。
    """
    backend = PaddleOCRBackend()
    try:
        import paddleocr  # noqa: F401
    except ImportError:
        assert backend.is_available() is False
    else:
        pytest.skip("paddleocr 已安装，is_available 应为 True")


def test_disabled_ocr_is_not_available() -> None:
    assert DisabledOCR().is_available() is False
