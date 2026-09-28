"""文档解析测试（不依赖 OCR 后端）。"""

from __future__ import annotations

from pathlib import Path

import pytest

from inner_rag.services.parser import parser


def test_get_file_type() -> None:
    assert parser.get_file_type("a.PDF") == "pdf"
    assert parser.get_file_type("a.docx") == "word"
    assert parser.get_file_type("a.xlsx") == "excel"
    assert parser.get_file_type("a.md") == "text"
    assert parser.get_file_type("a.png") == "image"
    assert parser.get_file_type("a.bin") == "unknown"


def test_parse_text_file(tmp_path: Path) -> None:
    target = tmp_path / "note.txt"
    target.write_text("inner-rag 解析测试内容。", encoding="utf-8")

    documents, meta = parser.parse(str(target), target.name)

    assert len(documents) == 1
    assert "inner-rag" in documents[0].page_content
    assert meta["file_type"] == "text"
    assert meta["total_chars"] == len("inner-rag 解析测试内容。")
    assert meta["ocr_enabled"] is False


def test_parse_gbk_text_file(tmp_path: Path) -> None:
    """编码检测：GBK 文件不能按 UTF-8 解出乱码。"""
    target = tmp_path / "gbk.txt"
    target.write_bytes("中文编码测试".encode("gbk"))

    documents, _ = parser.parse(str(target), target.name)

    assert documents[0].page_content.strip() == "中文编码测试"


def test_parse_pdf_pages(tmp_path: Path) -> None:
    import pymupdf

    path = tmp_path / "doc.pdf"
    with pymupdf.open() as doc:
        page = doc.new_page()
        page.insert_text((72, 100), "inner-rag PDF page one")
        doc.new_page()  # 第二页留空，用来验证「无文本页被跳过」
        doc.save(path)

    documents, meta = parser.parse(str(path), path.name)

    assert len(documents) == 1
    assert documents[0].metadata["page"] == 1
    assert meta["page_count"] == 1


def test_parse_unsupported_type(tmp_path: Path) -> None:
    target = tmp_path / "data.bin"
    target.write_bytes(b"\x00\x01")

    with pytest.raises(ValueError, match="不支持的文件类型"):
        parser.parse(str(target), target.name)


def test_parse_empty_text_raises(tmp_path: Path) -> None:
    """解析不到文本时必须显式报错，而不是写入占位文本污染索引。"""
    target = tmp_path / "blank.txt"
    target.write_text("   \n\n  ", encoding="utf-8")

    with pytest.raises(ValueError, match=r"未能从.*提取到任何文本"):
        parser.parse(str(target), target.name)


def test_parse_image_without_ocr(tmp_path: Path) -> None:
    from PIL import Image

    target = tmp_path / "scan.png"
    Image.new("RGB", (20, 20), "white").save(target)

    with pytest.raises(ValueError, match="OCR"):
        parser.parse(str(target), target.name)
