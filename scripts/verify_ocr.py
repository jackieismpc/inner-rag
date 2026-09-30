"""验证 paddle OCR 后端：生成含文字的图片，用 PaddleOCR 识别。

用于 Phase 9 落地验证：确认 paddle 后端能正确识别扫描件中的文字，
从而让图片/扫描件文档能进索引、被检索到。
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image, ImageDraw, ImageFont

from inner_rag.services.ocr import get_ocr_backend


def _make_scan_image(text: str, path: Path) -> None:
    """生成一张白底黑字的图片，模拟扫描件。"""
    width, height = 800, 200
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    # 尝试加载中文字体，找不到就用默认（PIL 默认字体不支持中文，但 PaddleOCR 仍能识别图形）
    try:
        font = ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 48)
    except Exception:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 48)
        except Exception:
            font = ImageFont.load_default()
    draw.text((40, 70), text, fill="black", font=font)
    image.save(path)
    print(f"[verify] 已生成图片: {path}")


def main() -> None:
    text = "路明非坐在窗边看书"
    tmp = Path("/tmp/ocr_verify.png")
    _make_scan_image(text, tmp)

    backend = get_ocr_backend()
    print(f"[verify] OCR 后端: {backend.name}, is_available={backend.is_available()}")

    if backend.name != "paddle":
        print(f"[verify] 失败：期望 paddle 后端，实际 {backend.name}。")
        print("请先设置 OCR_BACKEND=paddle（环境变量或 .env）")
        sys.exit(1)

    recognized = backend.ocr_bytes(tmp.read_bytes())
    print(f"[verify] 识别结果: {recognized!r}")

    # 断言识别结果包含关键文字（PaddleOCR 可能把空格/标点处理掉，只比对核心词）
    keyword = "路明非"
    if keyword in recognized.replace(" ", ""):
        print(f"[verify] ✅ 识别成功，包含关键字「{keyword}」")
        sys.exit(0)
    else:
        print(f"[verify] ❌ 识别结果不含关键字「{keyword}」")
        sys.exit(2)


if __name__ == "__main__":
    main()
