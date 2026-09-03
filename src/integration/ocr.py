"""
ocr.py — 图片 OCR 文本提取
===========================
使用 EasyOCR 从图片中提取文字，支持中英文混合。
模型加载移到后台线程，绝不阻塞事件循环。

优化:
- 启动时预加载模型（warmup）
- 大图自动缩小到 512px
- OCR 超时 30s，超时自动降级
"""

import asyncio
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

from src.core.logging import get_logger

logger = get_logger(__name__)

MAX_SIZE = 512
OCR_TIMEOUT = 45  # 单次 OCR 最长等待秒数
_ocr_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ocr")
_reader = None
_reader_ready = asyncio.Event()
_warmed_up = False


def _load_reader():
    """在后台线程加载 EasyOCR 模型（不阻塞事件循环）。"""
    global _reader
    import easyocr
    logger.info("EasyOCR 模型加载中（后台线程）...")
    t0 = time.time()
    _reader = easyocr.Reader(['ch_sim', 'en'], gpu=False)
    logger.info(f"EasyOCR 模型加载完成 ({time.time()-t0:.0f}s)")


async def warmup():
    """启动时调用：后台加载 OCR 模型。"""
    global _warmed_up
    if _warmed_up:
        return
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(_ocr_executor, _load_reader)
    _reader_ready.set()
    _warmed_up = True


def _preprocess(image_path: str) -> str:
    """大图缩小到 MAX_SIZE，返回处理后的路径。"""
    img = Image.open(image_path)
    w, h = img.size

    # 统一转 RGB（JPEG 不支持 RGBA）
    if img.mode in ('RGBA', 'P', 'LA'):
        img = img.convert('RGB')

    if max(w, h) <= MAX_SIZE and img.mode == 'RGB':
        return image_path

    if max(w, h) > MAX_SIZE:
        ratio = MAX_SIZE / max(w, h)
        new_size = (int(w * ratio), int(h * ratio))
        img = img.resize(new_size, Image.LANCZOS)

    tmp_path = image_path + '.ocr_tmp.jpg'
    img.save(tmp_path, 'JPEG', quality=80)
    logger.info(f"图片预处理: {w}x{h} -> {img.size[0]}x{img.size[1]} ({img.mode})")
    return tmp_path


def _do_ocr(image_path: str) -> str:
    """同步执行 OCR（在后台线程中调用）。"""
    reader = _reader
    if reader is None:
        _load_reader()
        reader = _reader
    results = reader.readtext(image_path, detail=0)
    return "\n".join(results)


async def extract_text(image_path: str) -> str:
    """从图片中提取文字，绝不阻塞事件循环。

    返回: 提取的文字；超时或失败时返回空字符串。
    """
    global _warmed_up
    if not _warmed_up:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(_ocr_executor, _load_reader)
        _reader_ready.set()
        _warmed_up = True

    loop = asyncio.get_running_loop()
    t0 = time.time()

    # 预处理（可能阻塞 IO，用 executor）
    ocr_path = image_path
    try:
        ocr_path = await loop.run_in_executor(_ocr_executor, _preprocess, image_path)
    except Exception as e:
        logger.warning(f"图片预处理失败: {e}")

    # OCR 识别（CPU 密集，用 executor + 超时）
    try:
        text = await asyncio.wait_for(
            loop.run_in_executor(_ocr_executor, _do_ocr, ocr_path),
            timeout=OCR_TIMEOUT,
        )
        elapsed = (time.time() - t0) * 1000
        logger.info(f"OCR 完成: {len(text)} 字符, {elapsed:.0f}ms")
        return text.strip()
    except asyncio.TimeoutError:
        logger.warning(f"OCR 超时 ({OCR_TIMEOUT}s)，跳过")
        return ""
    except Exception as e:
        logger.error(f"OCR 失败: {e}")
        return ""
    finally:
        if ocr_path != image_path:
            try:
                os.remove(ocr_path)
            except Exception:
                pass
