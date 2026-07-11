"""OCR-free text-region detection to prune scene frames from the VLM pass.

Purpose (CLAUDE.md "wart 1"): the VLM title-card OCR burns ~5 s per frame and
truncates on busy scene frames; on a long title with no card in the primary band
it grinds through hundreds (a 50-min Venture Bros special once took 44 min). A
cheap "does this frame contain text at all?" gate lets the VLM run only on
plausible card frames.

RECALL-FIRST: never drop a real card (that loses the episode); letting a scene
frame through only costs one VLM call. So the gate fails OPEN — if the detector is
unavailable it keeps every frame — and callers still guard with "if it would prune
the whole window, keep the window."

`frame_has_text` is the seam `verify_title` calls. The detector is **PaddleOCR**
PP-OCRv3 text detection via **RapidOCR** (onnxruntime): Apache-2.0, model bundled
with the pip package (~10 MB), no manual download. On a VB(stylized)+Enterprise
(plain) card corpus vs scene negatives: 100% recall on real episode cards, ~92%
scene rejection, ~170 ms/frame. Classical detectors (morphological gradient, MSER)
were evaluated and rejected first (~60% recall / ~30% rejection). Detector
unavailable → fail open.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union

ImageLike = Union[str, Path, "object"]


# ---------------------------------------------------------------------------
# backend 1: PaddleOCR PP-OCRv3 detection via RapidOCR (Apache-2.0) — preferred
# ---------------------------------------------------------------------------

_RAPID = None
_RAPID_TRIED = False


def _rapid():
    global _RAPID, _RAPID_TRIED
    if not _RAPID_TRIED:
        _RAPID_TRIED = True
        try:
            from rapidocr_onnxruntime import RapidOCR
            _RAPID = RapidOCR()
        except Exception:  # noqa: BLE001 — package/model missing
            _RAPID = None
    return _RAPID


def paddle_boxes(img: ImageLike) -> Optional[int]:
    """#text boxes PP-OCRv3 detects, or None if the backend is unavailable."""
    eng = _rapid()
    if eng is None:
        return None
    src = str(img) if isinstance(img, (str, Path)) else img
    res, _ = eng(src, use_det=True, use_cls=False, use_rec=False)
    return 0 if res is None else len(res)


def paddle_available() -> bool:
    return _rapid() is not None


def ocr_text(img: ImageLike) -> str:
    """Full PP-OCR recognition (det+rec) of an image → recognized text, boxes
    joined in reading order. "" if the backend is unavailable or nothing is read.
    Used for subtitle-bitmap OCR (PGS / VOBSUB), where PP-OCR is markedly more
    accurate than tesseract — especially on low-res DVD VOBSUB. `use_cls=False`
    skips angle classification — subtitles are horizontal, so it's wasted work."""
    eng = _rapid()
    if eng is None:
        return ""
    src = str(img) if isinstance(img, (str, Path)) else img
    res, _ = eng(src, use_cls=False)
    if not res:
        return ""
    return " ".join(line[1] for line in res).strip()


# --- parallel OCR over many frames (subtitle-card OCR is embarrassingly parallel;
# each PP-OCR call is one core, so a process pool = ~one frame per core at once) ---
_POOL_ENG = None


def _pool_init():
    """Per-worker: a SINGLE-THREAD RapidOCR (so N workers ≈ N cores, no
    oversubscription). OMP is pinned to 1 before the engine's onnx sessions load."""
    import os
    os.environ["OMP_NUM_THREADS"] = "1"
    global _POOL_ENG
    from rapidocr_onnxruntime import RapidOCR
    try:
        _POOL_ENG = RapidOCR(intra_op_num_threads=1)
    except Exception:  # noqa: BLE001
        _POOL_ENG = RapidOCR()


def _pool_ocr(path: str) -> str:
    res, _ = _POOL_ENG(path, use_cls=False)
    return " ".join(line[1] for line in res).strip() if res else ""


_POOL = None            # persistent worker pool, reused across titles in a run


def _get_pool(workers: int):
    """A PERSISTENT spawn pool — workers load the OCR models ONCE and stay warm
    for the whole run. Re-spawning per title (each worker re-importing onnx +
    re-loading models) dominated the cost otherwise. Torn down at process exit."""
    global _POOL
    if _POOL is None:
        import multiprocessing as mp
        import atexit
        _POOL = mp.get_context("spawn").Pool(workers, initializer=_pool_init)
        atexit.register(lambda: _POOL and _POOL.terminate())
    return _POOL


def ocr_texts(paths, workers: Optional[int] = None) -> list:
    """OCR a list of image paths, in order (so consecutive-dedup still works),
    across a persistent process pool. Falls back to serial for a short list / no
    backend / single worker. Order-preserving (`pool.map`)."""
    paths = [str(p) for p in paths]
    if _rapid() is None:
        return ["" for _ in paths]
    import os
    if workers is None:
        workers = max(1, (os.cpu_count() or 2) - 2)
    if workers <= 1 or len(paths) < 8:
        return [ocr_text(p) for p in paths]
    return _get_pool(workers).map(_pool_ocr, paths, chunksize=4)


# ---------------------------------------------------------------------------
# the seam
# ---------------------------------------------------------------------------


def frame_has_text(img: ImageLike, *, min_boxes: int = 1,
                   backend: Optional[str] = None) -> bool:
    """True if the frame plausibly bears text. Recall-first, fails OPEN.

    The detector is PaddleOCR (via RapidOCR). `VJ_TEXT_DETECTOR` is honored for
    compatibility ('paddle'/'auto'); an unavailable or unrecognized detector fails
    open — the filter must never silently drop frames it can't judge."""
    b = backend or os.environ.get("VJ_TEXT_DETECTOR", "auto")
    if b in ("auto", "paddle"):
        n = paddle_boxes(img)
        if n is not None:
            return n >= min_boxes
    return True                              # detector unavailable -> keep the frame
