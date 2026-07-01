"""OCR-free text-region detection to prune scene frames from the VLM pass.

Purpose (CLAUDE.md "wart 1"): the VLM title-card OCR burns ~5 s per frame and
truncates on busy scene frames; on a long title with no card in the primary band
it grinds through hundreds (a 50-min Venture Bros special once took 44 min). A
cheap "does this frame contain text at all?" gate lets the VLM run only on
plausible card frames.

RECALL-FIRST: never drop a real card (that loses the episode); letting a scene
frame through only costs one VLM call. So the gate fails OPEN — if no detector is
available it keeps every frame — and callers still guard with "if it would prune
the whole window, keep the window."

`frame_has_text` is the detector-agnostic seam `verify_title` calls. Backends,
preferred first:

1. **PaddleOCR** PP-OCRv3 text detection via **RapidOCR** (onnxruntime).
   Apache-2.0, model bundled with the pip package (~10 MB), no manual download.
   On a VB(stylized)+Enterprise(plain) card corpus vs scene negatives: 100%
   recall on real episode cards, ~92% scene rejection, ~170 ms/frame. Default.
2. **EAST** scene-text CNN (cv2.dnn). Same 100% real-card recall but only ~70%
   scene rejection at ~50 ms/frame. Its model (`frozen_east_text_detection.pb`,
   ~96 MB) is NOT shipped and its license is murky (GPL-3.0 upstream, dubious
   mirrors, research-terms ICDAR training data) — kept only as a fallback; point
   `VJ_EAST_MODEL` at a model to use it.

Classical detectors (morphological gradient, MSER) were evaluated and rejected
first (~60% recall / ~30% rejection). Force a backend with `VJ_TEXT_DETECTOR`
(`paddle` | `east` | `auto`, default auto). Neither available → fail open.
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


# ---------------------------------------------------------------------------
# backend 2: EAST (cv2.dnn) — optional fallback, license-murky
# ---------------------------------------------------------------------------

_DEFAULT_EAST = ("/run/media/ekovac/MediaScratc/video-juicer-artifacts/"
                 "models/frozen_east_text_detection.pb")
_NETS: dict = {}


def east_model_path(explicit: Optional[str] = None) -> str:
    return explicit or os.environ.get("VJ_EAST_MODEL", _DEFAULT_EAST)


def east_available(explicit: Optional[str] = None) -> bool:
    return Path(east_model_path(explicit)).is_file()


def east_score(img: ImageLike, *, model: Optional[str] = None,
               inp_w: int = 640, inp_h: int = 384) -> float:
    """Max per-cell text confidence [0,1] from EAST; 0.0 if unavailable."""
    import cv2
    path = east_model_path(model)
    if not Path(path).is_file():
        return 0.0
    net = _NETS.get(path)
    if net is None:
        net = _NETS[path] = cv2.dnn.readNet(path)
    im = cv2.imread(str(img)) if isinstance(img, (str, Path)) else img
    if im is None:
        return 0.0
    blob = cv2.dnn.blobFromImage(im, 1.0, (inp_w, inp_h),
                                 (123.68, 116.78, 103.94), swapRB=True, crop=False)
    net.setInput(blob)
    scores = net.forward("feature_fusion/Conv_7/Sigmoid")
    return float(scores[0, 0].max())


# ---------------------------------------------------------------------------
# the seam
# ---------------------------------------------------------------------------


def frame_has_text(img: ImageLike, *, min_boxes: int = 1, conf: float = 0.9,
                   backend: Optional[str] = None, model: Optional[str] = None) -> bool:
    """True if the frame plausibly bears text. Recall-first, fails OPEN.

    Backend order (default 'auto'): PaddleOCR (preferred), then EAST, else keep
    the frame. `VJ_TEXT_DETECTOR` overrides. A forced-but-unavailable backend
    fails open too — the filter must never silently drop frames it can't judge."""
    b = backend or os.environ.get("VJ_TEXT_DETECTOR", "auto")
    if b in ("auto", "paddle"):
        n = paddle_boxes(img)
        if n is not None:
            return n >= min_boxes
        if b == "paddle":
            return True                      # forced paddle, unavailable -> open
    if b in ("auto", "east"):
        if east_available(model):
            return east_score(img, model=model) >= conf
        if b == "east":
            return True                      # forced east, unavailable -> open
    return True                              # nothing available -> keep the frame
