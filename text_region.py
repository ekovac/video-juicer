"""OCR-free text-region detection to prune scene frames from the VLM pass.

Purpose (CLAUDE.md "wart 1"): the VLM title-card OCR burns ~5 s per frame and
truncates on busy scene frames; on a long title with no card in the primary band
it grinds through hundreds of them (a 50-min Venture Bros special once took
44 min). A cheap "does this frame contain text at all?" gate lets the VLM run
only on plausible card frames.

RECALL-FIRST: never drop a real card (that loses the episode); letting a scene
frame through only costs one VLM call. So the gate fails OPEN — if the model is
missing or unsure, keep the frame — and callers still guard with "if it would
prune everything, keep everything."

The gate is detector-agnostic: `verify_title` only calls `frame_has_text`. The
default backend is **EAST** (pretrained scene-text CNN via cv2.dnn). Classical
detectors were evaluated and rejected — a morphological gradient gate drowns on
stylized script over textured backgrounds (~60% card recall) and MSER over-fires
on scene edges (~30% scene rejection). EAST holds ~100% recall on real episode
cards across plain (Enterprise) and ornate/textured (Venture Bros) styles while
pruning ~70% of scene frames, at ~50 ms/frame.

EAST needs a ~96 MB model file (`frozen_east_text_detection.pb`); point
`VJ_EAST_MODEL` at it or drop it in the artifacts `models/` dir. Absent → the
gate fails open (no pruning). **Licensing:** the model is NOT shipped with this
repo and its license is murky — the upstream implementation (argman/EAST) is
GPL-3.0, common frozen mirrors relicense it dubiously, and it's trained on
research-terms ICDAR data. Fine for personal use (never redistributed here); for
a cleaner story swap in a permissively-licensed detector (PaddleOCR / docTR
DBNet, Apache-2.0) behind this same `frame_has_text` seam. See DESIGN.md.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union

import cv2

ImageLike = Union[str, Path, "cv2.typing.MatLike"]

_DEFAULT_MODEL = ("/run/media/ekovac/MediaScratc/video-juicer-artifacts/"
                  "models/frozen_east_text_detection.pb")
_NETS: dict = {}


def model_path(explicit: Optional[str] = None) -> str:
    return explicit or os.environ.get("VJ_EAST_MODEL", _DEFAULT_MODEL)


def model_available(explicit: Optional[str] = None) -> bool:
    return Path(model_path(explicit)).is_file()


def _net(path: str):
    net = _NETS.get(path)
    if net is None:
        net = cv2.dnn.readNet(path)
        _NETS[path] = net
    return net


def frame_text_score(img: ImageLike, *, model: Optional[str] = None,
                     inp_w: int = 640, inp_h: int = 384) -> float:
    """Max per-cell text confidence [0,1] EAST assigns the frame. Input dims
    must be multiples of 32. Returns 1.0 (fail open) if the model is missing."""
    path = model_path(model)
    if not Path(path).is_file():
        return 1.0
    im = cv2.imread(str(img)) if isinstance(img, (str, Path)) else img
    if im is None:
        return 1.0
    blob = cv2.dnn.blobFromImage(im, 1.0, (inp_w, inp_h),
                                 (123.68, 116.78, 103.94), swapRB=True, crop=False)
    net = _net(path)
    net.setInput(blob)
    scores = net.forward("feature_fusion/Conv_7/Sigmoid")
    return float(scores[0, 0].max())


def frame_has_text(img: ImageLike, *, conf: float = 0.9,
                   model: Optional[str] = None, **kw) -> bool:
    """True if the frame plausibly bears text (EAST conf >= `conf`).

    Recall-first: text cards score ~1.0, so a high threshold prunes more scenes
    without dropping cards. Fails OPEN (True) when the model is unavailable."""
    return frame_text_score(img, model=model, **kw) >= conf
