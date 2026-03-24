import os
import numpy as np
import torch
from PIL import Image

# ── Grounding DINO 설치 확인 ──────────────────────────────
# pip install groundingdino-py
# 가중치 다운로드:
#   mkdir -p weights
#   wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth -P weights/

try:
    from groundingdino.util.inference import load_model, predict, annotate
    GROUNDING_DINO_AVAILABLE = True
except ImportError:
    GROUNDING_DINO_AVAILABLE = False
    print("[grounding_detector] groundingdino 패키지 없음. pip install groundingdino-py")


CONFIG_PATH = "groundingdino/config/GroundingDINO_SwinT_OGC.py"
WEIGHTS_PATH = "weights/groundingdino_swint_ogc.pth"

_model = None


def load_grounding_model():
    global _model
    if not GROUNDING_DINO_AVAILABLE:
        raise ImportError("groundingdino-py 패키지를 설치해주세요.")
    if _model is None:
        _model = load_model(CONFIG_PATH, WEIGHTS_PATH)
    return _model


def detect_with_text(frame_bgr, text_prompt: str, threshold: float = 0.3):
    """
    frame_bgr: OpenCV BGR 이미지
    text_prompt: 자유 텍스트 ("a red chair", "exit door", "person in blue" 등)
    threshold: confidence 임계값
    returns: {
        "detected": bool,
        "cx": float or None,
        "cy": float or None,
        "frame_w": int,
        "frame_h": int,
        "conf": float,
        "box": tuple or None,   # (x1, y1, x2, y2) 픽셀 좌표
        "phrase": str,
    }
    """
    model = load_grounding_model()
    h, w = frame_bgr.shape[:2]

    boxes, logits, phrases = predict(
        model=model,
        image=frame_bgr,
        caption=text_prompt,
        box_threshold=threshold,
        text_threshold=threshold,
    )

    if len(boxes) == 0:
        return {
            "detected": False,
            "cx": None,
            "cy": None,
            "frame_w": w,
            "frame_h": h,
            "conf": 0.0,
            "box": None,
            "phrase": "",
        }

    # confidence 가장 높은 박스 선택
    best_idx = logits.argmax()
    box = boxes[best_idx].tolist()  # 정규화된 cx, cy, w, h

    # 정규화 좌표 → 픽셀 좌표 변환
    bx, by, bw, bh = box
    x1 = int((bx - bw / 2) * w)
    y1 = int((by - bh / 2) * h)
    x2 = int((bx + bw / 2) * w)
    y2 = int((by + bh / 2) * h)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    return {
        "detected": True,
        "cx": cx,
        "cy": cy,
        "frame_w": w,
        "frame_h": h,
        "conf": float(logits[best_idx]),
        "box": (x1, y1, x2, y2),
        "phrase": phrases[best_idx],
    }


def decide_velocity_grounding(detection: dict):
    """
    Grounding DINO bounding box 기반 속도 결정
    returns: (linear_vel, angular_vel, mode)
    """
    if not detection["detected"]:
        return 0.0, 0.15, "SEARCHING"

    cx = detection["cx"]
    frame_w = detection["frame_w"]
    center = frame_w / 2
    offset = cx - center  # 양수 = 오른쪽, 음수 = 왼쪽

    # 정면 ±10% 이내면 전진
    if abs(offset) < frame_w * 0.1:
        return 0.15, 0.0, "GO_FORWARD"

    # 좌우 회전
    angular = -0.2 if offset > 0 else 0.2
    return 0.05, angular, "TRACKING"