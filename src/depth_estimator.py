import numpy as np
from PIL import Image
from transformers import pipeline

# 모델 로드 (처음 한 번만)
depth_pipe = pipeline(
    "depth-estimation",
    model="depth-anything/Depth-Anything-V2-Small-hf"
)


def estimate_depth(frame_bgr, box=None):
    """
    frame_bgr: OpenCV BGR 이미지
    box: (x1, y1, x2, y2) bounding box — None이면 전체 프레임 평균
    returns: 상대적 깊이값 (작을수록 가까움, 0~255 범위)
    """
    img = Image.fromarray(frame_bgr[:, :, ::-1])  # BGR → RGB
    result = depth_pipe(img)
    depth_map = np.array(result["depth"], dtype=np.float32)

    if box is not None:
        x1, y1, x2, y2 = [int(v) for v in box]
        # 범위 클리핑
        h, w = depth_map.shape
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        roi = depth_map[y1:y2, x1:x2]
        if roi.size == 0:
            return float(depth_map.mean())
        return float(roi.mean())

    return float(depth_map.mean())


def get_depth_velocity(depth_value: float):
    """
    깊이값에 따라 전진 속도 결정
    depth_value: estimate_depth() 반환값 (작을수록 가까움)
    returns: (linear_vel, mode)
    """
    if depth_value < 50:
        # 매우 가까움 → 정지
        return 0.0, "DEPTH_STOP"
    elif depth_value < 100:
        # 가까움 → 천천히 접근
        return 0.05, "DEPTH_SLOW"
    elif depth_value < 180:
        # 중간 거리 → 보통 속도
        return 0.10, "DEPTH_MEDIUM"
    else:
        # 멀리 있음 → 빠르게 접근
        return 0.15, "DEPTH_FAST"


def get_full_depth_map(frame_bgr):
    """
    전체 깊이 맵 반환 (numpy array, shape: HxW)
    시각화나 장애물 감지에 활용
    """
    img = Image.fromarray(frame_bgr[:, :, ::-1])
    result = depth_pipe(img)
    return np.array(result["depth"], dtype=np.float32)