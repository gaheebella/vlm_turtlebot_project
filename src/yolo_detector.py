from ultralytics import YOLO
import cv2

# 모델 로드 (처음 한 번만)
model = YOLO("yolov8n.pt")


def detect_goal(frame_bgr, goal_label: str):
    """
    frame_bgr: OpenCV BGR 이미지
    goal_label: COCO 클래스명 ("chair", "person", "bottle", "cup" 등)
    returns: {
        "detected": bool,
        "cx": float or None,       # bounding box 중심 x 픽셀
        "frame_w": int,            # 프레임 너비
        "area_ratio": float,       # 화면 대비 크기 (클수록 가까움)
        "conf": float,             # confidence
        "box": tuple or None,      # (x1, y1, x2, y2)
    }
    """
    results = model(frame_bgr, verbose=False)

    best_box = None
    best_conf = 0.0

    for box in results[0].boxes:
        label = model.names[int(box.cls)]
        conf = float(box.conf)
        if label == goal_label and conf > best_conf:
            best_conf = conf
            best_box = box

    if best_box is None:
        return {
            "detected": False,
            "cx": None,
            "frame_w": frame_bgr.shape[1],
            "area_ratio": 0.0,
            "conf": 0.0,
            "box": None,
        }

    x1, y1, x2, y2 = best_box.xyxy[0].tolist()
    cx = (x1 + x2) / 2
    h, w = frame_bgr.shape[:2]
    area_ratio = ((x2 - x1) * (y2 - y1)) / (w * h)

    return {
        "detected": True,
        "cx": cx,
        "frame_w": w,
        "area_ratio": area_ratio,
        "conf": best_conf,
        "box": (x1, y1, x2, y2),
    }


def decide_velocity_yolo(detection: dict):
    """
    YOLO bounding box 기반 속도 결정
    returns: (linear_vel, angular_vel, mode)
    """
    if not detection["detected"]:
        # 목표 안 보이면 천천히 회전하며 탐색
        return 0.0, 0.15, "SEARCHING"

    cx = detection["cx"]
    frame_w = detection["frame_w"]
    area_ratio = detection["area_ratio"]
    center = frame_w / 2
    offset = cx - center  # 양수 = 오른쪽, 음수 = 왼쪽

    # 목표가 너무 가까우면 정지
    if area_ratio > 0.3:
        return 0.0, 0.0, "GOAL_REACHED"

    # 정면 ±10% 이내면 전진
    if abs(offset) < frame_w * 0.1:
        return 0.15, 0.0, "GO_FORWARD"

    # 오른쪽/왼쪽으로 회전
    angular = -0.2 if offset > 0 else 0.2
    return 0.05, angular, "TRACKING"


def draw_detection(frame_bgr, detection: dict, goal_label: str):
    """
    bounding box 및 정보를 프레임에 시각화
    """
    vis = frame_bgr.copy()

    if detection["detected"]:
        x1, y1, x2, y2 = [int(v) for v in detection["box"]]
        cx = int(detection["cx"])
        h = vis.shape[0]

        # bounding box
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 중심선
        cv2.line(vis, (cx, 0), (cx, h), (0, 255, 255), 1)

        # 레이블
        label = f"{goal_label} {detection['conf']:.2f}"
        cv2.putText(vis, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # area ratio
        cv2.putText(vis, f"area: {detection['area_ratio']:.3f}", (x1, y2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    else:
        cv2.putText(vis, f"Searching: {goal_label}...", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    return vis