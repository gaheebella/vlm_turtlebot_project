import torch
from PIL import Image
import open_clip


def compute_direction_scores(frame_bgr, text_goal: str, model, preprocess, device):
    """
    frame_bgr: OpenCV BGR image (numpy array)
    text_goal: ex) "a chair", "a door"
    returns: {"left": float, "center": float, "right": float}
    """
    if frame_bgr is None:
        raise ValueError("frame_bgr is None")

    h, w = frame_bgr.shape[:2]

    # 겹치게 crop (좌/중/우)
    crops = {
        "left":   frame_bgr[:, :int(w * 0.4)],
        "center": frame_bgr[:, int(w * 0.3):int(w * 0.7)],
        "right":  frame_bgr[:, int(w * 0.6):],
    }

    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    text = tokenizer([text_goal]).to(device)

    with torch.no_grad():
        text_feat = model.encode_text(text)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

    scores = {}
    for name, crop in crops.items():
        img = Image.fromarray(crop[:, :, ::-1])  # BGR -> RGB
        img_tensor = preprocess(img).unsqueeze(0).to(device)

        with torch.no_grad():
            img_feat = model.encode_image(img_tensor)
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            sim = (img_feat @ text_feat.T).item()

        scores[name] = sim

    return scores


def decide_velocity(scores: dict, obstacle_penalty: float = 0.0):
    """
    scores: {"left": float, "center": float, "right": float}
    obstacle_penalty: 0.0 ~ 1.0 (LiDAR 기반 전방 장애물 패널티)

    CLIP  → 목표를 향해 접근 (goal-seeking)
    LiDAR → 물리적 장애물 긴급 회피 (image_subscriber에서 처리)

    returns:
        linear_vel, angular_vel, best_name, margin, mode
    """
    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best_name, best_score = sorted_items[0]
    second_score = sorted_items[1][1]
    margin = best_score - second_score

    # LiDAR 장애물 회피 최우선 (CLIP 무시)
    if obstacle_penalty >= 0.5:
        return 0.0, 0.15, best_name, margin, "AVOIDANCE"

    # 점수 차이 너무 작으면 불확실 → 천천히 전진하며 탐색
    if margin < 0.005:
        return 0.08, 0.0, best_name, margin, "SEARCH_FORWARD"

    # 목표가 정면 → 전진
    if best_name == "center":
        return 0.15, 0.0, best_name, margin, "GO_FORWARD"

    # 목표가 좌/우이고 margin 작으면 → 전진하면서 살짝 회전
    if margin < 0.02:
        angular = 0.08 if best_name == "left" else -0.08
        return 0.10, angular, best_name, margin, "FORWARD_ADJUST"

    # 목표가 좌/우이고 margin 크면 → 제자리 회전으로 목표 정면으로
    if best_name == "left":
        return 0.0, 0.15, best_name, margin, "TURN_LEFT"
    else:
        return 0.0, -0.15, best_name, margin, "TURN_RIGHT"