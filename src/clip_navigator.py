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

    # 겹치게 crop
    crops = {
        "left": frame_bgr[:, :int(w * 0.4)],
        "center": frame_bgr[:, int(w * 0.3):int(w * 0.7)],
        "right": frame_bgr[:, int(w * 0.6):],
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
    obstacle_penalty: 현재는 사용하지 않거나 0.0으로 둠

    returns:
        linear_vel, angular_vel, best_name, margin, mode
    """
    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best_name, best_score = sorted_items[0]
    second_score = sorted_items[1][1]
    margin = best_score - second_score

    # 점수 차이가 너무 작으면 불확실하다고 보고 정지
    if margin < 0.02:
        return 0.0, 0.0, best_name, margin, "UNCERTAIN_STOP"

    # 장애물 회피는 나중에 붙일 수 있도록 자리만 남김
    if obstacle_penalty >= 0.5:
        return 0.0, 0.3, best_name, margin, "AVOIDANCE"

    # 방향 제어
    if best_name == "left":
        return 0.0, 0.25, best_name, margin, "TURN_LEFT"

    elif best_name == "right":
        return 0.0, -0.25, best_name, margin, "TURN_RIGHT"

    else:  # center
        return 0.08, 0.0, best_name, margin, "GO_FORWARD"