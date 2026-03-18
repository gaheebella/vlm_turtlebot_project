import torch
from PIL import Image
import open_clip
from collections import deque, Counter


class ClipDirectionNavigator:
    def __init__(
        self,
        model,
        preprocess,
        device,
        model_name="ViT-B-32",
        history_size=5,
        center_bias=0.015,
    ):
        self.model = model
        self.preprocess = preprocess
        self.device = device
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.history = deque(maxlen=history_size)
        self.center_bias = center_bias

    def compute_direction_scores(self, frame_bgr, text_goal: str):
        """
        frame_bgr: OpenCV BGR image (numpy array)
        text_goal: ex) "a chair", "a door"

        returns:
            {
                "left": float,
                "center": float,
                "right": float
            }
        """
        if frame_bgr is None:
            raise ValueError("frame_bgr is None")

        h, w = frame_bgr.shape[:2]
        if h == 0 or w == 0:
            raise ValueError("Invalid frame size")

        # 겹치는 ROI
        crops = {
            "left": frame_bgr[:, : int(w * 0.4)],
            "center": frame_bgr[:, int(w * 0.3): int(w * 0.7)],
            "right": frame_bgr[:, int(w * 0.6):],
        }

        text = self.tokenizer([text_goal]).to(self.device)

        with torch.no_grad():
            text_feat = self.model.encode_text(text)
            text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

        scores = {}
        for name, crop in crops.items():
            if crop.size == 0:
                scores[name] = -1.0
                continue

            img = Image.fromarray(crop[:, :, ::-1])  # BGR -> RGB
            img_tensor = self.preprocess(img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                img_feat = self.model.encode_image(img_tensor)
                img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
                sim = (img_feat @ text_feat.T).item()

            scores[name] = sim

        # center bias
        scores["center"] += self.center_bias

        return scores

    def smooth_direction(self, current_direction: str):
        self.history.append(current_direction)
        count = Counter(self.history)
        smoothed_direction = count.most_common(1)[0][0]
        return smoothed_direction

    def decide_velocity(self, scores: dict, obstacle_penalty: float = 0.0):
        """
        returns:
            linear_vel, angular_vel, best_raw, best_smooth, margin, mode
        """
        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        best_raw, best_score = sorted_items[0]
        second_score = sorted_items[1][1]
        margin = best_score - second_score

        best_smooth = self.smooth_direction(best_raw)

        # LiDAR 회피 최우선
        if obstacle_penalty >= 0.5:
            return 0.0, 0.18, best_raw, best_smooth, margin, "AVOIDANCE"

        # 확신이 너무 낮으면 천천히 전진
        if margin < 0.03:
            return 0.06, 0.0, best_raw, best_smooth, margin, "SEARCH_FORWARD"

        # 정면이면 전진
        if best_smooth == "center":
            if margin >= 0.08:
                return 0.16, 0.0, best_raw, best_smooth, margin, "GO_FORWARD_FAST"
            else:
                return 0.10, 0.0, best_raw, best_smooth, margin, "GO_FORWARD"

        # 좌우인데 확신이 중간 정도면 전진+약회전
        if margin < 0.08:
            angular = 0.10 if best_smooth == "left" else -0.10
            return 0.08, angular, best_raw, best_smooth, margin, "FORWARD_ADJUST"

        # 좌우 확신이 크면 회전
        if best_smooth == "left":
            return 0.0, 0.18, best_raw, best_smooth, margin, "TURN_LEFT"
        else:
            return 0.0, -0.18, best_raw, best_smooth, margin, "TURN_RIGHT"