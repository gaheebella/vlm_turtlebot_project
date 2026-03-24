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
        detect_threshold=0.24,
        margin_threshold=0.005,
        stop_area_ratio=0.38,
    ):
        self.model = model
        self.preprocess = preprocess
        self.device = device
        self.tokenizer = open_clip.get_tokenizer(model_name)

        self.history = deque(maxlen=history_size)
        self.center_bias = center_bias
        self.detect_threshold = detect_threshold
        self.margin_threshold = margin_threshold
        self.stop_area_ratio = stop_area_ratio

    def compute_direction_scores(self, frame_bgr, text_goal: str):
        if frame_bgr is None:
            raise ValueError("frame_bgr is None")

        h, w = frame_bgr.shape[:2]
        if h == 0 or w == 0:
            raise ValueError("Invalid frame size")

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

        scores["center"] += self.center_bias

        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        best_raw, best_score = sorted_items[0]
        second_score = sorted_items[1][1]
        margin = best_score - second_score

        best_smooth = self._smooth_direction(best_raw)

        target_visible = (
            best_score >= self.detect_threshold and margin >= self.margin_threshold
        )

        center_dominant = best_smooth == "center" and target_visible

        side_mean = (scores["left"] + scores["right"]) / 2.0
        approx_area_ratio = max(0.0, min(1.0, (scores["center"] - side_mean) * 8.0))

        meta = {
            "best_raw": best_raw,
            "best_smooth": best_smooth,
            "best_score": best_score,
            "margin": margin,
            "target_visible": target_visible,
            "center_dominant": center_dominant,
            "approx_area_ratio": approx_area_ratio,
        }

        return scores, meta

    def _smooth_direction(self, current_direction: str):
        self.history.append(current_direction)
        count = Counter(self.history)
        return count.most_common(1)[0][0]

    def estimate_arrival(self, meta: dict):
        # 기존 판정은 너무 쉽게 STOPPED로 들어감
        return (
            meta["center_dominant"]
            and meta["best_score"] >= 0.32
            and meta["approx_area_ratio"] >= 0.75
        )

    def decide_stateful_velocity(
        self,
        nav_state: str,
        scores: dict,
        meta: dict,
        obstacle_penalty: float = 0.0,
        search_turn_dir: int = 1,
    ):
        best_dir = meta["best_smooth"]
        margin = meta["margin"]
        target_visible = meta["target_visible"]
        center_dominant = meta["center_dominant"]

        if obstacle_penalty >= 0.5:
            angular = 0.35 if search_turn_dir >= 0 else -0.35
            return 0.0, angular, "AVOIDING", "AVOIDING"

        if self.estimate_arrival(meta):
            return 0.0, 0.0, "STOPPED", "STOPPED_AT_GOAL"

        if nav_state == "SEARCHING":
            if target_visible:
                if best_dir == "center":
                    return 0.10, 0.0, "APPROACHING", "TARGET_FOUND_CENTER_GO"
                return 0.0, 0.18 if best_dir == "left" else -0.18, "ALIGNING", "TARGET_FOUND_SIDE_ALIGN"

            return 0.0, 0.18 if search_turn_dir >= 0 else -0.18, "SEARCHING", "SEARCH_TURN"

        if nav_state == "ALIGNING":
            if not target_visible:
                return 0.0, 0.18 if search_turn_dir >= 0 else -0.18, "SEARCHING", "LOST_TARGET_RESEARCH"

            if center_dominant:
                return 0.10, 0.0, "APPROACHING", "ALIGNED_GO"

            if best_dir == "left":
                angular = 0.16 if margin < 0.08 else 0.22
                return 0.0, angular, "ALIGNING", "TURN_LEFT_TO_ALIGN"
            elif best_dir == "right":
                angular = -0.16 if margin < 0.08 else -0.22
                return 0.0, angular, "ALIGNING", "TURN_RIGHT_TO_ALIGN"
            else:
                return 0.10, 0.0, "APPROACHING", "CENTER_CONFIRMED_GO"

        if nav_state == "APPROACHING":
            if not target_visible:
                return 0.0, 0.18 if search_turn_dir >= 0 else -0.18, "SEARCHING", "LOST_TARGET_WHILE_APPROACHING"

            if not center_dominant:
                if best_dir == "left":
                    return 0.0, 0.16, "ALIGNING", "DRIFTED_LEFT_REALIGN"
                elif best_dir == "right":
                    return 0.0, -0.16, "ALIGNING", "DRIFTED_RIGHT_REALIGN"

            if meta["approx_area_ratio"] > 0.28:
                return 0.06, 0.0, "APPROACHING", "APPROACH_SLOW"
            else:
                return 0.12, 0.0, "APPROACHING", "APPROACH_FORWARD"

        if nav_state == "AVOIDING":
            if obstacle_penalty >= 0.5:
                angular = 0.35 if search_turn_dir >= 0 else -0.35
                return 0.0, angular, "AVOIDING", "CONTINUE_AVOIDING"

            if target_visible:
                if center_dominant:
                    return 0.10, 0.0, "APPROACHING", "RECOVERED_AFTER_AVOID_GO"
                return 0.0, 0.16 if best_dir == "left" else -0.16, "ALIGNING", "REALIGN_AFTER_AVOID"

            return 0.0, 0.18 if search_turn_dir >= 0 else -0.18, "SEARCHING", "SEARCH_AFTER_AVOID"

        if nav_state == "STOPPED":
            return 0.0, 0.0, "STOPPED", "STOPPED"

        return 0.0, 0.18 if search_turn_dir >= 0 else -0.18, "SEARCHING", "FALLBACK_SEARCH"