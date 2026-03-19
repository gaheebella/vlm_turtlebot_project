import cv2
from clip_navigator import compute_direction_scores, decide_velocity
from clip_core import load_clip_model

# ── 설정 ────────────────────────────────────────────────────
image_path = "test/test2.jpeg"
goal_text = "a chair"
# ────────────────────────────────────────────────────────────

frame = cv2.imread(image_path)
if frame is None:
    raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {image_path}")

loaded = load_clip_model()
if len(loaded) == 4:
    model, preprocess, _, device = loaded
elif len(loaded) == 3:
    model, preprocess, device = loaded
else:
    raise ValueError("load_clip_model() 반환값 개수를 확인하세요.")

scores = compute_direction_scores(
    frame_bgr=frame,
    text_goal=goal_text,
    model=model,
    preprocess=preprocess,
    device=device,
)

linear_v, angular_v, best_name, margin, mode = decide_velocity(
    scores=scores,
    obstacle_penalty=0.0,
)

print("=" * 40)
print(f"Goal      : {goal_text}")
print(f"Scores    : {scores}")
print(f"Best      : {best_name}")
print(f"Margin    : {margin:.4f}")
print(f"Mode      : {mode}")
print(f"cmd_vel   : v={linear_v:.2f}  w={angular_v:.2f}")
print("=" * 40)