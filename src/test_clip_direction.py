import cv2
from clip_navigator import compute_direction_scores
from clip_core import load_clip_model

image_path = "test/test2.jpeg"
goal_text = "a chair"

frame = cv2.imread(image_path)
if frame is None:
    raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {image_path}")

loaded = load_clip_model()

# tokenizer는 쓰지 않음
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

print(scores)
print("best =", max(scores, key=scores.get))

sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
margin = sorted_scores[0][1] - sorted_scores[1][1]
print("margin =", margin)