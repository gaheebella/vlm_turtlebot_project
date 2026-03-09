# scene label을 action으로 연결

from pathlib import Path

import torch
from PIL import Image
import open_clip
from config import prompts


def decide_action(label: str) -> str:
    if label == "an open corridor":
        return "forward"
    elif label == "free space for navigation":
        return "forward"
    elif label == "a chair blocking the path":
        return "turn"
    elif label == "a wall in front of the robot":
        return "rotate"
    elif label == "a door in front of the robot":
        return "approach"
    else:
        return "stop"


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32",
        pretrained="openai",
    )
    model = model.to(device)
    tokenizer = open_clip.get_tokenizer("ViT-B-32")

    text = tokenizer(prompts).to(device)

    image_dir = Path("data/images")
    image_paths = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.jpeg")) + list(image_dir.glob("*.png"))

    if not image_paths:
        raise FileNotFoundError(f"No images found in {image_dir}")

    print("\nCLIP + Action Decision Results:\n")

    with torch.no_grad():
        text_features = model.encode_text(text)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        for image_path in image_paths:
            image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)

            image_features = model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)

            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            scores = similarity[0].cpu().tolist()

            best_idx = max(range(len(scores)), key=lambda i: scores[i])
            best_label = prompts[best_idx]
            action = decide_action(best_label)

            print(f"Image: {image_path.name}")
            print(f"  Final scene label: {best_label}")
            print(f"  Action: {action}")
            print()


if __name__ == "__main__":
    main()