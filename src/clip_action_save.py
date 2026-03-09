# clip_action_save.py : 액션의 실험 결과 표 (v2)

from pathlib import Path
import csv

import torch
from PIL import Image
import open_clip


def decide_action(label: str) -> str:
    if label == "a long indoor corridor":
        return "forward"
    elif label == "free open floor space":
        return "forward"
    elif label == "a chair blocking the robot path":
        return "turn"
    elif label == "a close wall directly blocking the robot":
        return "rotate"
    elif label == "a door directly in front of the robot":
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

    # 수정된 prompt
    prompts = [
        "a door directly in front of the robot",
        "a chair blocking the robot path",
        "a table obstacle",
        "a long indoor corridor",
        "free open floor space",
        "a close wall directly blocking the robot",
    ]

    text = tokenizer(prompts).to(device)

    image_dir = Path("data/images")
    image_paths = (
        list(image_dir.glob("*.jpg"))
        + list(image_dir.glob("*.jpeg"))
        + list(image_dir.glob("*.png"))
    )

    if not image_paths:
        raise FileNotFoundError(f"No images found in {image_dir}")

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    output_csv = results_dir / "clip_action_results_v2.csv"

    rows = []

    print("\nCLIP + Action Decision Results (v2):\n")

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
            best_score = scores[best_idx]
            action = decide_action(best_label)

            print(f"Image: {image_path.name}")
            for prompt, score in zip(prompts, scores):
                print(f"  {prompt}: {score:.4f}")
            print(f"  Final scene label: {best_label}")
            print(f"  Best score: {best_score:.4f}")
            print(f"  Action: {action}")
            print()

            rows.append({
                "image": image_path.name,
                "scene_label": best_label,
                "best_score": round(best_score, 4),
                "action": action,
            })

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["image", "scene_label", "best_score", "action"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved results to: {output_csv}")


if __name__ == "__main__":
    main()