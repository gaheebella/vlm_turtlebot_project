from pathlib import Path

import torch
from PIL import Image
import open_clip
from config import prompts


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32",
        pretrained="openai",
    )
    model = model.to(device)
    tokenizer = open_clip.get_tokenizer("ViT-B-32")

    image_path = Path("data/images/corridor.jpeg")
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")


    image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    text = tokenizer(prompts).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    scores = similarity[0].cpu().tolist()

    print(f"\nInput image: {image_path.name}\n")
    for prompt, score in zip(prompts, scores):
        print(f"{prompt}: {score:.4f}")

    best_idx = max(range(len(scores)), key=lambda i: scores[i])
    print("\nFinal scene label:")
    print(prompts[best_idx])


if __name__ == "__main__":
    main()