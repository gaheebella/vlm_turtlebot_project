from pathlib import Path

from config import PROMPTS
from clip_core import load_clip_model, infer_image, get_device


def main() -> None:
    print(f"Using device: {get_device()}")

    model, preprocess, text_features, device = load_clip_model()

    image_path = Path("data/images/corridor.jpeg")
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    result = infer_image(image_path, model, preprocess, text_features, device)

    print(f"\nInput image: {result['image']}\n")
    for prompt, score in zip(PROMPTS, result["scores"]):
        print(f"{prompt}: {score:.4f}")

    print("\nFinal scene label:")
    print(result["best_label"])


if __name__ == "__main__":
    main()