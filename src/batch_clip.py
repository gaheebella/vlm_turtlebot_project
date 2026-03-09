from config import PROMPTS
from clip_core import load_clip_model, infer_image, get_image_paths, get_device


def main() -> None:
    print(f"Using device: {get_device()}")

    model, preprocess, text_features, device = load_clip_model()
    image_paths = get_image_paths()

    if not image_paths:
        raise FileNotFoundError("No images found in data/images")

    print("\nBatch CLIP inference results:\n")

    for image_path in image_paths:
        result = infer_image(image_path, model, preprocess, text_features, device)

        print(f"Image: {result['image']}")
        for prompt, score in zip(PROMPTS, result["scores"]):
            print(f"  {prompt}: {score:.4f}")
        print(f"  Final scene label: {result['best_label']}")
        print()


if __name__ == "__main__":
    main()