from config import ACTION_MAP, DEFAULT_ACTION
from clip_core import load_clip_model, infer_image, get_image_paths, get_device

CONF_THRESHOLD = 0.6


def decide_action(label: str, score: float) -> str:
    if score < CONF_THRESHOLD:
        return "stop"
    return ACTION_MAP.get(label, DEFAULT_ACTION)


def main() -> None:
    print(f"Using device: {get_device()}")

    model, preprocess, text_features, device = load_clip_model()
    image_paths = get_image_paths()

    if not image_paths:
        raise FileNotFoundError("No images found in data/images")

    print("\nCLIP + Action Decision Results:\n")

    for image_path in image_paths:
        result = infer_image(image_path, model, preprocess, text_features, device)

        best_label = result["best_label"]
        best_score = result["best_score"]
        action = decide_action(best_label, best_score)

        print(f"Image: {result['image']}")
        print(f"  Final scene label: {best_label}")
        print(f"  Best score: {best_score:.4f}")
        print(f"  Action: {action}")

        if best_score < CONF_THRESHOLD:
            print(f"  Warning: low confidence (< {CONF_THRESHOLD})")
        print()


if __name__ == "__main__":
    main()