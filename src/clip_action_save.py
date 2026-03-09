from pathlib import Path
import csv

from config import ACTION_MAP, DEFAULT_ACTION
from clip_core import load_clip_model, infer_image, get_image_paths, get_device

CONF_THRESHOLD = 0.6


def decide_action(label: str, score: float) -> str:
    if score < CONF_THRESHOLD:
        return "stop"
    return ACTION_MAP.get(label, DEFAULT_ACTION)


def get_confidence_status(score: float) -> str:
    if score < CONF_THRESHOLD:
        return "low_confidence"
    return "confident"


def main() -> None:
    print(f"Using device: {get_device()}")

    model, preprocess, text_features, device = load_clip_model()
    image_paths = get_image_paths()

    if not image_paths:
        raise FileNotFoundError("No images found in data/images")

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    output_csv = results_dir / "clip_action_results_v4.csv"

    rows = []

    print("\nCLIP + Action Decision Results (v4):\n")

    for image_path in image_paths:
        result = infer_image(image_path, model, preprocess, text_features, device)

        best_label = result["best_label"]
        best_score = result["best_score"]
        confidence_status = get_confidence_status(best_score)
        action = decide_action(best_label, best_score)

        print(f"Image: {result['image']}")
        print(f"  Final scene label: {best_label}")
        print(f"  Best score: {best_score:.4f}")
        print(f"  Confidence status: {confidence_status}")
        print(f"  Action: {action}")

        if best_score < CONF_THRESHOLD:
            print(f"  Warning: low confidence (< {CONF_THRESHOLD})")
        print()

        rows.append({
            "image": result["image"],
            "scene_label": best_label,
            "best_score": round(best_score, 4),
            "confidence_status": confidence_status,
            "action": action,
        })

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["image", "scene_label", "best_score", "confidence_status", "action"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved results to: {output_csv}")


if __name__ == "__main__":
    main()