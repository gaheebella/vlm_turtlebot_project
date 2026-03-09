from pathlib import Path

import torch
from PIL import Image
import open_clip

from config import PROMPTS


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_clip_model():
    device = get_device()

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32",
        pretrained="openai",
    )
    model = model.to(device)
    tokenizer = open_clip.get_tokenizer("ViT-B-32")

    text = tokenizer(PROMPTS).to(device)

    with torch.no_grad():
        text_features = model.encode_text(text)
        text_features /= text_features.norm(dim=-1, keepdim=True)

    return model, preprocess, text_features, device


def infer_image(image_path: Path, model, preprocess, text_features, device):
    image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    scores = similarity[0].cpu().tolist()
    best_idx = max(range(len(scores)), key=lambda i: scores[i])

    return {
        "image": image_path.name,
        "scores": scores,
        "best_label": PROMPTS[best_idx],
        "best_score": scores[best_idx],
    }


def get_image_paths(image_dir: str = "data/images"):
    path = Path(image_dir)
    image_paths = (
        list(path.glob("*.jpg"))
        + list(path.glob("*.jpeg"))
        + list(path.glob("*.png"))
    )
    return image_paths