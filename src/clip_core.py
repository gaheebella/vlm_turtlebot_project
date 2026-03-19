import torch
import open_clip


def load_clip_model(model_name="ViT-B-32", pretrained="openai"):
    """
    returns:
        model, preprocess, tokenizer, device
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained
    )
    model = model.to(device)
    model.eval()

    tokenizer = open_clip.get_tokenizer(model_name)

    return model, preprocess, tokenizer, device