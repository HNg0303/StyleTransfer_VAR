from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

# Use the builder that returns GramVAR with optimization-based guidance.
from models import build_style_vae_var as build_vae_var
from models.cnn import VGG16Encoder

import argparse
import random


root = Path(__file__).resolve().parent
model_dir = root / "model_dir"


def load_image(path, preprocess: transforms.Compose, device="cpu"):
    with Image.open(path) as image:
        return preprocess(image.convert("RGB")).unsqueeze(0).to(device)


def main():
    parser = argparse.ArgumentParser(description="Run GramVAR inference with different cutoff indices.")
    parser.add_argument("--opti_steps", type=int, default=10, help="Number of optimization steps for inference.")
    parser.add_argument("--start_idx", type=int, default=0, help="Starting index for inference.")
    parser.add_argument("--cutoff_idx_list", type=int, default=3, help="List of cutoff indices to run inference on.")
    parser.add_argument("--content", action="store_true", help="Whether to use content features in inference.")
    args = parser.parse_args()  

    content_path = root / "data\\imagenet-1k\\8_2.png"
    style_path = root / "data\\wikiart\\Impressionism\\000.jpg"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    opti_steps = args.opti_steps
    start_idx = args.start_idx
    cutoff_idx_list = args.cutoff_idx_list

    # Load Model:
    vae, var = build_vae_var(device=device, depth=16)
    vae.load_state_dict(torch.load(model_dir / "vae_ch160v4096z32.pth", map_location="cpu"))
    var.load_state_dict(torch.load(model_dir / "var_d16.pth", map_location="cpu"))
    vae.eval().requires_grad_(False)
    var.eval().requires_grad_(False)
    vgg = VGG16Encoder(input_range=(0, 1)).to(device).eval()

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
    ])

    content, style = load_image(content_path, preprocess), load_image(style_path, preprocess)
    with torch.no_grad():
        content_features = vgg(content) if args.content else None
        style_features = vgg(style)
    label_B = -1
    if not args.content:
        print("Content features are not used in inference. Set --content to use them.")
        label_B = 8 # Random label for style transfer without content guidance from ImageNet-1k. This is a placeholder and can be adjusted based on specific needs.
    for cutoff_idx in range(start_idx, cutoff_idx_list + 1):
        print(f"Running inference with cutoff_idx={cutoff_idx}...")
        result = var.autoregressive_infer_cfg(
            B=1, label_B=label_B, cfg=1.5, g_seed=42, top_k=900, top_p=0.95,
            vgg_encoder=vgg,
            content_features=content_features,
            style_features=style_features,
            content_layer="relu4_3", content_alpha=0.5,
            style_alpha={"relu1_2": 1.5, "relu3_3": 1.3, "relu5_3": 1.2},
            opti_steps=opti_steps, opti_lr=0.1,
            start_idx=start_idx, cutoff_idx=cutoff_idx,
        )

        if args.content:
            output_path = root / f"results/gramvar_inference_label{label_B}_optisteps{opti_steps}_start{start_idx}_cutoff{cutoff_idx}_content.png"
        else:
            output_path = root / f"results/gramvar_inference_label{label_B}_optisteps{opti_steps}_start{start_idx}_cutoff{cutoff_idx}_nocontent.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_image(torch.cat([content, style, result]), output_path, nrow=3)
        print(f"Saved content | style | result to {output_path}")

if __name__ == "__main__":
    main()