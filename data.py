from pathlib import Path
from collections import defaultdict
from datasets import load_dataset


SAVE_DIR = Path(__file__).parent / "data" / "imagenet-1k"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# ds = load_dataset(
#     "ILSVRC/imagenet-1k",
#     split="validation",
#     cache_dir=SAVE_DIR,
#     streaming=True
# )

# counts = defaultdict(int) # Count dictionary that count 5 images for each label.
# data_json = []

# for sample in ds:
#     label = sample["label"]
#     if counts[label] < 5:
#         counts[label] += 1
#         img = sample["image"].convert("RGB") # sample["image"] is a PIL Image object, convert it to RGB mode.
#         img.save(SAVE_DIR / f"{label}_{counts[label]}.png")
#         data_json.append({"label": label, "image_path": SAVE_DIR / f"{label}_{counts[label]}.png"})

# data_json_path =  SAVE_DIR.parent / "imagenet_1k_data.json"
# with open(data_json_path, "w") as f:
#     import json
#     json.dump(data_json, f, indent=4)

def create_imagenet_1k_json(data_path: Path = SAVE_DIR, json_dir: Path = SAVE_DIR.parent / "imagenet_1k_data.json"):
    data_json = []
    for img in data_path.glob("*.png"):
        label = img.stem.split("_")[0]
        data_json.append({"label": label, "image_path": str(img)})

    with open(json_dir, "w") as f:
        import json
        json.dump(data_json, f, indent=4)

# print(f"Saved {len(data_json)} images and data.json to {SAVE_DIR} and {json_dir}")

if __name__ == "__main__":
    create_imagenet_1k_json()