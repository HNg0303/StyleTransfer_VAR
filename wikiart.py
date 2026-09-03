from datasets import load_dataset
from collections import defaultdict
from pathlib import Path

SAVE_DIR = Path(__file__).parent / "data" / "wikiart"
N_PER_STYLE = 10

ds = load_dataset(
    "huggan/wikiart",
    split="train",
    streaming=True,
)

counts = defaultdict(int)

# Get readable style names
style_names = ds.features["style"].names

for sample in ds:
    style_id = sample["style"]

    if counts[style_id] >= N_PER_STYLE:
        continue

    style_name = style_names[style_id]

    # Make Windows-safe folder name
    safe_name = style_name.replace("/", "_").replace("\\", "_")

    out_dir = SAVE_DIR / safe_name
    out_dir.mkdir(parents=True, exist_ok=True)

    image = sample["image"].convert("RGB")

    image.save(
        out_dir / f"{counts[style_id]:03d}.jpg"
    )

    counts[style_id] += 1

    # Stop once every style has enough images
    if (
        len(counts) == len(style_names)
        and all(counts[i] >= N_PER_STYLE for i in range(len(style_names)))
    ):
        break

print("Done")
print("Styles:", len(counts))
print("Images:", sum(counts.values()))