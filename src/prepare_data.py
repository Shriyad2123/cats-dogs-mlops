import argparse
import random
import shutil
from pathlib import Path
from PIL import Image

VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def infer_label(path: Path):
    parts = [p.lower() for p in path.parts]
    name = path.name.lower()
    if any("cat" in p for p in parts) or name.startswith("cat"):
        return "cat"
    if any("dog" in p for p in parts) or name.startswith("dog"):
        return "dog"
    return None


def collect(source: Path):
    rows = []
    for p in source.rglob("*"):
        if p.is_file() and p.suffix.lower() in VALID_EXT:
            label = infer_label(p)
            if label:
                rows.append((p, label))
    return rows


def save_resized(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as img:
        img = img.convert("RGB").resize((224, 224))
        img.save(dst, format="JPEG", quality=90)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--output", default="data/processed")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-images", type=int, default=4000,
                    help="Cap for laptop/Colab-friendly baseline. Set 0 for all images.")
    args = ap.parse_args()

    source, output = Path(args.source), Path(args.output)
    rows = collect(source)
    if not rows:
        raise RuntimeError(f"No labelled cat/dog images found under {source}")

    random.Random(args.seed).shuffle(rows)
    if args.max_images and len(rows) > args.max_images:
        # keep classes approximately balanced
        cats = [r for r in rows if r[1] == "cat"][: args.max_images // 2]
        dogs = [r for r in rows if r[1] == "dog"][: args.max_images // 2]
        rows = cats + dogs
        random.Random(args.seed).shuffle(rows)

    if output.exists():
        shutil.rmtree(output)

    n = len(rows)
    n_train, n_val = int(n * 0.8), int(n * 0.1)
    splits = {
        "train": rows[:n_train],
        "val": rows[n_train:n_train+n_val],
        "test": rows[n_train+n_val:],
    }
    for split, items in splits.items():
        for i, (src, label) in enumerate(items):
            save_resized(src, output / split / label / f"{label}_{i:05d}.jpg")
        print(split, len(items))

if __name__ == "__main__":
    main()
