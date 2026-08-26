from pathlib import Path
from PIL import Image, ImageEnhance, ImageOps
import numpy as np

IMAGE_SIZE = (224, 224)


def load_rgb_224(path: str | Path) -> np.ndarray:
    """Load an image as normalized 224x224 RGB float32 array."""
    with Image.open(path) as img:
        img = img.convert("RGB").resize(IMAGE_SIZE)
        arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr


def augment_image(img: Image.Image, flip: bool = True, brightness: float = 1.0) -> Image.Image:
    img = img.convert("RGB").resize(IMAGE_SIZE)
    if flip:
        img = ImageOps.mirror(img)
    if brightness != 1.0:
        img = ImageEnhance.Brightness(img).enhance(brightness)
    return img


def flatten_image(path: str | Path) -> np.ndarray:
    return load_rgb_224(path).reshape(-1)
