from PIL import Image
from src.preprocess import load_rgb_224

def test_load_rgb_224(tmp_path):
    p = tmp_path / "x.png"
    Image.new("RGB", (50, 80), (10, 20, 30)).save(p)
    arr = load_rgb_224(p)
    assert arr.shape == (224, 224, 3)
    assert arr.dtype.name == "float32"
    assert 0.0 <= float(arr.min()) <= float(arr.max()) <= 1.0
