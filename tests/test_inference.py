import io
from PIL import Image
from app.main import image_bytes_to_features, predict_bytes

def fake_image_bytes():
    b = io.BytesIO(); Image.new("RGB", (30,30), (120,120,120)).save(b, format="JPEG"); return b.getvalue()

def test_feature_shape():
    x = image_bytes_to_features(fake_image_bytes())
    assert x.shape == (1, 224*224*3)

def test_prediction_contract():
    out = predict_bytes(fake_image_bytes())
    assert out["label"] in {"cat", "dog"}
    assert set(out["probabilities"]) == {"cat", "dog"}
    assert abs(sum(out["probabilities"].values()) - 1.0) < 1e-5
