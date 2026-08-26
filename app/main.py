import io
import logging
import time
from pathlib import Path
import joblib
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("cats-dogs-api")

MODEL_PATH = Path("model/model.joblib")
model = joblib.load(MODEL_PATH)
app = FastAPI(title="Cats vs Dogs MLOps API", version="1.0.0")
metrics = {"request_count": 0, "total_latency_ms": 0.0, "prediction_count": 0}


def image_bytes_to_features(content: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(content)) as img:
        img = img.convert("RGB").resize((224,224))
        x = np.asarray(img, dtype=np.float32) / 255.0
    return x.reshape(1, -1)


def predict_bytes(content: bytes):
    x = image_bytes_to_features(content)
    p_dog = float(model.predict_proba(x)[0,1])
    p_cat = 1.0 - p_dog
    label = "dog" if p_dog >= 0.5 else "cat"
    return {"label": label, "probabilities": {"cat": round(p_cat, 6), "dog": round(p_dog, 6)}}

@app.middleware("http")
async def monitor(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    latency = (time.perf_counter() - start) * 1000
    metrics["request_count"] += 1
    metrics["total_latency_ms"] += latency
    logger.info("method=%s path=%s status=%s latency_ms=%.2f", request.method, request.url.path, response.status_code, latency)
    return response

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL_PATH.exists()}

@app.get("/metrics")
def get_metrics():
    count = max(metrics["request_count"], 1)
    return {
        **metrics,
        "average_latency_ms": round(metrics["total_latency_ms"] / count, 3)
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload an image file")
    content = await file.read()
    try:
        result = predict_bytes(content)
        metrics["prediction_count"] += 1
        logger.info("prediction label=%s cat_prob=%s dog_prob=%s", result["label"], result["probabilities"]["cat"], result["probabilities"]["dog"])
        return result
    except Exception as e:
        logger.exception("prediction_failed")
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")
