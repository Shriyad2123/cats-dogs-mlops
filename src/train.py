import argparse
import json
from pathlib import Path
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix, classification_report
from src.preprocess import flatten_image, load_rgb_224, augment_image


def list_split(root: Path, split: str):
    pairs = []
    for label, y in [("cat", 0), ("dog", 1)]:
        for p in sorted((root / split / label).glob("*.jpg")):
            pairs.append((p, y))
    return pairs


def batches(pairs, batch_size, augment=False, rng=None):
    from PIL import Image
    rng = rng or np.random.default_rng(42)
    for i in range(0, len(pairs), batch_size):
        chunk = pairs[i:i+batch_size]
        feats = []
        for p, _ in chunk:
            if augment:
                with Image.open(p) as img:
                    flip = bool(rng.integers(0, 2))
                    brightness = float(rng.uniform(0.85, 1.15))
                    aug = augment_image(img, flip=flip, brightness=brightness)
                    arr = np.asarray(aug, dtype=np.float32) / 255.0
                feats.append(arr.reshape(-1))
            else:
                feats.append(flatten_image(p))
        X = np.stack(feats).astype(np.float32)
        y = np.asarray([y for _, y in chunk], dtype=np.int64)
        yield X, y


def evaluate(model, pairs, batch_size=32):
    ys, probs = [], []
    for X, y in batches(pairs, batch_size):
        p = model.predict_proba(X)[:, 1]
        ys.extend(y.tolist()); probs.extend(p.tolist())
    y = np.asarray(ys); p = np.asarray(probs)
    pred = (p >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "log_loss": float(log_loss(y, np.column_stack([1-p, p]), labels=[0,1])),
        "y": y, "pred": pred, "p": p,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--alpha", type=float, default=0.0001)
    args = ap.parse_args()

    root = Path(args.data)
    train, val, test = [list_split(root, s) for s in ["train", "val", "test"]]
    if not train or not val or not test:
        raise RuntimeError("Processed train/val/test data not found. Run prepare_data.py first.")

    Path("model").mkdir(exist_ok=True)
    Path("reports/figures").mkdir(parents=True, exist_ok=True)
    mlflow.set_experiment("cats-vs-dogs-baseline")

    model = SGDClassifier(loss="log_loss", alpha=args.alpha, random_state=42)
    epoch_train_loss, epoch_val_loss = [], []

    with mlflow.start_run() as run:
        mlflow.log_params({
            "model": "SGDClassifier-logistic",
            "input_size": "224x224x3",
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "alpha": args.alpha,
        })

        for epoch in range(args.epochs):
            rng = np.random.default_rng(42 + epoch)
            idx = rng.permutation(len(train))
            shuffled = [train[i] for i in idx]
            for X, y in batches(shuffled, args.batch_size, augment=True, rng=rng):
                if epoch == 0 and not hasattr(model, "classes_"):
                    model.partial_fit(X, y, classes=np.array([0, 1]))
                else:
                    model.partial_fit(X, y)
            tr = evaluate(model, train[: min(len(train), 500)], args.batch_size)
            va = evaluate(model, val, args.batch_size)
            epoch_train_loss.append(tr["log_loss"])
            epoch_val_loss.append(va["log_loss"])
            mlflow.log_metrics({"train_loss": tr["log_loss"], "val_loss": va["log_loss"], "val_accuracy": va["accuracy"]}, step=epoch)
            print(f"epoch={epoch+1} val_accuracy={va['accuracy']:.4f} val_loss={va['log_loss']:.4f}")

        te = evaluate(model, test, args.batch_size)
        mlflow.log_metrics({"test_accuracy": te["accuracy"], "test_log_loss": te["log_loss"]})

        cm = confusion_matrix(te["y"], te["pred"])
        fig, ax = plt.subplots(figsize=(4,4))
        ax.imshow(cm)
        ax.set_xticks([0,1], labels=["cat", "dog"]); ax.set_yticks([0,1], labels=["cat", "dog"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title("Confusion Matrix")
        for i in range(2):
            for j in range(2): ax.text(j, i, str(cm[i,j]), ha="center", va="center")
        fig.tight_layout(); fig.savefig("reports/figures/confusion_matrix.png"); plt.close(fig)

        fig, ax = plt.subplots(figsize=(6,4))
        ax.plot(range(1,args.epochs+1), epoch_train_loss, label="train")
        ax.plot(range(1,args.epochs+1), epoch_val_loss, label="validation")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Log loss"); ax.set_title("Loss Curve"); ax.legend()
        fig.tight_layout(); fig.savefig("reports/figures/loss_curve.png"); plt.close(fig)

        joblib.dump(model, "model/model.joblib", compress=3)
        report = classification_report(te["y"], te["pred"], target_names=["cat","dog"], output_dict=True)
        Path("reports/test_metrics.json").write_text(json.dumps({"accuracy": te["accuracy"], "log_loss": te["log_loss"], "classification_report": report}, indent=2))
        mlflow.log_artifact("reports/figures/confusion_matrix.png")
        mlflow.log_artifact("reports/figures/loss_curve.png")
        mlflow.log_artifact("reports/test_metrics.json")
        mlflow.sklearn.log_model(model, artifact_path="model")
        print("MLflow run:", run.info.run_id)
        print("Saved model/model.joblib")

if __name__ == "__main__":
    main()
