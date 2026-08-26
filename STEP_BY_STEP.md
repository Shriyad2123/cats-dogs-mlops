# Step-by-step execution guide

## Phase 1 - Create GitHub repository
1. Create a public GitHub repository named `cats-dogs-mlops`.
2. Upload all files from this project ZIP.
3. Commit to `main`.

## Phase 2 - Train without Docker (Google Colab)
1. Open Google Colab and start a new notebook.
2. Run:
```python
!git clone https://github.com/YOUR_USERNAME/cats-dogs-mlops.git
%cd cats-dogs-mlops
!pip install -q -r requirements-train.txt
import kagglehub
path = kagglehub.dataset_download("bhavikjikadara/dog-and-cat-classification-dataset")
print(path)
```
3. Preprocess the exact assignment dataset:
```python
!python src/prepare_data.py --source "$path" --output data/processed --max-images 4000
```
This creates 224x224 RGB images and an 80/10/10 train-validation-test split.
4. Train + track experiments:
```python
!python -m src.train --data data/processed --epochs 5 --batch-size 32
```
5. Inspect MLflow:
```python
!mlflow ui --host 0.0.0.0 --port 5000 &
```
For assignment evidence, screenshots of the run output, `reports/figures/loss_curve.png`, `reports/figures/confusion_matrix.png`, and the `mlruns` directory are sufficient. If you want the UI in Colab, use a tunnel or run MLflow locally later.
6. Download `model/model.joblib`, `reports/`, and optionally `mlruns/` from Colab, replace the bootstrap files in your GitHub repo, then commit and push.

## Phase 3 - DVC evidence
In Colab after preprocessing:
```bash
dvc init -f
dvc add data/processed
git add data/processed.dvc .dvc .gitignore
git commit -m "Track processed dataset with DVC"
```
If you do not push the full data to a remote, the `.dvc` metadata still demonstrates dataset versioning. Do not put raw Kaggle data in Git.

## Phase 4 - CI and image publishing (no Docker on your laptop)
1. Push the final trained `model/model.joblib` to GitHub.
2. Open GitHub -> Actions -> `CI-CD MLOps`.
3. The workflow automatically installs dependencies, runs pytest, builds Linux/amd64 Docker image, and pushes it to GitHub Container Registry (GHCR).
4. After the first successful image push, open your GitHub profile -> Packages -> `cats-dogs-mlops` -> Package settings -> Change visibility -> Public.

## Phase 5 - Deploy on Render from GHCR
1. Sign in to Render.
2. New -> Web Service -> Existing Image.
3. Image URL: `ghcr.io/YOUR_USERNAME/cats-dogs-mlops:latest`.
4. Choose the free instance if available to your account.
5. Start command is already inside the image. The app uses the `PORT` environment variable.
6. Deploy and copy the service URL, e.g. `https://cats-dogs-mlops.onrender.com`.
7. Render service -> Settings -> Deploy Hook -> copy the secret hook URL.

## Phase 6 - Connect CD back to GitHub
GitHub repo -> Settings -> Secrets and variables -> Actions -> New repository secret:
- `RENDER_DEPLOY_HOOK_URL` = the Render deploy hook
- `RENDER_SERVICE_URL` = your public Render service URL

Push one harmless change to `README.md`. GitHub Actions should now:
`test -> build -> push GHCR image -> call Render deploy hook -> smoke test /health -> smoke test /predict`.

## Phase 7 - Verify endpoints
Open:
- `/health`
- `/metrics`
- `/docs` for Swagger UI

Use Swagger `/docs` -> POST `/predict` -> upload a cat or dog image -> Execute.

## Phase 8 - Post-deployment model performance
From Colab:
```bash
python scripts/post_deploy_eval.py --url https://YOUR-SERVICE.onrender.com --test-dir data/processed/test
```
This writes `reports/post_deploy_metrics.json` from simulated post-deployment requests with known true labels.

## Phase 9 - Final ZIP
Before submitting, ensure ZIP contains source code, `.github/workflows/ci-cd.yml`, `dvc.yaml` or `data/processed.dvc`, Dockerfile, requirements files, app code, tests, scripts, trained `model/model.joblib`, report figures/metrics, and README.

## Phase 10 - <5 minute screen recording
Follow `RECORDING_SCRIPT.md`.
