# <5 minute recording script

**0:00-0:35 - Repository and M1**
Show GitHub repo structure. Say: "Git versions the source, DVC versions processed data, the images are resized to 224x224 RGB and split 80/10/10. The baseline is a binary logistic classifier trained with SGD. MLflow logs parameters, validation/test metrics, confusion matrix, loss curve, and the model artifact."

**0:35-1:20 - Experiment evidence**
Show `reports/figures/loss_curve.png`, `confusion_matrix.png`, `test_metrics.json`, and MLflow run/artifact folder or UI.

**1:20-2:00 - API and containerization**
Show `app/main.py`, `/health`, `/predict`, `/metrics`, `requirements.txt`, and `Dockerfile`. Mention Docker is built in cloud CI for reproducibility.

**2:00-3:05 - CI/CD**
Show `.github/workflows/ci-cd.yml`, then GitHub Actions successful run. Expand pytest, Docker build/push, deploy-hook, and smoke-test steps. Show GHCR package.

**3:05-4:05 - Deployment**
Show Render service running from `ghcr.io/...:latest`. Open `/health` and `/metrics`.

**4:05-4:40 - Prediction**
Open `/docs`, call POST `/predict` with one cat/dog image, show returned label and probabilities.

**4:40-4:58 - Monitoring/post-deployment**
Refresh `/metrics`, show request count/latency changed. Show `reports/post_deploy_metrics.json`. End by saying the deployment is automatically updated on main-branch pushes and smoke tests fail the pipeline if the service is unhealthy.
