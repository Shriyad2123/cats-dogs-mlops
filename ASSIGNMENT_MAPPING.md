# Assignment-to-file mapping
| Rubric | Evidence in repository |
|---|---|
| M1 Git | GitHub repository |
| M1 DVC | `dvc.yaml`, `.dvc/`/`data/processed.dvc` after execution |
| M1 Model | `src/train.py`, `model/model.joblib` |
| M1 MLflow | `src/train.py`, `mlruns/`, figures + metrics |
| M2 REST API | `app/main.py` |
| M2 pinned environment | `requirements.txt` |
| M2 Docker | `Dockerfile` |
| M3 tests | `tests/` |
| M3 CI | `.github/workflows/ci-cd.yml` |
| M3 registry | GHCR push in workflow |
| M4 target | Render image-backed web service |
| M4 CD | Render deploy hook from GitHub Actions |
| M4 smoke test | `scripts/smoke_test.py` + workflow |
| M5 logs/metrics | FastAPI middleware + `/metrics` |
| M5 post-deploy performance | `scripts/post_deploy_eval.py` |
