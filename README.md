# Cats vs Dogs - End-to-End MLOps Assignment

This repository implements the complete MLOps flow required by Assignment 2: Git + DVC, baseline image classifier with training-time horizontal-flip/brightness augmentation, MLflow tracking, FastAPI inference, Docker packaging, GitHub Actions CI/CD, GHCR artifact publishing, Render image-based deployment, smoke tests, request/latency logging, and post-deployment performance tracking.

## Architecture
`Kaggle -> preprocess 224x224 RGB -> DVC -> SGD logistic baseline -> MLflow -> model.joblib -> FastAPI -> Docker -> GitHub Actions -> GHCR -> Render -> smoke tests + metrics`

## Dataset
Assignment dataset: `bhavikjikadara/dog-and-cat-classification-dataset` on Kaggle.

## No local Docker required
Training can be done in Google Colab. Docker is built entirely on GitHub Actions and deployed as a prebuilt GHCR image to Render.

## Quick start
See `STEP_BY_STEP.md`. Do not submit the included bootstrap model as your final trained model. Run the Colab/training step and replace it first.
