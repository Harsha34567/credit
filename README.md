# Credit Scoring Model

A professional machine learning system to predict creditworthiness using financial history, debt, income, employment history, and payment behavior.

## Project Structure

- `data/`: raw, processed, and external datasets.
- `notebooks/`: exploratory data analysis and experimentation.
- `src/`: modular source code for preprocessing, feature engineering, training, evaluation, and utilities.
- `models/`: serialized model artifacts and versioned outputs.
- `reports/`: model summaries, evaluation reports, and documentation.
- `app/`: production-ready Streamlit application.
- `assets/`: visual assets, diagrams, and UI images.
- `tests/`: unit tests and validation checks.
- `logs/`: runtime logs and debugging output.

## Setup

1. Create and activate the virtual environment:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
2. Install dependencies:
   ```powershell
   python -m pip install -r requirements.txt
   ```
3. Install pre-commit hooks:
   ```powershell
   .\.venv\Scripts\python.exe -m pre_commit install
   ```
4. Run the starter script:
   ```powershell
   python main.py
   ```

## Dataset

The project expects Kaggle credentials stored in a `.env` file. Use `.env.example` as a template and never commit your real credentials.

```powershell
copy .env.example .env
# Then edit .env with your Kaggle username and API key.
```

The utility script in `src/utils/kaggle_utils.py` can download and unzip the dataset automatically.

## Training Pipeline

Use the training runner to download data, preprocess it, train candidate models, and persist the best model:

```powershell
python -m src.training.run_training --force-download
```

If you want to keep the raw data but skip writing the processed CSV, add:

```powershell
python -m src.training.run_training --skip-save-processed
```

## GitHub Remote

If you want to connect this repository to GitHub, authenticate using GitHub CLI and create a remote:

```powershell
"C:\Program Files\GitHub CLI\gh.exe" auth login --hostname github.com --web
"C:\Program Files\GitHub CLI\gh.exe" repo create credit-scoring-model --public --source . --remote origin --push
```

## Goals

- Build a scalable, production-ready credit scoring pipeline.
- Use best practices for modularity, reproducibility, and maintainability.
- Support model explainability, deployment, and portfolio-ready documentation.
