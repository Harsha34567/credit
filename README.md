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
3. Run the starter script:
   ```powershell
   python main.py
   ```

## Goals

- Build a scalable, production-ready credit scoring pipeline.
- Use best practices for modularity, reproducibility, and maintainability.
- Support model explainability, deployment, and portfolio-ready documentation.
