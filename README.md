# PredictAI - Student Early Warning System

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Machine Learning](https://img.shields.io/badge/Machine-Learning-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)
![Status](https://img.shields.io/badge/Status-MVP-green)

## Overview

PredictAI is an educational early warning prototype developed
by Team Medad AI as part of SHIRAA 2026.

It explores how machine learning and transparent academic
indicators can help educational counselors identify students
who may benefit from additional academic support.

## Team

- Maryam Ghanem
- Noor Sabra

## Features

- Individual student risk analysis
- Machine learning risk estimation
- Explainable AI using Logistic Regression coefficients
- Rule-based educational recommendations
- Class-level CSV analysis
- Input data validation
- Interactive counselor dashboard
- Baseline and AI ranking comparison
- Adjustable follow-up capacity
- Downloadable follow-up reports
- Model performance visualization

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Joblib
- Git and GitHub

## Dataset

UCI Student Performance Dataset - Mathematics.

- 395 student records
- 33 original variables
- 316 training records
- 79 testing records

The target is derived from the final mathematics grade G3.
G3 and G2 are excluded from the model inputs.

Source:
https://archive.ics.uci.edu/dataset/320/student+performance

Dataset credit: Paulo Cortez.
DOI: https://doi.org/10.24432/C5TG7T
License: CC BY 4.0

## Machine Learning Workflow

1. Exploratory Data Analysis
2. Data Preprocessing
3. Feature Selection
4. Train/Test Split
5. Random Forest Training
6. Logistic Regression
7. Cross-Validation
8. Hyperparameter Tuning
9. Permutation Feature Importance
10. Risk Ranking and Model Integration

## Experimental Results

| Model | Mean CV F1 |
|---|---:|
| G1 Baseline | 77.28% |
| Reduced Logistic Regression | 76.51% |
| Original Logistic Regression | 75.56% |
| Tuned Random Forest | 74.07% |
| Original Random Forest | 72.24% |

### Risk Ranking Comparison

Ten students were selected per validation fold,
with 50 selections across five folds.

| Method | Actual failing students identified |
|---|---:|
| G1 Baseline | 44/50 |
| Original Logistic Regression | 43/50 |
| Reduced Logistic Regression | 42/50 |

The traditional baseline outperformed the ML models
in these exploratory experiments.

PredictAI therefore uses the G1 baseline as its default
ranking method and presents ML risk scores as experimental
estimates.

## Installation

Clone the repository:

    git clone https://github.com/Maryamghanem64/ai-failure-predictor.git
    cd ai-failure-predictor

Create a virtual environment:

    python -m venv .venv

Activate it on Windows PowerShell:

    .\.venv\Scripts\Activate.ps1

Install dependencies:

    pip install -r requirements.txt

## Run the Application

    streamlit run app.py

Open:

    http://localhost:8501

## Demo Data

Use data/demo/students.csv to test class-level analysis.

The demo records are synthetic.

## Retraining

Place the official UCI student-mat.csv dataset inside data/raw
if it is not already present.

    python eda.py

The trained model is saved in models/predictai_model.joblib.

Only load trusted Joblib model files.

## Ethics and Limitations

- This application is a research and hackathon prototype.
- Risk scores are not confirmed student outcomes.
- Model explanations do not establish causation.
- Recommendations require human review.
- The dataset is small and context-specific.
- Probabilities have not been validated for calibration.
- Model selection results are exploratory.
- The timing of questionnaire features requires verification.
- Real-world educational effectiveness has not been established.
- Use synthetic or appropriately de-identified data.

## Project Status

Functional MVP developed for SHIRAA 2026.

Further validation and testing are required before
real-world educational deployment.

## Acknowledgments

SHIRAA 2026
Educational Mastar Central
UCI Machine Learning Repository

Developed by Team Medad AI.
