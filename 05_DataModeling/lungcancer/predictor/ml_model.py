"""
Lung Cancer Risk Prediction Model

Trains a Logistic Regression model on the lung_cancer_chance.csv dataset.
The model is trained once when this module is first imported, and then
the predict() function can be called to get predictions.
"""
import os
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

warnings.filterwarnings('ignore')

# Path to the CSV file (one level up from the Django project)
CSV_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'lung_cancer_chance.csv'
)

# Feature columns in order
FEATURE_COLUMNS = [
    'age', 'gender', 'education_years', 'income_level',
    'smoker', 'smoking_years', 'cigarettes_per_day', 'pack_years',
    'passive_smoking', 'air_pollution_index', 'occupational_exposure',
    'radon_exposure', 'family_history_cancer',
    'copd', 'asthma', 'previous_tb',
    'chronic_cough', 'chest_pain', 'shortness_of_breath', 'fatigue',
    'bmi', 'oxygen_saturation', 'fev1_x10', 'crp_level',
    'xray_abnormal', 'exercise_hours_per_week', 'diet_quality',
    'alcohol_units_per_week', 'healthcare_access',
]

# Human-readable labels for each feature
FEATURE_LABELS = {
    'age': 'Age',
    'gender': 'Gender (0=Female, 1=Male)',
    'education_years': 'Education (years)',
    'income_level': 'Income Level (1-5)',
    'smoker': 'Smoker (0=No, 1=Yes)',
    'smoking_years': 'Smoking Years',
    'cigarettes_per_day': 'Cigarettes Per Day',
    'pack_years': 'Pack Years',
    'passive_smoking': 'Passive Smoking (0=No, 1=Yes)',
    'air_pollution_index': 'Air Pollution Index',
    'occupational_exposure': 'Occupational Exposure (0=No, 1=Yes)',
    'radon_exposure': 'Radon Exposure (0=No, 1=Yes)',
    'family_history_cancer': 'Family History of Cancer (0=No, 1=Yes)',
    'copd': 'COPD (0=No, 1=Yes)',
    'asthma': 'Asthma (0=No, 1=Yes)',
    'previous_tb': 'Previous TB (0=No, 1=Yes)',
    'chronic_cough': 'Chronic Cough (0=No, 1=Yes)',
    'chest_pain': 'Chest Pain (0=No, 1=Yes)',
    'shortness_of_breath': 'Shortness of Breath (0=No, 1=Yes)',
    'fatigue': 'Fatigue (0=No, 1=Yes)',
    'bmi': 'BMI',
    'oxygen_saturation': 'Oxygen Saturation (%)',
    'fev1_x10': 'FEV1 (×10)',
    'crp_level': 'CRP Level',
    'xray_abnormal': 'X-Ray Abnormal (0=No, 1=Yes)',
    'exercise_hours_per_week': 'Exercise (hours/week)',
    'diet_quality': 'Diet Quality (1-5)',
    'alcohol_units_per_week': 'Alcohol (units/week)',
    'healthcare_access': 'Healthcare Access (1-5)',
}

# Feature groups for form organization
FEATURE_GROUPS = {
    'Demographics': ['age', 'gender', 'education_years', 'income_level'],
    'Smoking & Exposure': [
        'smoker', 'smoking_years', 'cigarettes_per_day', 'pack_years',
        'passive_smoking', 'air_pollution_index', 'occupational_exposure',
        'radon_exposure',
    ],
    'Medical History': [
        'family_history_cancer', 'copd', 'asthma', 'previous_tb',
    ],
    'Symptoms': [
        'chronic_cough', 'chest_pain', 'shortness_of_breath', 'fatigue',
    ],
    'Health Metrics': [
        'bmi', 'oxygen_saturation', 'fev1_x10', 'crp_level', 'xray_abnormal',
    ],
    'Lifestyle': [
        'exercise_hours_per_week', 'diet_quality',
        'alcohol_units_per_week', 'healthcare_access',
    ],
}


# ── Train the model once at import ──────────────────────────────

_model = None
_accuracy = None


def _train_model():
    global _model, _accuracy
    df = pd.read_csv(CSV_PATH)
    X = df.drop('lung_cancer_risk', axis=1)
    y = df['lung_cancer_risk']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=99
    )

    _model = LogisticRegression(max_iter=1000)
    _model.fit(X_train, y_train)

    y_pred = _model.predict(X_test)
    _accuracy = accuracy_score(y_test, y_pred) * 100


_train_model()


# ── Public API ──────────────────────────────────────────────────

def predict(input_dict: dict) -> dict:
    """
    Predict lung cancer risk.

    Args:
        input_dict: dict mapping feature names to numeric values

    Returns:
        dict with keys: prediction (0/1), probability (float 0-100),
        risk_level (str), accuracy (float)
    """
    values = [float(input_dict.get(col, 0)) for col in FEATURE_COLUMNS]
    X_input = np.array(values).reshape(1, -1)

    prediction = int(_model.predict(X_input)[0])
    probabilities = _model.predict_proba(X_input)[0]
    risk_probability = probabilities[1] * 100  # probability of class 1 (risk)

    # Determine top risk factors based on coefficient magnitude × input value
    coefficients = _model.coef_[0]
    contributions = []
    for i, col in enumerate(FEATURE_COLUMNS):
        contribution = coefficients[i] * values[i]
        contributions.append((col, FEATURE_LABELS[col], contribution, values[i]))

    # Sort by absolute contribution descending
    contributions.sort(key=lambda x: abs(x[2]), reverse=True)
    top_factors = contributions[:5]

    return {
        'prediction': prediction,
        'probability': round(risk_probability, 2),
        'risk_level': 'High Risk' if prediction == 1 else 'Low Risk',
        'accuracy': round(_accuracy, 2),
        'top_factors': [
            {
                'name': f[1],
                'contribution': round(f[2], 4),
                'value': f[3],
                'direction': 'increasing' if f[2] > 0 else 'decreasing',
            }
            for f in top_factors
        ],
    }
