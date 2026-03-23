from django import forms


class PredictionForm(forms.Form):
    """Form with all 29 features for lung cancer risk prediction."""

    # ── Demographics ─────────────────────────────────────────
    age = forms.IntegerField(
        label='Age', min_value=18, max_value=100, initial=50,
        widget=forms.NumberInput(attrs={'class': 'form-input', 'id': 'age'})
    )
    gender = forms.ChoiceField(
        label='Gender', choices=[(0, 'Female'), (1, 'Male')],
        widget=forms.Select(attrs={'class': 'form-input', 'id': 'gender'})
    )
    education_years = forms.IntegerField(
        label='Education (years)', min_value=0, max_value=25, initial=12,
        widget=forms.NumberInput(attrs={'class': 'form-input', 'id': 'education_years'})
    )
    income_level = forms.ChoiceField(
        label='Income Level', choices=[(i, str(i)) for i in range(1, 6)],
        initial=3,
        widget=forms.Select(attrs={'class': 'form-input', 'id': 'income_level'})
    )

    # ── Smoking & Environmental ──────────────────────────────
    smoker = forms.ChoiceField(
        label='Smoker', choices=[(0, 'No'), (1, 'Yes')],
        widget=forms.Select(attrs={'class': 'form-input', 'id': 'smoker'})
    )
    smoking_years = forms.IntegerField(
        label='Smoking Years', min_value=0, max_value=60, initial=0,
        widget=forms.NumberInput(attrs={'class': 'form-input', 'id': 'smoking_years'})
    )
    cigarettes_per_day = forms.IntegerField(
        label='Cigarettes Per Day', min_value=0, max_value=60, initial=0,
        widget=forms.NumberInput(attrs={'class': 'form-input', 'id': 'cigarettes_per_day'})
    )
    pack_years = forms.IntegerField(
        label='Pack Years', min_value=0, max_value=100, initial=0,
        widget=forms.NumberInput(attrs={'class': 'form-input', 'id': 'pack_years'})
    )
    passive_smoking = forms.ChoiceField(
        label='Passive Smoking', choices=[(0, 'No'), (1, 'Yes')],
        widget=forms.Select(attrs={'class': 'form-input', 'id': 'passive_smoking'})
    )
    air_pollution_index = forms.IntegerField(
        label='Air Pollution Index', min_value=0, max_value=200, initial=50,
        widget=forms.NumberInput(attrs={'class': 'form-input', 'id': 'air_pollution_index'})
    )
    occupational_exposure = forms.ChoiceField(
        label='Occupational Exposure', choices=[(0, 'No'), (1, 'Yes')],
        widget=forms.Select(attrs={'class': 'form-input', 'id': 'occupational_exposure'})
    )
    radon_exposure = forms.ChoiceField(
        label='Radon Exposure', choices=[(0, 'No'), (1, 'Yes')],
        widget=forms.Select(attrs={'class': 'form-input', 'id': 'radon_exposure'})
    )

    # ── Medical History ──────────────────────────────────────
    family_history_cancer = forms.ChoiceField(
        label='Family History of Cancer', choices=[(0, 'No'), (1, 'Yes')],
        widget=forms.Select(attrs={'class': 'form-input', 'id': 'family_history_cancer'})
    )
    copd = forms.ChoiceField(
        label='COPD', choices=[(0, 'No'), (1, 'Yes')],
        widget=forms.Select(attrs={'class': 'form-input', 'id': 'copd'})
    )
    asthma = forms.ChoiceField(
        label='Asthma', choices=[(0, 'No'), (1, 'Yes')],
        widget=forms.Select(attrs={'class': 'form-input', 'id': 'asthma'})
    )
    previous_tb = forms.ChoiceField(
        label='Previous TB', choices=[(0, 'No'), (1, 'Yes')],
        widget=forms.Select(attrs={'class': 'form-input', 'id': 'previous_tb'})
    )

    # ── Symptoms ─────────────────────────────────────────────
    chronic_cough = forms.ChoiceField(
        label='Chronic Cough', choices=[(0, 'No'), (1, 'Yes')],
        widget=forms.Select(attrs={'class': 'form-input', 'id': 'chronic_cough'})
    )
    chest_pain = forms.ChoiceField(
        label='Chest Pain', choices=[(0, 'No'), (1, 'Yes')],
        widget=forms.Select(attrs={'class': 'form-input', 'id': 'chest_pain'})
    )
    shortness_of_breath = forms.ChoiceField(
        label='Shortness of Breath', choices=[(0, 'No'), (1, 'Yes')],
        widget=forms.Select(attrs={'class': 'form-input', 'id': 'shortness_of_breath'})
    )
    fatigue = forms.ChoiceField(
        label='Fatigue', choices=[(0, 'No'), (1, 'Yes')],
        widget=forms.Select(attrs={'class': 'form-input', 'id': 'fatigue'})
    )

    # ── Health Metrics ───────────────────────────────────────
    bmi = forms.FloatField(
        label='BMI', min_value=10, max_value=50, initial=24,
        widget=forms.NumberInput(attrs={'class': 'form-input', 'id': 'bmi', 'step': '0.1'})
    )
    oxygen_saturation = forms.IntegerField(
        label='Oxygen Saturation (%)', min_value=70, max_value=100, initial=97,
        widget=forms.NumberInput(attrs={'class': 'form-input', 'id': 'oxygen_saturation'})
    )
    fev1_x10 = forms.IntegerField(
        label='FEV1 (×10)', min_value=0, max_value=50, initial=33,
        widget=forms.NumberInput(attrs={'class': 'form-input', 'id': 'fev1_x10'})
    )
    crp_level = forms.FloatField(
        label='CRP Level', min_value=0, max_value=50, initial=3,
        widget=forms.NumberInput(attrs={'class': 'form-input', 'id': 'crp_level', 'step': '0.1'})
    )
    xray_abnormal = forms.ChoiceField(
        label='X-Ray Abnormal', choices=[(0, 'No'), (1, 'Yes')],
        widget=forms.Select(attrs={'class': 'form-input', 'id': 'xray_abnormal'})
    )

    # ── Lifestyle ────────────────────────────────────────────
    exercise_hours_per_week = forms.IntegerField(
        label='Exercise (hours/week)', min_value=0, max_value=20, initial=3,
        widget=forms.NumberInput(attrs={'class': 'form-input', 'id': 'exercise_hours_per_week'})
    )
    diet_quality = forms.ChoiceField(
        label='Diet Quality', choices=[(i, str(i)) for i in range(1, 6)],
        initial=3,
        widget=forms.Select(attrs={'class': 'form-input', 'id': 'diet_quality'})
    )
    alcohol_units_per_week = forms.IntegerField(
        label='Alcohol (units/week)', min_value=0, max_value=50, initial=5,
        widget=forms.NumberInput(attrs={'class': 'form-input', 'id': 'alcohol_units_per_week'})
    )
    healthcare_access = forms.ChoiceField(
        label='Healthcare Access', choices=[(i, str(i)) for i in range(1, 6)],
        initial=3,
        widget=forms.Select(attrs={'class': 'form-input', 'id': 'healthcare_access'})
    )
