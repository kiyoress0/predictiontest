from django.shortcuts import render
from .forms import PredictionForm
from .ml_model import predict, FEATURE_GROUPS, FEATURE_COLUMNS


def home(request):
    """Render the prediction form."""
    form = PredictionForm()
    return render(request, 'predictor/home.html', {
        'form': form,
        'feature_groups': FEATURE_GROUPS,
    })


def predict_view(request):
    """Handle form submission and return prediction results."""
    if request.method == 'POST':
        form = PredictionForm(request.POST)
        if form.is_valid():
            # Build input dict from cleaned data
            input_data = {}
            for col in FEATURE_COLUMNS:
                input_data[col] = form.cleaned_data[col]

            result = predict(input_data)

            # Calculate gauge offset for SVG arc (total arc length = 251.2)
            gauge_offset = 251.2 * (1 - result['probability'] / 100)

            return render(request, 'predictor/result.html', {
                'result': result,
                'input_data': input_data,
                'gauge_offset': round(gauge_offset, 1),
            })
    else:
        form = PredictionForm()

    return render(request, 'predictor/home.html', {
        'form': form,
        'feature_groups': FEATURE_GROUPS,
    })
