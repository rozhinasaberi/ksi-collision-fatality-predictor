# KSI Collision Fatality Predictor

This project is a machine learning web application that predicts whether a Toronto traffic collision is likely to result in a fatal or non-fatal outcome.

## Tech stack

- Python
- scikit-learn
- Flask
- React (served through Flask static/template files)

## Features

- Predicts fatal vs non-fatal collision outcomes
- Uses trained machine learning pipelines saved as `.pkl` files
- Lets users choose a deployed model in the frontend
- Accepts collision details such as road class, visibility, lighting, impact type, hour, and risk-related flags

## Included models

- Decision Tree
- Random Forest
- Support Vector Machine

## Run locally

```bash
python3 -m pip install -r requirements.txt
python3 app.py
```

If port `5000` is busy on macOS, run:

```bash
python3 -c "from app import app; app.run(debug=True, host='127.0.0.1', port=5001)"
```

Then open `http://127.0.0.1:5000` or `http://127.0.0.1:5001`.

## Main files

- `script.py` - training, preprocessing, evaluation, and model saving
- `app.py` - Flask backend
- `templates/` and `static/` - frontend
- `ksi_*.pkl` - saved trained models
