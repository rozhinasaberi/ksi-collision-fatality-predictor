# KSI Collision Fatality Predictor

A Streamlit app that predicts whether a Toronto traffic collision is likely to be **Fatal** or **Non-Fatal** using machine learning models trained on Toronto Police Service KSI data.

Live app: [ksi-collision-fatality-predictor-rozhinasaberi.streamlit.app](https://ksi-collision-fatality-predictor-rozhinasaberi.streamlit.app/)

## Overview

This project lets users enter collision details such as road class, district, visibility, light condition, road surface, impact type, hour of day, and involvement flags like speeding or alcohol. The app then returns:

- the predicted outcome
- fatal probability
- non-fatal probability
- model confidence

The dataset referenced in the app is:

- Toronto Police Service KSI (2006-2023)

## Models Included

- Support Vector Machine (`ksi_svm.pkl`) - best model
- Decision Tree (`ksi_decision_tree.pkl`)
- Random Forest (`ksi_random_forest.pkl`)

Users can switch between models in the Streamlit interface and compare predictions.

## Tech Stack

- Python
- Streamlit
- pandas
- scikit-learn
- pickle

## Project Files

- `app.py` - main Streamlit application
- `requirements.txt` - Python dependencies
- `ksi_svm.pkl` - trained SVM model
- `ksi_decision_tree.pkl` - trained decision tree model
- `ksi_random_forest.pkl` - trained random forest model

## Run Locally

1. Clone the repository.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Start the app:

```bash
streamlit run app.py
```

4. Open the local URL shown by Streamlit in your browser.

## How To Use

1. Choose a model from the dropdown.
2. Enter the collision conditions and involvement flags.
3. Click `Predict Outcome`.
4. Review the predicted class and probability scores.

## Authors

- Rojina Saberi
- Elijah Robinson
