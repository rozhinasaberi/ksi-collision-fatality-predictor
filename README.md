# KSI Collision Fatality Predictor

An AI-powered Streamlit application that predicts whether a Toronto traffic collision is likely to be **Fatal** or **Non-Fatal** using supervised machine learning.

Live demo: [ksi-collision-fatality-predictor-rozhinasaberi.streamlit.app](https://ksi-collision-fatality-predictor-rozhinasaberi.streamlit.app/)

## Project Summary

This project applies machine learning to Toronto Police Service KSI data to support collision severity prediction. Users can enter roadway, environmental, and incident-related factors through an interactive Streamlit interface and receive:

- a predicted collision outcome
- fatal and non-fatal probability scores
- model confidence
- side-by-side model selection for comparison

The app is designed as a practical supervised learning project that demonstrates how classification models can be deployed in a user-facing interface.

## Dataset

- Toronto Police Service KSI dataset (2006-2023)

## Machine Learning Models

- Support Vector Machine (`ksi_svm.pkl`) - best-performing model
- Decision Tree (`ksi_decision_tree.pkl`)
- Random Forest (`ksi_random_forest.pkl`)

## Features

- Interactive Streamlit web interface
- Real-time collision outcome prediction
- Model switching from the UI
- Probability-based output for interpretability
- Simple local setup and cloud deployment

## Tech Stack

- Python
- Streamlit
- pandas
- scikit-learn
- SciPy
- pickle

## Repository Structure

- `app.py` - main Streamlit application
- `requirements.txt` - project dependencies
- `ksi_svm.pkl` - trained SVM model
- `ksi_decision_tree.pkl` - trained decision tree model
- `ksi_random_forest.pkl` - trained random forest model

## Running The App Locally

1. Clone the repository.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Launch the Streamlit app:

```bash
streamlit run app.py
```

4. Open the local URL provided by Streamlit in your browser.

## How It Works

1. Select one of the trained machine learning models.
2. Enter collision attributes such as road class, district, visibility, light condition, road surface, impact type, hour, and involvement flags.
3. Click `Predict Outcome`.
4. Review the predicted class, probability scores, and model confidence.

## Project Type

This is an **AI / machine learning classification project**, specifically a **supervised learning web application** for collision fatality prediction.

## Authors

- Rojina Saberi
- Elijah Robinson
