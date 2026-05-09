# KSI Collision Fatality Predictor

An AI-powered Streamlit application that predicts whether a Toronto traffic collision is likely to be **Fatal** or **Non-Fatal** using supervised machine learning.

Live demo: [ksi-collision-fatality-predictor-rozhinasaberi.streamlit.app](https://ksi-collision-fatality-predictor-rozhinasaberi.streamlit.app/)

## What This Project Is About

This project applies machine learning to Toronto Police Service KSI data to help estimate collision severity. Users enter roadway, environmental, and incident-related factors through a Streamlit interface and receive a prediction with probability scores.

This is an **AI / machine learning classification project** and a **deployed Streamlit web app**.

## Models Used

- Support Vector Machine (`ksi_svm.pkl`) - best-performing model
- Decision Tree (`ksi_decision_tree.pkl`)
- Random Forest (`ksi_random_forest.pkl`)

## Inputs And Outputs

Inputs include:

- road class
- district
- accident location
- traffic control
- visibility
- light condition
- road surface
- impact type
- hour of day
- involvement flags such as pedestrian, cyclist, speeding, alcohol, and aggressive driving

Outputs include:

- predicted class
- fatal probability
- non-fatal probability
- model confidence

## Tech Stack

- Python
- Streamlit
- pandas
- scikit-learn
- SciPy
- pickle

## Main Files

- `app.py` - Streamlit application
- `requirements.txt` - dependencies
- `ksi_svm.pkl` - trained SVM model
- `ksi_decision_tree.pkl` - trained decision tree model
- `ksi_random_forest.pkl` - trained random forest model

## Dataset

- Toronto Police Service KSI dataset (2006-2023)

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Authors

- Rojina Saberi
- Elijah Robinson
