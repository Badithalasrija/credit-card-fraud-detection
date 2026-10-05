import joblib
import pandas as pd


# Load saved model artifact
artifact = joblib.load(
    "models/fraud_detection_model.pkl"
)

model = artifact["model"]
threshold = artifact["threshold"]


def predict_fraud(transaction):
    """
    Predict whether a transaction is legitimate or fraudulent.

    Parameters
    ----------
    transaction : pandas DataFrame
        Transaction features in the same format used during training.

    Returns
    -------
    prediction : int
        0 = Legitimate
        1 = Fraud

    probability : float
        Predicted probability of fraud.
    """

    probability = model.predict_proba(transaction)[:, 1]

    prediction = (
        probability >= threshold
    ).astype(int)

    return prediction, probability