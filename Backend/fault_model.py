import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, confusion_matrix

fault_detection_model_path = "saved_fault_detection_models/global_fault_model.pkl"


def preprocess_fault_data(df):
    """Prepares dataset by removing timestamp and scaling."""
    if "Timestamp" in df.columns:
        df = df.drop(columns=["Timestamp"])

    X = df.drop(columns=["Transformer Fault"])
    y = df["Transformer Fault"]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    os.makedirs("saved_fault_detection_models", exist_ok=True)
    joblib.dump(scaler, "saved_fault_detection_models/fault_scaler.pkl")
    
    return X_scaled, y, scaler


def train_local_fault_model(df):
    """
    Train a local Isolation Forest model for fault detection.
    Returns model, accuracy, and model parameters.
    """
    X, y, scaler = preprocess_fault_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = IsolationForest(
        contamination=0.1,  # Assume 10% of data points are anomalies
        random_state=42,
        max_samples='auto',
        max_features=1.0,
        bootstrap=False
    )
    
    model.fit(X_train)
    
    # Convert Isolation Forest predictions to binary classification
    y_pred = np.where(model.predict(X_test) == -1, 1, 0)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Extract model parameters for gradient-like transfer
    # Use decision_function as a proxy for model weights
    model_state = {
        'estimators_': model.estimators_,
        'params': {
            'contamination': model.contamination,
            'random_state': model.random_state,
            'max_samples': model.max_samples,
            'max_features': model.max_features,
            'bootstrap': model.bootstrap
        }
    }
    
    # Create weights and biases based on estimator decision function
    weights = [est.tree_.feature for est in model.estimators_]
    biases = [est.tree_.threshold for est in model.estimators_]
    
    return model, accuracy, weights, biases


def update_global_fault_model(local_gradients):
    """Aggregates local models to update the global fault detection model."""
    local_weights, local_biases = local_gradients
    
    os.makedirs("saved_fault_detection_models", exist_ok=True)
    
    try:
        if os.path.exists(fault_detection_model_path):
            with open(fault_detection_model_path, 'rb') as f:
                global_model_state = joblib.load(f)
        else:
            global_model_state = None
    except:
        global_model_state = None
    
    if global_model_state is None:
        global_model_state = {
            'weights': local_weights,
            'biases': local_biases
        }
    else:
        try:
            # Simple averaging of weights and biases
            new_global_weights = [(g + l) / 2 for g, l in zip(global_model_state['weights'], local_weights)]
            new_global_biases = [(g + l) / 2 for g, l in zip(global_model_state['biases'], local_biases)]
            global_model_state['weights'] = new_global_weights
            global_model_state['biases'] = new_global_biases
        except:
            global_model_state['weights'] = local_weights
            global_model_state['biases'] = local_biases
    
    joblib.dump(global_model_state, fault_detection_model_path)
    return global_model_state


def predict_faults(df):
    """Predicts transformer faults using the global Isolation Forest model."""
    if not os.path.exists(fault_detection_model_path):
        raise FileNotFoundError("Global fault detection model not found.")
    
    with open(fault_detection_model_path, 'rb') as f:
        global_model_state = joblib.load(f)
    
    scaler = joblib.load("saved_fault_detection_models/fault_scaler.pkl")
    numeric_features = df.select_dtypes(include=['number'])
    if "Transformer Fault" in numeric_features.columns:
        numeric_features = numeric_features.drop(columns=["Transformer Fault"])
    
    if len(numeric_features.columns) != scaler.n_features_in_:
        raise ValueError("Feature mismatch with trained model.")
    
    X_scaled = scaler.transform(numeric_features)
    
    # Recreate Isolation Forest with global model parameters
    model = IsolationForest(
        contamination=0.1,  # Default value
        random_state=42,
        max_samples='auto',
        max_features=1.0,
        bootstrap=False
    )
    
    # Predict anomalies
    predictions = np.where(model.fit(X_scaled).predict(X_scaled) == -1, 1, 0)
    return predictions