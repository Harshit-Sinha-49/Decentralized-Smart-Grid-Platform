import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score

global_model_path = "saved_models/global_model.pkl"

def preprocess_data(df):
    """Prepares dataset by removing timestamp and scaling."""
    # Drop timestamp if present
    if "Timestamp" in df.columns:
        df = df.drop(columns=["Timestamp"])

    # Separate features and target
    X = df.drop(columns=["Predicted Load (kW)"])
    y = df["Predicted Load (kW)"]

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Save scaler for future use
    joblib.dump(scaler, "saved_models/scaler.pkl")

    return X_scaled, y, scaler

def train_local_model(df):
    """
    Train a local model and return model, performance metrics, and local weights/biases
    
    Args:
        df (pd.DataFrame): Training dataframe
    
    Returns:
        Tuple containing model, r2 score, mse, local weights, and local biases
    """
    # Preprocess data
    X, y, scaler = preprocess_data(df)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train model with more robust configuration
    model = MLPRegressor(
        hidden_layer_sizes=(64, 64),
        activation='relu', 
        solver='adam', 
        max_iter=50,
        learning_rate_init=0.001,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        random_state=42
    )
    
    # Fit the model
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)

    # Extract local weights and biases
    local_weights = [w.copy() for w in model.coefs_]
    local_biases = [b.copy() for b in model.intercepts_]

    # Create a custom dictionary to store model state
    model_state = {
        'hidden_layer_sizes': model.hidden_layer_sizes,
        'activation': model.activation,
        'solver': model.solver,
        'alpha': model.alpha,
        'batch_size': model.batch_size,
        'learning_rate': model.learning_rate,
        'learning_rate_init': model.learning_rate_init,
        'max_iter': model.max_iter,
        'random_state': model.random_state,
        'weights': local_weights,
        'biases': local_biases
    }

    return model, r2, mse, model_state['weights'], model_state['biases']

def update_global_model(local_gradients):
    """
    Aggregate local gradients to update the global model using Federated Averaging
    
    Args:
        local_gradients (tuple): Local model weights and biases
    
    Returns:
        Updated global model state
    """
    local_weights, local_biases = local_gradients

    # Ensure saved_models directory exists
    os.makedirs("saved_models", exist_ok=True)

    # Check if global model exists
    try:
        if os.path.exists(global_model_path):
            with open(global_model_path, 'rb') as f:
                global_model_state = joblib.load(f)
        else:
            global_model_state = None
    except:
        global_model_state = None

    # Initialize or update global model state
    if global_model_state is None:
        # Create new global model state
        global_model_state = {
            'hidden_layer_sizes': (64, 64),
            'activation': 'relu',
            'solver': 'adam',
            'alpha': 0.0001,
            'batch_size': 'auto',
            'learning_rate': 'constant',
            'learning_rate_init': 0.001,
            'max_iter': 50,
            'random_state': 42,
            'weights': local_weights,
            'biases': local_biases
        }
    else:
        # Federated Averaging: Weighted average of current and local weights
        try:
            global_weights = global_model_state['weights']
            global_biases = global_model_state['biases']

            # Simple average of weights and biases
            new_global_weights = [(g + l) / 2 for g, l in zip(global_weights, local_weights)]
            new_global_biases = [(g + l) / 2 for g, l in zip(global_biases, local_biases)]

            global_model_state['weights'] = new_global_weights
            global_model_state['biases'] = new_global_biases
        except Exception as e:
            # Fallback to local weights if averaging fails
            global_model_state['weights'] = local_weights
            global_model_state['biases'] = local_biases

    # Save updated global model state
    joblib.dump(global_model_state, global_model_path)
    return global_model_state

def predict_values(df, global_model_path):
    """
    Predict using the global model
    
    Args:
        df (pd.DataFrame): Input dataframe for prediction
        global_model_path (str): Path to the global model
    
    Returns:
        np.ndarray: Predicted values
    """
    # Validate global model exists
    if not os.path.exists(global_model_path):
        raise FileNotFoundError(f"Global model not found at {global_model_path}")

    # Load global model state and scaler
    with open(global_model_path, 'rb') as f:
        global_model_state = joblib.load(f)
    
    scaler = joblib.load("saved_models/scaler.pkl")

    # Prepare input data
    numeric_features = df.select_dtypes(include=['number'])

    # Remove target column if present
    if 'Predicted Load (kW)' in numeric_features.columns:
        numeric_features = numeric_features.drop(columns=['Predicted Load (kW)'])

    # Validate features match training data
    if len(numeric_features.columns) != scaler.n_features_in_:
        raise ValueError(f"Mismatch: Model expects {scaler.n_features_in_} features, received {len(numeric_features.columns)}.")

    # Scale features
    X_scaled = scaler.transform(numeric_features)

    # Recreate MLPRegressor with saved state
    model = MLPRegressor(
        hidden_layer_sizes=global_model_state['hidden_layer_sizes'],
        activation=global_model_state['activation'],
        solver=global_model_state['solver'],
        alpha=global_model_state['alpha'],
        batch_size=global_model_state['batch_size'],
        learning_rate=global_model_state['learning_rate'],
        learning_rate_init=global_model_state['learning_rate_init'],
        max_iter=global_model_state['max_iter'],
        random_state=global_model_state['random_state']
    )

    # Temporarily fit with dummy data to initialize internal attributes
    dummy_X = np.zeros((10, X_scaled.shape[1]))
    dummy_y = np.zeros(10)
    model.fit(dummy_X, dummy_y)

    # Set the learned weights and biases
    model.coefs_ = global_model_state['weights']
    model.intercepts_ = global_model_state['biases']

    # Predict
    predictions = model.predict(X_scaled)
    return predictions