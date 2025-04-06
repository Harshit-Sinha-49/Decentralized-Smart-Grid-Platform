from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import io
import joblib
import os
import numpy as np
from typing import List
from model import train_local_model, predict_values, update_global_model
from fault_model import train_local_fault_model, predict_faults, update_global_fault_model

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (change in production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure necessary directories exist
os.makedirs("saved_fault_detection_models", exist_ok=True)
os.makedirs("saved_models", exist_ok=True)
os.makedirs("data", exist_ok=True)

# # Initialize global model storage
global_model_path = "saved_models/global_model.pkl"
# if not os.path.exists(global_model_path):
#     joblib.dump(None, global_model_path)  # Placeholder
global_fault_model_path = "saved_fault_detection_models/global_fault_model.pkl"


@app.post("/train_local")
async def train_local(file: UploadFile = File(...)):
    """Trains a local model on the uploaded dataset and updates the global model."""
    contents = await file.read()
    df = pd.read_csv(io.StringIO(contents.decode("utf-8")))

    # Save file for reference
    file_path = f"data/{file.filename}"
    df.to_csv(file_path, index=False)

    # Train local model and extract gradients
    model, r2, mse, local_weights, local_biases = train_local_model(df)

    # Save model and gradients
    joblib.dump(model, "saved_models/local_model.pkl")
    joblib.dump((local_weights, local_biases), "saved_models/local_gradients.pkl")

    # Update the global model
    updated_global_model = update_global_model((local_weights, local_biases))
    joblib.dump(updated_global_model, global_model_path)  # Save updated model

    return {
        "message": "Local training complete",
        "R2 Score": r2,
        "MSE": mse,
        "model_path": "saved_models/local_model.pkl",
        "gradient_saved": "saved_models/local_gradients.pkl"
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Make predictions using the global model with a CSV file."""
    contents = await file.read()
    df = pd.read_csv(io.StringIO(contents.decode("utf-8")))

    if not os.path.exists(global_model_path):
        raise HTTPException(status_code=400, detail="Global model not trained yet.")

    predictions = predict_values(df, global_model_path)

    return {"predictions": predictions.tolist()}


@app.post("/train_local_fault_model")
async def train_local_fault_model_api(file: UploadFile = File(...)):
    """Trains a local transformer fault detection model on the uploaded dataset and updates the global fault model."""
    contents = await file.read()
    df = pd.read_csv(io.StringIO(contents.decode("utf-8")))

    # Save file for reference
    file_path = f"data/{file.filename}"
    df.to_csv(file_path, index=False)

    # Train local fault model
    model, accuracy, local_weights, local_biases = train_local_fault_model(df)

    # Save model and gradients
    joblib.dump(model, "saved_fault_detection_models/local_fault_model.pkl")
    joblib.dump((local_weights, local_biases), "saved_fault_detection_models/local_fault_gradients.pkl")

    # Update global fault model
    updated_global_fault_model = update_global_fault_model((local_weights, local_biases))
    joblib.dump(updated_global_fault_model, global_fault_model_path)  # Save updated model

    return {
        "message": "Local fault detection training complete",
        "Accuracy": accuracy,
        "model_path": "saved_fault_detection_models/local_fault_model.pkl",
        "gradient_saved": "saved_fault_detection_models/local_fault_gradients.pkl"
    }

@app.post("/predict_fault")
async def predict_fault(file: UploadFile = File(...)):
    """Predict transformer faults using the global fault detection model."""
    contents = await file.read()
    df = pd.read_csv(io.StringIO(contents.decode("utf-8")))

    if not os.path.exists(global_fault_model_path):
        raise HTTPException(status_code=400, detail="Global fault detection model not trained yet.")

    predictions = predict_faults(df)
    return {"fault_predictions": predictions.tolist()}
