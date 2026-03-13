"""
FastAPI Inference Service
--------------------------
Serves fraud detection predictions via a REST API.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import joblib
import pathlib
import sys
import logging

# Add src to path to allow importing Preprocessor
BASE = pathlib.Path(__file__).parents[2]
sys.path.append(str(BASE))

from src.preprocessing.preprocess import Preprocessor
from src.features.feature_engineering import add_amount_features, add_ratio_features
from src.features.graph_features import GraphFeatureExtractor

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

app = FastAPI(title="Fraud Detection API", version="1.0.0")

# Global variables for model and preprocessor
model = None
preprocessor = None
graph_extractor = None
selected_features = None

class Transaction(BaseModel):
    Time: float = Field(..., example=0.0)
    card_id: str = Field(..., example="C_123")
    merchant_id: str = Field(..., example="M_456")
    device_id: str = Field(..., example="D_789")
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float = Field(..., example=100.0)

@app.on_event("startup")
def load_artifacts():
    global model, preprocessor, graph_extractor, selected_features
    try:
        models_dir = BASE / "models"
        model = joblib.load(models_dir / "model.pkl")
        preprocessor = Preprocessor.load(str(models_dir / "preprocessor.pkl"))
        graph_extractor = joblib.load(models_dir / "graph_extractor.pkl")
        
        # Load selected features list from training results or assume all features from training data
        # For simplicity, we'll get the feature names from the model if possible
        if hasattr(model, "feature_names_in_"):
            selected_features = model.feature_names_in_.tolist()
        elif hasattr(model, "feature_name"): # LightGBM
            selected_features = model.feature_name()
        else:
            # Fallback: load from a saved list if available
            import json
            results_path = BASE / "data" / "training_results.json"
            if results_path.exists():
                with open(results_path, "r") as f:
                    results = json.load(f)
                    # This is a bit hacky, but we just need the column names
                    pass
        
        log.info("Model and preprocessor loaded successfully")
    except Exception as e:
        log.error(f"Error loading artifacts: {e}")
        # Note: In production, you might want to exit if artifacts fail to load
        pass

@app.post("/predict")
def predict(transaction: Transaction):
    if model is None or preprocessor is None or graph_extractor is None:
        raise HTTPException(status_code=503, detail="Model artifacts not loaded")
    
    try:
        # 1. Convert to DataFrame
        df = pd.DataFrame([transaction.dict()])
        
        # 2. Preprocess
        df_p = preprocessor.transform(df)
        
        # 3. Graph Features
        df_g = graph_extractor.transform(df_p)
        
        # 4. Feature Engineering
        # Note: Velocity features are skipped for single-transaction inference 
        # unless we have a stateful store (like Redis). 
        df_f = add_amount_features(df_g)
        df_f = add_ratio_features(df_f)
        
        # Drop Entity IDs as they were dropped during training
        df_f = df_f.drop(columns=["card_id", "merchant_id", "device_id"], errors="ignore")
        
        # 5. Ensure all training features are present (even velocity ones)
        if selected_features:
            for col in selected_features:
                if col not in df_f.columns:
                    df_f[col] = 0.0 # Default value for missing features
            df_final = df_f[selected_features]
        else:
            df_final = df_f
            
        # 6. Predict
        prob = model.predict_proba(df_final)[0][1]
        
        # Determine fraud based on some threshold (e.g., 0.5 or the best one from training)
        threshold = 0.5 # Default
        
        return {
            "fraud_probability": float(prob),
            "is_fraud": bool(prob >= threshold),
            "threshold_used": threshold
        }
    except Exception as e:
        log.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
