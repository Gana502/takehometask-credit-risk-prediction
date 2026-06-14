from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
import joblib
import logging

MODEL_PATH = Path(__file__).resolve().parents[1] / "artifacts" / "model.joblib"

app = FastAPI(title="ML Inference Service")

# Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Request payload expected by the prediction endpoint
class CustomerFeatures(BaseModel):
    customer_id: str
    txn_count: float
    total_debit: float
    total_credit: float
    avg_amount: float
    kw_rent: int = 0
    kw_netflix: int = 0
    kw_tesco: int = 0
    kw_payroll: int = 0
    kw_bonus: int = 0

model = None

@app.on_event("startup")
def load_model():
    """Load the trained model from the artifacts directory when the FastAPI application starts."""
    global model
    
    logger.info("Loading model from %s", MODEL_PATH)
    
    if not MODEL_PATH.exists():
        raise RuntimeError("Model file not found. Please place model.joblib in artifacts/")
    model = joblib.load(MODEL_PATH)
    
    logger.info("Model loaded successfully. Model type: %s", type(model))

@app.get("/health")
def health():
    logger.info("Health check endpoint called")
    """Health check endpoint used for monitoring and readiness checks."""
    return {"status": "ok"}

@app.post("/predict")
def predict(payload: CustomerFeatures):
    """ Generate a default risk prediction for a customer based on their transaction features."""
    logger.info("Received prediction request for customer_id: %s", payload.customer_id)
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    features = [[
        payload.txn_count,
        payload.total_debit,
        payload.total_credit,
        payload.avg_amount,
        payload.kw_rent,
        payload.kw_netflix,
        payload.kw_tesco,
        payload.kw_payroll,
        payload.kw_bonus,
    ]]
    proba = model.predict_proba(features)[0][1]
    pred = int(proba >= 0.5)
    logger.info("Prediction for customer_id %s: probability=%.4f, prediction=%d", payload.customer_id, proba, pred)
    return {
        "customer_id": payload.customer_id, 
        "probability": float(proba), 
        "prediction": pred
        }