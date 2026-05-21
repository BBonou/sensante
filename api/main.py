# api/main.py
# FastAPI API for SenSante - Medical pre-diagnostic assistant

from fastapi import FastAPI
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware

import joblib
import numpy as np

# Create the application
app = FastAPI(
    title="SenSante API",
    description="Medical pre-diagnostic assistant for Senegal",
    version="0.2.0"
)

# Allow requests from the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # In dev : accept all
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Basic route: verify that the API is working
@app.get("/health")
def health_check():
    """Checking API status."""
    return {
        "status": "ok",
        "message": "SensSante API is running"
    }

# --- Pydantic Schemas ---

class PatientInput(BaseModel):
    """Input data: a patient's symptoms."""
    age: int = Field(..., ge=0, le=120, description="Age in years")
    sexe: str = Field(..., description="Gender : M or F")
    temperature: float = Field(..., ge=35.0, le=42.0, description="Temperature in degrees Celsius")
    tension_sys: int = Field(...,  ge=60, le=250,description="Tension symptom")
    toux: bool = Field(..., description="Toux presence")
    fatigue: bool = Field(..., description="Fatigue presence")
    maux_tete: bool = Field(..., description="Maux tete presence")
    frissons: bool = Field(..., description="Frissons presence")
    nausee: bool = Field(..., description="Nausee presence")
    region: str = Field(..., description="Region of Senegal")

class DiagnosticOutput(BaseModel):
    """Output data: diagnostic result."""
    diagnostic: str = Field(..., description="Predicted diagnosis")
    probabilite: float = Field(..., description="Probability of diagnosis")
    confiance: str = Field(..., description="Confidence level")
    message: str = Field(..., description="Diagnostic message")


# --- Load the model and encoders at startup ---
print("Loading model...")
model = joblib.load("models/model.pkl")
le_sexe = joblib.load("models/encoder_sexe.pkl")
le_region = joblib.load("models/encoder_region.pkl")
feature_cols = joblib.load("models/features_cols.pkl")
print(f"Loaded model: {type(model).__name__}")
print(f"Classes : {list(model.classes_)}")


@app.post("/predict", response_model=DiagnosticOutput)
def predict(patient: PatientInput):
    """
    Predict a diagnosis based on a patient's symptoms.
    Receives symptoms in JSON format, returns the diagnosis,
    probability, and a recommendation.
    """
    # 1. Encode categorical variables
    try:
        sexe_enc = le_sexe.transform([patient.sexe])[0]
    except ValueError:
        return DiagnosticOutput(
            diagnostic="error",
            probabilite=0.0,
            confiance="aucune",
            message=f"Invalid sexe : {patient.sexe}. Use M or F."
        )

    try:
        region_enc = le_region.transform([patient.region])[0]
    except ValueError:
        return DiagnosticOutput(
            diagnostic="error",
            probabilite=0.0,
            confiance="aucune",
            message=f"Unknown region : {patient.region}."
        )

    # 2. Construct the feature vector
    features = np.array([[
        patient.age,
        sexe_enc,
        patient.temperature,
        patient.tension_sys,
        int(patient.toux),
        int(patient.fatigue),
        int(patient.maux_tete),
        int(patient.frissons),
        int(patient.nausee),
        region_enc
    ]])

    # 3. Predict
    diagnostic = model.predict(features)[0]
    probas = model.predict_proba(features)[0]
    proba_max = float(probas.max())

    # 4. To determine the level of confidence
    if proba_max >= 0.7:
        confiance = "haute"
    elif proba_max >= 0.4:
        confiance = "moyenne"
    else:
        confiance = "faible"

    # 5. Generate the recommendation
    messages = {
        "paludisme": "Suspicion de paludisme. Consultez un medecin rapidement.",
        "grippe": "Suspicion de grippe. Repos et hydratation recommandes.",
        "typhoide": "Suspicion de typhoïde. Consultation medicale necessaire.",
        "sain": "Pas de pathologie detectee. Continuer a surveiller."
    }

    # 6. Return the result
    return DiagnosticOutput(
        diagnostic=diagnostic,
        probabilite=round(proba_max, 2),
        confiance=confiance,
        message=messages.get(diagnostic, "Consultez un medecin.")
    )



# Exercise 1 : returns information about the model
@app.get("/model-info")
def model_info():
    """
    returns information about the model: type (RandomForestClassifier), number of trees,
    possible classes, and number of features.
    """
    return {
        "type": type(model).__name__,
        "num_trees": model.n_estimators,
        "classes": list(model.classes_),
        "num_features": len(feature_cols)
    }