import os
import io
import cv2
import numpy as np
import base64
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tensorflow as tf
from tensorflow.keras.models import load_model

app = FastAPI(title="Wildfire Prediction API")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Model Configuration
WILDFIRE_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "Wildfire-Prediction-from-Satellite-Imagery", "saved_model", "custom_best_model.h5")
WATERBODY_MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "Water_Body_Segmentation", "image_segmentation_model_UNet.h5")

wildfire_model = None
waterbody_model = None

@app.on_event("startup")
async def startup_event():
    global wildfire_model, waterbody_model
    
    if os.path.exists(WILDFIRE_MODEL_PATH):
        print(f"Loading Wildfire model from {WILDFIRE_MODEL_PATH}...")
        wildfire_model = load_model(WILDFIRE_MODEL_PATH)
        print("Wildfire Model loaded successfully.")
    else:
        print(f"Warning: Wildfire Model not found at {WILDFIRE_MODEL_PATH}")

    if os.path.exists(WATERBODY_MODEL_PATH):
        print(f"Loading Water Body model from {WATERBODY_MODEL_PATH}...")
        waterbody_model = load_model(WATERBODY_MODEL_PATH)
        print("Water Body Model loaded successfully.")
    else:
        print(f"Warning: Water Body Model not found at {WATERBODY_MODEL_PATH}")

class PredictionResponse(BaseModel):
    no_wildfire_prob: float
    wildfire_prob: float
    prediction: str

# Class names mapping
CLASS_MAP = {0: "Nowildfire", 1: "Wildfire"}

@app.post("/api/predict/wildfire", response_model=PredictionResponse)
async def predict_wildfire(file: UploadFile = File(...)):
    if not wildfire_model:
        raise HTTPException(status_code=500, detail="Wildfire model is not loaded.")
        
    try:
        # Read uploaded image bytes
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file.")

        # Preprocessing according to the notebook
        # 1. Convert BGR to RGB (OpenCV loads as BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 2. Resize to 224x224 and scale to [0, 1]
        img = cv2.resize(img, (224, 224)) / 255.0
        
        # 3. Add batch dimension
        img_batch = np.expand_dims(img, axis=0)

        # Predict
        preds = wildfire_model.predict(img_batch)
        no_wildfire_prob = float(preds[0][0])
        wildfire_prob = float(preds[0][1])

        prediction_label = "Wildfire" if wildfire_prob > no_wildfire_prob else "No Wildfire"

        return PredictionResponse(
            no_wildfire_prob=no_wildfire_prob,
            wildfire_prob=wildfire_prob,
            prediction=prediction_label
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class WaterBodyResponse(BaseModel):
    prediction: str
    mask_base64: str

@app.post("/api/predict/waterbody", response_model=WaterBodyResponse)
async def predict_waterbody(file: UploadFile = File(...)):
    if not waterbody_model:
        raise HTTPException(status_code=500, detail="Water Body model is not loaded.")
        
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file.")

        # OpenCV loads as BGR
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 1. Resize to 256x256 (UNet takes integers 0-255 based on its internal rescaling)
        img = cv2.resize(img, (256, 256))
        
        # 2. Add batch dimension
        img_batch = np.expand_dims(img, axis=0)

        # Predict
        prediction = waterbody_model.predict(img_batch)[0]
        
        # Create a binary mask where > 0.5 threshold is met
        mask = np.where(prediction > 0.5, 1, 0).astype(np.uint8)
        
        # Generate an RGBA color overlay for the mask (Cyan output: B=255, G=255, R=0, A=130)
        mask_rgba = np.zeros((256, 256, 4), dtype=np.uint8)
        mask_rgba[mask[:, :, 0] == 1] = [255, 255, 0, 130] 
        
        # Encode to Base64 PNG
        _, buffer = cv2.imencode('.png', mask_rgba)
        mask_base64 = base64.b64encode(buffer).decode('utf-8')

        return WaterBodyResponse(
            prediction="Water Body Segmentation Generated",
            mask_base64=mask_base64
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health")
async def health_check():
    return {"status": "ok", "wildfire_model_loaded": wildfire_model is not None, "waterbody_model_loaded": waterbody_model is not None}
