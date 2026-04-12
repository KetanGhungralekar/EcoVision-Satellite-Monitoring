import os
import io
import cv2
import numpy as np
import base64
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import tensorflow as tf
from tensorflow.keras.models import load_model

app = FastAPI(title="EcoVision Satellite Monitoring API")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Model paths ────────────────────────────────────────────────────────────────
WILDFIRE_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..",
    "Wildfire-Prediction-from-Satellite-Imagery",
    "saved_model", "custom_best_model.h5"
)
WATERBODY_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..",
    "Water_Body_Segmentation", "image_segmentation_model_UNet.h5"
)

wildfire_model = None
waterbody_model = None

@app.on_event("startup")
async def startup_event():
    global wildfire_model, waterbody_model

    if os.path.exists(WILDFIRE_MODEL_PATH):
        print(f"Loading Wildfire model…")
        wildfire_model = load_model(WILDFIRE_MODEL_PATH)
        print("Wildfire model loaded.")
    else:
        print(f"WARNING: Wildfire model not found at {WILDFIRE_MODEL_PATH}")

    if os.path.exists(WATERBODY_MODEL_PATH):
        print(f"Loading Water Body model…")
        waterbody_model = load_model(WATERBODY_MODEL_PATH)
        print("Water Body model loaded.")
    else:
        print(f"WARNING: Water Body model not found at {WATERBODY_MODEL_PATH}")


# ── Schemas ────────────────────────────────────────────────────────────────────
class PredictionResponse(BaseModel):
    no_wildfire_prob: float
    wildfire_prob: float
    prediction: str

class WaterBodyResponse(BaseModel):
    prediction: str
    mask_base64: str

CLASS_MAP = {0: "Nowildfire", 1: "Wildfire"}


# ── Wildfire endpoint (unchanged) ──────────────────────────────────────────────
@app.post("/api/predict/wildfire", response_model=PredictionResponse)
async def predict_wildfire(file: UploadFile = File(...)):
    if not wildfire_model:
        raise HTTPException(status_code=500, detail="Wildfire model not loaded.")

    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file.")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224)) / 255.0
        img_batch = np.expand_dims(img, axis=0)

        preds = wildfire_model.predict(img_batch)
        no_wildfire_prob = float(preds[0][0])
        wildfire_prob    = float(preds[0][1])
        prediction_label = "Wildfire" if wildfire_prob > no_wildfire_prob else "No Wildfire"

        return PredictionResponse(
            no_wildfire_prob=no_wildfire_prob,
            wildfire_prob=wildfire_prob,
            prediction=prediction_label
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Water Body endpoint (unchanged) ───────────────────────────────────────────
@app.post("/api/predict/waterbody", response_model=WaterBodyResponse)
async def predict_waterbody(file: UploadFile = File(...)):
    if not waterbody_model:
        raise HTTPException(status_code=500, detail="Water Body model not loaded.")

    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file.")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (256, 256))
        img_batch = np.expand_dims(img, axis=0)

        prediction = waterbody_model.predict(img_batch)[0]
        mask = np.where(prediction > 0.5, 1, 0).astype(np.uint8)

        mask_rgba = np.zeros((256, 256, 4), dtype=np.uint8)
        mask_rgba[mask[:, :, 0] == 1] = [255, 255, 0, 130]

        _, buffer = cv2.imencode('.png', mask_rgba)
        mask_base64 = base64.b64encode(buffer).decode('utf-8')

        return WaterBodyResponse(
            prediction="Water Body Segmentation Generated",
            mask_base64=mask_base64
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Burned Area endpoint (GeoTIFF + PyTorch Prithvi) ──────────────────────────
@app.post("/api/predict/burnscar")
async def predict_burnscar(file: UploadFile = File(...)):
    """
    Accepts a multi-band GeoTIFF (.tif / .tiff).
    Returns:
      - mask_base64 : orange overlay PNG of predicted burn scars
      - rgb_base64  : visible-light composite PNG for display
      - burned_pct  : estimated percentage of burned area
    """
    allowed_ext = {'.tif', '.tiff'}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed_ext:
        return JSONResponse(
            content={"error": "Only .tif / .tiff files are accepted for burn scar analysis."},
            status_code=400
        )

    temp_path = "/tmp/burnscar_upload.tif"
    try:
        from prithvi_inference import burn_scar_predictor

        contents = await file.read()
        with open(temp_path, "wb") as f:
            f.write(contents)

        mask_b64, rgb_b64 = burn_scar_predictor.predict(temp_path)

        return JSONResponse(content={
            "prediction_text": "Burned Area Segmentation Generated",
            "mask_base64": mask_b64,
            "rgb_base64":  rgb_b64,
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(content={"error": str(e)}, status_code=500)

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


# ── Deforestation endpoint (U-Net) ───────────────────────────────────────────
@app.post("/api/predict/deforestation")
async def predict_deforestation(file: UploadFile = File(...)):
    """
    Accepts an RGB image (PNG/JPEG).
    Returns:
      - mask_base64 : color-coded overlay PNG (Green=Forest, Red=Deforest)
    """
    try:
        from deforestation_inference import deforestation_predictor
        
        contents = await file.read()
        mask_b64 = deforestation_predictor.predict(contents)

        return JSONResponse(content={
            "prediction_text": "Deforestation Segmentation Generated",
            "mask_base64": mask_b64
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(content={"error": str(e)}, status_code=500)


# ── Wildfire Spread endpoint (TFRecord + PyTorch GNN) ──────────────────────────
@app.post("/api/predict/wildfire-spread")
async def predict_wildfire_spread(file: UploadFile = File(...)):
    """
    Accepts a .tfrecord or .npy file for next-day wildfire spread.
    Returns 5 base64 encoded pngs.
    """
    try:
        from wildfire_spread_inference import WildfireSpreadPredictor
        
        # Initialize the predictor using the saved model path
        model_path = os.path.join(
            os.path.dirname(__file__), "..",
            "Wildfire Spread Prediction", "model", "best_model.pth"
        )
        predictor = WildfireSpreadPredictor(model_path)
        
        # Preserve extension so the predictor can auto-detect the format
        ext = os.path.splitext(file.filename)[1].lower()
        temp_path = f"wildfire_spread_upload{ext}"
            
        contents = await file.read()
        with open(temp_path, "wb") as f:
            f.write(contents)
            
        result = predictor.predict(temp_path)
        
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        return JSONResponse(content=result)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(content={"error": str(e)}, status_code=500)


# ── Health check ───────────────────────────────────────────────────────────────
@app.get("/api/health")
async def health_check():
    return {
        "status": "ok",
        "wildfire_model_loaded":  wildfire_model is not None,
        "waterbody_model_loaded": waterbody_model is not None,
        "burnscar_model":         "lazy-loaded on first request (PyTorch / Prithvi)",
    }
