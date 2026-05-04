<div align="center">
  <h1>🌍 EcoVision AI</h1>
  <p><strong>Spatial Computing for Planetary Intelligence</strong></p>
  <p>A comprehensive platform for monitoring and analyzing our planet using advanced machine learning models applied to satellite imagery.</p>
  
  <p>
    <img src="https://img.shields.io/badge/React-19-blue" alt="React 19" />
    <img src="https://img.shields.io/badge/Vite-8-blue" alt="Vite 8" />
    <img src="https://img.shields.io/badge/FastAPI-Backend-green" alt="FastAPI" />
    <img src="https://img.shields.io/badge/AI-Prithvi_100M-orange" alt="Prithvi 100M" />
    <img src="https://img.shields.io/badge/AI-U_Net-orange" alt="U-Net" />
  </p>
</div>

<hr/>

## Key Features

EcoVision AI is currently specialized in a high-precision **Three-Stage Wildfire Analysis Pipeline**:

### 🔥 The Three-Stage Wildfire Pipeline
1.  **Wildfire Risk Prediction (Risk Assessment)**  
    Analyzes standard satellite imagery (JPG/PNG) using VGG-16 and Custom CNNs to evaluate the immediate risk of wildfire occurrence with ~95% accuracy.
    
2.  **Wildfire Spread Prediction (Dynamic Modeling)**  
    Uses a **Hybrid ResGNN-UNet** model that combines spatial image learning with environmental data (weather, terrain, vegetation) to predict the dynamic growth and direction of active fires.
    
3.  **Burned Area (Burn Scar) Segmentation**  
    Utilizes the **Prithvi-100M Foundation Model** on 6-band Sentinel-2 / HLS GeoTIFFs to precisely map post-fire damage areas and outline burn scars for recovery analysis.

## Tech Stack

- **Frontend:** React 19, Vite, Vanilla CSS, Lucide Icons.
- **Backend:** Python, FastAPI, PyTorch, TensorFlow.
- **AI Models:** 
    - **Prithvi-100M:** NASA/IBM Geospatial Foundation Model.
    - **Hybrid ResGNN-UNet:** Graph-based spatial-temporal modeling.
    - **VGG-16 / Custom CNN:** Deep learning for risk classification.


## Getting Started

### 1. Clone and Setup
```bash
git clone https://github.com/KetanGhungralekar/EcoVision-Satellite-Monitoring.git
cd EcoVision-Satellite-Monitoring
```

### 2. Configure Backend
Create a `backend/.env` file and add your Hugging Face token to access gated models:
```env
HF_TOKEN_MODEL_3=your_token_here
```
Install dependencies and run the server:
```bash
cd backend
# Recommended: create and activate a venv first
./venv/bin/python3 -m uvicorn main:app --reload
```

### 3. Run Frontend
```bash
cd ../frontend
npm install
npm run dev
```

## Project Structure

```
EcoVision-Satellite-Monitoring/
├── backend/                        # FastAPI server and Model Inference scripts
│   ├── main.py                     # API Entry point
│   ├── prithvi_inference.py        # Prithvi geospatial logic
│   └── wildfire_spread_inference.py# ResGNN-UNet spread modeling
├── frontend/                       # React 19 Client application
├── prithvi-pytorch/                # Prithvi model architecture source
└── Wildfire-Prediction-from-Satellite-Imagery/ # Risk prediction notebooks
```

## 📝 License

Distributed under the MIT License. See the `LICENSE` file for more details.
