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

## ✨ Key Features

EcoVision provides four major analytical tools optimized for environmental monitoring:

- 🔥 **Wildfire Prediction**  
  Analyzes standard satellite imagery (JPG/PNG) to predict the likelihood of active wildfire occurrences accurately.
  
- 💧 **Water Body Segmentation**  
  Automated semantic segmentation designed to map, track, and monitor vital water resources and shifting boundaries.

- 🗺️ **Burned Area (Burn Scar) Segmentation**  
  Utilizes the **Prithvi-100M foundation model** on 6-band Sentinel-2 / HLS GeoTIFFs to precisely map post-fire damage areas and outline burn scars dynamically.

- 🌲 **Deforestation Detection**  
  Leverages cutting-edge **U-Net** models to consistently detect forest land-cover transitions and expose instances of unauthorized logging.

## 💻 Stunning UI/UX

A completely overhauled, highly immersive dashboard with **Glassmorphism**, smooth micro-animations, dynamic gradients, and real-time inference displays. Experience planetary monitoring in style!

## 🛠 Tech Stack

- **Frontend:** React 19, Vite, Vanilla CSS, Lucide Icons.
- **Backend:** Python, FastAPI.
- **Machine Learning Models:** Custom Convolutional Neural Networks (CNNs), U-Net Architectures, and IBM/NASA's Prithvi-100M Foundation Model.

## 🚀 Getting Started

Follow these steps to get the project running locally.

### 1. Clone the repository
```bash
git clone https://github.com/KetanGhungralekar/EcoVision-Satellite-Monitoring.git
cd EcoVision-Satellite-Monitoring
```

### 2. Run the Backend (FastAPI)
The Python backend manages the ML inference and model loading.
```bash
cd backend
# Optional: It is recommended to create a virtual environment
# python -m venv venv
# source venv/bin/activate (or venv\Scripts\activate on Windows)

# Ensure dependencies are installed (e.g. fastapi, uvicorn, torch, etc.)
# pip install -r requirements.txt 

uvicorn main:app --reload
```
*The backend server will become available at `http://localhost:8000`*

### 3. Run the Frontend (React + Vite)
```bash
cd ../frontend
npm install
npm run dev
```
*The interactive dashboard will boot up at `http://localhost:5173`*

## 📁 Project Structure

```
EcoVision-Satellite-Monitoring/
├── backend/                                   # FastAPI server and Model Inference scripts
│   ├── main.py                                # System API Entry point
│   ├── prithvi_inference.py                   # Prithvi geospatial logic
│   └── deforestation_inference.py             # U-Net image segmentation
├── frontend/                                  # React 19 Client application
│   ├── src/
│   │   ├── App.jsx                            # Analytics Dashboard Wrapper
│   │   ├── Login.jsx                          # Glassmorphism entrance portal
│   │   └── index.css                          # Application styling
├── Water_Body_Segmentation/                   # Raw models and Jupyter notebooks
└── Wildfire-Prediction-from-Satellite-Imagery/# Raw models and Jupyter notebooks
```

## 🤝 Contributing

Contributions, issues, and feature requests are very welcome! Feel free to check the issues page.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'feat: Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

Distributed under the MIT License. See the `LICENSE` file for more details.
