# Wildfire Susceptibility Mapping System — Bursa, Turkey (2025)

CME434 GIS Final Project — Karabuk University, Department of Computer Engineering
Instructor: Dr. Sohaib K M Abujayyab | Spring Semester 2024-2025

## Overview
A web-based GIS decision support system for wildfire susceptibility mapping using machine learning.
Built for the July 26, 2025 wildfire incident in Bursa, Turkey.

## Folder Structure
- `models/` — Trained ML model (Random Forest)
- `parameters/` — Feature columns and scaler
- `predictions/` — Output susceptibility raster and report map
- `samplepoints/` — Burned/unburned sample point shapefiles
- `static/` — Web assets (rasters, images, JSON data)
- `templates/` — HTML templates

## How to Run
1. Install dependencies:
```
pip install -r requirements.txt
```
2. Start the web application:
```
python app.py
```
3. Open your browser and go to: **http://127.0.0.1:5000**

> Note: The trained model is already included. No need to re-run `train_model.py`.

## Features
- 15 environmental features (NDVI, NDWI, NBR, EVI, SAVI, dNBR, DEM, Slope, Aspect, LST, Wind, Precipitation, LULC, Population, SoilMoisture)
- 1,000 sample points (500 burned, 500 unburned)
- Best model: Random Forest (Accuracy: 100%, AUC: 1.000)
- Interactive susceptibility map with click-to-query functionality
- Scenario explorer for real-time risk assessment
- Model comparison dashboard
