from flask import Flask, render_template, jsonify, request
import numpy as np
import pandas as pd
import joblib
import rasterio
import json
import os

app = Flask(__name__)

model        = joblib.load('models/best_model_RF.pkl')
feature_cols = joblib.load('parameters/feature_cols.pkl')

raster_band_to_feature = {
    1: 'NDVI', 2: 'NDWI', 3: 'NBR',  4: 'EVI',  5: 'SAVI',
    6: 'dNBR', 7: 'DEM',  8: 'Slope', 9: 'Aspect', 10: 'LST',
    11: 'Wind', 12: 'Precipitation', 13: 'LULC',
    14: 'Population', 15: 'SoilMoisture',
}
feature_to_band0 = {v: k-1 for k, v in raster_band_to_feature.items()}
band_read_order  = [feature_to_band0[f] for f in feature_cols]


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    data     = request.json
    features = pd.DataFrame([data], columns=feature_cols)
    features = features.fillna(0)
    prob      = model.predict_proba(features)[0][1]
    risk_level = 'High' if prob > 0.7 else 'Medium' if prob > 0.4 else 'Low'
    return jsonify({'susceptibility': round(float(prob), 3), 'risk_level': risk_level})


@app.route('/susceptibility_map')
def susceptibility_map():
    with rasterio.open('static/Features_Raster_Bursa.tif') as src:
        data   = src.read().astype(float)
        bounds = src.bounds
        rows, cols = src.height, src.width

    data_reordered = data[band_read_order, :, :]
    pixels         = data_reordered.reshape(15, -1).T
    pixels_df      = pd.DataFrame(pixels, columns=feature_cols)

    for col in pixels_df.columns:
        col_data   = pixels_df[col]
        median_val = col_data[(col_data > -1e9) & (col_data < 1e9)].median()
        pixels_df[col] = col_data.where((col_data > -1e9) & (col_data < 1e9), median_val)
    pixels_df = pixels_df.fillna(pixels_df.median())

    susceptibility = model.predict_proba(pixels_df)[:, 1]
    susc_map       = susceptibility.reshape(rows, cols)

    step       = 5
    susc_small = susc_map[::step, ::step]

    return jsonify({
        'data': susc_small.tolist(),
        'bounds': {'north': bounds.top, 'south': bounds.bottom,
                   'east': bounds.right, 'west': bounds.left},
        'rows': susc_small.shape[0],
        'cols': susc_small.shape[1]
    })


@app.route('/model_comparison')
def model_comparison():
    with open('static/model_comparison.json') as f:
        return jsonify(json.load(f))


@app.route('/hyperparameter_tuning')
def hyperparameter_tuning():
    with open('static/hyperparameter_tuning.json') as f:
        return jsonify(json.load(f))


@app.route('/susceptibility_at_point')
def susceptibility_at_point():
    lat = float(request.args.get('lat'))
    lng = float(request.args.get('lng'))

    with rasterio.open('static/susceptibility_raster.tif') as src:
        bounds = src.bounds
        if not (bounds.left <= lng <= bounds.right and bounds.bottom <= lat <= bounds.top):
            return jsonify({'error': 'Point outside study area'})
        val = list(src.sample([(lng, lat)]))[0][0]

    prob = float(np.clip(val, 0, 1))
    risk_level = 'High' if prob > 0.7 else 'Medium' if prob > 0.4 else 'Low'
    return jsonify({'susceptibility': round(prob, 3), 'risk_level': risk_level})


if __name__ == '__main__':
    app.run(debug=True)
