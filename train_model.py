"""
train_model.py - Wildfire Susceptibility Mapping — Bursa, Turkey (2025)
Trains 3 classifiers (Random Forest, Gradient Boosting, Logistic Regression)
on 15 features, selects the best model, and generates the susceptibility map.
"""

import numpy as np
import pandas as pd
import rasterio
import joblib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ─── 1. BAND MAPPING ────────────────────────────────────────────────────────
raster_band_to_feature = {
    1:  'NDVI',
    2:  'NDWI',
    3:  'NBR',
    4:  'EVI',
    5:  'SAVI',
    6:  'dNBR',
    7:  'DEM',
    8:  'Slope',
    9:  'Aspect',
    10: 'LST',
    11: 'Wind',
    12: 'Precipitation',
    13: 'LULC',
    14: 'Population',
    15: 'SoilMoisture',
}

# Training data column order (from Inputs.txt header)
feature_cols = ['Aspect', 'DEM', 'EVI', 'LST', 'LULC', 'NBR', 'NDVI',
                'NDWI', 'Population', 'Precipitation', 'SAVI', 'Slope',
                'SoilMoisture', 'Wind', 'dNBR']

feature_to_band0 = {v: k-1 for k, v in raster_band_to_feature.items()}
band_read_order = [feature_to_band0[f] for f in feature_cols]
print("Band read order (0-indexed):", band_read_order)

# ─── 2. LOAD TRAINING DATA ──────────────────────────────────────────────────
X = pd.read_csv('Inputs.txt')
y = pd.read_csv('Label.txt', header=None, names=['label'])['label']
X = X.fillna(X.median())
print(f"Training data: {X.shape[0]} samples, {X.shape[1]} features")
print(f"Label distribution: {y.value_counts().to_dict()}")

# ─── 3. TRAIN-TEST SPLIT ────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {len(X_train)} | Test: {len(X_test)}")

# Scale for Logistic Regression
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ─── 4. TRAIN 3 CLASSIFIERS ─────────────────────────────────────────────────
models = {
    'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'Gradient Boosting':   GradientBoostingClassifier(n_estimators=100, learning_rate=0.1,
                                                       max_depth=4, random_state=42),
    'Logistic Regression': LogisticRegression(C=1.0, max_iter=1000, random_state=42),
}

results = {}
best_auc  = -1
best_name = None
best_model = None

print("\n" + "="*60)
print("MODEL COMPARISON RESULTS")
print("="*60)

for name, model in models.items():
    if name == 'Logistic Regression':
        model.fit(X_train_sc, y_train)
        y_pred  = model.predict(X_test_sc)
        y_proba = model.predict_proba(X_test_sc)[:, 1]
        cv_scores = cross_val_score(model, scaler.transform(X), y, cv=5, scoring='accuracy')
    else:
        model.fit(X_train, y_train)
        y_pred  = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')

    acc     = accuracy_score(y_test, y_pred)
    auc     = roc_auc_score(y_test, y_proba)
    cv_mean = cv_scores.mean()
    cv_std  = cv_scores.std()

    results[name] = {
        'accuracy': round(float(acc), 4),
        'auc':      round(float(auc), 4),
        'cv_mean':  round(float(cv_mean), 4),
        'cv_std':   round(float(cv_std), 4),
    }

    print(f"\n{name}:")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  ROC-AUC  : {auc:.4f}")
    print(f"  CV Acc   : {cv_mean:.4f} ± {cv_std:.4f}")
    print(classification_report(y_test, y_pred, target_names=['Unburned', 'Burned']))

    if auc > best_auc:
        best_auc  = auc
        best_name = name
        best_model = model

print(f"\n>>> Best model: {best_name} (AUC={best_auc:.4f})")

# Save model comparison JSON for the web app
with open('static/model_comparison.json', 'w') as f:
    json.dump({'results': results, 'best': best_name}, f, indent=2)
print("Saved static/model_comparison.json")

# Save best model, scaler, and feature list
joblib.dump(best_model, 'models/best_model_RF.pkl')
joblib.dump(scaler, 'parameters/scaler.pkl')
joblib.dump(feature_cols, 'parameters/feature_cols.pkl')
print(f"Saved best_model_RF.pkl ({best_name})")

# ─── 5. GENERATE SUSCEPTIBILITY MAP ─────────────────────────────────────────
print("\nGenerating susceptibility map from raster...")

with rasterio.open('static/Features_Raster_Bursa.tif') as src:
    data   = src.read().astype(float)
    bounds = src.bounds
    rows, cols_n = src.height, src.width
    print(f"Raster: {rows}x{cols_n}, bounds: {bounds}")

# Reorder bands to match training column order
data_reordered = data[band_read_order, :, :]
pixels = data_reordered.reshape(15, -1).T
pixels_df = pd.DataFrame(pixels, columns=feature_cols)

# Replace extreme nodata values, then fill with median
for col in pixels_df.columns:
    col_data = pixels_df[col]
    median_val = col_data[(col_data > -1e9) & (col_data < 1e9)].median()
    pixels_df[col] = col_data.where((col_data > -1e9) & (col_data < 1e9), median_val)
pixels_df = pixels_df.fillna(pixels_df.median())

# Predict
if best_name == 'Logistic Regression':
    susceptibility = best_model.predict_proba(scaler.transform(pixels_df))[:, 1]
else:
    susceptibility = best_model.predict_proba(pixels_df)[:, 1]

susc_map = susceptibility.reshape(rows, cols_n)
print(f"Susceptibility range: {susc_map.min():.3f} – {susc_map.max():.3f}, mean: {susc_map.mean():.3f}")

# ─── 6. SAVE MAP AS PNG ─────────────────────────────────────────────────────
cmap = mcolors.LinearSegmentedColormap.from_list(
    'wildfire', ['#2ecc71', '#f1c40f', '#e74c3c'], N=256
)

# Web overlay: pure pixel array — no margins, colorbar, or title.
# Leaflet imageOverlay stretches the entire image to the bounds, so any
# extra content (colorbar, axes) would appear mis-placed on the map.
plt.imsave('static/Susceptibility_Map_Bursa.png', susc_map,
           cmap=cmap, vmin=0, vmax=1, origin='upper')
print("Saved static/Susceptibility_Map_Bursa.png (overlay-ready)")

# Report figure: decorated version with colorbar + sample points
fig, ax = plt.subplots(figsize=(14, 8))
im = ax.imshow(susc_map, cmap=cmap, vmin=0, vmax=1,
               extent=[bounds.left, bounds.right, bounds.bottom, bounds.top],
               origin='upper', aspect='auto')
cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
cbar.set_label('Susceptibility (0=Low, 1=High)', fontsize=11)

with open('static/samplepoints.json') as f:
    pts = json.load(f)
ax.scatter([p['lng'] for p in pts if p['label']==0],
           [p['lat'] for p in pts if p['label']==0],
           c='#27ae60', s=6, alpha=0.6, label='Unburned (0)',
           zorder=5, edgecolors='white', linewidths=0.3)
ax.scatter([p['lng'] for p in pts if p['label']==1],
           [p['lat'] for p in pts if p['label']==1],
           c='#e74c3c', s=6, alpha=0.8, label='Burned (1)',
           zorder=6, edgecolors='white', linewidths=0.3)

ax.set_title(f'Wildfire Susceptibility Map — Bursa, Turkey (2025)\n'
             f'Best Model: {best_name} | AUC: {best_auc:.3f} | 15 Features',
             fontsize=13, fontweight='bold', pad=12)
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.legend(loc='lower right', fontsize=9)
plt.tight_layout()
plt.savefig('predictions/Susceptibility_Map_Bursa_Report.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved predictions/Susceptibility_Map_Bursa_Report.png (report figure)")

# Save susceptibility raster as GeoTIFF so the API can look up exact values
import rasterio
from rasterio.transform import from_bounds
transform = from_bounds(bounds.left, bounds.bottom, bounds.right, bounds.top, cols_n, rows)
for out_path in ['static/susceptibility_raster.tif', 'predictions/susceptibility_raster.tif']:
    with rasterio.open(
        out_path, 'w',
        driver='GTiff', height=rows, width=cols_n,
        count=1, dtype='float32', crs='EPSG:4326', transform=transform
    ) as dst:
        dst.write(susc_map.astype('float32'), 1)
print("Saved susceptibility_raster.tif to static/ and predictions/")

# Save actual bounds for reference
with open('static/map_bounds.json', 'w') as f:
    json.dump({'north': bounds.top, 'south': bounds.bottom,
               'east': bounds.right, 'west': bounds.left,
               'best_model': best_name, 'best_auc': round(best_auc, 4)}, f, indent=2)
print("Saved static/map_bounds.json")
print("\nAll done!")
