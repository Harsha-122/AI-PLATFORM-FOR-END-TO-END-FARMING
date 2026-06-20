# ============================================================
#  app.py  —  Agri-Smart Flask Application
#  Run: python app.py   →   Open: http://127.0.0.1:5000
# ============================================================
from flask import Flask, render_template, request, jsonify
import joblib, numpy as np, os

app = Flask(__name__)

# ── Load all 5 models at startup ──────────────────────────
MODELS = {}
model_files = {
    'fertilizer' : 'models/fertilizer_model.pkl',
    'crop'        : 'models/crop_model.pkl',
    'fertility'   : 'models/fertility_model.pkl',
    'irrigation'  : 'models/irrigation_model.pkl',
    'disease'     : 'models/disease_model.pkl',
}
for name, path in model_files.items():
    if os.path.exists(path):
        MODELS[name] = joblib.load(path)
        print(f"✅ Loaded: {name}")
    else:
        print(f"❌ Missing: {path} — run train_models.py first!")

SOIL_TYPES = ['Sandy', 'Loamy', 'Black', 'Red', 'Clayey']
CROP_TYPES = ['Maize', 'Sugarcane', 'Cotton', 'Tobacco', 'Paddy',
              'Barley', 'Wheat', 'Millets', 'Oil seeds', 'Pulses', 'Ground Nuts']

# ── Pages ─────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/fertilizer')
def fertilizer_page():
    return render_template('fertilizer.html',
                           soil_types=SOIL_TYPES, crop_types=CROP_TYPES)

@app.route('/crop')
def crop_page():
    return render_template('crop.html')

@app.route('/fertility')
def fertility_page():
    return render_template('fertility.html')

@app.route('/irrigation')
def irrigation_page():
    return render_template('irrigation.html',
                           soil_types=SOIL_TYPES, crop_types=CROP_TYPES)

@app.route('/disease')
def disease_page():
    return render_template('disease.html', crop_types=CROP_TYPES)


# ── Prediction APIs ────────────────────────────────────────

@app.route('/predict/fertilizer', methods=['POST'])
def predict_fertilizer():
    try:
        d = request.get_json()
        pkg = MODELS['fertilizer']
        soil_enc = pkg['le_soil'].transform([d['soil_type']])[0]
        crop_enc = pkg['le_crop'].transform([d['crop_type']])[0]
        X = [[float(d['temperature']), float(d['humidity']),
              float(d['moisture']), soil_enc, crop_enc,
              float(d['nitrogen']), float(d['potassium']),
              float(d['phosphorous'])]]
        X_sc = pkg['scaler'].transform(X)
        pred = pkg['model'].predict(X_sc)[0]
        proba = pkg['model'].predict_proba(X_sc)[0].max()
        label = pkg['le_target'].inverse_transform([pred])[0]
        return jsonify({'result': label, 'confidence': round(proba*100, 1)})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/predict/crop', methods=['POST'])
def predict_crop():
    try:
        d = request.get_json()
        pkg = MODELS['crop']
        X = [[float(d['nitrogen']), float(d['phosphorous']),
              float(d['potassium']), float(d['temperature']),
              float(d['humidity']), float(d['ph']),
              float(d['rainfall'])]]
        X_sc = pkg['scaler'].transform(X)
        pred = pkg['model'].predict(X_sc)[0]
        proba = pkg['model'].predict_proba(X_sc)[0].max()
        label = pkg['le_target'].inverse_transform([pred])[0]
        return jsonify({'result': label, 'confidence': round(proba*100, 1)})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/predict/fertility', methods=['POST'])
def predict_fertility():
    try:
        d = request.get_json()
        pkg = MODELS['fertility']
        X = [[float(d['nitrogen']), float(d['phosphorous']),
              float(d['potassium']), float(d['ph']),
              float(d['organic_matter']), float(d['moisture'])]]
        X_sc = pkg['scaler'].transform(X)
        pred = pkg['model'].predict(X_sc)[0]
        proba = pkg['model'].predict_proba(X_sc)[0].max()
        label = pkg['le_target'].inverse_transform([pred])[0]
        return jsonify({'result': label, 'confidence': round(proba*100, 1)})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/predict/irrigation', methods=['POST'])
def predict_irrigation():
    try:
        d = request.get_json()
        pkg = MODELS['irrigation']
        soil_enc = pkg['le_soil'].transform([d['soil_type']])[0]
        crop_enc = pkg['le_crop'].transform([d['crop_type']])[0]
        X = [[float(d['temperature']), float(d['humidity']),
              float(d['moisture']), float(d['rainfall']),
              soil_enc, crop_enc]]
        X_sc = pkg['scaler'].transform(X)
        pred = pkg['model'].predict(X_sc)[0]
        proba = pkg['model'].predict_proba(X_sc)[0].max()
        label = pkg['le_target'].inverse_transform([pred])[0]
        return jsonify({'result': label, 'confidence': round(proba*100, 1)})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/predict/disease', methods=['POST'])
def predict_disease():
    try:
        d = request.get_json()
        pkg = MODELS['disease']
        crop_enc = pkg['le_crop'].transform([d['crop_type']])[0]
        X = [[float(d['temperature']), float(d['humidity']),
              float(d['rainfall']), float(d['moisture']), crop_enc]]
        X_sc = pkg['scaler'].transform(X)
        pred = pkg['model'].predict(X_sc)[0]
        proba = pkg['model'].predict_proba(X_sc)[0].max()
        label = pkg['le_target'].inverse_transform([pred])[0]
        return jsonify({'result': label, 'confidence': round(proba*100, 1)})
    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True)