from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = joblib.load("habitability_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    features = np.array([[
        data['planet_radius'],
        data['planet_mass'],
        data['orbital_period'],
        data['equilibrium_temp'],
        data['stellar_temp']
    ]])

    scaled_features = scaler.transform(features)
    prediction = model.predict(scaled_features)[0]
    probability = model.predict_proba(scaled_features)[0][1]

    result = {
        "habitability": "Habitable" if prediction == 1 else "Non-Habitable",
        "confidence": round(probability * 100, 2)
    }

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
