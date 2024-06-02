from flask import Flask, request, jsonify, render_template
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Muat model yang telah disimpan
model = load_model('population.h5')

# Fungsi untuk normalisasi dan denormalisasi
def normalize(x, min_x, max_x):
    return (x - min_x) / (max_x - min_x)

def denormalize(x, min_y, max_y):
    return x * (max_y - min_y) + min_y

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # Mengambil data JSON dari request body
    year = float(data['year'])
    year_normalized = normalize(year, 2014, 2023)
    prediction_normalized = model.predict(np.array([[year_normalized]]))
    prediction = denormalize(prediction_normalized, 2833, 2896)
    return jsonify({'prediction': float(prediction)})

if __name__ == '__main__':
    app.run(debug=True)
