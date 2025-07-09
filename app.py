from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pickle

# Load trained model and scaler
model = pickle.load(open('linear_regression_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    sqft = data.get('sqft')
    bed = data.get('bed')
    bath = data.get('bath')
    input_data = np.array([[sqft, bed, bath]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    return jsonify({'predicted_price': round(float(prediction), 2)})

if __name__ == '__main__':
    app.run(debug=True)
