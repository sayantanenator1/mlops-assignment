from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the model
model = joblib.load('model.joblib')

@app.route('/')
def home():
    return "Welcome to the Flask App!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    print(f"Received data: {data}")  # Print the received data
    features = np.array(data['features']).reshape(1, -1)
    print(f"Features: {features}")  # Print the features
    prediction = model.predict(features)
    print(f"Prediction: {prediction}")  # Print the prediction
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
