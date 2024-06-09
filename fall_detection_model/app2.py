from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
import pickle

app = Flask(__name__)

# Load the model architecture
with open('sensor_model_architecture.pkl', 'rb') as pkl_file:
    model_pkl = pickle.load(pkl_file)

model = tf.keras.models.model_from_json(model_pkl)

# Load the model weights
model.load_weights('sensor_model.weights.h5')

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Load the scaler and label encoder
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    if 'readings' not in data:
        return jsonify({'error': 'No readings provided'}), 400
    
    readings = data['readings']
    
    # Convert readings to a numpy array and scale
    readings_scaled = scaler.transform(np.array(readings).reshape(-1, 6))

    # Ensure the readings array has the correct shape
    if readings_scaled.shape[0] != 10:
        return jsonify({'error': 'Readings must contain 10 sequences of 6 features each'}), 400

    # Reshape to match the input shape expected by the model
    readings_seq = readings_scaled.reshape(1, 10, 6)

    # Predict using the model
    prediction = model.predict(readings_seq)
    predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])

    return jsonify({'prediction': predicted_label[0]})

if __name__ == '__main__':
    app.run(debug=True)
