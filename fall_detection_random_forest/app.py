from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the saved model, scaler, and label encoder
model = joblib.load('model_random_forest.pkl')
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    new_value = np.array(data['input']).reshape(1, -1)
    
    # Scale the new value
    new_value_scaled = scaler.transform(new_value)
    
    # Predict the label for the new data point
    new_prediction = model.predict(new_value_scaled)
    predicted_label = label_encoder.inverse_transform(new_prediction)
    
    return jsonify({'predicted_label': predicted_label[0]})

if __name__ == '__main__':
    app.run(debug=True)

# passing the input to postman (for example) is like this:
# {
#     "input": [4.79, 2.47, -2.9, 17.58, -3.91, -16.3]
# }
