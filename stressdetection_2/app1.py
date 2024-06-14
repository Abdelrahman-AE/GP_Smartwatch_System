import pandas as pd
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('stress_model22.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.json
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)
    return jsonify({'stress_level': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)

