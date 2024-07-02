import pickle
from flask import Flask, request, url_for, jsonify, render_template
import numpy as np 
import pandas as pd
import os
# Load the trained model

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)


with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')



@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json  # Get JSON data from the request
        data = {k: float(v) for k, v in data.items()}  # Convert to float
        print("Processed data:", data)
        # Define feature names based on the dataset
        feature_names = ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
        
        # Create a DataFrame with the correct feature names
        input_df = pd.DataFrame([data], columns=feature_names)
        print("Created DataFrame:", input_df)
        # Scale the input data using the loaded scaler
        scaled_data = scaler.transform(input_df)
        print("Scaled Data:", scaled_data)
        # Make prediction using the loaded model
        prediction = model.predict(scaled_data)
        print("Predicted:", prediction)
        # Return prediction as JSON response
        return jsonify({'prediction': int(prediction[0])})
    
    except Exception as e:
        return jsonify({'error': str(e)})
    


if __name__ == '__main__':
    app.run(debug=True, port=os.getenv("PORT", default=5000), host='0.0.0.0')



    