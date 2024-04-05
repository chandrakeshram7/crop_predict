# Importing essential libraries and modules
from flask import Flask, jsonify, request
import requests
import pandas as pd
import io
import joblib
from flask_cors import CORS
import numpy as np

crop_recommendation_model_path = 'crop_app'
crop_recommendation_model = joblib.load(crop_recommendation_model_path)

df = pd.read_csv('msp.csv')
app = Flask(__name__)
CORS(app, methods=['GET', 'POST', 'OPTIONS'])


@ app.route('/')
def home():
    return "Server is running on port 5000"




@ app.route('/predict', methods=['POST'])
def crop_prediction():
        crop_profile = request.get_json()
        print("Received farmer profile:", crop_profile)
    
        # N = 23
        # P = 23
        # K = 34
        # ph = 4
        # rainfall = 123.32
        # temperature = 34.33
        # humidity = 22.1
        values = [float(value) for value in crop_profile.values()]
        input_array = np.array(values)

        my_prediction = crop_recommendation_model.predict(input_array.reshape(1,-1))
        final_prediction = my_prediction[0]

        return jsonify({'prediction' : final_prediction })

@app.route('/msp')
def get_msp():
    data = request.get_json()
    crop_name = data.get('crop_name')
    # crop_name ="Jowar"
    if crop_name is None:
        return jsonify({'error': 'Crop name parameter is missing'}), 400
    
    crop_data = df[df['crop_name'] == crop_name.title()]  # Filter data for the specified crop name
    
    if crop_data.empty:
        return jsonify({'error': 'Crop not found'}), 404
    
    msp_dict = {
        'Crop': crop_name.title(),
        'Varieties': crop_data['variety'].tolist(),
        'MSP': {
            '2021-22': crop_data['2021-22'].tolist(),
            '2022-23': crop_data['2022-23'].tolist(),
            '2023-24': crop_data['2023-24'].tolist()
        }
    }
    
    return jsonify(msp_dict)

    



# ===============================================================================================
if __name__ == '__main__':
    app.run(debug=False)
