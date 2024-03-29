# Importing essential libraries and modules
from flask import Flask, jsonify
import requests
import io
import joblib
crop_recommendation_model_path = 'crop_app'
crop_recommendation_model = joblib.load(crop_recommendation_model_path)

app = Flask(__name__)


@ app.route('/')
def home():
    return "Server is running on port 5000"




@ app.route('/predict')
def crop_prediction():
    
        N = 23
        P = 23
        K = 34
        ph = 4
        rainfall = 123.32
        temperature = 34.33
        humidity = 22.1
       
        data = [[N, P, K, temperature, humidity, ph, rainfall]]
        my_prediction = crop_recommendation_model.predict(data)
        final_prediction = my_prediction[0]

        return jsonify({'prediction ' : final_prediction })

        
#This is just 


# ===============================================================================================
if __name__ == '__main__':
    app.run(debug=False)
