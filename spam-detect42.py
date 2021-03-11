#!/usr/bin/env python
import os
import time
import joblib 
import numpy as np

import custom.deploy_models as dp
from flask import Flask, render_template, url_for, request

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict_fun(): 

    XGboost_mod1_PATH = os.path.join("data", 
                                     "5_deployment", 
                                     "XGboost_mod1.joblib")
    
    with open(XGboost_mod1_PATH, 'rb') as f:
        XGboost_mod1 = joblib.load(f)

    if request.method == 'POST': 

        message = request.form['message']

        try:
            new_data = np.array([message])
        except:
            new_data = [message]
        try:
            X_test_processed = dp.transform_newdata(new_data)
            y_pred = XGboost_mod1.predict(X_test_processed)
        except:
            y_pred = np.array([0])

    return render_template('result.html', 
                            prediction = y_pred[0])   

if __name__ == '__main__':
	app.run(debug=True)
    #app.run(debug=True, host='0.0.0.0', port=80)
	
