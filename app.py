from __future__ import division, print_function
from flask import Flask, redirect, url_for, request, render_template

from werkzeug.utils import secure_filename
import numpy as np
import pickle
import os
import cv2

app = Flask(__name__)

CATEGORIES = ["FRAUDULENT","ORIGINAL"]

signatureRF = pickle.load(open('signature-rf.pkl','rb'))

def model_predict(img_path, model):
    
    def random_forest_predictions(testX, signatureRF):
        predictions = signatureRF.predict(testX)    
        return predictions

    # Read image
    img_array = cv2.imread(img_path)

    # Resize
    img_array = cv2.resize(img_array,(100,100))

    # Convert to grayscale
    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)

    # Flatten
    new_array = img_array.reshape(10000)

    # Prediction
    predictedIndex = random_forest_predictions([new_array], signatureRF)

    preds = CATEGORIES[signatureRF.classes_[predictedIndex[0]]]

    print("THE SIGNATURE is",preds)
    return preds
    

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if username == 'admin' and password == 'admin':
            return redirect(url_for('predict'))
        else:
            return render_template('login.html', error='Invalid Credentials')

    return render_template('login.html', error=None)


@app.route('/predict', methods=['GET', 'POST'])
def predict():

    prediction = None

    if request.method == 'POST':

        f = request.files['file']

        basepath = os.path.dirname(__file__)

        file_path = os.path.join(
            basepath,
            'uploads',
            secure_filename(f.filename)
        )

        f.save(file_path)

        preds = model_predict(file_path, signatureRF)

        result = preds

        if result:
            return redirect(url_for('result', result=result))

    return render_template('predict.html', prediction=prediction)


@app.route('/result')
def result():

    res = request.args.get('result')

    if res == 'ORIGINAL':
        bg_image = '../static/img/p2.jpg'
    else:
        bg_image = '../static/img/p5.jpg'

    print("FINAL OUTPUT:",res)

    return render_template(
        'result.html',
        result=res,
        bg_image=bg_image
    )


@app.route('/logout')
def logout():
    return redirect(url_for('login'))


if __name__ == '__main__':
    app.run(debug=True, port=5001)