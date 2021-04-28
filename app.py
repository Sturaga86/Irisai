import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('IrisPredict.model', 'rb'))
# ohe = pickle.load(open('StateEncoder.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    slength = request.form['slength']
    swidth = request.form['swidth']
    plength = request.form['plength']
    pwidth = request.form['pwidth']
    finalFeatures = np.array([[slength,swidth,plength,pwidth]])
    prediction = model.predict(finalFeatures)

    return render_template('index.html', prediction_text='Predicted Results: {}'.format(prediction))


if __name__ == "__main__":
    app.run(debug=True)