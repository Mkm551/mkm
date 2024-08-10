#importing required libraries
import numpy as np
import pandas as pd
from sklearn import metrics 
import pickle
from feature import FeatureExtraction
import os
from flask import Flask, request, render_template, session, redirect, url_for
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

file = open("pickle/model.pkl","rb")
gbc = pickle.load(file)
file.close()


@app.route("/")
def home():
    return render_template("index.html")

@app.route('/index')
def index():

    return render_template("index.html")


@app.route("/error", methods=['GET', 'POST'])
def error():

    if request.method == "POST":

        url = request.form["url"]
        obj = FeatureExtraction(url)
        x = np.array(obj.getFeaturesList()).reshape(1,30) 
        y_pro_non_phishing = gbc.predict_proba(x)[0,1]
        return render_template('error.html',xx =round(y_pro_non_phishing,2),url=url )
    return render_template("error.html", xx =-1)


if __name__ == "__main__":
    
    app.run(debug=True)
