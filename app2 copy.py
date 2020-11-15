import flask
import numpy as np
import pandas as pd
import sys
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder

model = None

app = flask.Flask(__name__)

def load_model(wine_type):

    with open('red_scaler.pkl', 'rb') as s:
        scaler = pickle.load(s)
    
    if wine_type == "Red":

        with open('red_scaler.pkl', 'rb') as s:
            scaler = pickle.load(s)
        with open('red_model.pkl', 'rb') as f: 
            model = pickle.load(f)

    elif wine_type == "White":

        with open('white_scaler.pkl', 'rb') as s:
            scaler = pickle.load(s)

        with open('white_model.pkl', 'rb') as f: 
            model = pickle.load(f)

    return model, scaler

def load_data(wine_type):
    if wine_type == "Red":

        wine_df = pd.read_csv("Red.csv")

    elif wine_type == "White":

        wine_df = pd.read_csv("White.csv")

    return wine_df

@app.route('/',methods=['GET', 'POST'])
def home():
    if request.method =='GET':
        return render_template('index2.html')

@app.route('/predict',methods=['GET', 'POST'])
def predict():

    if flask.request.method =='POST':

        color = flask.request.form['color']
        vintage= flask.request.form['vintage']
        appellation = flask.request.form['appellation']
        vineyards = flask.request.form['vineyards']
        
        inputV = np.array([color, vintage, appellation, vineyards]).reshape(1,4)
        wine_df = pd.DataFrame()
        model = []
        final_input = []

        if color == "0":

            model, scaler = load_model("Red")
            wine_df = load_data(wine_type= "Red")

        elif color == "1":
            
            model, scaler = load_model("White")
            wine_df = load_data(wine_type= "White")

        print(wine_df.shape, file=sys.stderr)

        wine_input = wine_df.loc[wine_df['vintage'] == int(vintage)]
        prev_input = wine_input

        if len(wine_input) > 0:
            wine_input = wine_input.loc[wine_df['appellation_' + appellation] == appellation]
        if len(wine_input) > 0:
            prev_input = wine_input
            wine_input = wine_input.loc[wine_df['vineyards_' + vineyards] == vineyards]
        if len(wine_input) > 0:
            final_input = wine_input # will return wine input
        else:
            final_input = prev_input

        temp = final_input.iloc[0].to_numpy().reshape(-1, 1)[1:]
        scaled_final_input = scaler.fit_transform(temp).reshape(1, -1)
        prediction = model.predict(scaled_final_input)
        
    return render_template(prediction[0])

if __name__ == "__main__":
    app.run(debug=True)
