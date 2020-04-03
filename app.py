import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template, url_for
import pickle
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler


app = Flask(__name__)
app.config.from_object('config')
standard_x = StandardScaler()
model = pickle.load(open('house_model.pkl', 'rb'))


@app.route('/')
def home():
    data = pd.read_csv('seloger-cleaned.csv', encoding="utf-8")
    villes = data['ville_raw'].dropna()
    villes = villes.sort_values()
    villes = villes.unique()
    quartiers = data['quartier_raw'].dropna()
    quartiers = quartiers.sort_values()
    quartiers = quartiers.unique()
    return render_template('index.html', villes = list(villes), quartiers = list(quartiers))


@app.route('/contact')
def contact():
    return render_template('contact.html')


@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    standard_value = standard_x.fit_transform(final_features)
    prediction = model.predict(standard_x.transform(standard_value))

    output = int(prediction[0])

    return render_template('index.html', prediction_text='Le prix estimé du bien est de {} €'.format(output))


@app.route('/predict_api', methods=['POST'])
def predict_api():
    '''
    For direct API calls through request
    '''
    data = request.get_json(force=True)
    standard_value = standard_x.fit_transform([np.array(list(data.values()))])
    prediction = model.predict(standard_x.transform(standard_value))

    output = int(prediction[0])
    return jsonify(output)


if __name__ == "__main__":
    app.run(debug=True)