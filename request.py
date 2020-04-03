import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url, json={'type of logement_raw': 1,
                             'ville_raw': 1,
                             'quartier_raw': 0, 
                             'room_nb_raw': 2,
                             'bedroom_nb_raw': 5,
                             'surface_raw': 50})

print(r.json())
