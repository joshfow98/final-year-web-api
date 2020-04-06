from flask import request, current_app, jsonify
from flask_api import FlaskAPI
import json
import pandas as pd
import pickle

app = FlaskAPI(__name__)
app.model = pickle.load(open('model.sav', 'rb'))

@app.route('/predictSingle', methods=['GET'])
def predictSingle():
    instance = parse_single_instance(request.get_json())
    return predict_single_instance(instance)

@app.route('/predictMultiple', methods=['GET'])
def predictMultiple():
    instances = parse_multiple_instances(request.get_json())
    return predict_multiple_instances(instances)

def parse_single_instance(jsonData):
    #Use a try catch to user the typ='series'
    data_frame = pd.read_json(json.dumps(jsonData), orient='columns', typ='series')
    return data_frame.values

def parse_multiple_instances(jsonData):
    data_frame = pd.read_json(json.dumps(jsonData), orient='columns')
    return data_frame.values

def predict_single_instance(instance):
    # Needs to be a 2D array to pass into model.predict which it is not as only one instance has been passed in
    instance = make_2D_array(instance)
    result = predict(instance)
    return parse_to_json(result)

def make_2D_array(instances):
    return instances.reshape(1, -1)

def predict_multiple_instances(instances):
    result = current_app.model.predict(instances)
    return parse_to_json(result)

def predict(instances):
    return current_app.model.predict(instances)

def parse_to_json(result):
    return jsonify(result.tolist())

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')