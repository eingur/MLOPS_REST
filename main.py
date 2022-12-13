import os
import sys
from flask import Flask, request, jsonify
import flask
from flask_restx import Api, Resource, fields, abort, reqparse
import joblib
import traceback
import pandas as pd
import json
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import pathlib
import pickle
from fit_and_save_model import fit_model, load_model
app = Flask(__name__)
api = Api()
api.init_app(app)

main_path = ''#os.getcwd()+'/'


def get_path(path, id):
    if id == 1:
        return main_path + 'models/rf/' + 'rf.pkl'
    elif id == 2:
        return main_path + 'models/lr/' + 'lr.pkl'
    else:
        return None


@api.route('/')
class Descr(Resource):
    def get(self):
        """Get available models"""
        return jsonify({'Models': {1: "Random Forest Classifier", 2: "Logistic Regression"}})

    @api.doc(params={'id': {'description': '1 :  rf, 2 : LR', 'type': int, 'default': 1}},
             responses={200: 'Request is correct', 400: 'Bad request', 404:"File {file_path} doesn't exist"})
    def delete(self):
        """Delete model by model id"""
        path = get_path(main_path, request.args['id'])
        if os.path.isfile(path):
            os.remove(path)
            return jsonify({'status': 'file removed', 'file': path}), 200
        abort(404, "File {} doesn't exist".format(path))

parser = reqparse.RequestParser()
parser.add_argument('data', type=json.loads, help='data for predict')
parser.add_argument('parameters',json.loads)

@api.route('/prediction/<int:id>')
class Prediction(Resource):
    @api.doc(params={'id': {'description': '1 :  rf, 2 : LR', 'type': int, 'default': 1}},
             responses={200: 'Request is correct', 400: 'Bad request'})
    def post(self, id):
        """Get prediction with posted data"""
        args = parser.parse_args()#request.json
        params = args['parameters']
        model = load_model(id,params)
        # path = get_path(main_path, id)
        # if os.path.isfile(path):
        #     model = joblib.load(path)
        # else:
        if not model:
            return "model with params {} isn't fitted or mismatch.".format(params), 404
        else :
            try:
                data_ = args['data']#request.json
                print(data_)
                query = pd.DataFrame(data_)
                query.columns = rnd_columns
                predict = list(model.predict(query))
                return jsonify({'prediction': predict})
            except:
                return jsonify({'trace': traceback.format_exc()})



@api.route('/refit/<int:id>')
class Fitting(Resource):
#   @api.doc(params = {'id':{'type':int, 'default':1},})
    def post(self, id):
        args = request.json#(force=True)
        train_data = pd.DataFrame.from_dict(json.loads(args['train_data']), orient="columns")
        train_target = pd.DataFrame.from_dict(json.loads(args['train_target']), orient="index")
        params = args['params']

        flag = fit_model(train_data, train_target, id, params)
        if flag == 1:
            return jsonify({'status': 'fitted', 'model id': id, 'parameters': params})
        else:
            return jsonify({'trace': traceback.format_exc()})

 
if __name__ == '__main__':
    try:
        print(sys.argv[1])
        port = int(sys.argv[1])
    except:
        port = 5000
        print(main_path + 'models/' + 'cols.pkl')
        rnd_columns = joblib.load(main_path + 'models/' + 'cols.pkl')
        app.run(host = '0.0.0.0', port=port, debug=True)
