import os
import sys
from flask import Flask, request, jsonify
import flask
from flask_restx import Api, Resource, fields, abort, marshal
import joblib
import traceback
import pandas as pd
import json
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import pathlib
import pickle
from fit_and_save_model import fit_model
app = Flask(__name__)
api = Api(title="MLOps. Flask with swagger.",
          description="Realization of 2 ML models.")
api.init_app(app)

main_path = os.getcwd()+'\\'


def get_path(path, id):
    if id == 1:
        return main_path + 'models\\' + 'rf.pkl'
    elif id == 2:
        return main_path + 'models\\' + 'lr.pkl'
    else:
        return None


@api.route('/')
class Descr(Resource):
    def get(self):
        """Get available models"""
        return jsonify({'Models': {1: "Random Forest Classifier", 2: "Logistic Regression"}})


@api.route('/Models/')
class DeleteModel(Resource):
    def get(self):
        return jsonify({'available models with rf': os.listdir(main_path + 'models\\'),
                        'available models with lr': os.listdir(main_path + 'models\\')})

    @api.doc(params={'id': {'description': '1 :  rf, 2 : LR', 'type': int, 'default': 1}},
             responses={200: 'Request is correct', 400: 'Bad request', 404:"File {file_path} doesn't exist"})
    def delete(self):
        """Delete model by model id"""
        path = get_path(main_path, request.args['id'])
        if os.path.isfile(path):
            os.remove(path)
            return jsonify({'status': 'file removed', 'file': path}), 200
        abort(404, "File {} doesn't exist".format(path))


# predict_model = api.model(
#     "Item",
#     {
#         "train_data": fields.String,
#         "train_target": fields.String,
#         "params": fields.String,
#     },
# )


@api.route('/prediction/<int:id>')
class Prediction(Resource):
    @api.doc(params={'id': {'description': '1 :  rf, 2 : LR', 'type': int, 'default': 1}},
             responses={200: 'Request is correct', 400: 'Bad request'})
    def post(self, id):
        """Get prediction with posted data"""
        path = get_path(main_path, id)
        if os.path.isfile(path):
            model = joblib.load(path)
        else:
            return "File {} doesn't exist".format(path), 404
        if model:
            try:
                data_ = request.json
                print(data_)
                query = pd.DataFrame(data_)
                query.columns = rnd_columns
                predict = list(model.predict(query))
                return jsonify({'prediction': predict})
            except:
                return jsonify({'trace': traceback.format_exc()})
        else:
            return "Mismatch parameters with model", 400


@api.route('/refit/<int:id>')
class Fitting(Resource):
    def post(self):
        """Fit and refit model, posted json has to include \" train_data \", \"test_data\", \"params\""""
        args = request.json
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
        port = 12345
        rnd_columns = joblib.load(main_path + 'models\\' + 'cols.pkl')
        app.run(port=port, debug=True)
