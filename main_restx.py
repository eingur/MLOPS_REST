import os
import sys
from flask import Flask, request, jsonify
import flask
from flask_restx import Api, Resource, fields, abort
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
api = Api()
api.init_app(app)

main_path = ''#os.getcwd()+'/'


def get_path(path, id):
    if id == 1:
        return main_path + 'models/' + 'rf.pkl'
    elif id == 2:
        return main_path + 'models/' + 'lr.pkl'
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

    # def post(self, id):
    #     #
    #     # X_train = pd.read_csv('train.csv')
    #     # y_train = pd.read_csv('y_train.csv')
    #     data = request.args.get('train_data')
    #     target = request.args.get('train_target')
    #
    #     if id !=1 and id !=2:
    #         return 'Nice try, bro'
    #     else:
    #         try:
    #             query = request.json
    #             print(query)
    #             if id == 1:
    #                 type = 'rf.pkl'
    #                 model = RandomForestClassifier(**query)
    #             elif id == 2:
    #                 type = 'lr.pkl'
    #                 model = LogisticRegression(**query)
    #             model.fit(train_data,train_target)
    #             pickle.dump(model, open(main_path+type, 'wb'))
    #             return (f'model {type} refitting with parameters {query}')
    #         except:
    #             return jsonify({'trace': traceback.format_exc()})
    # def post(self, id):
    #     X_train = pd.read_csv('train.csv')
    #     y_train = pd.read_csv('y_train.csv')
    #     if id !=1 and id !=2:
    #         return 'Nice try, bro'
    #     else:
    #         try:
    #             query = request.json
    #             print(query)
    #             if id == 1:
    #                 type = 'rf.pkl'
    #                 model = RandomForestClassifier(**query)
    #             elif id == 2:
    #                 type = 'lr.pkl'
    #                 model = LogisticRegression(**query)
    #             model.fit(train_data,train_target)
    #             pickle.dump(model, open(main_path+type, 'wb'))
    #             return (f'model {type} refitting with parameters {query}')
    #         except:
    #             return jsonify({'trace': traceback.format_exc()})


if __name__ == '__main__':
    try:
        print(sys.argv[1])
        port = int(sys.argv[1])
    except:
        port = 12345
        print(main_path + 'models/' + 'cols.pkl')
        rnd_columns = joblib.load(main_path + 'models/' + 'cols.pkl')
        app.run(port=port, debug=True)
