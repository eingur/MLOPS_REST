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
from fit_and_save_model import fit_model, load_model, load_all_models, delete_models, get_id
app = Flask(__name__)
api = Api()
api.init_app(app)

main_path = ''#os.getcwd()+'/'


@api.route('/models')
class Descr(Resource):
    def get(self):
        """Get available models"""
        return jsonify({'Models': {1: "Random Forest Classifier", 2: "Logistic Regression"}})

@api.route('/models_in_db')
class Descr(Resource):
    def get(self):
        """Get fitted models in database"""
        return jsonify({'Models': load_all_models()})

    @api.doc(params={'id': {'description': '1 :  rf, 2 : LR', 'type': int, 'default': 1}},
             responses={200: 'Request is correct', 400: 'Bad request', 404:"File {file_path} doesn't exist"})
    def delete(self):
        """Delete model by model_id"""
        flg = delete_models(get_id(id))
        if flg:
            return jsonify({'status': 'models removed from database', 'model type': get_id(id)}), 200
        abort(404, "Models {} doesn't exist in database".format(get_id(id)))


parser = reqparse.RequestParser()
parser.add_argument('data', type = json.loads,location='json', required = True, help='data for predict')
parser.add_argument('parameters',type = json.loads,location='json', required = True)

@api.route('/prediction/<int:id>')
class Prediction(Resource):
    @api.doc(params={'id': {'description': '1 :  RandomForestClassifier, 2 : LogisticRegression', 'type': int, 'default': 1}},
             responses={200: 'Request is correct', 400: 'Bad request'})
    def post(self, id):
        """Get prediction with posted data"""
        args = request.json#parser.parse_args()#request.json
        params = args['parameters']
        model = load_model(id,params)

        if not model:
            return "model with params {} isn't fitted or mismatch.".format(params), 404
        else :
            try:
                data_ = pd.DataFrame.from_dict(json.loads(args['data']),orient='columns')
                predict = model.predict(data_)
                return jsonify({'prediction': predict.tolist()})
            except:
                return jsonify({'trace': traceback.format_exc()}), 400



@api.route('/refit/<int:id>')
class Fitting(Resource):
#   @api.doc(params = {'id':{'type':int, 'default':1},})
    def post(self, id):
        args = request.json
        train_data = pd.DataFrame.from_dict(json.loads(args['train_data']), orient="columns")
        train_target = pd.DataFrame.from_dict(json.loads(args['train_target']), orient="columns")
        params = args['parameters']
        status_value = {'status':None}
        flag = fit_model(train_data, train_target, id, params, status_value)
        if flag == 1:
            return jsonify({'status': status_value['status'], 'model id': get_id(id), 'parameters': params})
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
