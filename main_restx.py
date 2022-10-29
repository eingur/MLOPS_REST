import os
import sys
from flask import Flask, request, jsonify
import flask
from flask_restx import Api, Resource,fields
import joblib
import traceback
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import pathlib
import pickle
app = Flask(__name__)
api = Api()
api.init_app(app)

main_path = os.getcwd()+'\\'

@api.route('/')
class Descr(Resource):
    def get(self):
        return "1: Random Forest Classifier, 2: Logistic Regression"

@api.route('/<int:id>')
class del_model(Resource):
    def delete(id):
        if id == 1:
            path = main_path + 'rf.pkl'
        elif id == 2:
            path = main_path + 'lr.pkl'
        else:
            return "Nice try, bro \n again..."
        if os.path.isfile(path):
            os.remove(path)
            return f'{path} deleted!!!'
        else:
            return "No such file :c"

@api.route('/prediction/<int:id>')
class Prediction(Resource):
    def post(self, id):
        if id == 1:
            path = main_path + 'rf.pkl'
            if os.path.isfile(path):
                model = joblib.load(path)
            else:
                return "Fit at first"
        elif id == 2:
            path = main_path + 'lr.pkl'
            if os.path.isfile(path):
                model = joblib.load(path)
            else:
                return "Fit at first"
        else:
            return 'Nice try, bro'

        if model:
            try:
                data_ = request.json
                print(data_)
                query = pd.DataFrame(data_)
                query.columns = rnd_columns
                predict = list(model.predict(query))
                return jsonify({'prediction': str(predict)})
            except:
                return jsonify({'trace': traceback.format_exc()})
        else:
            print ('Model not good')
            return ('Model is not good')

@api.route('/refit/<int:id>')
class Fitting(Resource):
    def get(self, id):
        X_train = pd.read_csv('train.csv')
        y_train = pd.read_csv('y_train.csv')
        if id !=1 and id !=2:
            return 'Nice try, bro'
        else:
            try:
                if id == 1:
                    type = 'rf.pkl'
                    model = RandomForestClassifier()
                elif id == 2:
                    type = 'lr.pkl'
                    model = LogisticRegression()
                model.fit(X_train,y_train)
                pickle.dump(model, open(main_path+type, 'wb'))
                return (f'model {type} fit')
            except:
                return jsonify({'trace': traceback.format_exc()})

    def post(self, id):
        X_train = pd.read_csv('train.csv')
        y_train = pd.read_csv('y_train.csv')
        if id !=1 and id !=2:
            return 'Nice try, bro'
        else:
            try:
                query = request.json
                print(query)
                if id == 1:
                    type = 'rf.pkl'
                    model = RandomForestClassifier(**query)
                elif id == 2:
                    type = 'lr.pkl'
                    model = LogisticRegression(**query)
                model.fit(X_train,y_train)
                pickle.dump(model, open(main_path+type, 'wb'))
                return (f'model {type} refitting with parameters {query}')
            except:
                return jsonify({'trace': traceback.format_exc()})

if __name__ == '__main__':
    try:
        print(sys.argv[1])
        port = int(sys.argv[1])
    except:
        port = 12345
        rnd_columns = joblib.load(main_path + 'cols.pkl') # Load “rnd_columns.pkl”
        app.run(port=port, debug=True)
