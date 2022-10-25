import os
import sys
from flask import Flask, request, jsonify
import joblib
import traceback
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import pathlib
import pickle
app = Flask(__name__)
# api.model()

main_path = os.getcwd()+'\\'

@app.route('/', methods=['GET'])
def avail():
    return "1: Random Forest Classifier, 2: Logistic Regression"

@app.route('/<int:id>', methods=['DELETE'])
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

@app.route('/prediction/<int:id>', methods=['POST'])
def predict(id):
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

@app.route('/refit/<int:id>', methods=['POST','GET'])
def refit(id):
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
