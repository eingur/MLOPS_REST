import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib
import pandas as pd
import pickle
import io
import psycopg2
from sqlalchemy import select, insert, update,delete
from db.db import connection, Weights


main_path = os.getcwd()+'\\models\\'

class clf():
    def __init__(self,id : int, hyperparams : dict = {}):
        self.id = id
        if self.id == 1:
            self.model = RandomForestClassifier(**hyperparams)
        elif self.id == 2:
            self.model = LogisticRegression(**hyperparams)
        else:
            raise ValueError('not determined type of model.') 
    
def get_id(id):
    if id == 1:
        return 'rf'
    elif id == 2:
        return 'lr'
    else:
        return None

def load_all_models():
    statement = select([Weights.model])
    result = connection.execute(statement).fetchall()
    return [v[0] for v in result]

def delete_models(model_type : str):
    result = delete(Weights).where(Weights.retired.notilike(f'%{model_type}%'))
    connection.execute(result)
    return 1

def load_model(logs : str):
    statement = select(Weights).where(Weights.model == logs)
    result = connection.execute(statement).fetchone() 
    if result:
        result = connection.execute(statement)
        for _, parameters in result:
            return pickle.loads(parameters)
    else:
        return False

def predict_model(id : int, hyperparams : dict, data):
    model_type = get_id(id)
    logs = '_'.join([str(k)+'_'+str(v) for k, v in hyperparams.items()])
    logs = model_type + logs
    model = load_model(logs)
    return model.predict(data)

def fit_model(train_data, train_target, id : int, hyperparams : dict, status_value : dict):

    logs = '_'.join([str(k)+'_'+str(v) for k, v in hyperparams.items()])
    model_type = get_id(id)
    classifier = clf(hyperparams, id)
    logs = model_type + '_' + logs
    classifier.model_.fit(train_data, train_target)
    row = pickle.dumps(classifier.model_)
    statement = select(Weights).where(Weights.model == logs)
    result = connection.execute(statement).fetchone() 
    if (result):
        statement = update(Weights).where(Weights.model == logs).values(parameters = row)
        connection.execute(statement) 
        status_value['status'] = 'refited'
    else:
        statement = insert(Weights).values(model = logs, parameters = row)
        connection.execute(statement)
        status_value['status'] = 'fited'
    return 1
