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

def get_id(id):
    if id == 1:
        return 'rf'
    else:
        return 'lr'

def load_all_models():
    statement = select([Weights.model])
    result = connection.execute(statement).fetchall()
    return [v[0] for v in result]

def delete_models(model_type):
    result = delete(Weights).where(Weights.retired.notilike(f'%{model_type}%'))
    connection.execute(result)
    return 1

def load_model(id, hyperparams):
    model_type = 'rf' if id == 1 else 'lr'
    logs = '_'.join([str(k)+'_'+str(v) for k, v in hyperparams.items()])
    logs = model_type + logs
    statement = select(Weights).where(Weights.model == logs)
    result = connection.execute(statement).fetchone() 
    if result:
        result = connection.execute(statement)
        for _, parameters in result:
            model_ = pickle.loads(parameters)
    else:
        return False
    return model_

def fit_model(train_data, train_target, id, hyperparams, status_value):

    logs = '_'.join([str(k)+'_'+str(v) for k, v in hyperparams.items()])
    try:
        if id == 1:
            model_type = 'rf'
            model_ = RandomForestClassifier(**hyperparams)
        elif id == 2:
            model_type = 'lr'
            model_ = LogisticRegression(**hyperparams)
        else:
            return 0
    except ValueError:
        return 0
    logs = model_type + '_' + logs
    model_.fit(train_data, train_target)
    row = pickle.dumps(model_)
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
