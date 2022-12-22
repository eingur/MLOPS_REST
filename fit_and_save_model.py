import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib
import pandas as pd
import pickle
import io
import psycopg2
from sqlalchemy import select, insert, update

from db.db import connection, Weights


main_path = os.getcwd()+'\\models\\'


# def fit_model(train_data, train_target, id, params):
#     path = main_path
#     try:
#         if id == 1:
#             type = 'rf'
#             model = RandomForestClassifier(**params)
#         elif id == 2:
#             type = 'lr'
#             model = LogisticRegression(**params)
#         else:
#             return 0
#         model.fit(train_data, train_target)
#         logs = '_'.join([str(k)+'_'+str(v) for k, v in params.items()])
#         pickle.dump(model, open(path + type + '\\' + logs + '.pkl', 'wb'))
#         return 1
#     except:
#         return 0
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

def fit_model(train_data, train_target, id, hyperparams):

    logs = '_'.join([str(k)+'_'+str(v) for k, v in hyperparams.items()])
    # bytes_io = io.BytesIO()
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
    logs = model_type + logs
    model_.fit(train_data, train_target)
    # pickle.dump(model_, bytes_io)#, pickle.HIGHEST_PROTOCOL)
    raw = pickle.dumps(model_)#, pickle.HIGHEST_PROTOCOL)
    statement = select(Weights).where(Weights.model == logs)
    result = connection.execute(statement).fetchone() 
    if (result):
        # bytes_io.seek(0) 
        statement = update(Weights).values(model = logs, parameters = raw)
        # statement = update(Weights).values(model = logs, parameters = psycopg2.Binary(bytes_io))
        connection.execute(statement) 
    else:
        # bytes_io.seek(0) 
        statement = insert(Weights).values(model = logs, parameters = raw)
        # statement = insert(Weights).values(model = logs, parameters = psycopg2.Binary(bytes_io))
        connection.execute(statement)
    return 1
