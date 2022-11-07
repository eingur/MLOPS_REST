import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib
import pandas as pd
import pickle
main_path = os.getcwd()+'\\models\\'


def fit_model(train_data, train_target, id, params):
    path = main_path
    try:
        if id == 1:
            type = 'rf'
            model = RandomForestClassifier(**params)
        elif id == 2:
            type = 'lr'
            model = LogisticRegression(**params)
        else:
            return 0
        model.fit(train_data, train_target)
        logs = '_'.join([str(k)+'_'+str(v) for k, v in params.items()])
        pickle.dump(model, open(path + type + '\\' + logs + '.pkl', 'wb'))
        return 1
    except:
        return 0
