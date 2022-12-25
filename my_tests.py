import pytest
import numpy as np
import pandas as pd
from fit_and_save_model import clf, predict_model, load_model
import pickle
from sklearn.exceptions import NotFittedError
import sklearn

test_params_lr = {'penalty':'l2','C':0.3}
test_params_rf = {{'max_depth': 5,'n_estimators':100}}
test_bad_params_lr = {'penalty':'l3','C':0.3}
test_bad_params_rf = {{'max_depth':'a','n_estimators':100}}
test_bad_params = {'penalty':'l3','n_estimators':100}

logs_bad = 'rf_n_estimators_100'
@pytest.fixture()
def load_model_replace_bad():
    yield pickle.load(open('tests/'+logs_bad+'.json','rb'))

logs = 'lr_penalty_l2_C_0.3'
@pytest.fixture()
def load_model_replace():
    yield pickle.load(open('tests/'+logs+'.json','rb'))



def test_init():
    data_test = pd.read_csv('datasets/data_train.csv')
    target_test = pd.read_csv('datasets/target_train.csv')
    with pytest.raises(ValueError):
        model = clf(1,test_bad_params_rf).model.fit(data, target)
    with pytest.raises(ValueError):
        model = clf(2, test__bad_params_lr).model.fit(data, target)
    
    with pytest.raises(ValueError):
        model = clf(3)

    with pytest.raises(TypeError):
        model = clf(1, test_bad_params)
        model = clf(2, test_bad_params)

    with pytest.raises(TypeError):
        model = clf(1, test_bad_params)
        model = clf(2, test_bad_params)
    
    model = clf(1, test_params_rf)
    assert type(model) == sklearn.ensemble._forest.RandomForestClassifier

    model = clf(2, test_params_lr)
    assert type(model) == sklearn.linear_model._logistic.LogisticRegression
    
def check_predict(load_model_replace,load_model_replace_bad): 
    
    data_test = pd.read_csv('datasets/data_test.csv')
    target_test = pd.read_csv('datasets/target_test.csv')

    mocker.patch.object("load_model","lol")
    with pytest.raises(TypeError):
        y_pred = predict_model(1,test_params_rf,data_test)
        y_pred = predict_model(2,test_params_lr,data_test)

    with pytest.raises(ValueError):
        y_pred = predict_model(1,test_params_rf,"fgfd")
        y_pred = predict_model(2,test_params_lr,"fgfd")

    mocker.patch.object("load_model",load_model_replace)
    with pytest.raises(NotFittedError):
        y_pred = predict_model(1,test_params_rf,data_test)
    
    mocker.patch.object("load_model",load_model_replace)
    y_pred = predict_model(2,test_params_lr,data_test)
    assert y_pred.shape[0] == X_test.shape[0]


    
    

     






        

