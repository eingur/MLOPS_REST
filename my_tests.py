import pytest
import numpy as np
import pandas as pd
import fit_and_save_model
from fit_and_save_model import clf, predict_model, load_model
import pickle
from sklearn.exceptions import NotFittedError
import sklearn

test_params_lr = {'penalty':'l2','C':0.3}
test_params_rf = {'max_depth': 5,'n_estimators':100}
test_bad_params_lr = {'penalty':'l3','C':0.3}
test_bad_params_rf = {'max_depth':-1,'n_estimators':100}
test_bad_params = {'penalty':'l3','n_estimators':100}

logs_bad = 'rf_n_estimators_100'
@pytest.fixture()
def fixture_replace_load_bad():
    _model = pickle.load(open('tests/'+logs_bad+'.json','rb'))
    print(f" is ok {_model} {type(_model)}")
    yield _model
    print(f"stil ok")
    
logs = 'lr_penalty_l2_C_0.3'
@pytest.fixture()
def fixture_replace_load():
    yield pickle.load(open('tests/'+logs+'.json','rb'))



def test_init():
    data_train = pd.read_csv('datasets/data_train.csv')
    target_train = pd.read_csv('datasets/target_train.csv')

    with pytest.raises(ValueError):
        model = clf(1,test_bad_params_rf).model.fit(data_train, target_train)
    with pytest.raises(ValueError):
        model = clf(2, test_bad_params_lr).model.fit(data_train, target_train)
    
    with pytest.raises(ValueError):
        model = clf(3)

    with pytest.raises(TypeError):
        model = clf(1, test_bad_params)
        model = clf(2, test_bad_params)

    with pytest.raises(TypeError):
        model = clf(1, test_bad_params)
        model = clf(2, test_bad_params)
    
    model = clf(1, test_params_rf)
    assert type(model.model) == sklearn.ensemble._forest.RandomForestClassifier

    model = clf(2, test_params_lr)
    assert type(model.model) == sklearn.linear_model._logistic.LogisticRegression
    
def test_predict(mocker, fixture_replace_load, fixture_replace_load_bad): 
    data_test = pd.read_csv('datasets/data_test.csv')
    target_test = pd.read_csv('datasets/target_test.csv')

    mocker.patch.object(fit_and_save_model, "load_model", "lol")
    with pytest.raises(TypeError):
        y_pred = predict_model(1,test_params_rf,data_test)
        y_pred = predict_model(2,test_params_lr,data_test)
    
    clff = pickle.load(open('tests/'+logs_bad+'.json','rb'))
    mocker.patch.object(fit_and_save_model, "load_model", return_value = fixture_replace_load_bad)
    with pytest.raises(NotFittedError):
        y_pred = predict_model(1,{'n_estimators':100},data_test)
    

    mocker.patch.object(fit_and_save_model, "load_model", return_value = fixture_replace_load)
    with pytest.raises(ValueError):
        y_pred = predict_model(1,test_params_rf,"fgfd")
        y_pred = predict_model(2,test_params_lr,"fgfd")

    y_pred = predict_model(2,test_params_lr,data_test)
    assert y_pred.shape[0] == data_test.shape[0]


    
    

     






        

