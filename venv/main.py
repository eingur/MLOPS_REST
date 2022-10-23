from flask import Flask, request, jsonify
import joblib
import traceback
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)
# api.model()

@app.route('/', methods=['GET'])
def avail():
    

@app.route('/prediction/<int:id>', methods=['POST'])
def predict(id):
    if id == 1:
        model = rf
    elif id == 2:
        model = lr
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
            if id == 2:
                type = 'RFclf'
                model = RandomForestClassifier(**query)
            elif id == 1:
                type = 'LR'
                model = LogisticRegression(**query)
            model.fit(X_train,y_train)
            return (f'model {type} refitting with parameters {query}')
        except:
            return jsonify({'trace': traceback.format_exc()})

if __name__ == '__main__':
    try:
        print(sys.argv[1])
        port = int(sys.argv[1])
    except:
        port = 12345
        rf = joblib.load("C:/Users/user/ipynbs/MLOPS/HW1/MLOPS_REST/rf.pkl")
        lr = joblib.load("C:/Users/user/ipynbs/MLOPS/HW1/MLOPS_REST/lr.pkl")
        print ('Model loaded')
        rnd_columns = joblib.load('C:/Users/user/ipynbs/MLOPS/HW1/MLOPS_REST/cols.pkl') # Load “rnd_columns.pkl”
        app.run(port=port, debug=True)
