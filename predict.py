import subprocess
subprocess.run("pip3 install -r requirements.txt",  shell=True)

import os
import pandas as pd
import mlflow

n_features=20
def predict(args):
    with open(".best_model_uri", "r") as f:
        best_model_uri=f.readline().strip()

    # Load model as a PyFuncModel.
    loaded_model = mlflow.pyfunc.load_model(best_model_uri)

    # Predict on a Pandas DataFrame.
    #X = [[args['feature_'+str(i)] for i in range(n_features)]]
    X = pd.DataFrame([args])
    X = X.loc[:, ['feature_'+str(i) for i in range(n_features)]]

    pred = loaded_model.predict(X)
    return {'result':pred[0]}