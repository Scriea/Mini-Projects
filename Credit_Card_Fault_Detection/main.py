import os
import argparse
import numpy as np
import pandas as pd
from gde import *
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.model_selection import train_test_split

models = {
    "IF": IsolationForest(),
    "GDE": GDE(),
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", '--data', type=str, help='data directory', required=True)
    parser.add_argument("-m",'--model', type=str, default='GDE', help='Type of model to use IF(Isolation Forest(), LOF(Local Outlier Factor),OCS(One-Class SVM), or GDE(Gaussian Density Estimation)') 
    args = parser.parse_args()

    # Load the data
    data = pd.read_csv(os.path.abspath(args.data))

    # Split the data into features and labels
    X = data.drop('Class', axis=1).to_numpy()
    Y = data['Class'].to_numpy()

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)

    model = models[args.model]
    model.fit(X_train, Y_train)
    y_val_pred = model.predict(X_val)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    print(f"Validation Accuracy: {np.mean(y_val_pred == Y_val)}")
    print(f"Train Accuracy: {np.mean(y_train_pred == Y_train)}")
    print(f"Test Accuracy: {np.mean(y_test_pred == Y_test)}")
