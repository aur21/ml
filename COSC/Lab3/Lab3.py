#############################################################
#
# Name: Minh Tao
# 
# Collaborators: Max Spieler, Andrew Agriantonis
#
# ############################################################

from pyexpat import model
from unicodedata import decomposition
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyparsing import col
import seaborn as sns
import sklearn
import sklearn.datasets
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import sklearn.preprocessing
import sklearn.model_selection
import sklearn.metrics
import sklearn.decomposition
import sklearn.experimental
import sklearn.impute
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer 
from sklearn.impute import SimpleImputer 

def fit_predict():
    """Complete this function as described in Lab3.pdf"""
    X_train = pd.read_csv("Lab3_X_train.csv")
    y_train = pd.read_csv("Lab3_y_train.csv").squeeze("columns")
    X_test = pd.read_csv("Lab3_X_test.csv")

    # X_plot = pd.read_csv("Lab3_X_train.csv")
    # X_plot['RainTomorrow'] = list(y_train)
    # X_plot.drop(columns=["Location", "WindGustDir", "WindDir9am", "WindDir3pm"])
    # sns.pairplot(data=X_plot.head(1000), hue="RainTomorrow", kind = 'kde')
    # plt.savefig("PairPlot.pdf")

    print(X_train)

    X_train = pd.get_dummies(X_train, columns=["Location", "WindGustDir", "WindDir9am", "WindDir3pm"])
    print(X_train)
    X_test = pd.get_dummies(X_test, columns=["Location", "WindGustDir", "WindDir9am", "WindDir3pm"])

    imp = IterativeImputer(max_iter=3, random_state=0)
    #imp = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value =-1)
    X_train = imp.fit_transform(X_train)
    X_test = imp.transform(X_test)

    # Create a StandardScaler object
    scl = sklearn.preprocessing.StandardScaler()
    X_train = scl.fit_transform(X_train)
    X_test = scl.transform(X_test)

    # maxIter = [200, 300, 400, 500]
    # for i in [1,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2, 0,1]:
    #     for j in maxIter:
    #         logi = LogisticRegression(C=i, max_iter= j)
    #         scores = sklearn.model_selection.cross_val_score(logi, X_train, y_train, cv=5,scoring="precision")
    #         print(f"Average cross-val precision: {i} : {scores.mean()} \n")

    # solve = ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
    # for i in solve:
    #     logi = LogisticRegression(solver=i, C=0.4)
    #     scores = sklearn.model_selection.cross_val_score(logi, X_train, y_train, cv=5,scoring="precision")
    #     print(f"Average cross-val precision: {i} : {scores.mean()} \n")

    # Initialize
    logi = LogisticRegression()
    logi.fit(X_train, y_train)

    return logi.predict(X_test)

def main():
    """This function is for your own testing. It will not be called by the leaderboard."""
    fit_predict()
    pass


if __name__ == "__main__":
    main()