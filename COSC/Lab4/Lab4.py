################################################
#
# Name: Minh Tao
#
# Partner (if applicable): Max Spieler
#
################################################

import pandas as pd
import numpy as np
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, learning_curve
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import sklearn.model_selection
import sklearn.metrics
import matplotlib.pyplot as plt

# ADDITIONAL IMPORTS AS NEEDED

def fit_predict():
    X_train = pd.read_csv("movies_X_train.csv")
    y_train = pd.read_csv("movies_y_train.csv").squeeze("columns")
    X_test = pd.read_csv("movies_X_test.csv")

    model = RandomForestRegressor(max_depth=16, max_leaf_nodes=90, max_features='auto', min_samples_leaf=7, min_samples_split = 9, min_weight_fraction_leaf = 0.1)
    model.fit(X_train, y_train)

    # return predicted test labels 
    return model.predict(X_test)

def optimize():
    X_train = pd.read_csv("movies_X_train.csv")
    y_train = pd.read_csv("movies_y_train.csv").squeeze("columns")

    # Cross-validation folds
    k = 10

    # Hyperparameters to tune:
    param = {"splitter":["best","random"],
            "max_depth" : [1,5,7,9],
            "min_samples_leaf":[1,4,7],
            "min_weight_fraction_leaf":[0.1,0.3,0.5],
            "max_features":["auto","log2","sqrt",None],
            "max_leaf_nodes":[None,10,20,40,70] 
            }

    # Initialize GridSearchCV object with decision tree regressor and hyperparameters
    grid_tree = GridSearchCV(estimator=DecisionTreeRegressor(random_state=0), param_grid = param, cv = k, return_train_score=True, scoring='neg_mean_absolute_error', refit=True)

    # Train and cross-validate, print results
    grid_tree.fit(X_train, y_train)
    grid_tree_result = pd.DataFrame(grid_tree.cv_results_).sort_values(by=['mean_test_score'], ascending=False)
    print(grid_tree_result[['param_splitter', 'param_max_depth', 'param_min_samples_leaf', 'param_min_weight_fraction_leaf', 'param_max_features', 'param_max_leaf_nodes', 'mean_test_score']].head)


def forest():
    X_train = pd.read_csv("movies_X_train.csv").head(1000)
    y_train = pd.read_csv("movies_y_train.csv").squeeze("columns").head(1000)

    # Cross-validation folds
    k = 5

    # Hyperparameters to tune:
    # param = {"n_estimators": range(1, 100, 20),
    #         "max_depth" : [1,3,5,7],
    #         "min_samples_leaf":[1,4,7],
    #         "min_weight_fraction_leaf":[0.1,0.3,0.5],
    #         }

    param = {"n_estimators": range(100, 500, 100),
            "criterion": ['squared_error'],
            "max_depth" : [16],
            "min_samples_leaf":[1,2,3,4,5,6,7,8,9],
            "min_samples_split":[1,2,3,4,5,6,7,8,9],
            "min_weight_fraction_leaf":[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8],
            "max_features":["auto"],
            "max_leaf_nodes":[90],
            "bootstrap":[True]
            }
    
    # Initialize GridSearchCV object with decision tree regressor and hyperparameters
    grid_tree = GridSearchCV(estimator=RandomForestRegressor(n_jobs=-1), param_grid = param, cv = k, return_train_score=True, scoring='neg_mean_absolute_error', refit=True, verbose=10)

    # Train and cross-validate, print results
    grid_tree.fit(X_train, y_train)
    grid_tree_result = pd.DataFrame(grid_tree.cv_results_).sort_values(by=['mean_test_score'], ascending=False)
    print(grid_tree_result[['param_n_estimators', 'param_max_depth', 'param_min_samples_split', 'param_min_weight_fraction_leaf', 'param_max_features', 'param_max_leaf_nodes', 'mean_test_score']].head)

def evaluate():
    X_train = pd.read_csv("movies_X_train.csv")
    y_train = pd.read_csv("movies_y_train.csv").squeeze("columns")

    forest = RandomForestRegressor(criterion='squared_error', max_depth=16, max_leaf_nodes=90, max_features='auto', min_samples_leaf=7, min_samples_split = 9, min_weight_fraction_leaf = 0.1)     

    train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(estimator=forest, X=X_train, y=y_train, cv=30, return_times=True, n_jobs=-1, scoring='neg_mean_absolute_error')

    plt.plot(train_sizes,np.mean(train_scores,axis=1))
    plt.plot(train_sizes,np.mean(test_scores,axis=1))
    plt.show()

def main():
    """Use this function for your own testing. It will not be called by the leaderboard"""
    evaluate()
    pass


if __name__ == "__main__":
    main()
