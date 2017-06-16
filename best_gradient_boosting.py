from sklearn.ensemble import GradientBoostingRegressor

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_results(X,y, gs):

    plt.scatter(y, gs.predict(X), color = 'blue')
    plt.plot(y,y, color = 'red')

    plt.title('Gradient Boosting')
    plt.xlabel('True Y')
    plt.ylabel('Predicted Y')
    plt.show()
    return(0)

def best_model(X_train, X_test, y_train, y_test, plot_ind, eval_parm):

    X = np.concatenate((X_train, X_test), axis=0)
    y = np.concatenate((y_train, y_test), axis=0)

    if eval_parm == 'deep':
        parameters = {'n_estimators': (50, 100), \
                      'loss': ('ls', 'lad', 'huber'), \
                    'criterion': ('friedman_mse', 'mse', 'mae'), \
                      'max_depth': (3, 4, 5), \
                 'min_samples_split': (2, 3, 4), \
                 'min_samples_leaf': (2, 3), \
                 'max_features': ('auto', 'log2', None) \
                 }
    elif eval_parm == 'test':
        parameters = {'n_estimators': (50, 100), \
                      'loss': ('ls', 'lad'), \
                 'min_samples_leaf': (2, 3), \
                 'max_features': ('auto', 'log2', None) \
                 }
    
    regressor = GradientBoostingRegressor(random_state = 0)

    gs = GridSearchCV(regressor, parameters, cv=5)
    gs.fit(np.concatenate((X_train, X_test), axis=0), \
           np.concatenate((y_train, y_test), axis=0))

# Predicting the Test set results
    y_pred = gs.predict(X_test)

    score = r2_score(y_test, y_pred)   

    return_parm= {'trained_model': gs.best_estimator_, 'score': score}

    if plot_ind:
        plot_results(X,y, gs)

    return (return_parm)
