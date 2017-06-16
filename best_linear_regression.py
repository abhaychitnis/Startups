from sklearn.linear_model import LinearRegression

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_results(X,y, gs):

    plt.scatter(y, gs.predict(X), color = 'blue')
    plt.plot(y,y, color = 'red')

    plt.title('Linear Regression')
    plt.xlabel('True Y')
    plt.ylabel('Predicted Y')
    plt.show()
    return(0)

def best_model(X_train, X_test, y_train, y_test, plot_ind, eval_parm):

    if eval_parm == 'deep':
        parameters = {'fit_intercept': (True, False) \
                      }
    elif eval_parm == 'test':
        parameters = {'fit_intercept': (True, False) \
                      }
    
    regressor = LinearRegression()

    gs = GridSearchCV(regressor, parameters, cv=5)
    gs.fit(X_train, y_train)

# Predicting the Test set results
    y_pred = gs.predict(X_test)

    score = r2_score(y_test, y_pred)   

    return_parm= {'trained_model': gs.best_estimator_, 'score': score}

    if plot_ind:
        X = np.concatenate((X_train, X_test), axis=0)
        y = np.concatenate((y_train, y_test), axis=0)
        plot_results(X,y, gs)

    return (return_parm)
