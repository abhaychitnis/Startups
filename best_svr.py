from sklearn.svm import SVR

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_results(X,y, gs):

    plt.scatter(y, gs.predict(X), color = 'blue')
    plt.plot(y,y, color = 'red')

    plt.title('SVR')
    plt.xlabel('True Y')
    plt.ylabel('Predicted Y')
    plt.show()
    return(0)
 


def best_model(X_train, X_test, y_train, y_test, plot_ind, eval_parm):
    #return_parm=[]

    if eval_parm == 'deep':
        parameters = {'C': (0.5, 1., 2.), \
                    'epsilon': (0.05, 0.1, 0.2, 0.3), \
                 'kernel': ('linear', 'rbf', 'sigmoid', 'poly'), \
                 'degree': (2, 3, 4), \
                 'coef0': (0.05, 0.1), \
                 'gamma': (0.1, 1, 10) \
                 }
    elif eval_parm == 'test':
        parameters = {'C': (1., 2.), \
                    'epsilon': (0.05, 0.1), \
                 'kernel': ('linear', 'rbf', 'poly'), \
                 'degree': (2, 3), \
                 }
    
    regressor = SVR()

    gs = GridSearchCV(regressor, parameters, cv=5)
    gs.fit(X_train, \
           y_train)

# Predicting the Test set results
    y_pred = gs.predict(X_test)

    score = r2_score(y_test, y_pred)   

    return_parm= {'trained_model': gs.best_estimator_, 'score': score}

    if plot_ind:
        X = np.concatenate((X_train, X_test), axis=0)
        y = np.concatenate((y_train, y_test), axis=0)
        plot_results(X,y, gs)

    return (return_parm)
