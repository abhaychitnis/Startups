import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.preprocessing import PolynomialFeatures
import statsmodels.formula.api as sm

def get_best_features(X, y):

    poly_reg = PolynomialFeatures(degree = 2)
    X = poly_reg.fit_transform(X)

    last_rsquared_adj = 0
    X = np.append(arr = np.ones((X.shape[0], 1)).astype(int), values = X, axis = 1)

    X_columns = [i for i in range(X.shape[1])]
    X_opt = np.float64(X[:, X_columns])
    regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()

    drop_list = [np.argmax(regressor_OLS.pvalues)]

    while not (not drop_list):
#        print ("Old rsquared : ", regressor_OLS.rsquared_adj)
        X_opt = np.delete(X_opt,drop_list , axis=1)
        last_rsquared_adj = regressor_OLS.rsquared_adj
        regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
        drop_list = []
#        print ("New rsquared : ", regressor_OLS.rsquared_adj)
        
        if regressor_OLS.rsquared_adj >= last_rsquared_adj \
        and (max(regressor_OLS.pvalues) - 0.05) > 0.02:
                drop_list.append(np.argmax(regressor_OLS.pvalues))

    return (X_opt)
