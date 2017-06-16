import pandas as pd
import pickle

import get_best_features as bf

def get_best_model(X,y):

    model_eval = []
    X = bf.get_best_features(X, y)

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

    import importlib
    model_to_evaluate = pd.read_csv('regression_models_list.csv')

    return_list = []
    best_score = 0
    best_model = None
    for i in range(len(model_to_evaluate.index)):
        #print ('Scores for regression model, ', model_to_evaluate['model_name'][i], ' are : ')

        eval_parm = model_to_evaluate['eval_parm'][i]
    
        brf = importlib.import_module(model_to_evaluate['file_name'][i])
        return_parm = brf.best_model(X_train, X_test, y_train, y_test, False, eval_parm)
        
        return_parm["model_name"] = model_to_evaluate['model_name'][i]

        return_list.append(return_parm)

    best_model_found = evalModelList(X_train, y_train, return_list)
 
    return(pickle.dumps(best_model_found))

def evalModelList (X_train, y_train, return_list):

    modelStore = pd.DataFrame(return_list)

    print (modelStore)

    bestModelRow = modelStore.ix[modelStore['score'].idxmax()]

    print (bestModelRow)

    best_model_found = bestModelRow['trained_model']

    return(best_model_found)
