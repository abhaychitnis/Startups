import pickle
import matplotlib.pyplot as plt


import get_best_features as bf

def plot_best_model(X,y, best_model):

    X = bf.get_best_features(X, y)

    select_model = pickle.loads(best_model)

    y_predict = select_model.predict(X)

    plt.scatter(y, y_predict, color = 'green')
    plt.plot(y,y, color = 'red')

    plt.title('Best Model')
    plt.xlabel('True Y')
    plt.ylabel('Predicted Y')
    plt.show()
    
    return(0)

