from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score

import pickle
import numpy as np
import ast
from sklearn.model_selection import train_test_split, cross_val_score

with open("keypoint_data.txt","r") as myfile:
    data = myfile.read()
    data = data.split("\n")

X = []
Y = []


label_to_no = {"tree": 0, "warrior1":1, "warrior2":2, "childs":3,"downwarddog":4,"plank":5,"mountain":6,"trianglepose":7}
for line in data:
    if line != "":
        line = line.split(":")
        X.append(ast.literal_eval(line[1]))
        Y.append(label_to_no[line[0]])
mlp = MLPClassifier()

#To get the allowed parameters values for hyperparameters of the MLP classifier
print(mlp.get_params())

X_train, X_test, Y_train, Y_test = train_test_split(X, Y)

tuned_params = {
'learning_rate': ["constant", "invscaling", "adaptive"],
'hidden_layer_sizes': [(30,30,30),(10,10,10)],
'solver': ['sgd', 'adam'],
'activation': ["relu", "logistic", "tanh"],
'max_iter' : [10000]
}

#Can also use normal KFold instead of StratifiedKFold
cv_method = StratifiedKFold(n_splits=5, shuffle=True)

grid = GridSearchCV(estimator=mlp, param_grid=tuned_params, cv=cv_method, scoring='accuracy', verbose = 10)
grid.fit(X,Y)

print('Best parameters found:\n', grid.best_params_)
print("Best score: ",grid.best_score_)



