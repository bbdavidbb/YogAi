from sklearn.neural_network import MLPClassifier
import pickle

import ast
from sklearn.model_selection import train_test_split
with open("keypoint_data.txt","r") as myfile:
    data = myfile.read()
    data = data.split("\n")
'''
Best parameters found:
 {'activation': 'tanh', 'hidden_layer_sizes': (30, 30, 30), 'learning_rate': 'invscaling', 'max_iter': 10000, 'solver': 'adam'}
Best score:  0.9332161687170475
'''
X = []
Y = []
label_to_no = {"tree": 0, "warrior1":1, "warrior2":2, "childs":3,"downwarddog":4,"plank":5,"mountain":6,"trianglepose":7}
no_to_label = {0: "tree", 1: "warrior1", 2:"warrior2"}
for line in data:
    if line != "":
        line = line.split(":")
        X.append(ast.literal_eval(line[1]))
        Y.append(label_to_no[line[0]])
mlp = MLPClassifier(hidden_layer_sizes=(30,30,30), max_iter= 10000, learning_rate = 'invscaling', solver='adam',activation= 'tanh')

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42, shuffle=True)
mlp.fit(X_train,y_train)
pickle.dump(mlp, open("mlp_model_best.sav", 'wb'))
print(mlp.score(X_test, y_test))
