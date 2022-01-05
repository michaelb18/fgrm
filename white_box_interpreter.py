from sklearn import tree
import numpy as np
from matplotlib import pyplot as plt
import graphviz
with open('dt_train_features.npy', 'rb') as f:
    train_features = np.load(f, allow_pickle=True)

with open('dt_train_targets.npy', 'rb') as f:
    train_targets = np.load(f, allow_pickle=True)

with open('dt_test_features.npy', 'rb') as f:
    test_features = np.load(f, allow_pickle=True)

with open('dt_test_targets.npy', 'rb') as f:
    test_targets = np.load(f, allow_pickle=True)

def get_accuracy(test_features, test_targets, interpreter):
    num_right = 0
    for t in range(len(test_features)):
        if interpreter.predict([test_features[t]]) == test_targets[t]:
            num_right = num_right + 1

    return num_right/len(test_features)
x = []
y = []
model = tree.DecisionTreeClassifier().fit(train_features, train_targets)
accuracy = get_accuracy(test_features, test_targets, model)
combo = 0
ma = 15
mi = 2
for depth in range(mi, ma):
    for split in range(mi, ma):
        for leaves in range(mi, ma):
            if combo % (((ma-mi)**3)//100) == 0:
                print(str(combo)+'/'+str(((ma-mi)**3))+'('+str(combo/(((ma-mi)**3))*100)+' percent )')
                print('Top accuracy: '+str(accuracy))
            interpreter = tree.DecisionTreeClassifier(max_leaf_nodes = leaves, max_depth = depth, min_samples_split = split)
            interpreter = interpreter.fit(train_features, train_targets)
            a = get_accuracy(test_features, test_targets, interpreter)
            if a > accuracy:
                model = interpreter
                accuracy = a
            combo = combo + 1
print(accuracy)

dot_data = tree.export_graphviz(interpreter, out_file=None, 
                      feature_names=['required yardage(yards)', 'wind in y direction(sideline to sideline)(mph)', 'wind in x direction(endzone to endzone)(mph)', 'field goal percentage of the kicker(%)', 'precipitation(in)'],  
                      class_names=['Made', 'Missed'],  
                      filled=True, rounded=True,  
                      special_characters=True)  
graph = graphviz.Source(dot_data)

graph.view()