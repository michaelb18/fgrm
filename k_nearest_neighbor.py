import numpy as np
import math
from matplotlib import pyplot as plt
import time
import itertools
from numpy.lib.function_base import _quantile_unchecked
from metric_learn import LMNN

train_features = []
train_targets = []
test_features = []
test_targets = []
#different distances experimented with

def euclidean_distance(a, b):
    c = np.array(a)
    d = np.array(b)
    f = c - d
    return np.linalg.norm(c-d)

def mahalanobis_distance(a, b):
    c = np.array(a)[2:]
    d = np.array(b)[2:]
    M = np.eye(c.shape[0])
    return math.sqrt(np.matmul(np.matmul((c-d).T, M), c-d))
    
def keralized_distance(x, b):
    return np.linalg.norm(rbf(x,x) + rbf(b, b) - 2*rbf(x, b))**(1/2)

def rbf(x, x_1):
    s = .01
    return np.exp(-(euclidean_distance(x, x_1)**2)/(2*s**2))

#pca(feature ranking/impact)
def mu_x(D):
    sum = np.zeros(np.array(D[0]).shape)
    for d in D:
        sum = sum + np.array(d)
    return sum/len(D)

def S_x(D):
    mu = mu_x(D)
    sum = np.zeros(np.matmul(np.array(D[0] - mu).reshape(len(mu), 1), np.array(D[0] - mu).reshape(len(mu), 1).T).shape)
    for d in D:
        sum = sum + np.matmul(np.array(D[0] - mu).reshape(len(mu), 1), np.array(D[0] - mu).reshape(len(mu), 1).T)
    return sum/len(D)

def pca(D):
    S = S_x(D)
    vals, vecs = np.linalg.eig(S)
    return vecs[:-2]

def remove_idx(a, i):
    l = []
    for a_x in range(len(a)):
        if a_x != i:
            l.append(a[a_x])
    return np.array(l)

def remove_pca(D, idx):
    for d in range(len(D)):
        D[d] = remove_idx(D[d], idx)

#knn functions
def get_knn(x, distance_metric, D, k = 1):
    nearest_indices = []
    for i in range(k):
        nearest_indices.append([-1, math.inf])
    nearest_indices = np.array(nearest_indices)
    for v in range(len(D)):
        nearest = False
        for n in nearest_indices:
            if distance_metric(D[v], x) < n[1]:
                nearest = True
                break
        if nearest:
            nearest_indices[k-1] = [v, distance_metric(D[v], x)]
            nearest_indices = nearest_indices[np.argsort(nearest_indices[:, 1])]
            
    return nearest_indices

def predict(N_k, targets, return_distance=False):
    labels = {'Kick Attempt Good' : 0, 'Kick Attempt No Good' : 0}
    distance = 0
    for neighbor in N_k:
        target = targets[int(neighbor[0])]
        distance = distance + neighbor[1]
        if target == 'Kick Attempt Good':
            labels['Kick Attempt Good'] = labels['Kick Attempt Good'] + 1
        else:
            labels['Kick Attempt No Good'] = labels['Kick Attempt No Good'] + 1
    if labels['Kick Attempt Good'] >= labels['Kick Attempt No Good']:
        if return_distance:
            return 'Kick Attempt Good', distance/len(N_k)
        else:
            return 'Kick Attempt Good'
    else:
        if return_distance:
            return 'Kick Attempt No Good', distance/len(N_k)
        else:
            return 'Kick Attempt No Good'

def get_accuracy(features, targets, train_features, train_targets, metric, k):
    num_right = 0
    cm = [[0, 0], [0, 0]]
    for t in range(len(features)):
        n = get_knn(features[t], metric, train_features, k=k)
        pred = predict(n, train_targets)
        if pred == targets[t]:
            num_right = num_right + 1
        
        if pred == 'Kick Attempt Good' and targets[t] == 'Kick Attempt No Good':
            cm[0][1] = cm[0][1] + 1
        elif pred == 'Kick Attempt Good' and targets[t] == 'Kick Attempt Good':
            cm[0][0] = cm[0][0] + 1
        elif pred == 'Kick Attempt No Good' and targets[t] == 'Kick Attempt Good':    
            cm[1][0] = cm[1][0] + 1
        elif pred == 'Kick Attempt No Good' and targets[t] == 'Kick Attempt No Good':    
            cm[1][1] = cm[1][1] + 1
    return num_right/len(features), cm[0][0]/(cm[0][1] + cm[0][0]), cm[0][0]/(cm[0][0] + cm[1][0]), cm[1][1]/(cm[1][1] + cm[0][1]), cm[1][1]/(cm[1][1] + cm[1][0]), cm

def cross_validation(val_features, val_targets, train_features, train_targets):
    k = 1
    last_val = get_accuracy(val_features, val_targets, train_features, train_targets, euclidean_distance, k)
    #print(last_val)
    #print(get_accuracy(val_features, val_targets, val_features, k+1))
    while last_val < get_accuracy(val_features, val_targets, train_features, train_targets, euclidean_distance, k+1):
        print(last_val)
        k = k + 1
        last_val = get_accuracy(val_features, val_targets, train_features, train_targets, euclidean_distance, k)
    print(last_val)
    return k

def cross_validation_plot(val_features, val_targets, train_features, train_targets):
    k_x = []
    errors = []
    #print(last_val)
    #print(get_accuracy(val_features, val_targets, val_features, k+1))
    for k in range(10):
        errors.append(1 - get_accuracy(val_features, val_targets, train_features, train_targets, 1+k)[0])
        k_x.append(1+k)
    plt.xlabel('K')
    plt.ylabel('Validation Error')
    plt.plot(k_x, errors)
    return 10

with open('train_features.npy', 'rb') as f:
    train_features = np.load(f, allow_pickle=True)

with open('train_targets.npy', 'rb') as f:
    train_targets = np.load(f, allow_pickle=True)

with open('test_features.npy', 'rb') as f:
    test_features = np.load(f, allow_pickle=True)

with open('test_targets.npy', 'rb') as f:
    test_targets = np.load(f, allow_pickle=True)
projection_matrix = np.array(pca(train_features))

def create_dt_data(train_features, train_targets, test_features, test_targets, projection_matrix = np.eye(5)):
    dt_train_features = []
    dt_train_targets = []
    
    saved_5d_train = np.copy(train_features)
    saved_5d_test = np.copy(test_features)
    train_features = np.copy(train_features)
    test_features = np.copy(test_features)
    tr_f = np.zeros((train_features.shape[0], 3))
    for t in range(len(train_features)):
        a = np.reshape(train_features[t], (5,1))
        tr_f[t] = np.matmul(projection_matrix, a).flatten()
    train_features = tr_f

    tr_f = np.zeros((test_features.shape[0], 3))
    for t in range(len(test_features)):
        a = np.reshape(test_features[t], (5,1))
        tr_f[t] = np.matmul(projection_matrix, a).flatten()

    test_features = tr_f
    i = 0
    for sample in range(len(train_features)):
        n = get_knn(train_features[sample], euclidean_distance, train_features, k=1)
        dt_train_features.append(saved_5d_train[sample])
        pred = predict(n, train_targets)
        dt_train_targets.append(pred)
        i = i + 1
    dt_test_features = []
    dt_test_targets = []

    i = 0
    for sample in range(len(test_features)):
        n = get_knn(test_features[sample], euclidean_distance, test_features, k=1)
        dt_test_features.append(saved_5d_test[sample])
        pred = predict(n, test_targets)
        dt_test_targets.append(pred)
        i = i + 1
    
    with open('dt_train_features.npy', 'wb') as f:
        np.save(f, np.array(dt_train_features))
    with open('dt_test_features.npy', 'wb') as f:
        np.save(f, np.array(dt_test_features))
    with open('dt_train_targets.npy', 'wb') as f:
        np.save(f, np.array(dt_train_targets))
    with open('dt_test_targets.npy', 'wb') as f:
        np.save(f, np.array(dt_test_targets))

create_dt_data(train_features, train_targets, test_features, test_targets, projection_matrix)

tr_f = np.zeros((train_features.shape[0], 3))
for t in range(len(train_features)):
    a = np.reshape(train_features[t], (5,1))
    tr_f[t] = np.matmul(projection_matrix, a).flatten()
train_features = tr_f

tr_f = np.zeros((test_features.shape[0], 3))
for t in range(len(test_features)):
    a = np.reshape(test_features[t], (5,1))
    tr_f[t] = np.matmul(projection_matrix, a).flatten()

test_features = tr_f

#Uncomment following to run cross validation
#val_index = int(len(train_features) * 1/4)
#val_features = train_features[:val_index]
#val_targets = train_targets[:val_index]
#train_targets = train_targets[val_index:]
#train_features = train_features[val_index:]

n_made = 0
n_missed = 0
for f in train_targets:
    if f == 'Kick Attempt Good':
        n_made = n_made + 1
    else:
        n_missed = n_missed + 1

print(n_made/len(train_targets))
print(n_missed/len(train_targets))

#K=1 decided using cross validation. Uncomment this line to run cross validation
k = 1#cross_validation(val_features, val_targets, train_features, train_targets)

print('K Chosen:')
print(k)
print('Train accuracy(acc, precision, recall, specificity, negative predictive value, confusion matrix):')
a = get_accuracy(train_features, train_targets, train_features, train_targets, euclidean_distance, k)
print(a)

print('Test accuracy(acc, precision, recall, specificity, negative predictive value, confusion matrix):')
a = get_accuracy(test_features, test_targets, train_features, train_targets, euclidean_distance, k)
print(a)
cm = np.stack(a[5])
fig, ax = plt.subplots(figsize=(3, 3))
ax.matshow(cm, cmap=plt.cm.Blues)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(x=j, y=i,s=cm[i, j], va='center', ha='center')
 
plt.xlabel('Predictions', fontsize=12)
plt.ylabel('Targets', fontsize=12)
plt.xticks([0, 1],['Good', 'No Good'], fontsize=12)
plt.yticks([0, 1],['Good', 'No Good'], fontsize=12)
plt.title('Confusion Matrix', fontsize=12)
plt.show()
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
for t in range(len(train_features)):
    x = train_features[t][0]
    y = train_features[t][1]
    z = train_features[t][2]
    if train_targets[t] == 'Kick Attempt Good':
        s = 'x'
        c = 'green'
    else:
        s = 'o'
        c = 'red'
    ax.scatter(x, y, z, marker=s, color=c)
plt.show()
def target_zones(wind_y, wind_x, fgp, precip, metric):
    yds = 17
    x = []
    y = []
    d = []
    n = get_knn(np.matmul(projection_matrix,np.array([yds, wind_x, wind_y, fgp, precip])), metric, train_features, k=k)
    while yds < 66:
        yds = yds + 1
        x.append(yds)
        n = get_knn(np.matmul(projection_matrix,np.array([yds, wind_x, wind_y, fgp, precip])), metric, train_features, k=k)
        pred, dist = predict(n, train_targets, return_distance=True)
        d.append(dist)
        y.append('g' if pred == 'Kick Attempt Good' else 'r')
    return x, y, d

x, y, d = target_zones(12 * math.sin(270 - 90), 12 * math.cos(270 - 90), 75, 0, euclidean_distance)
for i in range(len(x)):
    plt.axvline(x = x[i], color = y[i])
plt.xlabel("Required Yardage")
plt.show()