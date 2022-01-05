import numpy as np
import math
from matplotlib import pyplot as plt
import time
import itertools
from numpy.lib.function_base import _quantile_unchecked
from pandas import read_csv

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

with open('train_features.npy', 'rb') as f:
    train_features = np.load(f, allow_pickle=True)

with open('train_targets.npy', 'rb') as f:
    train_targets = np.load(f, allow_pickle=True)

with open('test_features.npy', 'rb') as f:
    test_features = np.load(f, allow_pickle=True)

with open('test_targets.npy', 'rb') as f:
    test_targets = np.load(f, allow_pickle=True)
projection_matrix = np.array(pca(train_features))
tr_f = np.zeros((train_features.shape[0], 3))
#Uncomment to generate training data for interpreter
for t in range(len(train_features)):
    a = np.reshape(train_features[t], (5,1))
    tr_f[t] = np.matmul(projection_matrix, a).flatten()
train_features = tr_f

tr_f = np.zeros((test_features.shape[0], 3))
for t in range(len(test_features)):
    a = np.reshape(test_features[t], (5,1))
    tr_f[t] = np.matmul(projection_matrix, a).flatten()

test_features = tr_f
def target_zones(wind_y, wind_x, fgp, precip, metric):
    yds = 17
    x = []
    y = []
    d = []
    n = get_knn(np.matmul(projection_matrix,np.array([yds, wind_x, wind_y, fgp, precip])), metric, train_features, k=1)
    while yds < 66:
        yds = yds + 1
        x.append(yds)
        n = get_knn(np.matmul(projection_matrix,np.array([yds, wind_x, wind_y, fgp, precip])), metric, train_features, k=1)
        pred, dist = predict(n, train_targets, return_distance=True)
        d.append(dist)
        y.append('g' if pred == 'Kick Attempt Good' else 'r')
    return x, y, d
stadiums = {}
stad_csv = read_csv('./archive/stadium_coordinates.csv')
for idx, s in stad_csv.iterrows():
    stadiums[s['StadiumName']] = (float(s['StadiumAzimuthAngle']), s['RoofType'])
again = 'y'
while(again == 'y'):
    wind_speed = 0
    precipitation = 0
    stadium = input('Enter name of stadium being played in:')
    while(stadium not in stadiums):
        print('That stadium could not be found')
        stadium = input('Enter name of stadium being played in:')
    azi, type = stadiums[stadium]
    #azi = float(azi)
    fgp = float(input('Enter field goal percentage of kicker:'))
    temp = 0
    if type != 'Indoors':
        wind_speed = float(input('Enter Wind Speed in mph:'))
        wind_dir = float(input('Enter wind direction in degrees:'))
        temp = float(input('What is the temperature in degrees fahrenheit?'))
        precipitation = float(input('How much precipitation is there in inches?'))
        if type == 'Retractable' and (wind_speed > 40 or temp < 40 or precipitation > 0):
            wind_speed = 0
            precipitation = 0
    x, y, d = target_zones(wind_speed * math.sin(wind_dir - azi), wind_speed * math.cos(wind_dir - azi), fgp, precipitation, euclidean_distance)
    for i in range(len(x)):
        plt.axvline(x = x[i], color = y[i])
    plt.xlabel("Required Yardage")
    plt.show()
    again = input('Run another prediction(y/n)?')
