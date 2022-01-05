import numpy as np
import h5py
with open('saved_fg.npy', 'rb') as f:
    field_goals = np.load(f, allow_pickle=True)

np.random.shuffle(field_goals)
for fg in field_goals:
    fg[0] = np.array(fg[0])
import random

train = field_goals[:int(.7 * len(field_goals))]
test = field_goals[int(.7 * len(field_goals)) + 1:]

for t in train[:,0]:
    for s in test[:,0]:
        if s[0] == t[0] and s[1] == t[1] and s[2] == t[2] and s[3] == t[3] and s[4] == t[4]:
            print('-' * 50)
            print(t[0], t[1])
            print('=' * 50)
            print(s[0], s[1])
            print('-' * 50)
with open('train_features.npy', 'wb') as f:
    np.save(f, np.stack(train[:,0]))
with open('test_features.npy', 'wb') as f:
    np.save(f, np.stack(test[:,0]))
with open('train_targets.npy', 'wb') as f:
    np.save(f, np.stack(train[:,1]))
with open('test_targets.npy', 'wb') as f:
    np.save(f, np.stack(test[:,1]))
