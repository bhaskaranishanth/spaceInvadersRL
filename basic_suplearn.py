import gym 
import sys
import numpy as np
from gym.utils import play as p
import pickle
import time
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn import svm
# env = gym.make('SpaceInvaders-v0')

full_data = pickle.load(open("./human_play/sunash_Score_90.data", "rb"))
print(type(full_data['PriorGS']))
X = np.array([full_data['PriorGS'][i].flatten() for i in range(len(full_data))])
print(X.shape)
y = np.array(full_data['Action'])
print(np.shape(X), np.shape(y))

clf = svm.LinearSVC(C = 0.1)
clf = clf.fit(X, y)

# game name, prolly nvr gonna change this
game_name = 'SpaceInvaders-v0'
env = gym.make(game_name)
for i_episode in range(2):
    observation = env.reset()
    for t in range(1000):
        time.sleep(0.03)
        env.render()
        action = clf.predict(np.array([observation.flatten()]))
        observation, reward, done, info = env.step(action[0])
print("ass")