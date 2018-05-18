import gym 
import sys
from gym.utils import play as p
import numpy as np
from pygit2 import Repository
import pandas as pd
import datetime 
import pickle

# Things to fix / look at

# Notes, pickling for now, is this the best way to store the info?
# pandas dataframe creation might be n^2 the way im doing it rn , look at later


# Adjust game characteristics here
# frames per second, 30 is the default
fps_rate = 10
# game name, prolly nvr gonna change this
game_name = 'SpaceInvaders-v0'
# gets branch name.
head = Repository('.').head.shorthand
# Folder to dump in
filePath = './human_play/'

# Class to play the game + serialize the dataframe w play stats. 
class PlayGame(object):
    def __init__(self):
        self.game_data = pd.DataFrame()
        self.score = 0
    def callbackFunction(self,obs_t, obs_tp1, action, rew, done, info):
        # GS stands for game state. 
        column_names = ["PriorGS", "PostGS", "Action", "Reward", "Done?", "Info"]
        self.score += rew
        self.game_data = self.game_data.append(pd.DataFrame([[obs_t,\
            obs_tp1,action,rew,done,info]], columns = column_names), ignore_index = True)
        return 1
    def run(self):
        env = gym.make(game_name)
        p.play(env, zoom = 2, fps = fps_rate, callback = self.callbackFunction)
        now = datetime.datetime.now()
        filename = filePath + head + "_Score_" + str(int(self.score)) + ".data"
        f = open(filename, "wb")
        pickle.dump(self.game_data, f)
        f.close()

if __name__ == '__main__':
    g = PlayGame()
    g.run()    
