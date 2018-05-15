import gym 
import sys
from gym.utils import play as p
env = gym.make('SpaceInvaders-v0')

p.play(env, zoom = 2, fps = 10)
# 6 actions possible, {0,1,2,3,4,5}
# Box space is 210 x 
print(env.observation_space)