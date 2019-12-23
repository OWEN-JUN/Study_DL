
import gym
import numpy as np
import matplotlib.pyplot as plt
from gym.envs.registration import register
import random as pr

def rargmax(vector):
    m = np.amax(vector)
    indices = np.nonzero(vector == m)[0]
    return pr.choice(indices)

register(id="FrozenLake-v3", entry_point='gym.envs.toy_text:FrozenLakeEnv',kwargs={'map_name' : '4x4',
                                                                                   'is_slippery':False})


env = gym.make('FrozenLake-v3')





q = np.zeros([env.observation_space.n,env.action_space.n])

num_episodes = 100

rlist = []

for i in range(num_episodes):

    state = env.reset()
    rall = 0
    done = False

    while not done:
        action = rargmax(q[state,:])

        new_state, reward, done, _ = env.step(action)

        q[state, action] = reward + np.max(q[new_state, :])


        rall += reward
        state = new_state
    rlist.append(rall)



print("success rate: " + str(sum(rlist)/num_episodes))
print("final qtable values")
print("left down right up")
print(q)
plt.plot(range(len(rlist)),rlist, color="blue")
plt.show()