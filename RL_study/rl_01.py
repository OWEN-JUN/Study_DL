import numpy as np
import _pickle
import tensorflow as tf
import matplotlib.pyplot as plt
import math


# Load the CartPole Env.
import gym
env = gym.make('CartPole-v0')



observation = env.reset()
random_episodes = 0
reward_sum = 0
total_reward = []
while random_episodes < 20:
    env.render()
    action = env.action_space.sample()
#     print(observation, action)
    observation, reward, done, _ = env.step(action)
    reward_sum += reward
    if done:
        random_episodes += 1
        total_reward.append(reward_sum)
        print("Reward for this episode was: {}, Average reward for episode so far is: {}".format(reward_sum, np.round(np.mean(total_reward),1)))
        reward_sum = 0
        env.reset()