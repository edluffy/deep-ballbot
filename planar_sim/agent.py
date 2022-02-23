import random
import numpy as np

#import gym
import env
from vpg import VPG
from lqr import LQR

env = env.BallBotEnv()
#env = gym.make('LunarLander-v2')

#agent = VPG(input_size=4, output_size=2, alpha=0.0005)
#
#for ep in range(5000):
#    state = env.reset()
#    reward_sum = 0
#
#    # Run Episode
#    done = False
#    t = 0
#    while not done:
#        t += 1
#        env.render(stats=True)
#        action = agent.policy(state)
#        next_state, reward, done = env.step(action)
#
#        agent.store(state, action, reward)
#        state = next_state
#
#        reward_sum += reward
#
#    agent.learn()
#    print('Episode', ep, 'Rewards:', reward_sum)

agent = LQR()

for ep in range(5000):
    state = env.reset()

    t = 0
    done = False
    while not done:
        t+=1
        env.render(stats=True)
        action = agent.policy(state)
        state, _, done = env.step(action)
