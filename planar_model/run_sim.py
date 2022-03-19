import random
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

#import gym
import env
from vpg import VPG
from lqr import LQR

sns.set_theme()

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

for ep in range(1):
    state = env.reset()
    history = []
    torques = []
    done = False
    t = 0
    while not done:
        t+=1
        env.render()

        if t % 4 == 0:
            action = agent.policy(state)
        else:
            action = 0

        state, _, done = env.step(action)

        history.append(state)
        torques.append(action)

print(len(history))
norm = lambda d: d/(max(d)-min(d))

history = np.array(history).T
plt.plot(norm(history[0]), label='ball angle')
plt.plot(norm(history[1]), label='ball angular velocity')
plt.plot(norm(history[2]), label='rod angle')
plt.plot(norm(history[3]), label='rod angular velocity')
plt.plot(norm(torques), label='torque')

plt.legend()
plt.show()
