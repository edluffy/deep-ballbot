import random
import numpy as np

import env

env = env.BallBotEnv()

state = env.reset()
while True:
    action = np.random.randint(-1000, 1000)
    env.render(stats=True)
    next_state, reward, done = env.step(action)
    print(reward)
    #env.ball.torque = 10000000
    #print(env.ball.torque)

    if done:
        env.reset()
