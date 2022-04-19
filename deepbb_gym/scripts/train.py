#! /usr/bin/env python3
import cProfile
import rospy
import gym
import deepbb_balance_env
import numpy as np

from agents import ddpg

# To run as fast as possible:
# roslaunch tiago_gym start_training.launch gazebo:=false
# gz physics -u 0 -s 0.0025; gz stats

def train():
    rospy.init_node('deepbb_gym')

    env = gym.make('DeepBBBalanceEnv-v0')
    o_dims = len(env.observation_space.sample())
    a_dims = env.action_space.shape[0]
    a_high = env.action_space.high

    agent = ddpg.DDPG(env, o_dims, a_dims, a_high)
    agent.run(999999, logdir='vel20_step=0.02_0.0002_halflr')

if __name__ == '__main__':
    cProfile.run('train()', '~/dev_ws/src/deep-ballbot/deepbb_gym/training.prof')

    #for i in range(10):
    #    env.reset()
    #    done = False
    #    while not done:
    #        action = np.random.uniform(-1, 1, 3)
    #        next_state, reward, done, _ = env.step(action)

    #        state = next_state
