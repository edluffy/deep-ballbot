#! /usr/bin/env python3
import rospy
import gym
import deepbb_balance_env

# To run as fast as possible:
# roslaunch tiago_gym start_training.launch gazebo:=false
# gz physics -u 0 -s 0.0025; gz stats

if __name__ == '__main__':
    rospy.init_node('deepbb_gym')

    env = gym.make('DeepBBBalanceEnv-v0')

    for i in range(10):
        env.reset()
        done = False
        print('next episode')
        while not done:
            next_state, reward, done, _ = env.step([0.1, 0.1, 0.1])

            state = next_state


    #agent = dqn.DQN(env, input_size=o_dims, output_size=o_dims, alpha=0.01, epsilon_decay=0.95)
    #agent.run(100)
