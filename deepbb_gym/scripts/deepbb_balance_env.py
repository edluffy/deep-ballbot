from gym import spaces
import deepbb_env
from gym.envs.registration import register
import rospy
import math
import numpy as np
from tf.transformations import euler_from_quaternion

max_episode_steps = 1000

register(
        id='DeepBBBalanceEnv-v0',
        entry_point='deepbb_balance_env:DeepBBBalanceEnv',
        max_episode_steps=max_episode_steps,
    )

class DeepBBBalanceEnv(deepbb_env.DeepBBEnv):
    """
    Observation (all as measured by the onboard IMU):
        Type: Box(9)
        Num    Observation
        0      Orientation x
        1      Orientation y
        2      Orientation z
        3      Angular Velocity x
        4      Angular Velocity y
        5      Angular Velocity z

    Actions:
        Type: Box(3)
        Num    Action
        0      Motor 1 Torque Command
        1      Motor 2 Torque Command
        2      Motor 3 Torque Command

    Reward:
    """
    def __init__(self):
        o_high = np.ones(12)
        self.observation_space = spaces.Box(-o_high, o_high)

        a_high = np.array([1, 1, 1])
        self.action_space = spaces.Box(-a_high, a_high)

        self.max_torque = 1.0
        self.min_torque = -1.0

        super(DeepBBBalanceEnv, self).__init__()


    def _set_init_pose(self):
        self.move_joints([0, 0, 0])

    def _init_env_variables(self):
        """
        Inits variables needed to be initialised each time we reset at the start
        of an episode.
        :return:
        """
        self.initial_joints_position = self.joints.position


    def _set_action(self, action):
        torques = np.clip(action, self.min_torque, self.max_torque)
        self.move_joints(torques)


    def _get_obs(self):
        q = [self.imu.orientation.x, self.imu.orientation.y,
                self.imu.orientation.z, self.imu.orientation.w]
        roll, pitch, yaw = euler_from_quaternion(q)

        joints_position = np.subtract(self.joints.position, self.initial_joints_position)

        obs = [
            roll,
            pitch,
            yaw,
            self.imu.angular_velocity.x,
            self.imu.angular_velocity.y,
            self.imu.angular_velocity.z,
            joints_position[0],
            joints_position[1],
            joints_position[2],
            self.joints.velocity[0],
            self.joints.velocity[1],
            self.joints.velocity[2],
        ]

        return obs

    def _is_done(self, obs):
        roll, pitch, yaw = obs[:3]
        tilt_angle = math.atan(math.sqrt(math.tan(roll)**2 + math.tan(pitch)**2))

        done = False
        if tilt_angle > 0.524:
            done = True

        return done

    def _compute_reward(self, obs, done):
        roll, pitch, yaw = obs[:3]
        tilt_angle = math.atan(math.sqrt(math.tan(roll)**2 + math.tan(pitch)**2))

        #reward = 0.2-tilt_angle - 0.1*(self.joints.effort[0]**2 + self.joints.effort[1]**2 + self.joints.effort[2]**2)
        reward = -math.log(tilt_angle)

        return reward

    # Internal TaskEnv Methods
