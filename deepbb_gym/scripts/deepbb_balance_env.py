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
        self.action_count = 0

        self.max_vel = 20
        self.min_vel = -20

        self.step_size = 0.02

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
        self.action_count = 0
        self.sim_time = rospy.get_rostime().to_sec()

    def _set_action(self, action):
        vels = np.clip(action*20, self.min_vel, self.max_vel)
        print('ACTION', self.action_count,  vels, end='')
        self.move_joints(vels)
        self.action_count += 1
        rospy.sleep(self.step_size)


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
        print(' TILT:', format(np.degrees(tilt_angle), '.2f'), end='')

        done = False
        if abs(roll) > math.pi/12:
            done = True
        if abs(pitch) > math.pi/12:
            done = True
        if abs(yaw) > math.pi/12:
            done = True

        return done

    def _compute_reward(self, obs, done):
        roll, pitch, yaw = obs[:3]
        #tilt_angle = math.atan(math.sqrt(math.tan(roll)**2 + math.tan(pitch)**2))

        if not done:
            #reward = max(0, 1 - (abs(tilt_angle) / (math.pi/12)))
            #reward = max(0, math.sqrt(3*(math.pi/12)**2) - math.sqrt(roll**2 + pitch**2 + yaw**2))
            reward = max(0, 1-np.tanh(20*(roll**2) + 20*(pitch**2) + 10*(yaw**2)))
        else:
            reward = 0

        current_time = rospy.get_rostime().to_sec()
        self.drift = (current_time-self.sim_time) - self.step_size
        self.sim_time = current_time

        print(' REWARD:', format(reward, '.2f'), 'TIME:', format(self.sim_time, '.5f'))

        return reward

    # Internal TaskEnv Methods
