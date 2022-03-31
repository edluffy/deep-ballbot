from gym import spaces
import deepbb_env
from gym.envs.registration import register
import rospy
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
        6      Linear Acceleration x
        7      Linear Acceleration y
        8      Linear Acceleration z

    Actions:
        Type: Box(3)
        Num    Action
        0      Motor 1 Torque Command
        1      Motor 2 Torque Command
        2      Motor 3 Torque Command

    Reward:
    """
    def __init__(self):
        number_actions = 3
        self.action_space = spaces.Discrete(number_actions)

        high = np.ones(9)
        self.observation_space = spaces.Box(-high, high)

        self.max_torque = 1.0
        self.min_torque = -1.0

        self.max_lean_angle = 0.5
        self.max_angular_velocity = 10

        self.end_episode_penalty = -10

        super(DeepBBBalanceEnv, self).__init__()


    def _set_init_pose(self):
        self.move_joints([0, 0, 0])

    def _init_env_variables(self):
        """
        Inits variables needed to be initialised each time we reset at the start
        of an episode.
        :return:
        """
        # TODO


    def _set_action(self, action):
        torques = np.clip(action, self.min_torque, self.max_torque)
        self.move_joints(torques)


    def _get_obs(self):
        q = [self.imu.orientation.x, self.imu.orientation.y,
                self.imu.orientation.z, self.imu.orientation.w]
        roll, pitch, yaw = euler_from_quaternion(q)

        obs = [
            roll,
            pitch,
            yaw,
            self.imu.angular_velocity.x,
            self.imu.angular_velocity.y,
            self.imu.angular_velocity.z,
            self.joints.position[0],
            self.joints.position[1],
            self.joints.position[2],
            self.joints.velocity[0],
            self.joints.velocity[1],
            self.joints.velocity[2],
        ]

        return obs

    def _is_done(self, obs):
        # Unbounded yaw is ok
        roll = obs[0]
        pitch = obs[1]

        angular_velocity_x = obs[3]
        angular_velocity_y = obs[4]
        angular_velocity_z = obs[5]

        done = False

        if abs(roll) > self.max_lean_angle:
            rospy.logerr("Roll exceeded maximum lean angle=>"+str(roll))
            done = True

        if abs(pitch) > self.max_lean_angle:
            rospy.logerr("Pitch exceeded maximum lean angle=>"+str(pitch))
            done = True

        if abs(angular_velocity_x) > self.max_angular_velocity:
            rospy.logerr("Angular velocity in x exceeded the maximum=>"\
                    +str(angular_velocity_x))
            done = True

        if abs(angular_velocity_y) > self.max_angular_velocity:
            rospy.logerr("Angular velocity in y exceeded the maximum=>"\
                    +str(angular_velocity_y))
            done = True

        if abs(angular_velocity_z) > self.max_angular_velocity:
            rospy.logerr("Angular velocity in z exceeded the maximum=>"\
                    +str(angular_velocity_z))
            done = True

        return done

    def _compute_reward(self, obs, done):
        roll = obs[0]
        pitch = obs[1]

        if not done:
            reward = 1-abs(roll + pitch)
        else:
            reward = self.end_episode_penalty

        return reward

    # Internal TaskEnv Methods
