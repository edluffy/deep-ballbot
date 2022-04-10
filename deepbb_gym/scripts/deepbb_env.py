import rospy
from openai_ros import robot_gazebo_env
from sensor_msgs.msg import JointState, Imu
from std_msgs.msg import Float64

class DeepBBEnv(robot_gazebo_env.RobotGazeboEnv):
    def __init__(self):
        # Variables that we give through the constructor.

        # Internal Vars
        self.controllers_list = ['joint_state_controller', 'motor1_velocity_controller',
                'motor2_velocity_controller', 'motor3_velocity_controller']

        self.robot_name_space = "deep_ballbot"

        reset_controls_bool = True

        super(DeepBBEnv, self).__init__(controllers_list=self.controllers_list,
                                                robot_name_space=self.robot_name_space,
                                                reset_controls=reset_controls_bool)
        self.gazebo.unpauseSim()
        self.controllers_object.reset_controllers()
        self._check_all_sensors_ready()


        rospy.Subscriber('/deep_ballbot/joint_states', JointState, self._joints_callback)
        rospy.Subscriber('/imu', Imu, self._imu_callback)

        self._motor1_velocity_pub = rospy.Publisher('/deep_ballbot/motor1_velocity_controller/command',
                Float64, queue_size=1)
        self._motor2_velocity_pub = rospy.Publisher('/deep_ballbot/motor2_velocity_controller/command',
                Float64, queue_size=1)
        self._motor3_velocity_pub = rospy.Publisher('/deep_ballbot/motor3_velocity_controller/command',
                Float64, queue_size=1)

        self._check_publishers_connection()
        self.gazebo.pauseSim()


    # Methods needed by the RobotGazeboEnv
    # ----------------------------

    def _check_all_systems_ready(self):
        self._check_all_sensors_ready()
        self._check_publishers_connection()
        return True

    # DeepBBEnv virtual methods
    # ----------------------------

    def _check_all_sensors_ready(self):
        self._check_joint_states_ready()
        self._check_imu_ready()
        rospy.logdebug("ALL SENSORS READY")

    def _check_joint_states_ready(self):
        self.joints = None
        while self.joints is None and not rospy.is_shutdown():
            try:
                self.joints = rospy.wait_for_message("/deep_ballbot/joint_states", JointState, timeout=1.0)
                rospy.logdebug("Current deep_ballbot/joint_states READY=>" + str(self.joints))

            except:
                rospy.logerr("Current deep_ballbot/joint_states not ready yet, retrying getting joint_states")
        return self.joints

    def _check_imu_ready(self):
        self.imu = None
        while self.imu is None and not rospy.is_shutdown():
            try:
                self.imu = rospy.wait_for_message("/imu", Imu, timeout=1.0)
                rospy.logdebug("Current /imu READY=>" + str(self.imu))

            except:
                rospy.logerr("Current /imu not ready yet, retrying getting imu")

        return self.imu

    def _joints_callback(self, data):
        self.joints = data

    def _imu_callback(self, data):
        self.imu = data

    def _check_publishers_connection(self):
        rate = rospy.Rate(10)
        while self._motor1_velocity_pub.get_num_connections() == 0 \
                and self._motor2_velocity_pub.get_num_connections() == 0 \
                and self._motor3_velocity_pub.get_num_connections() == 0 \
                and not rospy.is_shutdown():
            rospy.logdebug("No subscribers to _motor_velocity_pubs, waiting and trying again")
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                pass

            rospy.logdebug("_motor_velocity_pubs connected")
            rospy.logdebug("All Publishers READY")

    # Methods that the TaskEnvironment will need to define here as virtual
    # because they will be used in RobotGazeboEnv GrandParentClass and defined in the
    # TaskEnvironment.
    # ----------------------------
    def _set_init_pose(self):
        """Sets the Robot in its init pose
        """
        raise NotImplementedError()

    def _init_env_variables(self):
        """Inits variables needed to be initialised each time we reset at the start
        of an episode.
        """
        raise NotImplementedError()

    def _compute_reward(self, observations, done):
        """Calculates the reward to give based on the observations given.
        """
        raise NotImplementedError()

    def _set_action(self, action):
        """Applies the given action to the simulation.
        """
        raise NotImplementedError()

    def _get_obs(self):
        raise NotImplementedError()

    def _is_done(self, observations):
        """Checks if episode done based on observations given.
        """
        raise NotImplementedError()

    # Methods that the TaskEnvironment will need.
    # ----------------------------

    def move_joints(self, vels):
        rospy.logdebug("Motor vels >> " + str(vels))
        self.publish_velocity(vels[0], self._motor1_velocity_pub)
        self.publish_velocity(vels[1], self._motor2_velocity_pub)
        self.publish_velocity(vels[2], self._motor3_velocity_pub)
        #self.wait_to_reach_velocity(vels)

    def publish_velocity(self, vel, pub):
        vel_msg = Float64()
        vel_msg.data = vel
        pub.publish(vel_msg)

    def wait_to_reach_velocity(self, target):
        #rate = rospy.Rate(1000)
        start_time = rospy.get_rostime().to_sec()
        end_time = 0.0
        bound = 1

        while not rospy.is_shutdown():
            joint_states = self._check_joint_states_ready()
            actual = joint_states.velocity

            if (actual[0]<=target[0]+bound and actual[0]>=target[0]-bound
            and actual[1]<=target[1]+bound and actual[1]>=target[1]-bound
            and actual[2]<=target[2]+bound and actual[2]>=target[2]-bound):
                end_time = rospy.get_rostime().to_sec()
                break
            #rate.sleep()

        wait_time = end_time-start_time
        rospy.logdebug('Velocity wait time:' + str(wait_time))
        print('Velocity wait time:' + str(wait_time))
        return wait_time

    def get_joints(self):
        return self.joints

    def get_imu(self):
        return self.imu

