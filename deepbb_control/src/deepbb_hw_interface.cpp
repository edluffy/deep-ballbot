#include <deepbb_control/deepbb_hw_interface.h>

namespace deepbb_ns
{
DeepbbHWInterface::DeepbbHWInterface(ros::NodeHandle& nh, urdf::Model* urdf_model)
  : ros_control_boilerplate::GenericHWInterface(nh, urdf_model)
{
	motor_sub1 = nh.subscribe("/motor_state1", 1, &DeepbbHWInterface::motorCallback1, this);
	motor_sub2 = nh.subscribe("/motor_state2", 1, &DeepbbHWInterface::motorCallback2, this);
	motor_sub3 = nh.subscribe("/motor_state3", 1, &DeepbbHWInterface::motorCallback3, this);

	motor_pub1 = nh.advertise<deepbb_control::MotorState>("/motor_command1", 1);
	motor_pub2 = nh.advertise<deepbb_control::MotorState>("/motor_command2", 1);
	motor_pub3 = nh.advertise<deepbb_control::MotorState>("/motor_command3", 1);

	ROS_INFO("DeepbbHWInterface consttucted");

}

/* MotorState.msg:
	uint32 counter
	float32 current
	float32 shaft_angle
	float32 shaft_velocity
*/
void DeepbbHWInterface::motorCallback1(const deepbb_control::MotorState::ConstPtr &msg)
{
	joint_position_[0] = msg->shaft_angle;
	joint_velocity_[0] = msg->shaft_velocity;
}
void DeepbbHWInterface::motorCallback2(const deepbb_control::MotorState::ConstPtr &msg)
{
	joint_position_[1] = msg->shaft_angle;
	joint_velocity_[1] = msg->shaft_velocity;
}
void DeepbbHWInterface::motorCallback3(const deepbb_control::MotorState::ConstPtr &msg)
{
	joint_position_[2] = msg->shaft_angle;
	joint_velocity_[2] = msg->shaft_velocity;
}


void DeepbbHWInterface::init()
{
  // Call parent class version of this function
  GenericHWInterface::init();

  ROS_INFO("DeepbbHWInterface Ready.");
}

void DeepbbHWInterface::read(ros::Duration& elapsed_time)
{
  // No need to read since our write() command populates our state for us
	ros::spinOnce();
}

void DeepbbHWInterface::write(ros::Duration& elapsed_time)
{
  // Safety
  //enforceLimits(elapsed_time);
	static deepbb_control::MotorState motor_command1;
	static deepbb_control::MotorState motor_command2;
	static deepbb_control::MotorState motor_command3;

	motor_command1.shaft_angle = joint_position_command_[0];
	motor_command1.shaft_velocity = joint_velocity_command_[0];

	motor_command2.shaft_angle = joint_position_command_[1];
	motor_command2.shaft_velocity = joint_velocity_command_[1];

	motor_command3.shaft_angle = joint_position_command_[2];
	motor_command3.shaft_velocity = joint_velocity_command_[2];

	motor_pub1.publish(motor_command1);
	motor_pub2.publish(motor_command2);
	motor_pub3.publish(motor_command3);
}

void DeepbbHWInterface::enforceLimits(ros::Duration& period)
{
  // Enforces position and velocity
  //pos_jnt_sat_interface_.enforceLimits(period);
}

}  // namespace deepbb_ns
