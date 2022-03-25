#include "deepbb_hardware_interface/deepbb_hardware_interface.hpp"

#include <limits>
#include <vector>

namespace deepbb_hardware_interface
{
	
	bool DeepBBHardwareInterface::init(ros::NodeHandle & /*root_nh*/, ros::NodeHandle & robot_hw_nh)
	{
		if(!robot_hw_nh.getParam("joint_names", joint_names_))
		{
			ROS_ERROR("Cannot find required parameter 'joint_names' on the parameter server.");
			throw std::runtime_error("Cannot find required parameter "
			"'joint_names' on the parameter server.");
		}
		
		size_t num_joints = joint_names_.size();
		ROS_INFO_NAMED("DeepBBHardwareInterface", "Found %zu joints.", num_joints);
		
		hw_position_states_.resize(num_joints, std::numeric_limits<double>::quiet_NaN());
		hw_velocity_states_.resize(num_joints, std::numeric_limits<double>::quiet_NaN());
		hw_effort_states_.resize(num_joints, std::numeric_limits<double>::quiet_NaN());
		hw_effort_commands_.resize(num_joints, std::numeric_limits<double>::quiet_NaN());
		
		// Create ros_control interfaces
		for(size_t i = 0; i < num_joints; i++)
		{
			// Create joint state interface for all joints
			joint_state_interface_.registerHandle(
					hardware_interface::JointStateHandle(
						joint_names_[i], &hw_position_states_[i], &hw_velocity_states_[i], &hw_effort_states_[i]));
			
			// Create joint effort control interface
			effort_command_interface_.registerHandle(
					hardware_interface::JointHandle(
						joint_state_interface_.getHandle(joint_names_[i]), &hw_effort_commands_[i]));
		}
		
		registerInterface(&joint_state_interface_);
		registerInterface(&effort_command_interface_);
		
		// stat execution on hardware
		ROS_INFO_NAMED("DeepBBHardwareInterface", "Starting...");
		
		// in this simple example reset state to initial positions
		for(size_t i = 0; i < num_joints; i++){
			hw_position_states_[i] = 0.0;
			hw_velocity_states_[i] = 0.0;
			hw_effort_states_[i] = 0.0;
			hw_effort_commands_[i] = hw_effort_states_[i];
		}
		
		return true;
	}
	
	bool DeepBBHardwareInterface::read(const ros::Time time, const ros::Duration period)
	{
		// read robot states from hardware, in this example print only
		ROS_INFO_NAMED("DeepBBHardwareInterface", "Reading...");
		
		for(size_t i = 0; i < hw_position_states_.size(); i++){
			ROS_INFO_NAMED("DeepBBHardwareInterface",
					"Got state %.2f for motor joint %zu!", hw_position_states_[i], i);
		}
		
		return true;
	}
	
	bool DeepBBHardwareInterface::write(const ros::Time time, const ros::Duration period)
	{
		// write command to hardware, in this example do mirror command to states
		for(size_t i = 0; i < hw_effort_commands_.size(); i++){
			hw_effort_states_[i] = hw_effort_commands_[i];
			//hw_effort_states_[i] = hw_effort_states_[i] +
			//	(hw_effort_commands_[i] - hw_effort_states_[i]) / 100.0;
		}
		
		return true;
	}
	
}  // namespace deepbb_hardware_interface
