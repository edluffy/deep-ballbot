#ifndef DEEPBB_HW_INTERFACE_H
#define DEEPBB_HW_INTERFACE_H

#include <ros_control_boilerplate/generic_hw_interface.h>
#include <deepbb_control/MotorState.h>

namespace deepbb_ns
{
/** \brief Hardware interface for a robot */
class DeepbbHWInterface : public ros_control_boilerplate::GenericHWInterface
{
public:
  /**
   * \brief Constructor
   * \param nh - Node handle for topics.
   */
  DeepbbHWInterface(ros::NodeHandle& nh, urdf::Model* urdf_model = NULL);

  /** \brief Initialize the robot hardware interface */
  virtual void init();

  /** \brief Read the state from the robot hardware. */
  virtual void read(ros::Duration& elapsed_time);

  /** \brief Write the command to the robot hardware. */
  virtual void write(ros::Duration& elapsed_time);

  /** \breif Enforce limits for all values before writing */
  virtual void enforceLimits(ros::Duration& period);

protected:
  ros::Subscriber motor_sub1;
  ros::Subscriber motor_sub2;
  ros::Subscriber motor_sub3;

  void motorCallback1(const deepbb_control::MotorState::ConstPtr &msg);
  void motorCallback2(const deepbb_control::MotorState::ConstPtr &msg);
  void motorCallback3(const deepbb_control::MotorState::ConstPtr &msg);

  ros::Publisher motor_pub1;
  ros::Publisher motor_pub2;
  ros::Publisher motor_pub3;

};  // class

}  // namespace ros_control_boilerplate

#endif
