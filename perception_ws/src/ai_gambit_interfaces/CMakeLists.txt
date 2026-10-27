cmake_minimum_required(VERSION 3.8)
project(ai_gambit_interfaces)
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/BoardObservation.msg" "msg/RackObservation.msg"
  "srv/Snapshot.srv" "srv/SetBaseline.srv" "srv/VerifyState.srv"
  "msg/Connect4Observation.msg" "srv/Connect4Baseline.srv" "action/ObserveConnect4.action"
  DEPENDENCIES std_msgs builtin_interfaces)
ament_export_dependencies(rosidl_default_runtime)
ament_package()
