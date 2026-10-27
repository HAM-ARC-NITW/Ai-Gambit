#!/usr/bin/env bash
# WSL Ubuntu 22.04. Separate Linux output avoids changing other ROS workspaces.
# bash ros_tools/run_connect4_ros.sh build|run|test
set -eo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
runtime_dir="$HOME/.local/share/ai-gambit-connect4"
venv_dir="${AI_GAMBIT_ROS_VENV:-$HOME/.local/share/ai-gambit/venv}"
if [[ ! -f /opt/ros/humble/setup.bash || ! -f "$venv_dir/bin/activate" ]]; then
  echo "Need apt ROS Humble and a Linux Python 3.10 venv with system-site-packages. See connect_4_perception/README.md." >&2
  exit 1
fi
unset AMENT_PREFIX_PATH COLCON_PREFIX_PATH CMAKE_PREFIX_PATH PYTHONPATH LD_LIBRARY_PATH
source /opt/ros/humble/setup.bash
source "$venv_dir/bin/activate"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-52}"
cd "$project_dir"
if [[ "${1:-run}" == build ]]; then
  exec python /usr/bin/colcon --log-base "$runtime_dir/log" build --base-paths perception_ws/src \
    --build-base "$runtime_dir/build" --install-base "$runtime_dir/install" --symlink-install \
    --cmake-args -DPython3_EXECUTABLE="$venv_dir/bin/python" -DPYTHON_EXECUTABLE="$venv_dir/bin/python" -DBUILD_TESTING=OFF
fi
source "$runtime_dir/install/setup.bash"
if [[ "${1:-run}" == test ]]; then
  exec python connect_4_perception/check_ros.py
fi
if [[ "${1:-run}" != run ]]; then echo "Use build, run, or test" >&2;exit 1;fi
shutdown_nodes() {
  trap - INT TERM HUP EXIT
  kill -TERM "$camera_pid" "$perception_pid" 2>/dev/null || true
  wait "$camera_pid" "$perception_pid" 2>/dev/null || true
}
# Python nodes only; stop exactly our two children, not other ROS applications.
python connect_4_perception/publish_camera.py &
camera_pid=$!
python "$runtime_dir/install/ai_gambit_perception/lib/ai_gambit_perception/connect4_node" --ros-args -p max_frame_age:=2.0 &
perception_pid=$!
trap shutdown_nodes INT TERM HUP EXIT
wait -n "$camera_pid" "$perception_pid"
