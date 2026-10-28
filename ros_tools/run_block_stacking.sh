#!/usr/bin/env bash
# Keep ROS in WSL; the Windows .venv cannot load Humble's Linux libraries.
set -eo pipefail
project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
venv_dir="${AI_GAMBIT_ROS_VENV:-$HOME/.local/share/ai-gambit/venv}"
if [[ ! -f /opt/ros/humble/setup.bash || ! -f "$venv_dir/bin/activate" ]]; then
  echo "Need apt ROS Humble and a Linux Python 3.10 venv. See block_stacking_sim/README.md." >&2
  exit 1
fi
unset AMENT_PREFIX_PATH COLCON_PREFIX_PATH CMAKE_PREFIX_PATH PYTHONPATH LD_LIBRARY_PATH
source /opt/ros/humble/setup.bash
source "$venv_dir/bin/activate"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-53}"
cd "$project_dir"
exec python block_stacking_sim/stack_engine.py "$@"
