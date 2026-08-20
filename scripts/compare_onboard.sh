#!/usr/bin/env bash
set -euo pipefail

# Compare shared autonomy source files with a running onboard container.
# Usage: scripts/compare_onboard.sh jeonbotdae@192.168.1.7

host=${1:?usage: compare_onboard.sh USER@HOST}
remote_root=/home/misys/shared_dir/autonomy_ws/src
files=(
  control/control/pure_pursuit_node.py
  control/control/unicorn_l1_node.py
  control/control/forza_map_node.py
  control/control/linear_mpc_node.py
  control/launch/control.launch.py
  planning/planning/local_obstacle_planner_node.py
  planning/planning/local_planner_core.py
  planning/launch/planning.launch.py
  planning/config/params.yaml
  planning/waypoints/track03_raceline.csv
  f1tenth_bringup/launch/autonomy.launch.py
  f1tenth_bringup/config/vehicle_model.yaml
  f1tenth_bringup/config/tracks.yaml
)

status=0
for file in "${files[@]}"; do
  local_sum=$(sha256sum "algorithms/$file" | awk '{print $1}')
  remote_sum=$(ssh "$host" "docker exec f1tenth sha256sum '$remote_root/$file'" \
    | awk '{print $1}')
  if [[ $local_sum == "$remote_sum" ]]; then
    printf 'SAME  %s\n' "$file"
  else
    printf 'DIFF  %s\n' "$file"
    status=1
  fi
done
exit "$status"
