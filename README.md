# F1TENTH ROS 2 Humble

실차에서 녹화하고 SLAM Toolbox로 생성한 `track02` 맵을 F1TENTH Gym에 적용한 ROS 2 Humble 자율주행 스택입니다. Docker 기반 시뮬레이터, AMCL 위치 추정, raceline 전역 경로, Pure Pursuit, Linear MPC와 반복 실험 도구를 포함합니다.

## 구성

```text
track02 map + LaserScan
          │
          ├─ AMCL ────────────────┐
          │   (map → odom TF)       │
safe raceline → /planning/path          │
          │                              ▼
          └─ Pure Pursuit / Linear MPC → /drive → F1TENTH Gym
```

| 영역 | 현재 구현 |
|---|---|
| Simulation | F1TENTH Gym, Docker Compose, RViz2 |
| Localization | Nav2 AMCL, `map → odom → ego_racecar/base_link` |
| Global path | `track02_raceline_safe.csv` 순환 경로 |
| Control | Pure Pursuit, Linear Time-Varying MPC (CVXPY + OSQP) |
| Evaluation | CTE, lap, collision, solver time CSV/plot |

> 현재 MPC는 정적 raceline 추종 단계입니다. 미등록 장애물 회피를 위한 local planner와 AEB는 아직 포함되지 않았습니다.

## 빠른 시작

### 1. 준비

- Ubuntu + Docker Engine + Docker Compose
- X11 GUI(RViz2)
- NVIDIA GPU는 필수가 아니며, 기본 Compose는 software rendering을 사용합니다.

```bash
git clone https://github.com/Kimz1xq/f1tenth.git
cd f1tenth
xhost +local:docker
docker compose up -d --build
```

컨테이너 접속:

```bash
docker compose exec sim bash
cd /sim_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
```

코드를 수정한 뒤에는 컨테이너에서 다시 빌드합니다.

```bash
cd /sim_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select \
  f1tenth_gym_ros localization planning control vehicle_interface f1tenth_bringup
source install/setup.bash
```

### 2. 시뮬레이터 + AMCL

첫 번째 컨테이너 터미널:

```bash
source /opt/ros/humble/setup.bash
source /sim_ws/install/setup.bash
ros2 launch f1tenth_gym_ros gym_bridge_launch.py
```

RViz2에서 `2D Pose Estimate`로 차량의 초기 위치와 방향을 지정합니다. 설정된 기본 맵은 `maps/track02.png` + `maps/track02.yaml`입니다.

### 3. Raceline 게시

두 번째 터미널:

```bash
docker compose exec sim bash
source /opt/ros/humble/setup.bash
source /sim_ws/install/setup.bash
ros2 launch planning planning.launch.py \
  waypoint_csv:=/sim_ws/src/planning/waypoints/track02_raceline_safe.csv
```

### 4. 컨트롤러 실행

Pure Pursuit:

```bash
ros2 launch control control.launch.py controller:=pure_pursuit
ros2 service call /control/enable std_srvs/srv/SetBool "{data: true}"
```

Linear MPC:

```bash
ros2 launch control control.launch.py controller:=mpc
```

MPC는 기본적으로 disabled dry-run 상태입니다. `/drive` 정지 명령과 제안 제어량을 먼저 확인한 후 주행을 허용하세요.

```bash
ros2 topic echo /drive --once
ros2 topic echo /mpc/proposed_drive --once
ros2 topic echo /mpc/solve_time_ms
ros2 service call /control/enable std_srvs/srv/SetBool "{data: true}"
```

즉시 정지:

```bash
ros2 service call /control/enable std_srvs/srv/SetBool "{data: false}"
```

세부 모델, 파라미터, 튜닝 순서는 [MPC 가이드](algorithms/control/MPC_GUIDE.md)를 참고하세요.

## 상태 확인

```bash
ros2 topic hz /scan
ros2 topic hz /planning/path
ros2 run tf2_ros tf2_echo map ego_racecar/base_link
ros2 run tf2_tools view_frames
rqt_graph
```

정상 TF 체인:

```text
map → odom → ego_racecar/base_link → ego_racecar/laser
```

## 반복 실험

3랩 closed-loop 테스트:

```bash
python3 /sim_ws/src/control/scripts/closed_loop_test.py \
  --duration 210 \
  --laps 3.0 \
  --max-error 0.30 \
  --output /sim_ws/src/control/results/track02_mpc_3laps.csv
```

ROS bag 녹화 예시:

```bash
ros2 bag record \
  /scan /ego_racecar/odom /ground_truth/odom \
  /amcl_pose /particle_cloud /tf /tf_static \
  /planning/path /drive \
  /mpc/reference_path /mpc/predicted_path /mpc/solve_time_ms \
  /ego_racecar/collision \
  -o /sim_ws/src/control/results/mpc_bags/track02_run
```

bag DB3는 크기가 크므로 Git에는 포함하지 않습니다. CSV와 비교 그래프는 `algorithms/control/results/`, `results/`에 있습니다.

## track02 측정 결과

| Controller | Lap [s] | Mean CTE [m] | P95 CTE [m] | Max CTE [m] | Collision |
|---|---:|---:|---:|---:|---:|
| Pure Pursuit safe | 48.45 | 0.068 | 0.116 | 0.164 | 0 |
| MPC baseline | 44.13 | 0.117 | 0.211 | 0.241 | 0 |
| MPC tuned v1 | 42.78 | 0.102 | 0.240 | 0.311 | 0 |
| MPC tuned v2 | 56.56 | 0.070 | 0.111 | 0.140 | 0 |

MPC tuned v2의 solver time은 mean 14.73ms, p95 19.81ms, max 34.33ms였습니다. 현재 파라미터는 최대 속도보다 경로 오차와 안정성에 중점을 둔 보수적 설정입니다.

![Controller comparison](results/track02_controller_comparison.png)

## 맵 교체

1. 맵 이미지(`.pgm` 또는 `.png`)와 YAML을 `maps/`에 놓습니다.
2. YAML의 `image`, `resolution`, `origin`, occupancy threshold를 확인합니다.
3. `config/sim.yaml`의 `map_path`, `map_img_ext`를 수정합니다.
4. 시뮬레이터를 재시작하고 맵과 LaserScan이 일치하는지 확인합니다.

## 주요 폴더

```text
algorithms/localization/       위치 추정 패키지
algorithms/planning/           raceline 게시·생성·검증
algorithms/control/            Pure Pursuit·Linear MPC·평가 스크립트
algorithms/vehicle_interface/  시뮬레이터/실차 제어 인터페이스
config/                        Gym·AMCL 설정
maps/                          track02 포함 occupancy map
launch/                        Gym bridge·RViz2 런치
results/                       컨트롤러 비교 결과
```

## 다음 개발 단계

- 곡률·가속·감속 제약을 반영한 velocity profile
- 랜덤 정적 장애물 2개를 위한 simulator obstacle manager
- LaserScan 기반 local planner + MPC local trajectory tracking
- iTTC 기반 AEB 및 안전 오버라이드
- AMCL/ground truth 오차, lap time, clearance, collision 통합 벤치마크

## License

[MIT License](LICENSE). 이 저장소는 F1TENTH Gym ROS 코드를 ROS 2 Humble 환경과 본 프로젝트에 맞게 확장한 버전입니다.
