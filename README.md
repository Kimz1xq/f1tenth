# F1TENTH Simulation

실차에서 만든 지도를 F1TENTH Gym에 적용하고 AMCL·전역경로·MPC·정적 장애물 회피를 반복 검증하는 ROS 2 Humble 환경입니다.

실차 온보드 코드는 [Kimz1xq/f1tenth-onboard](https://github.com/Kimz1xq/f1tenth-onboard)에서 관리합니다.

## 1. 최초 실행

```bash
git clone https://github.com/Kimz1xq/f1tenth.git
cd f1tenth

xhost +SI:localuser:root
docker compose up -d --build
docker compose exec sim bash
```

컨테이너 안에서 한 번 빌드합니다.

```bash
cd /sim_ws
source /opt/ros/humble/setup.bash

colcon build --symlink-install --packages-select \
  f1tenth_gym_ros localization planning control f1tenth_bringup

source install/setup.bash
```

코드를 수정하지 않았다면 다시 빌드할 필요가 없습니다.

## 2. 바로 시뮬레이션하기

장애물 없이 MPC 1.0 m/s:

```bash
ros2 launch f1tenth_bringup autonomy.launch.py \
  track:=track03 \
  controller:=mpc \
  mpc_profile:=speed_1.0 \
  obstacles:=false
```

랜덤 정적 장애물 2개 포함:

```bash
ros2 launch f1tenth_bringup autonomy.launch.py \
  track:=track03 \
  controller:=mpc \
  mpc_profile:=speed_1.0 \
  obstacles:=true
```

속도는 별도 설정 파일 없이 숫자만 바꿉니다.

```text
mpc_profile:=speed_0.5
mpc_profile:=speed_1.4
mpc_profile:=speed_2.0
```

launch가 Gym, RViz, Map Server, AMCL, 전역경로, local planner와 MPC를 모두 실행합니다. MPC는 안전을 위해 정지 상태로 시작합니다.

```bash
# 주행 시작
ros2 service call /control/enable std_srvs/srv/SetBool "{data: true}"

# 즉시 정지
ros2 service call /control/enable std_srvs/srv/SetBool "{data: false}"
```

## 3. 우리가 진행한 검증 순서

### 3-1. 맵과 AMCL만 확인

```bash
ros2 launch f1tenth_bringup autonomy.launch.py \
  track:=track03 controller:=none obstacles:=false
```

RViz에서 다음을 확인합니다.

- Map과 LaserScan이 겹치는지
- AMCL particle이 차량 주변으로 수렴하는지
- 차량 이동과 RViz 이동이 일치하는지

```bash
ros2 run tf2_ros tf2_echo map ego_racecar/base_link
ros2 run tf2_tools view_frames
rqt_graph
```

정상 TF:

```text
map -> odom -> ego_racecar/base_link -> ego_racecar/laser
```

### 3-2. 전역경로 확인

현재 실행 경로는 다음 파일입니다.

```text
algorithms/planning/waypoints/track03_raceline.csv
```

RViz의 `/planning/global_path`가 벽을 통과하지 않고 폐곡선을 이루는지 확인합니다. 새 맵의 중심선이 필요하면 다음처럼 생성합니다.

```bash
python3 /sim_ws/src/planning/scripts/generate_centerline.py \
  --map-yaml /sim_ws/src/f1tenth_gym_ros/maps/track03.yaml \
  --output /sim_ws/src/planning/waypoints/track03_centerline.csv \
  --preview /tmp/track03_centerline.png
```

### 3-3. MPC 검증

처음에는 장애물을 끄고 저속부터 올립니다.

```text
speed_0.5 -> speed_0.8 -> speed_1.0 -> 필요한 목표 속도
```

주행 전:

```bash
ros2 topic echo /mpc/proposed_drive --once
ros2 topic echo /mpc/solve_time_ms
ros2 service call /control/enable std_srvs/srv/SetBool "{data: true}"
```

### 3-4. 장애물 회피 검증

`obstacles:=true`일 때 local planner가 `/scan`과 `/map`으로 회피 경로를 만들고 MPC가 그 경로를 추종합니다.

```bash
ros2 topic echo /planning/local_status
ros2 topic echo /safety/emergency_stop

ros2 service call /simulation/randomize_obstacles std_srvs/srv/Trigger "{}"
ros2 service call /simulation/clear_obstacles std_srvs/srv/Trigger "{}"
```

상태 의미:

```text
GLOBAL_PATH_CLEAR       장애물 없음
AVOIDING ...            회피 경로 선택됨
NO_COLLISION_FREE_PATH  안전한 후보 경로 없음
AEB_STOP                긴급 제동
```

장애물은 시작할 때와 한 랩을 돌 때마다 다시 배치됩니다.

## 4. 새 맵 적용

최종 파일만 다음 이름으로 관리합니다.

```text
maps/<track>.pgm 또는 .png
maps/<track>.yaml
algorithms/planning/waypoints/<track>_centerline.csv
algorithms/planning/waypoints/<track>_raceline.csv
```

그다음 `config/tracks.yaml`에 맵, 시작 자세, 경로와 마찰계수를 한 번 등록합니다. `sim.yaml`이나 launch 파일을 맵마다 복제하지 않습니다.

```yaml
tracks:
  track03:
    map_path: /sim_ws/src/f1tenth_gym_ros/maps/track03
    map_ext: .pgm
    centerline: /sim_ws/src/planning/waypoints/track03_centerline.csv
    raceline: /sim_ws/src/planning/waypoints/track03_raceline.csv
    start: [0.2985, 0.5926, -0.6205]
    friction_mu: 1.0489
```

## 5. 문제 발생 시

```bash
ros2 node list
ros2 topic hz /scan
ros2 topic hz /planning/path
ros2 run tf2_ros tf2_echo map ego_racecar/base_link
```

- 맵이 안 바뀜: `config/tracks.yaml`의 `map_path`, `map_ext` 확인
- MPC가 바로 멈춤: AMCL, heading error, `/safety/emergency_stop` 확인
- 벽으로 감: RViz 초기 자세 방향과 raceline 진행 방향 확인
- 장애물을 못 피함: `/planning/local_status`가 `AVOIDING`인지 먼저 확인

상세 MPC 파라미터는 [MPC_GUIDE.md](algorithms/control/MPC_GUIDE.md)를 참고합니다.

## 주요 설정 파일

```text
config/tracks.yaml                       맵·경로·시작 자세·마찰계수
algorithms/control/config/mpc_params.yaml MPC 공통 파라미터
algorithms/planning/config/params.yaml    장애물 검출·회피 파라미터
```

## License

[MIT License](LICENSE)
