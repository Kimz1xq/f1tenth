# F1TENTH 시뮬레이션

실차에서 만든 지도 위에서 AMCL, 전역 경로, UNICORN L1, Linear MPC와 LiDAR 장애물 회피를 검증하는 ROS 2 Humble 환경입니다.

- 실차 저장소: [Kimz1xq/f1tenth-onboard](https://github.com/Kimz1xq/f1tenth-onboard)
- 시뮬레이터: [F1TENTH Gym ROS](https://github.com/f1tenth/f1tenth_gym_ros)
- 현재 지도/경로: `track03`

## 구성 원칙

시뮬레이션과 실차에서 아래 세 패키지의 소스와 파라미터를 동일하게 유지합니다.

```text
algorithms/planning           전역 경로와 LiDAR 로컬 회피
algorithms/control            UNICORN L1, Linear MPC, Pure Pursuit
algorithms/f1tenth_bringup     공통 autonomy launch
```

실차 저장소에서는 같은 패키지가 `autonomy_ws/src/` 아래에 있습니다. 차이가 필요한 부분은 입출력 계층뿐입니다.

| 구분 | 시뮬레이션 | 실차 |
|---|---|---|
| 차량/센서 | Gym bridge | VESC, URG LiDAR, joystick |
| 기준 프레임 | `ego_racecar/base_link` | `base_link` |
| Odometry | `/ego_racecar/odom` | `/odom` |
| 출력 | `/drive` | `/auto` → Ackermann mux → VESC |
| 초기 자세 | track 설정에서 자동 주입 | RViz에서 직접 지정 |

코드가 같아도 마찰계수, 타이어, 조향 영점, 모터 deadband, LiDAR 지연은 실차에서 별도 측정해야 합니다.

## 처음 한 번

```bash
git clone https://github.com/Kimz1xq/f1tenth.git
cd f1tenth
xhost +SI:localuser:root
docker compose up -d --build
docker compose exec sim bash
```

컨테이너 안에서 빌드합니다.

```bash
cd /sim_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select \
  f1tenth_gym_ros localization planning control f1tenth_bringup
source install/setup.bash
```
이후에는 새 터미널마다 다음만 실행합니다.

```bash
cd ~/f1tenth
docker compose up -d
docker compose exec sim bash
```

```bash
source /opt/ros/humble/setup.bash
source /sim_ws/install/setup.bash
```

## 제어기 선택과 실행

모든 제어기는 안전을 위해 비활성 상태로 시작합니다. 속도는 파일을 만들지 않고 launch 인자로 지정합니다.

### 기본 UNICORN L1

```bash
ros2 launch f1tenth_bringup autonomy.launch.py \
  mode:=sim track:=track03 \
  controller:=unicorn_l1 \
  mpc_profile:=speed_3.0 \
  obstacles:=true rviz:=true
```

### 동적 회피 속도 제한 UNICORN L1

```bash
ros2 launch f1tenth_bringup autonomy.launch.py \
  mode:=sim track:=track03 \
  controller:=unicorn_l1_dynamic \
  mpc_profile:=speed_3.0 \
  obstacles:=true rviz:=true
```

### Linear MPC 비교

```bash
ros2 launch f1tenth_bringup autonomy.launch.py \
  mode:=sim track:=track03 \
  controller:=mpc \
  mpc_profile:=speed_3.0 \
  obstacles:=true rviz:=true
```

주행 시작/정지:

```bash
ros2 service call /control/enable std_srvs/srv/SetBool "{data: true}"
ros2 service call /control/enable std_srvs/srv/SetBool "{data: false}"
```

`obstacles:=false`로 먼저 경로 추종을 확인한 다음 `true`로 회피를 검증합니다. 실차에서는 장애물 플래너가 항상 실행되며 LiDAR 관측에 따라 자동 전환합니다.

## 사용 중인 모델

- `AMCL`: Nav2의 particle-filter 기반 2D map localization
- 전역 경로: 지도 free-space에서 생성한 폐곡선 raceline CSV
- `unicorn_l1`: [HMCL-UNIST UNICORN Racing Stack](https://github.com/HMCL-UNIST/unicorn-racing-stack)의 속도·곡률 기반 L1/Pure-Pursuit 전략을 ROS 2 Humble 인터페이스로 이식한 제어기
- `unicorn_l1_dynamic`: 같은 L1 제어기에 로컬 플래너의 `/planning/speed_limit`을 추가 적용한 모드
- `mpc`: kinematic bicycle model을 선형화하여 경로 오차와 조향 입력을 최적화하는 저장소 내 Linear MPC 비교 구현
- 장애물 회피: `/scan`과 `/map`으로 정적 장애물을 추적하고 충돌하지 않는 offset 후보 경로를 선택하며 AEB를 별도로 적용

UNICORN L1은 MPC가 아닙니다. 같은 지도, 경로, 속도, 장애물 seed로 제어기별 lap time, CTE, 충돌, safety stop 횟수를 비교합니다.

## 검증 명령

```bash
ros2 lifecycle get /map_server
ros2 lifecycle get /amcl
ros2 run tf2_ros tf2_echo map ego_racecar/base_link
timeout 5 ros2 topic hz /scan
timeout 5 ros2 topic hz /ego_racecar/odom
timeout 5 ros2 topic hz /planning/path
ros2 topic echo /planning/local_status
ros2 topic echo /planning/avoidance_active
ros2 topic echo /safety/emergency_stop
```

정상 TF:

```text
map -> odom -> ego_racecar/base_link -> ego_racecar/laser
```

랜덤 장애물 제어:

```bash
ros2 service call /simulation/randomize_obstacles std_srvs/srv/Trigger "{}"
ros2 service call /simulation/clear_obstacles std_srvs/srv/Trigger "{}"
```

## 새 맵 적용

맵별 launch 파일을 만들지 않고 `config/tracks.yaml`에 한 번 등록합니다.

```text
maps/<track>.pgm
maps/<track>.yaml
algorithms/planning/waypoints/<track>_centerline.csv
algorithms/planning/waypoints/<track>_raceline.csv
```

주요 설정:

```text
config/tracks.yaml                          맵, 경로, 시작 자세, 마찰계수
algorithms/planning/config/params.yaml      검출, 회피, AEB
algorithms/control/config/mpc_params.yaml   Linear MPC 공통 파라미터
config/amcl.yaml                            시뮬레이션 AMCL 프레임과 초기 자세
```

## 변경 후 확인

```bash
cd /sim_ws
colcon build --symlink-install --packages-select planning control f1tenth_bringup
colcon test --packages-select planning control f1tenth_bringup
colcon test-result --verbose
source install/setup.bash
```
