# F1TENTH 공통 자율주행 스택

ROS 2 Humble에서 실차와 F1TENTH Gym이 같은 경로·플래너·제어기·차량
파라미터를 사용하도록 구성한 저장소입니다. 실행할 때 핵심 차이는
`mode:=sim`과 `mode:=real`입니다.

## 공통 구조

```text
algorithms/planning          raceline, LiDAR 정적 장애물 회피, AEB
algorithms/control           Pure Pursuit, UNICORN L1, ForzaETH MAP, MPC/MPCC
algorithms/f1tenth_bringup    공통 launch, track/vehicle/AMCL 설정
onboard                      실차 map 및 입출력 설정 스냅샷
```

공통 제어 출력은 `AckermannDriveStamped`입니다. 환경별 차이는 필요한
입출력 어댑터에만 남깁니다.

| 항목 | 시뮬레이션 | 실차 |
|---|---|---|
| 차량/센서 | Gym bridge | VESC + URG LiDAR |
| base frame | `ego_racecar/base_link` | `base_link` |
| odometry | `/ego_racecar/odom` | `/odom` |
| 제어 출력 | `/drive` | `/auto` → mux → VESC |
| 초기 자세 | raceline에서 자동 설정 | RViz `2D Pose Estimate` |

위 어댑터 외에는 동일한 소스와 설정을 사용합니다. 실제 타이어 마찰,
조향 영점, actuator 지연, encoder 오차는 bag 데이터로 측정해 Gym 모델을
보정해야 합니다.

## 시뮬레이션

처음 한 번:

```bash
git clone https://github.com/Kimz1xq/f1tenth.git
cd f1tenth
xhost +SI:localuser:root
docker compose up -d --build
docker compose exec sim bash
```

컨테이너 안에서:

```bash
cd /sim_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select \
  f1tenth_gym_ros localization planning control f1tenth_bringup
source install/setup.bash

cd /sim_ws/src/f1tenth_gym_ros
./run_autonomy.sh \
  mode:=sim track:=track03 controller:=pure_pursuit \
  speed:=5.0 maximum_speed:=20.0 \
  obstacles:=true obstacle_seed:=-1 rviz:=true
```

## 실차

노트북에서 온보드 컨테이너 접속:

```bash
ssh -tt jeonbotdae@192.168.1.7 \
  'docker start f1tenth >/dev/null 2>&1 || true; docker exec -it f1tenth bash'
```

온보드 컨테이너에서 환경 설정:

```bash
export ROS_DOMAIN_ID=30
export ROS_LOCALHOST_ONLY=0
export ROS2CLI_NO_DAEMON=1
source /opt/ros/humble/setup.bash
source /home/misys/f1tenth_ws/install/setup.bash
source /home/misys/shared_dir/autonomy_ws/install/setup.bash
```

터미널 1 — 하드웨어(조이스틱/VESC/LiDAR):

```bash
ros2 launch f1tenth_stack bringup_launch.py
```

터미널 2 — localization + planning + controller:

```bash
cd /home/misys/shared_dir
./run_autonomy.sh \
  mode:=real track:=track03 controller:=pure_pursuit \
  speed:=1.0 maximum_speed:=20.0
```

이 명령이 map server와 공통 `amcl_common.yaml`도 자동 실행합니다. 별도의
map server/AMCL lifecycle 명령은 입력하지 않습니다. 실차에서는
`base_frame_id:=base_link`, `/odom`, `/auto` 어댑터만 자동 선택되며 시작
자세는 RViz `2D Pose Estimate`로 지정합니다.

실차 속도는 1 m/s부터 단계적으로 올립니다. `maximum_speed`는 소프트웨어
명령 허용 상한일 뿐이며 VESC·배터리·모터·기어·타이어의 안전 한계를
해제하지 않습니다.

## 시작·정지

모든 제어기는 비활성 상태로 시작합니다.

```bash
ros2 service call /control/enable std_srvs/srv/SetBool "{data: true}"
ros2 service call /control/enable std_srvs/srv/SetBool "{data: false}"
```

## 제어기 선택

launch의 `controller:=` 값만 변경합니다.

```text
pure_pursuit   기본 기하학적 경로 추종
unicorn_l1     HMCL-UNIST의 adaptive L1/Pure-Pursuit 계열 전략
forza_map      ForzaETH Model- and Acceleration-based Pursuit
mpc            선형 kinematic bicycle MPC
mpcc           contour/lag error 기반 nonlinear MPCC
```

UNICORN L1과 ForzaETH MAP은 MPC가 아닙니다. 먼저 PP 계열을 기준선으로
검증하고 같은 map/raceline/speed/obstacle seed에서 CTE, lap time, 충돌,
AEB 횟수를 비교합니다.

참고 구현은 [F1TENTH Adaptive Pure Pursuit](https://github.com/f1tenth-dev/pure_pursuit),
[UNICORN Racing Stack](https://github.com/HMCL-UNIST/unicorn-racing-stack),
[ForzaETH Race Stack](https://github.com/ForzaETH/race_stack)입니다. 이 저장소는
해당 저장소 전체를 복사하지 않고 ROS 2 Humble 공통 입출력 규약에 맞춘
어댑터와 필요한 제어 전략만 유지합니다.

## 현재 재현 기준선

`track03`, 실차와 같은 0.12초 조향 지연·2.5 m/s² 가속 제한, Pure Pursuit
조건에서 확인한 값입니다.

| 조건 | 결과 | 실제 최고속도 | emergency stop |
|---|---:|---:|---:|
| 요청 5.0 m/s, 장애물 없음, 2랩 | PASS | 4.21 m/s | 0.00 s |
| 요청 5.0 m/s, 정적 장애물 2개/랩, 3랩 | PASS | 3.41 m/s | 0.12 s |

장애물은 seed 42에서 시작해 랩마다 43, 44로 재배치했습니다. 이 결과는
충돌 없는 기준선이며 고속 회피 튜닝 완료를 뜻하지 않습니다. `track03`은
약 23 m의 짧고 굽은 코스이므로 10 m/s 성능을 검증하기에는 부적합합니다.
더 높은 속도는 대회 크기 맵에서 같은 검증 절차로 확인해야 합니다.

별도의 411 m 고속 검증 맵에서는 Pure Pursuit가 10.0 m/s 명령에 실제
최고 9.97 m/s, 실제 경로오차 최대 0.44 m로 30초 주행했습니다. 하지만
AMCL 오차가 고속에서 크게 누적됐으므로 이는 제어기 단독 검증 결과이며,
실차 10 m/s 허가 기준이 아닙니다. UNICORN L1은 같은 조건에서 8.8초 후
경로오차 한계를 넘어 현재 고속 기준선으로 채택하지 않았습니다.

## Sim-to-real 검증

공통 파라미터와 출력 구조 검사:

```bash
./scripts/verify_sim2real.py
```

온보드 컨테이너와 소스 해시 비교:

```bash
./scripts/compare_onboard.sh jeonbotdae@192.168.1.7
```

런타임 확인:

```bash
ros2 lifecycle get /map_server
ros2 lifecycle get /amcl
timeout 5 ros2 topic hz /scan
timeout 5 ros2 topic hz /planning/path
ros2 run tf2_ros tf2_echo map base_link
```

시뮬레이션 TF의 base frame은 `ego_racecar/base_link`입니다. 정상 트리는
`map → odom → base_link → laser`이며 namespace만 환경에 따라 다릅니다.

## 새 맵

맵마다 launch 파일을 만들지 않습니다. 다음 파일을 추가하고
`algorithms/f1tenth_bringup/config/tracks.yaml`에 한 번 등록합니다.

```text
maps/<track>.pgm
maps/<track>.yaml
algorithms/planning/waypoints/<track>_raceline.csv
```

차량 공통 모델은
`algorithms/f1tenth_bringup/config/vehicle_model.yaml`, AMCL 공통 모델은
`algorithms/f1tenth_bringup/config/amcl_common.yaml`에서 관리합니다.

## 변경 후 검사

```bash
./scripts/verify_sim2real.py
PYTHONPATH="$PWD:$PWD/algorithms/planning:$PWD/algorithms/control" \
  python3 -m pytest -q \
  test/test_static_obstacles.py \
  algorithms/planning/test/test_local_planner_core.py
```
