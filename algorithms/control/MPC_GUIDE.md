# track02 Linear MPC 사용법

이 구현은 MIT 라이선스의 F1TENTH Lab 7 MPC 예제에서 kinematic bicycle
model과 QP 구조를 참고하고, 현재 ROS 2/AMCL/raceline 구성에 맞게 수정했다.

- 상태: `[map_x, map_y, speed, yaw]`
- 입력: `[acceleration, steering_angle]`
- solver: CVXPY + OSQP
- 제어 주기: 10 Hz
- horizon: 16 step × 0.1 s = 1.6 s
- 기본 상태: disabled. disabled 상태에서도 해와 예측 경로만 계산하며 `/drive`에는
  항상 정지 명령을 보낸다.
- 안전 정지: collision, stale odometry/path, TF 오류, 큰 경로/방향 오차,
  연속 solver 실패

원본 참고 코드:
<https://github.com/jasonf27/f1tenth_autonomous_anonymous/tree/main/lab-7-model-predictive-control-autonomous-anonymous-main>

## 실행

Gym, AMCL, planning이 실행된 상태에서 다음을 사용한다.

```bash
docker exec -it f1tenth_gym_ros_humble-sim-1 bash
source /opt/ros/humble/setup.bash
source /sim_ws/install/setup.bash
ros2 launch control control.launch.py controller:=mpc
```

다른 터미널에서 dry-run 출력을 확인한다.

```bash
ros2 topic echo /drive --once
ros2 topic echo /mpc/proposed_drive --once
ros2 topic echo /mpc/solve_time_ms
```

`/drive`가 `speed: 0.0`인 것을 확인한 뒤에만 주행을 시작한다.

```bash
ros2 service call /control/enable std_srvs/srv/SetBool "{data: true}"
```

정지:

```bash
ros2 service call /control/enable std_srvs/srv/SetBool "{data: false}"
```

RViz에서는 `/mpc/reference_path`가 주황색, `/mpc/predicted_path`가 파란색이다.

## 설정과 튜닝 순서

현재 적용 파일은 `config/mpc_params.yaml`이다. 실험했던 원본 설정은
`mpc_params_baseline.yaml`, 첫 튜닝은 `mpc_params_tuned_v1.yaml`로 보존했다.

1. `target_speed`, `corner_slowdown_gain`, `min_reference_speed`로 속도를 먼저
   맞춘다. 오차가 큰 급커브에서는 slowdown gain을 올리거나 min speed를 내린다.
2. `q_x`, `q_y`, `q_yaw`를 올리면 경로/방향 오차를 더 강하게 줄인다.
3. `r_steering`, `rd_steering`을 올리면 조향은 부드러워지지만 급커브 반응이
   느려진다.
4. `horizon_steps`를 올리면 더 멀리 보지만 계산량이 증가한다. 반드시
   `/mpc/solve_time_ms`의 p95가 100 ms보다 작은지 확인한다.
5. 한 번에 한 종류만 바꾸고 동일 시작 자세, 1랩, collision 0 조건으로 비교한다.

## 이번 측정 결과

| Controller | Lap [s] | Mean CTE [m] | P95 CTE [m] | Max CTE [m] | Collision |
|---|---:|---:|---:|---:|---:|
| Pure Pursuit safe | 48.45 | 0.068 | 0.116 | 0.164 | 0 |
| MPC baseline | 44.13 | 0.117 | 0.211 | 0.241 | 0 |
| MPC tuned v1 | 42.78 | 0.102 | 0.240 | 0.311 | 0 |
| MPC tuned v2 | 56.56 | 0.070 | 0.111 | 0.140 | 0 |

MPC tuned v2 solver 시간은 mean 14.73 ms, p95 19.81 ms, max 34.33 ms로
10 Hz deadline인 100 ms 이내였다.

## Bag 분석

```bash
python3 /sim_ws/src/control/scripts/analyze_mpc_bag.py \
  /sim_ws/src/control/results/mpc_bags/track02_mpc_tuned_v2_01
```

## 장애물 회피 범위

현재 단계는 최적화된 safe raceline을 따라가므로 지도에 이미 있는 벽은 피한다.
하지만 주행 중 새로 놓인 동적/미등록 장애물을 우회하는 local planning은 아직 없다.
LaserScan으로 장애물을 검출해 단순 정지만 붙이는 것은 emergency safety이고,
회피라 부르지 않는다. 다음 단계에서는 scan 기반 obstacle cluster와 local occupancy를
만들고, raceline 좌우의 후보 경로를 평가하거나 sequential convex MPC에 obstacle
distance constraint를 추가해야 한다.
