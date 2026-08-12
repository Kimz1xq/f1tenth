# Linear MPC 가이드

현재 controller는 F1TENTH Lab 7의 kinematic bicycle model과 QP 구조를 참고해
ROS 2, AMCL, local path 환경에 맞춘 Linear Time-Varying MPC입니다.

- 참고 코드: <https://github.com/jasonf27/f1tenth_autonomous_anonymous/tree/main/lab-7-model-predictive-control-autonomous-anonymous-main>
- 상태: `[map_x, map_y, speed, yaw]`
- 입력: `[acceleration, steering_angle]`
- solver: CVXPY + OSQP
- 기본 상태: disabled dry-run
- 안전 정지: collision, AEB, stale topic/TF, 큰 경로 오차, solver 연속 실패

MPC는 장애물을 센서에서 찾는 알고리즘이 아닙니다. planning이 `/scan`과 `/map`으로
충돌 없는 `/planning/path`를 만들고, MPC는 차량의 조향·가속·속도 제약을 고려해 그
경로를 예측 추종합니다.

## 실행

```bash
ros2 launch f1tenth_bringup autonomy.launch.py \
  track:=track03 controller:=mpc mpc_profile:=speed_0.55 \
  friction:=auto obstacles:=false
```

dry-run 확인 후 시작합니다.

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

RViz의 `/mpc/reference_path`는 주황색, `/mpc/predicted_path`는 파란색입니다.

## 프로필과 튜닝

모든 설정은 `config/mpc_params.yaml` 한 파일에 있습니다. `common`은 안전·차량
공통값이고 `speed_template`은 모든 동적 속도 실행에서 공유합니다. 속도별 파일이나
YAML 항목을 만들지 않고 `mpc_profile:=speed_0.85`처럼 실행 시 지정합니다.

권장 순서:

1. 동일 맵, 시작 자세, 마찰계수, 장애물 seed를 고정합니다.
2. `target_speed`, `max_speed`, `corner_slowdown_gain`으로 속도 프로필을 맞춥니다.
3. `q_x`, `q_y`, `q_yaw`로 경로와 방향 오차를 조절합니다.
4. `r_steering`, `rd_steering`으로 급격한 조향을 억제합니다.
5. 필요할 때만 `horizon_steps`를 늘리고 solver p95가 제어주기 100 ms보다 작은지
   확인합니다.
6. 한 번에 한 파라미터 묶음만 바꾸고 충돌, lap time, CTE, solver time을 기록합니다.

마찰계수 비교는 MPC 파일을 복사하지 않고 launch에서 선택합니다.

```bash
ros2 launch f1tenth_bringup autonomy.launch.py \
  track:=track03 controller:=mpc mpc_profile:=speed_0.85 friction:=0.70
```

## 현재 검증 범위

| Profile | Lap [s] | True CTE mean/P95/max [m] | Collision |
|---|---:|---:|---:|
| baseline | 44.13 | 0.136 / 0.266 / 0.306 | 0 |
| tuned_v1 | 42.78 | 0.118 / 0.301 / 0.347 | 0 |
| tuned_v2 | 56.56 | 0.077 / 0.150 / 0.182 | 0 |

tuned_v2 solver 시간은 mean 14.73 ms, p95 19.81 ms, max 34.33 ms였습니다.
이 결과는 track02, 저속, 장애물 없는 1랩 조건의 기준선입니다. 고속·저마찰·다중랩과
랜덤 장애물 조건은 별도 검증이 필요합니다.

## 장애물 조건 확인

```bash
ros2 service call /simulation/randomize_obstacles std_srvs/srv/Trigger "{}"
ros2 topic echo /planning/local_status
ros2 topic echo /safety/emergency_stop
ros2 topic echo /planning/path --once
```

정상은 `GLOBAL_PATH_CLEAR`, 회피 중에는 `AVOIDING ...`, 안전 정지는 `AEB_STOP`
또는 `NO_COLLISION_FREE_PATH`로 표시됩니다. 처음에는 초기 pose를 다시 맞춘 뒤 seed당
한 랩씩 검증합니다.
