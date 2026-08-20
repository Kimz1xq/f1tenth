"""
Nonlinear model-predictive contouring controller for F1TENTH.

The formulation follows the standard MPCC structure used by F1TENTH racing
research: a nonlinear kinematic bicycle prediction model, contour/lag errors,
input-rate penalties, and hard actuator constraints.  It consumes the same
``nav_msgs/Path`` and safety topics as the other controllers, so simulation
and the physical car differ only in launch-time topic/frame configuration.
"""

import time

import casadi as ca
import numpy as np
import rclpy
from rclpy._rclpy_pybind11 import RCLError
from rclpy.executors import ExternalShutdownException

from control.linear_mpc_node import LinearMpcNode


class NonlinearMpccNode(LinearMpcNode):
    """CasADi/FATROP nonlinear contouring controller."""

    NODE_NAME = 'nonlinear_mpcc_node'
    DISPLAY_NAME = 'Nonlinear MPCC'

    def build_mpc_problem(self):
        n = self.horizon
        physical_nx = self.STATE_SIZE
        # Previous steering is an augmented state. This makes the steering
        # rate constraint local to each OCP stage, which is the sparse
        # structure expected by FATROP.
        nx = physical_nx + 1
        nu = self.INPUT_SIZE

        state_variables = [
            ca.MX.sym('state_%d' % step, nx) for step in range(n + 1)]
        control_variables = [
            ca.MX.sym('control_%d' % step, nu) for step in range(n)]
        states = ca.horzcat(*state_variables)
        controls = ca.horzcat(*control_variables)
        parameter_count = physical_nx * (n + 1) + nu * n
        parameters = ca.MX.sym('parameters', parameter_count)

        cursor = 0
        reference = ca.reshape(
            parameters[cursor:cursor + physical_nx * (n + 1)],
            physical_nx, n + 1)
        cursor += physical_nx * (n + 1)
        reference_input = ca.reshape(
            parameters[cursor:cursor + nu * n], nu, n)

        q_contour = float(self.q[1, 1])
        q_lag = float(self.q[0, 0])
        q_speed = float(self.q[2, 2])
        q_yaw = float(self.q[3, 3])
        r_acceleration = float(self.r[0, 0])
        r_steering = float(self.r[1, 1])
        rd_acceleration = float(self.rd[0, 0])
        rd_steering = float(self.rd[1, 1])

        objective = 0
        constraints = []
        constraint_lower = []
        constraint_upper = []

        def physical_dynamics(state, control):
            speed, yaw = state[2], state[3]
            acceleration, steering = control[0], control[1]
            return ca.vertcat(
                speed * ca.cos(yaw),
                speed * ca.sin(yaw),
                acceleration,
                speed * ca.tan(steering) / self.wheelbase,
            )

        def rk4(state, control):
            k1 = physical_dynamics(state, control)
            k2 = physical_dynamics(state + 0.5 * self.dt * k1, control)
            k3 = physical_dynamics(state + 0.5 * self.dt * k2, control)
            k4 = physical_dynamics(state + self.dt * k3, control)
            return state + self.dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6

        for step in range(n):
            error_x = states[0, step] - reference[0, step]
            error_y = states[1, step] - reference[1, step]
            reference_yaw = reference[3, step]
            contour_error = (
                ca.sin(reference_yaw) * error_x
                - ca.cos(reference_yaw) * error_y)
            lag_error = (
                ca.cos(reference_yaw) * error_x
                + ca.sin(reference_yaw) * error_y)
            yaw_error = ca.atan2(
                ca.sin(states[3, step] - reference_yaw),
                ca.cos(states[3, step] - reference_yaw))
            speed_error = states[2, step] - reference[2, step]
            acceleration_error = (
                controls[0, step] - reference_input[0, step])
            steering_error = (
                controls[1, step] - reference_input[1, step])

            objective += (
                q_contour * contour_error ** 2
                + q_lag * lag_error ** 2
                + q_speed * speed_error ** 2
                + q_yaw * yaw_error ** 2
                + r_acceleration * acceleration_error ** 2
                + r_steering * steering_error ** 2)

            steering_delta = controls[1, step] - states[4, step]
            if step > 0:
                acceleration_delta = (
                    controls[0, step] - controls[0, step - 1])
                objective += rd_acceleration * acceleration_delta ** 2
            objective += rd_steering * steering_delta ** 2
            next_state = ca.vertcat(
                rk4(states[:physical_nx, step], controls[:, step]),
                controls[1, step],
            )
            # FATROP requires dynamics first, followed by this stage's local
            # path constraints.
            constraints.append(states[:, step + 1] - next_state)
            constraint_lower.extend([0.0] * nx)
            constraint_upper.extend([0.0] * nx)
            constraints.append(steering_delta)
            steering_step_limit = self.max_steering_rate * self.dt
            constraint_lower.append(-steering_step_limit)
            constraint_upper.append(steering_step_limit)

        terminal_x = states[0, n] - reference[0, n]
        terminal_y = states[1, n] - reference[1, n]
        terminal_yaw_ref = reference[3, n]
        terminal_contour = (
            ca.sin(terminal_yaw_ref) * terminal_x
            - ca.cos(terminal_yaw_ref) * terminal_y)
        terminal_lag = (
            ca.cos(terminal_yaw_ref) * terminal_x
            + ca.sin(terminal_yaw_ref) * terminal_y)
        terminal_yaw = ca.atan2(
            ca.sin(states[3, n] - terminal_yaw_ref),
            ca.cos(states[3, n] - terminal_yaw_ref))
        terminal_speed = states[2, n] - reference[2, n]
        terminal_scale = float(self.qf[0, 0] / max(self.q[0, 0], 1e-9))
        objective += terminal_scale * (
            q_contour * terminal_contour ** 2
            + q_lag * terminal_lag ** 2
            + q_speed * terminal_speed ** 2
            + q_yaw * terminal_yaw ** 2)

        decision_parts = []
        for step in range(n + 1):
            decision_parts.append(state_variables[step])
            if step < n:
                decision_parts.append(control_variables[step])
        decision = ca.vertcat(*decision_parts)
        problem = {
            'x': decision,
            'f': objective,
            'g': ca.vertcat(*constraints),
            'p': parameters,
        }
        options = {
            'print_time': False,
            'expand': True,
            'structure_detection': 'manual',
            'N': n,
            'nx': [nx] * (n + 1),
            'nu': [nu] * n + [0],
            'ng': [1] * n + [0],
            'convexify_strategy': 'regularize',
            'fatrop': {
                'print_level': 0,
                # Keep the real-time OCP solve bounded, while allowing enough
                # iterations for the larger reference change at corner entry.
                'max_iter': 100,
                'tol': 1e-4,
                'acceptable_tol': 2e-3,
                'acceptable_iter': 2,
            },
        }
        self.mpcc_solver = ca.nlpsol(
            'nonlinear_mpcc_solver', 'fatrop', problem, options)

        decision_count = nx * (n + 1) + nu * n
        self.mpcc_lbx = np.full(decision_count, -np.inf)
        self.mpcc_ubx = np.full(decision_count, np.inf)
        cursor = 0
        for step in range(n + 1):
            self.mpcc_lbx[cursor + 2] = 0.0
            self.mpcc_ubx[cursor + 2] = self.max_speed
            cursor += nx
            if step < n:
                self.mpcc_lbx[cursor] = -self.max_acceleration
                self.mpcc_ubx[cursor] = self.max_acceleration
                self.mpcc_lbx[cursor + 1] = -self.max_steering_angle
                self.mpcc_ubx[cursor + 1] = self.max_steering_angle
                cursor += nu

        self.mpcc_lbg = np.asarray(constraint_lower, dtype=float)
        self.mpcc_ubg = np.asarray(constraint_upper, dtype=float)
        self.mpcc_warm_start = None
        self.mpcc_state_size = nx

    def pack_decision(self, states, controls):
        """Pack stage variables in FATROP's x0,u0,...,xN order."""
        parts = []
        for step in range(self.horizon + 1):
            parts.append(states[:, step])
            if step < self.horizon:
                parts.append(controls[:, step])
        return np.concatenate(parts)

    def unpack_decision(self, decision):
        """Unpack FATROP stage variables into state/control matrices."""
        states = np.empty((self.mpcc_state_size, self.horizon + 1))
        controls = np.empty((self.INPUT_SIZE, self.horizon))
        cursor = 0
        for step in range(self.horizon + 1):
            states[:, step] = decision[
                cursor:cursor + self.mpcc_state_size]
            cursor += self.mpcc_state_size
            if step < self.horizon:
                controls[:, step] = decision[
                    cursor:cursor + self.INPUT_SIZE]
                cursor += self.INPUT_SIZE
        return states, controls

    def solve_mpc(self, current_state, reference, reference_input):
        n = self.horizon
        augmented_initial = np.concatenate((
            np.asarray(current_state, dtype=float),
            [self.previous_steering],
        ))
        parameters = np.concatenate((
            np.asarray(reference, dtype=float).reshape(-1, order='F'),
            np.asarray(reference_input, dtype=float).reshape(-1, order='F'),
        ))
        control_guess = np.asarray(reference_input, dtype=float).copy()
        state_guess = np.empty((self.mpcc_state_size, n + 1))
        state_guess[:self.STATE_SIZE] = np.asarray(reference, dtype=float)
        state_guess[:self.STATE_SIZE, 0] = current_state
        state_guess[4, 0] = self.previous_steering
        state_guess[4, 1:] = control_guess[1]
        initial_guess = self.pack_decision(state_guess, control_guess)
        if (self.mpcc_warm_start is not None
                and len(self.mpcc_warm_start) == len(initial_guess)):
            initial_guess = self.mpcc_warm_start
        initial_guess[:self.mpcc_state_size] = augmented_initial
        self.mpcc_lbx[:self.mpcc_state_size] = augmented_initial
        self.mpcc_ubx[:self.mpcc_state_size] = augmented_initial

        started = time.perf_counter()
        try:
            solution = self.mpcc_solver(
                x0=initial_guess,
                lbx=self.mpcc_lbx,
                ubx=self.mpcc_ubx,
                lbg=self.mpcc_lbg,
                ubg=self.mpcc_ubg,
                p=parameters,
            )
        except RuntimeError as error:
            raise RuntimeError('nonlinear solver failed: %s' % error) from error
        solve_ms = (time.perf_counter() - started) * 1000.0
        stats = self.mpcc_solver.stats()
        fatrop_stats = stats.get('fatrop', {})
        return_flag = int(fatrop_stats.get('return_flag', -1))
        # FATROP's public return enum defines 0 as Success and 2 as
        # StopAtAcceptablePoint.  The latter is a usable MPC iterate; flag 1
        # is MaxIterExceeded and remains a hard failure.
        solver_succeeded = (
            bool(stats.get('success', False)) or return_flag in (0, 2))
        if not solver_succeeded:
            raise RuntimeError(
                'nonlinear solver status: %s (FATROP flag %d)'
                % (stats.get('return_status', 'unknown'), return_flag))

        decision = np.asarray(solution['x']).reshape(-1)
        predicted_augmented, controls = self.unpack_decision(decision)
        # Receding-horizon warm start: discard the control/state already
        # applied and append the terminal sample. Reusing the unshifted
        # solution makes the next NLP start one full control period behind
        # the vehicle and can exhaust IPOPT iterations after enabling.
        if self.enabled:
            shifted_states = np.column_stack((
                predicted_augmented[:, 1:], predicted_augmented[:, -1]))
            shifted_controls = np.column_stack((
                controls[:, 1:], controls[:, -1]))
            self.mpcc_warm_start = self.pack_decision(
                shifted_states, shifted_controls)
        else:
            # A disabled controller publishes zero speed, so its state does
            # not advance. Keep the solved horizon aligned until enabling.
            self.mpcc_warm_start = decision.copy()

        # Keep publication in the shared ROS wrapper without depending on a
        # CasADi type outside this method.
        from std_msgs.msg import Float64
        message = Float64()
        message.data = solve_ms
        self.solve_time_pub.publish(message)
        return controls, predicted_augmented[:self.STATE_SIZE], solve_ms


def main(args=None):
    rclpy.init(args=args)
    node = NonlinearMpccNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException, RCLError):
        pass
    finally:
        try:
            if rclpy.ok():
                node.publish_stop()
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    main()
