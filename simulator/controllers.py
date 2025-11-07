import jax
import jax.numpy as jnp
import equinox as eqx
from dynamaxsys.parafoil import getInertialToBodyRotationMatrix


class PID(eqx.Module):
    kp: float
    ki: float
    kd: float

    def compute(self, ctrl_state, error, dt):
        integral, prev_error = ctrl_state
        integral += error * dt
        derivative = jax.lax.cond(
            dt > 0.0,  # conditional to avoid division by zero
            lambda d: (error - prev_error) / d,  # true branch
            lambda d: 0.0,  # false branch
            dt,
        )
        prev_error = error
        control_input = self.kp * error + self.ki * integral + self.kd * derivative
        new_ctrl_state = jnp.array([integral, prev_error])

        return control_input, new_ctrl_state


class DummyController:
    """
    A dummy controller for testing simulation
    """

    def __init__(self):
        pass

    def __call__(self, x, ctrl_state, dt=0.1):
        return jnp.array(1), ctrl_state


class TwelveStateHeadingController(PID):
    """
    A heading controller for a 12-state aircraft using PID control.
    Assumes the aircraft is desired to land at the inertial origin (x,y,z = 0,0,0)

    Args:
        kp: Proportional gain for the PID controller.
        ki: Integral gain for the PID controller.
        kd: Derivative gain for the PID controller.
    """

    def get_heading(self, x):
        phi, theta, psi = x[6], x[7], x[8]  # roll, pitch, yaw angles
        inertial_to_body = getInertialToBodyRotationMatrix(phi, theta, psi)
        body_to_inertial = inertial_to_body.T  # rotation matrix is orthogonal
        u, v, w = x[2], x[3], x[4]  # body-frame velocities
        xyz_dot = body_to_inertial @ jnp.array([u, v, w])
        x_inertial_velocity = xyz_dot[0]
        y_inertial_velocity = xyz_dot[1]
        heading_rad = jnp.arctan2(y_inertial_velocity, x_inertial_velocity)
        # heading_deg = heading_rad * 180.0 / jnp.pi
        return heading_rad

    def get_desired_heading(self, x):
        position = x[0:2]
        desired_heading_rad = (
            jnp.arctan2(position[1], position[0]) + jnp.pi
        )  # point towards origin
        # desired_heading_deg = desired_heading_rad * 180 / jnp.pi
        return desired_heading_rad

    def get_heading_error(self, x):
        current_heading = self.get_heading(x)  # rad
        desired_heading = self.get_desired_heading(x)  # rad
        heading_error = desired_heading - current_heading
        heading_error_wrapped = jnp.mod(heading_error + jnp.pi, 2*jnp.pi) - jnp.pi
        return heading_error_wrapped

    def __call__(self, x, ctrl_state, dt=0.1):
        heading_error = self.get_heading_error(x)
        return self.compute(ctrl_state, heading_error, dt)


class ModelPredictiveController:
    """
    A placeholder for a model predictive controller (MPC).
    """

    def __init__(self):
        pass

    def __call__(self, x, dt=0.1):
        # Placeholder: return zero control input
        return 0.0


# Example usage:
if __name__ == "__main__":
    DEG_TO_RAD = jnp.pi / 180.0
    heading_controller = TwelveStateHeadingController(kp=1, ki=0.01, kd=0.1)
    slegers_initial_state = {
        "x": 1000.0,  # ft
        "y": 1000.0,
        "z": -2000.0,
        "u": 10.0,  # ft/s
        "v": 0.1,
        "w": 10.0,
        "phi": 20 * DEG_TO_RAD,  # deg -> rad
        "theta": 2 * DEG_TO_RAD,
        "psi": 0.0 * DEG_TO_RAD,
        "p": 0.0 * DEG_TO_RAD,  # deg/s -> rad/s
        "q": 0.0 * DEG_TO_RAD,
        "r": 0.0 * DEG_TO_RAD,
    }
    x_sample = jnp.array(list(slegers_initial_state.values()))
    initial_error = heading_controller.get_heading_error(x_sample)
    ctrl_state0 = jnp.array([0.0, initial_error])  # [integral, prev_err]
    u, new_state = heading_controller(x_sample, ctrl_state0, 0.01)  # non-jitted call
    print(u, new_state)  # regular Python-level print works here
