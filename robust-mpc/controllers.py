import jax
import jax.numpy as jnp
import equinox as eqx
from dynamaxsys.parafoil import getInertialToBodyRotationMatrix

def Pid(Kp=None, Ki=None, Kd=None, freq_lpf=None, units=None):
    """Create a PID controller continuous-time transfer function.

    Args:
        Kp: Proportional gain.
        Ki: Integral gain.
        Kd: Derivative gain.
        freq_lpf_Hz: Cutoff frequency of the low-pass filter on the derivative term (in Hz).

    Returns:
        A control.TransferFunction representing the PID controller.
    """
    # proportional term
    if Kp is not None:
        C_p = ct.zpk([], [], Kp)
    else:
        C_p = ct.zpk([], [], 0)

    # integral term
    if Ki is not None:
        C_i = ct.zpk([], 0, Ki)
    else:
        C_i = ct.zpk([], [], 0)

    # derivative term
    if Kd is not None:
        if freq_lpf is not None:  # check for low-pass filter frequency and units
            if units is not None:
                match units:
                    case "Hz":
                        freq_lpf_rad_per_s = freq_lpf
                    case "rad/s":
                        freq_lpf_rad_per_s = 2 * np.pi * freq_lpf
                    case _:
                        raise ValueError("'units' must either be 'Hz' or 'rad/s'")

                C_d = ct.zpk([], [-freq_lpf_rad_per_s], Kd * freq_lpf_rad_per_s)
            else:
                raise ValueError("'units' of frequency must be specified")
        else:
            raise ValueError("A value for Kd was specified, but 'freq_lpf' was not")
    else:
        C_d = ct.zpk([], [], 0)

    # connect transfer functions in parallel
    C_pid = C_p + C_i + C_d
    return C_pid


class PID(eqx.Module):
    kp: float
    ki: float
    kd: float
    max_output: float = None  # upper bound saturation
    min_output: float = None  # lower bound saturation
    rate_limit: float = None  # max rate of change of output
    # TODO: make actuator class instead of rate limit here

    def compute(self, ctrl_state, error, dt):
        integral, prev_error, prev_control_input, spiraling = ctrl_state
        integral += error * dt
        derivative = jax.lax.cond(
            dt > 0.0,  # conditional to avoid division by zero
            lambda d: (error - prev_error) / d,  # true branch
            lambda d: 0.0,  # false branch
            dt,
        )
        prev_error = error
        control_input = self.kp * error + self.ki * integral + self.kd * derivative

        # Rate limiting (TODO: move to actuator class)
        if self.rate_limit is not None:
            max_delta = self.rate_limit * dt
            delta = control_input - prev_control_input
            delta = jnp.clip(delta, -max_delta, max_delta)
            control_input = prev_control_input + delta

        # Saturate output
        if self.max_output is not None:
            control_input = jnp.clip(control_input, self.min_output, self.max_output)

        # Update control state
        new_ctrl_state = integral, prev_error, control_input, spiraling

        return control_input, new_ctrl_state


# class PID2(eqx.Module):
#     kp: float
#     ki: float
#     kd: float
#     freq_lpf: float = None
#     # TODO: make actuator class instead of rate limit here


#     def __init__(self, kp, ki, kd, freq_lpf=None, max_output=None, min_output=None, rate_limit=None):
        
#     def compute(self, ctrl_state, error, dt):
#         integral, prev_error, prev_control_input, spiraling = ctrl_state
#         integral += error * dt
#         derivative = jax.lax.cond(
#             dt > 0.0,  # conditional to avoid division by zero
#             lambda d: (error - prev_error) / d,  # true branch
#             lambda d: 0.0,  # false branch
#             dt,
#         )
#         prev_error = error
#         control_input = self.kp * error + self.ki * integral + self.kd * derivative

#         # Rate limiting (TODO: move to actuator class)
#         if self.rate_limit is not None:
#             max_delta = self.rate_limit * dt
#             delta = control_input - prev_control_input
#             delta = jnp.clip(delta, -max_delta, max_delta)
#             control_input = prev_control_input + delta

#         # Saturate output
#         if self.max_output is not None:
#             control_input = jnp.clip(control_input, self.min_output, self.max_output)

#         # Update control state
#         new_ctrl_state = integral, prev_error, control_input, spiraling

#         return control_input, new_ctrl_state


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
        TODO: UPDATE
    """

    spiral_mode: bool = False  # spiral once above target
    inner_spiral_range: float = 0.0
    outer_spiral_range: float = 0.0

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
        heading_error_wrapped = jnp.mod(heading_error + jnp.pi, 2 * jnp.pi) - jnp.pi
        return heading_error_wrapped

    def get_horiz_distance_to_target(self, x):
        position = x[0:2]
        distance = jnp.linalg.norm(position)
        return distance

    def __call__(self, x, ctrl_state, dt):
        distance = self.get_horiz_distance_to_target(x)

        # compute normal PID output
        heading_error = self.get_heading_error(x)
        pid_control_input, new_ctrl_state = self.compute(ctrl_state, heading_error, dt)
        
        # get spiraling flag from current ctrl state
        _, _, _, spiraling = ctrl_state

        # Handle spiral mode with JAX ops
        # Convert static Python bool -> JAX scalar, and use logical ops on traced distance
        spiral_flag = jnp.array(self.spiral_mode)
        spiral_range = jnp.where(
            spiraling,
            jnp.array(self.outer_spiral_range), # use outer range if already spiraling
            jnp.array(self.inner_spiral_range), # use inner range to start spiraling
        ) 
        spiral_cond = jnp.logical_and(spiral_flag, distance < spiral_range)

        # control input when spiralling (scalar)
        spiral_control = jnp.array(
            self.max_output if self.max_output is not None else 0.0
        )

        # choose control and controller state using JAX selection
        # Use JAX ops for control selection to ensure tracing compatibility and avoid static Python control flow issues
        control_input = jnp.where(spiral_cond, spiral_control, pid_control_input)

        # update spiraling flag and ctrl state
        spiraling_new = jnp.where(spiral_cond, True, spiraling)
        integral_new, prev_error_new, prev_u_new, _ = new_ctrl_state
        new_ctrl_state = integral_new, prev_error_new, prev_u_new, spiraling_new

        return control_input, new_ctrl_state


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
    heading_controller = TwelveStateHeadingController(
        kp=1,
        ki=0.0,
        kd=0.1,
        max_output=4.0,
        min_output=-4.0,
        rate_limit=0.5,
        spiral_mode=True,
        inner_spiral_range=50.0,
        outer_spiral_range=100.0,
    )
    slegers_initial_state = {
        "x": -500.0,  # ft
        "y": 100.0,
        "z": 300.0,
        "u": 10.0,  # ft/s
        "v": 0.1,
        "w": 5.0,
        "phi": 10 * DEG_TO_RAD,  # deg -> rad
        "theta": 2 * DEG_TO_RAD,
        "psi": 180.0 * DEG_TO_RAD,
        "p": 0.0 * DEG_TO_RAD,  # deg/s -> rad/s
        "q": 0.0 * DEG_TO_RAD,
        "r": 0.0 * DEG_TO_RAD,
    }
    x_0 = jnp.array(list(slegers_initial_state.values()))
    ctrl_state0 = (
        jnp.array(0.0),     # integral 
        jnp.array(0.0),     # prev_error
        jnp.array(0.0),     # prev_control_input
        jnp.array(False),   # spiraling flag
    )  
    u, new_state = heading_controller(x_0, ctrl_state0, 0.01)  # non-jitted call
    print(u, new_state)
