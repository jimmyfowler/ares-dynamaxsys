import jax.numpy as jnp
from dynamaxsys.parafoil import getInertialToBodyRotationMatrix


class PID: 
    def __init__(self, kp, ki, kd, setpoint=0.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.integral = 0.0
        self.prev_error = 0.0

    def compute(self, measurement, dt):
        error = self.setpoint - measurement
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt > 0 else 0.0
        self.prev_error = error

        output = (
            self.kp * error +
            self.ki * self.integral +
            self.kd * derivative
        )
        return output


class TwelveStateHeadingController:
    """
    A heading controller for a 12-state parafoil model using PID control.
    Assumes the parafoil is desired to land at the inertial origin (x,y,z = 0,0,0)

    Args:
        continuous_dynamics: function representing the continuous-time dynamics of the parafoil.
        kp: Proportional gain for the PID controller.
        ki: Integral gain for the PID controller.
        kd: Derivative gain for the PID controller.
        setpoint: Desired heading angle in radians.

    """
    def __init__(self, kp, ki, kd, setpoint=0.0):
        self.pid = PID(kp, ki, kd, setpoint)

    def get_heading(self, x):
        phi, theta, psi = x[6], x[7], x[8]  # roll, pitch, yaw angles
        inertial_to_body = getInertialToBodyRotationMatrix(phi, theta, psi)
        body_to_inertial = inertial_to_body.T  # rotation matrix is orthogonal
        u, v, w = x[2], x[3], x[4]  # body-frame velocities
        xyz_dot = body_to_inertial @ jnp.array([u, v, w])
        x_inertial_velocity = xyz_dot[0]
        y_inertial_velocity = xyz_dot[1]
        heading = jnp.arctan2(y_inertial_velocity, x_inertial_velocity) * 180 / jnp.pi
        return heading
    
    def get_desired_heading(self, x):
        position = x[0:2]
        desired_heading_rad = jnp.arctan2(position[1], position[0]) + jnp.pi # point towards origin
        desired_heading_deg = desired_heading_rad * 180 / jnp.pi
        return desired_heading_deg
    
    def get_heading_error(self, x):
        current_heading = self.get_heading(x)
        desired_heading = self.get_desired_heading(x)
        heading_error = desired_heading - current_heading
        if heading_error > 180:
            heading_error -= 360
        elif heading_error < -180:
            heading_error += 360
        return heading_error
    
    def __call__(self, x, dt=0.1):
        heading_error = self.get_heading_error(x)
        control_input = self.pid.compute(heading_error, dt)
        return jnp.array([control_input])  # return as jax array for compatibility