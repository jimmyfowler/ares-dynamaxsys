import jax
import jax.numpy as jnp
from dynamaxsys.utils import runge_kutta_integrator, linearize
import functools
import abc



class Dynamics(metaclass=abc.ABCMeta):
    state_dim: int
    control_dim: int

    @abc.abstractmethod
    def ode_dynamics(self, state, control, time=0):
        """Implements the continuous-time dynamics ODE."""

    def discrete_step(self, state, control, time=0, dt=0.1):
        return runge_kutta_integrator(self.ode_dynamics, dt)(state, control, time)

    def linearized_dynamics(self, state0, control0, time):
        A, B, C = linearize(self.ode_dynamics, state0, control0, time)
        open_loop_dynamics = lambda x, t: A @ x + C
        control_jacobian = lambda x, t: B
        return LinearizedNonlinearDynamics(open_loop_dynamics, control_jacobian)

    def __call__(self, state, control, time=0):
        return self.ode_dynamics(state, control, time)



class ControlAffineDynamics(Dynamics):

    def ode_dynamics(self, state, control, time=0):
        return self.open_loop_dynamics(state, time) + self.control_jacobian(state, time) @ control

    @abc.abstractmethod
    def open_loop_dynamics(self, state, time):
        """Implements the open loop dynamics `f(x, t)`."""

    @abc.abstractmethod
    def control_jacobian(self, state, time):
        """Implements the control Jacobian `G_u(x, t)`."""


class LinearizedNonlinearDynamics(ControlAffineDynamics):

    def __init__(self, open_loop_dynamics, control_jacobian):
        self.old = open_loop_dynamics
        self.cj = control_jacobian

    def open_loop_dynamics(self, state, time=0):
        return self.old(state, time)

    def control_jacobian(self, state, time=0):
        return self.cj(state, time)