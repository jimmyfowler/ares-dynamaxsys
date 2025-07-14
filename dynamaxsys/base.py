from dynamaxsys.utils import runge_kutta_integrator, linearize
from typing import Callable, Union
import jax.numpy as jnp
import equinox

class Dynamics(equinox.Module):
    dynamics_func: Callable
    state_dim: int
    control_dim: int

    def __init__(self, dynamics_func, state_dim, control_dim):
        self.dynamics_func = dynamics_func
        self.state_dim = state_dim
        self.control_dim = control_dim

    def linearize(self, state0, control0, time):
        A, B, C = linearize(self.dynamics_func, state0, control0, time)
        return A @ state0 + B @ control0 + C

    def __call__(self, state, control, time=0):
        return self.dynamics_func(state, control, time)

class ControlAffineDynamics(Dynamics):
    drift_dynamics: Callable
    control_jacobian: Callable
    state_dim: int
    control_dim: int


    def __init__(self, drift_dynamics, control_jacobian, state_dim, control_dim):
        self.drift_dynamics = drift_dynamics
        self.control_jacobian = control_jacobian
        def dynamics_func(x, u, t):
            return drift_dynamics(x,t) + control_jacobian(x,t) @ u
        super().__init__(dynamics_func, state_dim, control_dim)

    def open_loop_dynamics(self, state, time=0.):
        return self.drift_dynamics(state, time)



class LTIDynamics(ControlAffineDynamics):
    A: Union[jnp.ndarray]
    B: Union[jnp.ndarray]
    C: Union[jnp.ndarray, None] = None

    def __init__(self, A, B, C=None):
        self.A = A
        self.B = B
        state_dim = A.shape[0]
        control_dim = B.shape[1]
        if C is None:
            self.C = jnp.zeros((state_dim,))
        else:
            assert C.shape == (state_dim,)
            self.C = C


        def drift_dynamics(x, t):
            return A @ x + self.C
        def control_jacobian(x, t):
            return B

        super().__init__(drift_dynamics, control_jacobian, state_dim, control_dim)



def get_discrete_time_dynamics(continuous_time_dynamics: Dynamics, dt: float) -> Dynamics:
    discete_dynamics = runge_kutta_integrator(continuous_time_dynamics, dt)
    return Dynamics(discete_dynamics, continuous_time_dynamics.state_dim, continuous_time_dynamics.control_dim)

def get_linearized_dynamics(dynamics: Dynamics, state0, control0, time):
    A, B, C = linearize(dynamics, state0, control0, time)
    return LTIDynamics(A, B, C)