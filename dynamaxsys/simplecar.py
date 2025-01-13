import jax.numpy as jnp
from dynamaxsys.base import ControlAffineDynamics, Dynamics



class SimpleCar(Dynamics):
    state_dim: int = 3
    control_dim: int = 2
    wheelbase: float

    def __init__(self, wheelbase):
        self.wheelbase = wheelbase

    def ode_dynamics(self, state, control, time=0):
        x, y, th = state
        v, delta = control
        return jnp.array(
            [
                v * jnp.cos(th),
                v * jnp.sin(th),
                v / self.wheelbase * jnp.tan(delta)
            ]
        )


class DynamicallyExtendedSimpleCar(ControlAffineDynamics):
    state_dim: int = 4
    control_dim: int = 2
    wheelbase: float

    def __init__(self, wheelbase):
        self.wheelbase = wheelbase

    def open_loop_dynamics(self, state, time=0):
        x, y, th, v = state
        # tandelta, a = control
        return jnp.array(
            [
                v * jnp.cos(th),
                v * jnp.sin(th),
                0.,
                0.,
            ]
        )

    def control_jacobian(self, state, time=0):
        # tandelta, a = control, tandelta = tan(delta)
        x, y, th, v = state
        return jnp.array(
            [
                [0., 0.],
                [0., 0.],
                [v / self.wheelbase, 0.],
                [0., 1.]
            ]
        )
