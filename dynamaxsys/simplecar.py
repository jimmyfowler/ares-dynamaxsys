import jax.numpy as jnp
from dynamaxsys.base import ControlAffineDynamics, Dynamics



class SimpleCar(Dynamics):
    state_dim: int = 3
    control_dim: int = 2
    wheelbase: float


    def __init__(self, wheelbase):
        self.wheelbase = wheelbase

        def dynamics_func(state, control, time=0):
            x, y, th = state
            v, tandelta = control
            return jnp.array(
                [
                    v * jnp.cos(th),
                    v * jnp.sin(th),
                    v / self.wheelbase * tandelta
                ]
            )


        super().__init__(dynamics_func, self.state_dim, self.control_dim)

class DynamicallyExtendedSimpleCar(ControlAffineDynamics):
    state_dim: int = 4
    control_dim: int = 2
    wheelbase: float

    def __init__(self, wheelbase):
        self.wheelbase = wheelbase

        def drift_dynamics(state, time=0):
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

        def control_jacobian(state, time=0):
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

        super().__init__(drift_dynamics, control_jacobian, self.state_dim, self.control_dim)
