import jax.numpy as jnp
from dynamaxsys.base import ControlAffineDynamics

 
class Parafoil(ControlAffineDynamics):
    state_dim: int = 4
    control_dim: int = 1

    def __init__(self):
        def drift_dynamics(state: jnp.ndarray, time: float) -> jnp.ndarray:
            x, y, vx, vy = state
            g = 9.81  # gravity
            return jnp.array([vx, vy, 0.0, -g])

        def control_jacobian(state: jnp.ndarray, time: float) -> jnp.ndarray:
            x, y, vx, vy = state
            # u = control (steering input)
            return jnp.array([[0.0], [0.0], [0.0], [1.0]])

        super().__init__(
            drift_dynamics, control_jacobian, self.state_dim, self.control_dim
        )