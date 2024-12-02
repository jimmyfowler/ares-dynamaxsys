import jax.numpy as jnp
from dynamaxsys.base import ControlAffineDynamics

class Unicycle(ControlAffineDynamics):
    state_dim: int = 3
    control_dim: int = 2

    def open_loop_dynamics(self, state, time=0):
        return jnp.zeros(self.state_dim)

    def control_jacobian(self, state, time=0):
        # v, om = control
        x, y, th = state
        return jnp.array(
            [
                [jnp.cos(th), 0.],
                [jnp.sin(th), 0.],
                [0., 1.]
            ]
        )

class DynamicallyExtendedUnicycle(ControlAffineDynamics):
    state_dim: int = 4
    control_dim: int = 2

    # def ode_dynamics(self, state, control, time=0):
    #     x, y, th, v = state
    #     a, om = control
    #     return jnp.array([v * jnp.cos(th),
    #                       v * jnp.sin(th),
    #                       om,
    #                       a])


    def open_loop_dynamics(self, state, time=0):
        x, y, th, v = state
        # om, a = control
        return jnp.array(
            [
                v * jnp.cos(th),
                v * jnp.sin(th),
                0.,
                0.,
            ]
        )

    def control_jacobian(self, state, time=0):
        # om, a = control
        return jnp.array(
            [
                [0., 0.],
                [0., 0.],
                [1., 0.],
                [0., 1.]
            ]
        )



class RelativeUnicycle(ControlAffineDynamics):
    state_dim: int = 3
    control_dim: int = 4

    # def ode_dynamics(self, state, control, time=0):
    #     xrel, yrel, threl = state
    #     v1, om1, v2, om2 = control
    #     return jnp.array([v2 * jnp.cos(threl) + om1 * yrel - v1,
    #                       v2 * jnp.sin(threl) - om1 * xrel,
    #                       om2 - om1])


    def open_loop_dynamics(self, state, time=0):
        xrel, yrel, threl = state
        # v1, om1, v2, om2 = control
        return jnp.zeros(self.state_dim)

    def control_jacobian(self, state, time=0):
        xrel, yrel, threl = state
        # v1, om1, v2, om2 = control
        return jnp.array(
            [
                [-1., yrel, jnp.cos(threl), 0.],
                [0., -xrel, jnp.sin(threl), 0.],
                [0., -1., 0., 1.]
            ]
        )


class RelativeDynamicallyExtendedUnicycle(ControlAffineDynamics):
    state_dim: int = 5
    control_dim: int = 4

    # def ode_dynamics(self, state, control, time=0):
    #     xrel, yrel, threl, v1, v2 = state
    #     om1, a1, om2, a2 = control
    #     return jnp.array([v2 * jnp.cos(threl) + om1 * yrel - v1,
    #                       v2 * jnp.sin(threl) - om1 * xrel,
    #                       om2 - om1,
    #                       a1,
    #                       a2])

    def open_loop_dynamics(self, state, time=0):
        xrel, yrel, threl, v1, v2 = state
        # om1, a1, om2, a1 = control
        return jnp.array(
            [
                -v1 + v2 * jnp.cos(threl),
                v2 * jnp.sin(threl),
                0.,
                0.,
                0.
            ]
        )

    def control_jacobian(self, state, time=0):
        xrel, yrel, threl, v1, v2 = state
        # om1, a1, om2, a1 = control
        return jnp.array(
            [
                [yrel, 0., 0., 0.],
                [xrel, 0., 0., 0.],
                [-1., 0., 1., 0.],
                [0., 1., 0., 0.],
                [0., 0., 0., 1.]
            ]
        )
