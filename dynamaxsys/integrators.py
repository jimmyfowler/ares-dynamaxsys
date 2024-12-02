import jax.numpy as jnp
from dynamaxsys.base import ControlAffineDynamics



class IntegratorND(ControlAffineDynamics):
    integrator_dim: int
    N_dim: int

    def __init__(self, integrator_dim, N_dim):
        self.integrator_dim = integrator_dim
        self.N_dim = N_dim
        self.state_dim = self.integrator_dim * self.N_dim
        self.control_dim = self.N_dim

        self.A = jnp.eye(self.state_dim, k=self.N_dim)
        self.B = jnp.zeros([self.state_dim, self.control_dim])
        self.B = self.B.at[-self.N_dim:].set(jnp.eye(self.N_dim))

    def open_loop_dynamics(self, state, time=0):
        return self.A @ state

    def control_jacobian(self, state, time=0):
        return self.B

    # def ode_dynamics(self, state, control, t):
    #     return jnp.concatenate([state[self.N_dim:], control])

def DoubleIntegrator2D():
    return IntegratorND(2, 2)

def DoubleIntegrator1D():
    return IntegratorND(2, 1)

def SingleIntegrator2D():
    return IntegratorND(1, 2)

def SingleIntegrator1D():
    return IntegratorND(1, 1)




class TwoPlayerRelativeIntegratorND(ControlAffineDynamics):
    integrator_dim: int
    N_dim: int

    def __init__(self, integrator_dim, N_dim):
        self.integrator_dim = integrator_dim
        self.N_dim = N_dim
        self.state_dim = self.integrator_dim * self.N_dim
        self.control_dim = self.N_dim * 2

        self.A = jnp.eye(self.state_dim, k=self.N_dim)
        B = jnp.zeros([self.state_dim, self.N_dim])
        B = B.at[-self.N_dim:].set(jnp.eye(self.N_dim))
        self.B2 = jnp.concatenate([-B, B], axis=-1)


    def open_loop_dynamics(self, state, time=0):
        return self.A @ state

    def control_jacobian(self, state, time=0):
        return self.B2
