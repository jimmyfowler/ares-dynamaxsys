import jax.numpy as jnp
import numpy as np
from base import *
from simplecar import *
from unicycle import *

def test_dynamics():
    # test on some made up dynamics
    def dynamics_func(state: jnp.ndarray, control: jnp.ndarray, time: float) -> jnp.ndarray:
        return state**2 + control**2

    state_dim = 2
    control_dim = 2
    dt = 0.1


    ct_dynamics = Dynamics(
        dynamics_func=dynamics_func,
        state_dim=state_dim,
        control_dim=control_dim,
    )
    dt_dynamics = get_discrete_time_dynamics(ct_dynamics, dt)


    state = jnp.ones(state_dim)
    control = jnp.ones(control_dim)
    time = 0.0
    state_derivative = ct_dynamics(state, control, time)
    xdot_linearize = ct_dynamics.linearize(state, control, time)

    xnext = dt_dynamics(state, control, time)
    xnext_linearize = dt_dynamics.linearize(state, control, time)

    print("Passed: Dynamics class")

def test_control_affine_dynamics():
    # test on some made up dynamics
    def drift_dynamics(state: jnp.ndarray, time: float) -> jnp.ndarray:
        return state**2
    def control_jacobian(state: jnp.ndarray,  time: float) -> jnp.ndarray:
        return state**2

    state_dim = 2
    control_dim = 2
    dt = 0.1


    ct_dynamics = ControlAffineDynamics(
        drift_dynamics=drift_dynamics,
        control_jacobian=control_jacobian,
        state_dim=state_dim,
        control_dim=control_dim,
    )
    dt_dynamics = get_discrete_time_dynamics(ct_dynamics, dt)


    state = jnp.ones(state_dim)
    control = jnp.ones(control_dim)
    time = 0.0
    state_derivative = ct_dynamics(state, control, time)
    xdot_linearize = ct_dynamics.linearize(state, control, time)

    xnext = dt_dynamics(state, control, time)
    xnext_linearize = dt_dynamics.linearize(state, control, time)

    print("Passed: ControlAffineDynamics class")


def test_linear_dynamics():
    # test on some made up dynamics
    state_dim = 9
    control_dim = 6
    dt = 0.1
    A = jnp.array(np.random.randn(state_dim, state_dim))
    B = jnp.array(np.random.randn(state_dim, control_dim))
    C = None

    ct_dynamics = LTIDynamics(
        A=A,
        B=B,
        C=C
    )
    dt_dynamics = get_discrete_time_dynamics(ct_dynamics, dt)


    state = jnp.ones(state_dim)
    control = jnp.ones(control_dim)
    time = 0.0
    state_derivative = ct_dynamics(state, control, time)
    xdot_linearize = ct_dynamics.linearize(state, control, time)

    xnext = dt_dynamics(state, control, time)
    xnext_linearize = dt_dynamics.linearize(state, control, time)


    print("Passed: LTIDynamics class")

def check_continuous_time_unicycle(state, control):
    x, y, theta = state
    v, omega = control
    ct_dynamics = Unicycle()
    time = 0.
    linear_dynamics = get_linearized_dynamics(ct_dynamics, state, control, time)
    linear_dynamics.A, linear_dynamics.B, linear_dynamics.C

    analytic_A = jnp.array([[0., 0., -v * jnp.sin(theta)], [0., 0., v * jnp.cos(theta)], [0., 0., 0.]])
    analytic_B = jnp.array([[jnp.cos(theta), 0.], [jnp.sin(theta), 0.], [0., 1.]])
    analytic_C = jnp.array([v * jnp.cos(theta), v * jnp.sin(theta), omega]) - analytic_A @ state - analytic_B @ control
    assert jnp.allclose(linear_dynamics.A, analytic_A)
    assert jnp.allclose(linear_dynamics.B, analytic_B)
    assert jnp.allclose(linear_dynamics.C, analytic_C)
    print("Passed: Unicycle")

def check_continuous_time_dynamic_unicycle(state, control):
    x, y, theta, v = state
    omega, a = control
    ct_dynamics = DynamicallyExtendedUnicycle()
    time = 0.
    linear_dynamics = get_linearized_dynamics(ct_dynamics, state, control, time)
    linear_dynamics.A, linear_dynamics.B, linear_dynamics.C

    analytic_A = jnp.array([[0., 0., -v * jnp.sin(theta), jnp.cos(theta)], [0., 0., v * jnp.cos(theta), jnp.sin(theta)], [0., 0., 0., 0.], [0., 0., 0., 0.]])
    analytic_B = jnp.array([[0., 0.], [0., 0.], [1., 0.], [0., 1.]])
    analytic_C = jnp.array([v * jnp.cos(theta), v * jnp.sin(theta), omega, a]) - analytic_A @ state - analytic_B @ control
    assert jnp.allclose(linear_dynamics.A, analytic_A)
    assert jnp.allclose(linear_dynamics.B, analytic_B)
    assert jnp.allclose(linear_dynamics.C, analytic_C)
    print("Passed: Dynamically Extended Unicycle")


def check_continuous_time_simplecar(state, control):
    x, y, theta = state
    v, tand = control
    ct_dynamics = SimpleCar(wheelbase=1.0)
    time = 0.
    linear_dynamics = get_linearized_dynamics(ct_dynamics, state, control, time)
    linear_dynamics.A, linear_dynamics.B, linear_dynamics.C

    analytic_A = jnp.array([[0., 0., -v * jnp.sin(theta)], [0., 0., v * jnp.cos(theta)], [0., 0., 0.]])
    analytic_B = jnp.array([[jnp.cos(theta), 0.], [jnp.sin(theta), 0.], [tand / ct_dynamics.wheelbase, v / ct_dynamics.wheelbase]])
    analytic_C = jnp.array([v * jnp.cos(theta), v * jnp.sin(theta), v / ct_dynamics.wheelbase * tand]) - analytic_A @ state - analytic_B @ control
    assert jnp.allclose(linear_dynamics.A, analytic_A)
    assert jnp.allclose(linear_dynamics.B, analytic_B)
    assert jnp.allclose(linear_dynamics.C, analytic_C)
    print("Passed: Simple Car")

def check_continuous_time_dynamic_simplecar(state, control):
    x, y, theta, v = state
    tand, a = control
    ct_dynamics = DynamicallyExtendedSimpleCar(wheelbase=1.0)
    time = 0.
    linear_dynamics = get_linearized_dynamics(ct_dynamics, state, control, time)
    linear_dynamics.A, linear_dynamics.B, linear_dynamics.C

    analytic_A = jnp.array([[0., 0., -v * jnp.sin(theta), jnp.cos(theta)], [0., 0., v * jnp.cos(theta), jnp.sin(theta)], [0., 0., 0., tand / ct_dynamics.wheelbase], [0., 0., 0., 0.]])
    analytic_B = jnp.array([[0., 0.], [0., 0.], [v / ct_dynamics.wheelbase, 0.], [0., 1.]])
    analytic_C = jnp.array([v * jnp.cos(theta), v * jnp.sin(theta), v / ct_dynamics.wheelbase * tand, a]) - analytic_A @ state - analytic_B @ control
    assert jnp.allclose(linear_dynamics.A, analytic_A)
    assert jnp.allclose(linear_dynamics.B, analytic_B)
    assert jnp.allclose(linear_dynamics.C, analytic_C)
    print("Passed: Dynamically Extended Simple Car")

if __name__ == '__main__':

    test_dynamics()
    test_control_affine_dynamics()
    test_linear_dynamics()

    check_continuous_time_unicycle(jnp.array([1., 2., 3.]), jnp.array([4., 5.]))
    check_continuous_time_dynamic_unicycle(jnp.array([1., 2., 3., 4.]), jnp.array([5., 6.]))
    check_continuous_time_simplecar(jnp.array([1., 2., 3.]), jnp.array([4., 5.]))
    check_continuous_time_dynamic_simplecar(jnp.array([1., 2., 3., 4.]), jnp.array([5., 6.]))

