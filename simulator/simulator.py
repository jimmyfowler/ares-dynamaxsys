import equinox
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from dynamaxsys.parafoil import JannParafoil4DOF
from dynamaxsys.base import get_discrete_time_dynamics


@equinox.filter_jit
def simulate(x0, us, ts, discrete_dynamics):
    def scan_fn(x, ut):
        u, t = ut
        xn = discrete_dynamics(x, u, t)
        return xn, xn

    _, xs = jax.lax.scan(scan_fn, x0, (us, ts))

    return jnp.concatenate([x0[None], xs], axis=0)


body_to_inertial = jnp.array(  # TODO: fill this in to simulate in inertial frame
    [
        [0.0, 1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)

jann_continuous_dynamics = JannParafoil4DOF(
    m=122.0,
    S=23.36,
    C_L0=0.502,
    C_D0=0.173,
    C_L_delta_s=0.892,
    C_D_delta_s=1.086,
    K_phi=0.504,
    T_phi=0.994,
)

jann_discrete_dynamics = get_discrete_time_dynamics(jann_continuous_dynamics, dt=0.1)

dt = 0.1  # seconds
time_horizon = 20  # seconds
N = int(time_horizon / dt)

us = jnp.array(
    [
        [jnp.ones(N) * 0.1],
        [jnp.zeros(N)]  # delta_a, delta_s
    ]
)
# shape (T, m) aka (time horizon, control dim)


ts = jnp.arange(0, time_horizon, dt)

x0 = jnp.array([3.0, 3.0, 0.0, 0.0])  # u, w, phi, psi

xs = simulate(x0, us, ts, jann_discrete_dynamics)


fig, axs = plt.subplots(4, 1, figsize=(10, 6))
axs[0].plot(ts, xs[:-1, 0], label="u (forward vel)")
axs[0].set_ylabel("fwd speed (m/s)")

axs[1].plot(ts, xs[:-1, 1], label="w (down vel)")
axs[1].set_ylabel("down speed (m/s)")

axs[2].plot(ts, xs[:-1, 2], label="phi (roll)")
axs[2].set_ylabel("roll (rad)")

axs[3].plot(ts, xs[:-1, 3], label="psi (yaw)")
axs[3].set_ylabel("yaw (rad)")

axs[3].set_xlabel("Time (s)")

plt.tight_layout()
plt.show()
