import time

import equinox
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.subplots as sp
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

from dynamaxsys.base import get_discrete_time_dynamics
from dynamaxsys.parafoil import JannParafoil4DOF, SlegersParafoil6DOF

RAD_TO_DEG = 180.0 / jnp.pi
DEG_TO_RAD = jnp.pi / 180.0
KG_M3_TO_SLUG_FT3 = 0.00194032  # 1 kg/m^3 = 0.00194032 slug/ft^3


@equinox.filter_jit
def simulate(x0, us, ts, discrete_dynamics):
    """
    Simulates the system dynamics over time using a discrete-time model.

    Args:
        x0: Initial state, shape (state_dim,).
        us: Control input sequence, shape (N, control_dim).
        ts: Time steps, shape (N,).
        discrete_dynamics: Function that computes next state given (x, u, t).

    Returns:
        Array of states over time, shape (N+1, state_dim).
    """

    def scan_fn(x, ut):
        u, t = ut
        xn = discrete_dynamics(x, u, t)
        return xn, xn

    # The scan input is a tuple (us, ts), where:
    #   us: control input sequence, shape (N, control_dim)
    #   ts: time steps, shape (N,)
    # This allows scan_fn to receive both the control input and time at each step.
    _, xs = jax.lax.scan(scan_fn, x0, (us, ts))

    return jnp.concatenate([x0[None], xs], axis=0)


@equinox.filter_jit
def simulate_with_controller(x0, controller, ts, discrete_dynamics):
    """
    Simulates the system dynamics over time using a discrete-time model
    and controller.

    Args:
        x0: Initial state, shape (state_dim,).
        controller: a function that takes the current state as an argument and outputs a control.
        ts: Time steps, shape (N,).
        discrete_dynamics: Function that computes next state given (x, u, t).

    Returns:
        Array of states over time, shape (N+1, state_dim).
    """

    def scan_fn(x, t):
        u = controller(x)
        xn = discrete_dynamics(x, u, t)
        return xn, xn

    _, xs = jax.lax.scan(scan_fn, x0, ts)

    return jnp.concatenate([x0[None], xs], axis=0)


def plot_jann_body(xs, ts):
    fig, axs = plt.subplots(4, 1, figsize=(10, 6))
    axs[0].plot(ts, xs[:-1, 0], label="u (forward vel)")
    axs[0].set_ylabel("fwd speed (m/s)")

    axs[1].plot(ts, xs[:-1, 1], label="w (down vel)")
    axs[1].set_ylabel("down speed (m/s)")

    axs[2].plot(ts, xs[:-1, 2] * RAD_TO_DEG, label="phi (roll)")
    axs[2].set_ylabel("roll (deg)")

    axs[3].plot(ts, xs[:-1, 3] * RAD_TO_DEG, label="psi (yaw)")
    axs[3].set_ylabel("yaw (deg)")

    axs[3].set_xlabel("Time (s)")

    plt.tight_layout()
    plt.show()


def get_state_label(idx):
    labels = [
        "x (ft)",
        "y (ft)",
        "z (ft)",
        "u (ft/s)",
        "v (ft/s)",
        "w (ft/s)",
        "phi (deg)",
        "theta (deg)",
        "psi (deg)",
        "p (deg/s)",
        "q (deg/s)",
        "r (deg/s)",
    ]
    return labels[idx] if 0 <= idx < len(labels) else f"State {idx}"


def plot_selected_states(xs, state_indices, us, ts):
    fig = sp.make_subplots(rows=len(state_indices) + 1, cols=1, shared_xaxes=True)
        
    for i, idx in enumerate(state_indices):
        fig.add_trace(
            go.Scatter(x=ts, y=xs[:-1, idx], name=get_state_label(idx)),
            row=i + 1,
            col=1,
        )
        fig.update_yaxes(title_text=get_state_label(idx), row=i + 1, col=1)
    
    fig.add_trace(
        go.Scatter(x=ts, y=us, name="Control Input"),
        row=len(state_indices) + 1,
        col=1,
    )
    fig.update_yaxes(title_text="Asymetric Deflection", row=len(state_indices) + 1, col=1)

    fig.update_xaxes(title_text="Time (s)", row=len(state_indices) + 1, col=1)
    fig.update_layout(height=200 * (len(state_indices) + 1), showlegend=False)
    fig.show()


def plot_slegers_3D(xs, ts):
    # Get positions and forward speed
    x = xs[:, 0]
    y = xs[:, 1]
    z = -xs[:, 2]
    fwd_speed = xs[:, 3]
    dwn_speed = xs[:, 5]

    # Create a hover text that includes time
    hover_text = [f'Time: {t:.2f}s<br>Down Speed: {ds:.2f} ft/s' for t, ds in zip(ts, dwn_speed)]

    fig = go.Figure(
        data=[go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode="lines",
            line=dict(color=dwn_speed, colorscale="viridis", width=4),
            marker=dict(
                color=dwn_speed,
                colorscale="viridis",
                size=4,
            ),
            text=hover_text,
            hoverinfo='text'
        )]
    )

    fig.update_layout(
        scene=dict(
            xaxis_title="X Position (ft)",
            yaxis_title="Y Position (ft)",
            zaxis_title="Altitude (ft)",
            aspectmode="data",
        ),
        title="3D Trajectory of Slegers Parafoil",
        coloraxis_colorbar=dict(title="Downward Speed (ft/s)"),
    )

    fig.show()


################################
## Simulation Hyperparameters ##
################################
dt = 0.01  # time step (seconds)
time_horizon = 20  # total time (seconds)
N = int(time_horizon / dt)  # number of timesteps
ts = jnp.arange(0, time_horizon, dt)


##############################
## BUILD JANN 4DOF DYNAMICS ##
##############################
jann_params = {
    "m": 122.0,  # kg
    "S": 23.36,  # m^2
    "C_L0": 0.502,
    "C_D0": 0.173,
    "C_L_delta_s": 0.892,
    "C_D_delta_s": 1.086,
    "K_phi": 0.504,
    "T_phi": 0.994,
    "g": 9.81,  # m/s^2
}

jann_continuous_dynamics = JannParafoil4DOF(jann_params)

jann_discrete_dynamics = get_discrete_time_dynamics(jann_continuous_dynamics, dt)

# control sequence:
us_jann = jnp.array(
    [
        jnp.ones(N) * 0.1,  # delta_a
        jnp.zeros(N),  # delta_s
    ]
).T
# columns: delta_a, delta_s
# shape (N, m) aka (time steps, control dim)

# initial state:
x0_jann = jnp.array(
    [3.0, 3.0, 10 * DEG_TO_RAD, 0.0]
)  # u (m/s), w (m/s) phi (rad), psi (rad)


#################################
## BUILD SLEGERS 6DOF DYNAMICS ##
#################################
slegers_params = {
    "m": 0.062,  # slugs
    "S": 7.5,  # ft^2
    "b": 4.25,  # ft
    "c": 1.0,  # ft
    "mmoi": jnp.array(
        [
            [0.1357, 0.0, 0.0025],
            [0.0, 0.1506, 0.0],
            [0.0025, 0.0, 0.0203],
        ]
    ),  # slug/ft^2
    "C_L0": 0.502,
    "C_D0": 0.173,
    "C_L_alpha": 3.256,
    "C_D_alpha2": 1.984,
    "C_L_delta_a": 0.892,
    "C_D_delta_a": 0.298,
    "C_lphi": -0.0100,
    "C_lp": -0.0520,
    "C_l_delta_a": 0.0021,
    # Pitching-moment coefficients (reasonable for parafoil)
    "C_m0": 0.02,  # zero-lift pitching moment
    "C_m_alpha": -0.05,  # pitching moment due to angle of attack (-0.05)
    "C_mq": -0.4,  # pitching moment due to pitch rate
    "C_m_delta_s": -0.02,  # pitching moment due to trailing edge deflection
    "C_n_r": -0.0850,
    "C_n_delta_a": 0.0010,
    "rho": 0.0023769,  # slug/ft^3 (sea level)
    "g": 32.174,  # ft/s^2
}

slegers_continuous_dynamics = SlegersParafoil6DOF(slegers_params)

slegers_discrete_dynamics = get_discrete_time_dynamics(slegers_continuous_dynamics, dt)

# control sequence:
# us_slegers = jnp.ones(N) * 1  # delta_a
us_slegers = jnp.zeros(N) # no control
# us_slegers = us_slegers.at[N // 2 :].set(5.0) # second half turn

# initial state:
x0_slegers = jnp.array(
    [0.0, 0.0, -200.0, 10, 0.1, 10, 20 * DEG_TO_RAD, 2 * DEG_TO_RAD, 0, 0.0, 0.0, 0.0]
)  # x, y, z, u, v, w, phi, theta, psi, p, q, r (imperial units)


#######################
## SIMULATE AND PLOT ##
#######################
start_time = time.time()
# xs = simulate(x0_jann, us_jann, ts, jann_discrete_dynamics)
xs = simulate(x0_slegers, us_slegers, ts, slegers_discrete_dynamics)
end_time = time.time()
print(f"Simulation run time: {end_time - start_time:.4f} seconds")
print(f"(total simulation frames: {N})")

# STATES
# 0: x
# 1: y
# 2: z
# 3: u (forward speed)
# 4: v (side speed)
# 5: w (down speed)
# 6: phi (roll)
# 7: theta (pitch)
# 8: psi (yaw)
# 9: p (roll rate)
# 10: q (pitch rate)
# 11: r (yaw rate)

# convert angles to degrees for plotting
for i in range(6, 12):
    xs = xs.at[:, i].set(xs[:, i] * RAD_TO_DEG)

plot_slegers_3D(xs, ts)
plot_selected_states(xs, [3, 4, 5, 6, 7, 8], us_slegers, ts)

# plot_jann_body(xs, ts)
