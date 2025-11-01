import time

import equinox
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.subplots as sp

from dynamaxsys.base import get_discrete_time_dynamics
from dynamaxsys.parafoil import JannParafoil4DOF, SlegersParafoil6DOF

import controllers

from model_parameters import slegers_6dof_nonlinear_params, jann_4dof_params


RAD_TO_DEG = 180.0 / jnp.pi
DEG_TO_RAD = jnp.pi / 180.0
KG_M3_TO_SLUG_FT3 = 0.00194032  # 1 kg/m^3 = 0.00194032 slug/ft^3


@equinox.filter_jit
def simulate(x0, us, ds, ts, discrete_dynamics):
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

    def scan_fn(x: jnp.ndarray, udt: tuple):
        u, d, t = udt
        xn = discrete_dynamics(x, u, d, t)
        return xn, xn

    # The scan input is a tuple (us, ts), where:
    #   us: control input sequence, shape (N, control_dim)
    #   ts: time steps, shape (N,)
    # This allows scan_fn to receive both the control input and time at each step.
    _, xs = jax.lax.scan(scan_fn, x0, (us, ds, ts))

    return jnp.concatenate([x0[None], xs], axis=0)


@equinox.filter_jit
def simulate_with_controller(x0, discrete_dynamics, controller, ds, ts):
    """
    Simulates the system dynamics over time using a discrete-time model,
    disturbances, and a controller.

    Args:
        x0: Initial state, shape (state_dim,).
        discrete_dynamics: Function that computes next state given (x, u, d, t).
        controller: a function that takes the current state as an argument and outputs a control.
        ds: Disturbance sequence, shape (N, disturbance_dim).
        ts: Time steps, shape (N,).

    Returns:
        Array of states over time, shape (N+1, state_dim).
    """
    dts = jnp.ones(N) * (ts[1] - ts[0])  # assume uniform time steps

    def scan_fn(carry, ctrl_disturb_t_dt):
        x, controller = carry
        d, t, dt = ctrl_disturb_t_dt
        u = controller(x, dt)
        xn = discrete_dynamics(x, u, d, t)
        return (xn, controller), xn

    _, xs = jax.lax.scan(scan_fn, (x0, controller), (ds, ts, dts))

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

def plot_states(xs, state_labels, state_indices, us, ts):
    fig = sp.make_subplots(rows=len(state_labels) + 1, cols=1, shared_xaxes=True)
    for i, idx in enumerate(state_indices):
        fig.add_trace(
            go.Scatter(x=ts, y=xs[:-1, state_indices[i]], name=state_labels[i]),
            row=i + 1,
            col=1,
        )
        fig.update_yaxes(title_text=state_labels[i], row=i + 1, col=1)
    
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
time_horizon = 50  # total time (seconds)
N = int(time_horizon / dt)  # number of timesteps
ts = jnp.arange(0, time_horizon, dt)


##############################
## BUILD JANN 4DOF DYNAMICS ##
##############################
jann_continuous_dynamics = JannParafoil4DOF(jann_4dof_params)
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
slegers_continuous_dynamics = SlegersParafoil6DOF(slegers_6dof_nonlinear_params)
slegers_discrete_dynamics = get_discrete_time_dynamics(slegers_continuous_dynamics, dt)

# control sequence:

# us_slegers = jnp.ones(N) * 1  # delta_a
# us_slegers = jnp.zeros(N) # no control
# us_slegers = us_slegers.at[N // 2 :].set(2.0) # second half turn

# build a control input that starts at zero, ramps up to ramp_max 
# from T/2 to (T/2 + ramp_time), and stays at ramp_max indefinitely
ramp_max = 0
ramp_time = 3.0 # seconds
ramp_start_time = ts[N//2]
u_interp = jnp.array([0, 0, ramp_max, ramp_max]) 
t_interp = jnp.array([0, ramp_start_time, ramp_start_time+ramp_time, ts[-1]])
us_slegers = jnp.interp(ts, t_interp, u_interp)

# disturbance sequence (wind in x,y,z)
wind_x = jnp.zeros(N)
wind_y = jnp.ones(N) * -5 # ft/s
wind_z = jnp.zeros(N)
ds_slegers = jnp.stack([wind_x, wind_y, wind_z], axis=1) # shape (N, 3) for jax.lax.scan
# OR
# ds_slegers = jnp.zeros((N, 3))  # no wind disturbance

# Initial state (imperial units):
slegers_initial_state = {
    "x": 0.0, # ft
    "y": 0.0,
    "z": -200.0,
    "u": 10.0, # ft/s
    "v": 0.1,
    "w": 10.0,
    "phi": 20 * DEG_TO_RAD, # deg -> rad
    "theta": 2 * DEG_TO_RAD,
    "psi": 0.0 * DEG_TO_RAD, 
    "p": 0.0 * DEG_TO_RAD, # deg/s -> rad/s
    "q": 0.0 * DEG_TO_RAD,
    "r": 0.0 * DEG_TO_RAD,
}

x0_slegers = jnp.array(list(slegers_initial_state.values()))


#######################
## SIMULATE AND PLOT ##
#######################
start_time = time.time()
# xs = simulate(x0_jann, us_jann, ds_jann, ts, jann_discrete_dynamics)
# xs = simulate(x0_slegers, us_slegers, ds_slegers, ts, slegers_discrete_dynamics)

heading_controller = controllers.TwelveStateHeadingController(kp=0.5, ki=0.0, kd=0.1)
xs = simulate_with_controller(x0_slegers, slegers_discrete_dynamics,
                              heading_controller, 
                              ds_slegers, ts)
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

label_idx_to_plot = {
        # "x (ft)": 0,
        # "y (ft)": 1,
        # "z (ft)": 2,
        "u (ft/s)": 3,
        "v (ft/s)": 4,
        "w (ft/s)": 5,
        "phi (deg)": 6,
        "theta (deg)": 7,
        "psi (deg)": 8,
        # "p (deg/s)": 9,
        # "q (deg/s)": 10,
        # "r (deg/s)": 11,
    }
labels = list(label_idx_to_plot.keys())
indices = list(label_idx_to_plot.values())

plot_slegers_3D(xs, ts)
plot_states(xs, labels, indices, us_slegers, ts)

# plot_jann_body(xs, ts)
