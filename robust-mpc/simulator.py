import equinox
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.subplots as sp

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

# @equinox.filter_jit
# def simulate_with_controller(x0, discrete_dynamics, controller, ds, ts):
#     """
#     Simulates the system dynamics over time using a discrete-time model,
#     disturbances, and a controller.

#     Args:
#         x0: Initial state, shape (state_dim,).
#         discrete_dynamics: Function that computes next state given (x, u, d, t).
#         controller: a function that takes the current state and dt and outputs a control.
#         ds: Disturbance sequence, shape (N, disturbance_dim).
#         ts: Time steps, shape (N,).

#     Returns:
#         xs: Array of states over time, shape (N, state_dim).
#         us: Array of control inputs over time, shape (N, control_dim).
#     """
#     N = len(ts)
#     dts = jnp.diff(ts, prepend=0.0)  # time step differences
#     xs = jnp.zeros((12, N))
#     xs = xs.at[:,0].set(x0)
#     for i in range(N-1):
#         u = controller(xs[:,i], dts[i])
#         xs = xs.at[:, i+1].set( discrete_dynamics(jnp.array(xs[:,i]), u, ds[i], ts[i]) )
#     us = jnp.array([controller(xs[:,i], dts[i]) for i in range(N)])
#     return xs, us



@equinox.filter_jit #TODO: fix jax implementation
def simulate_with_controller(x0, discrete_dynamics, controller, ctrl_state0, ds, ts):
    """
    Simulates the system dynamics over time using a discrete-time model,
    disturbances, and a controller.

    Args:
        x0: Initial state, shape (state_dim,).
        discrete_dynamics: Function that computes next state given (x, u, d, t).
        controller: an Equinox module (pytree) with signature controller(x, ctrl_state, dt) -> (u, new_state)
        ctrl_state0: initial controller state (e.g., (jnp.array(0.0), jnp.array(0.0)))
        ds: disturbances (N, disturbance_dim)
        ts: times (N,)

    Returns:
        Array of states over time, shape (N+1, state_dim).
    """
    dts = jnp.diff(ts, prepend=ts[0])

    init_carry = (x0, ctrl_state0)

    def scan_fn(carry, inputs):
        x, ctrl_state = carry
        d, t, dt = inputs
        u, new_ctrl_state = controller(x, ctrl_state, dt)
        xn = discrete_dynamics(x, u, d, t)
        return (xn, new_ctrl_state), (xn, u)

    _, outputs = jax.lax.scan(scan_fn, init_carry, (ds, ts, dts))
    xs, us = outputs

    return jnp.concatenate([x0[None], xs], axis=0), us


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
    fig.update_yaxes(
        title_text="Asymetric Deflection", row=len(state_indices) + 1, col=1
    )

    fig.update_xaxes(title_text="Time (s)", row=len(state_indices) + 1, col=1)
    fig.update_layout(height=200 * (len(state_indices) + 1), showlegend=False)
    fig.show()


def plot_3D_traj(xs, ts):
    # Get positions and forward speed
    x = xs[:, 0]
    y = xs[:, 1]
    z = xs[:, 2]
    fwd_speed = xs[:, 3]
    dwn_speed = xs[:, 5]

    # Create a hover text that includes time
    hover_text = [
        f"Time: {t:.2f}s<br>Down Speed: {ds:.2f} ft/s" for t, ds in zip(ts, dwn_speed)
    ]

    # main trajectory trace colored by downward speed
    traj_trace = go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode="lines+markers",
        line=dict(color=dwn_speed, colorscale="viridis", width=0.5),
        marker=dict(
            # color=dwn_speed,
            # colorscale="viridis",
            size=4,
            # colorbar=dict(title="Downward Speed (ft/s)"),
        ),
        text=hover_text,
        hoverinfo="text",
        name="Trajectory",
    )

    # large green starting marker
    start_x, start_y, start_z = x[0], y[0], z[0]
    start_trace = go.Scatter3d(
        x=[start_x],
        y=[start_y],
        z=[start_z],
        mode="markers",
        marker=dict(size=12, color="green", symbol="circle"),
        name="Start",
        hovertext=f"Start<br>Time: {ts[0]:.2f}s",
        hoverinfo="text",
        showlegend=True,
    )

    target_zone_trace = go.Scatter3d(
        x=[0.0],
        y=[0.0],
        z=[0.0],
        mode="markers",
        marker=dict(size=8, color="red", symbol="circle"),
        name="Target Zone",
        hovertext="Target Zone Center",
    )

    fig = go.Figure(data=[traj_trace, start_trace, target_zone_trace])

    # xr = max(x) - min(x)
    # yr = max(y) - min(y)
    # zr = max(z) - min(z)

    fig.update_layout(
        scene=dict(
            xaxis_title="X Position (ft)",
            yaxis_title="Y Position (ft)",
            zaxis_title="Altitude (ft)",
            # xaxis=dict(nticks=4, range=[-500, 500], title="X Position (ft)"),
            # yaxis=dict(nticks=4, range=[-500, 500], title="Y Position (ft)"),
            # zaxis=dict(nticks=4, range=[0, 500], title="Altitude (ft)"),
            aspectmode="data",
            # aspectmode = "manual",
            # aspectratio = dict(x=500, y=500, z=500),
        ),
        title="3D Trajectory",
    )

    fig.show()