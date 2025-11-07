import jax.numpy as jnp
import time
from dynamaxsys.base import get_discrete_time_dynamics
from dynamaxsys.parafoil import JannParafoil4DOF, SlegersParafoil6DOF
from model_parameters import slegers_6dof_nonlinear_params, jann_4dof_params
import controllers

from simulator import (
    simulate,
    simulate_with_controller,
    DEG_TO_RAD,
    RAD_TO_DEG,
    plot_states,
    plot_jann_body,
    plot_slegers_3D,
)

################################
## Simulation Hyperparameters ##
################################
dt = 0.01  # time step (seconds)
time_horizon = 60  # total time (seconds)
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
us_slegers = jnp.zeros(N) # no control
# us_slegers = us_slegers.at[N // 2 :].set(2.0) # second half turn

# build a control input that starts at zero, ramps up to ramp_max
# from T/2 to (T/2 + ramp_time), and stays at ramp_max indefinitely
ramp_max = 0.5 # asymetric deflection (rad)
ramp_time = 3.0  # seconds
ramp_start_time = ts[N // 2]
u_interp = jnp.array([0, 0, ramp_max, ramp_max])
t_interp = jnp.array([0, ramp_start_time, ramp_start_time + ramp_time, ts[-1]])
# us_slegers = jnp.interp(ts, t_interp, u_interp)

# # disturbance sequence (wind in x,y,z)
wind_x = jnp.zeros(N)
wind_y = jnp.ones(N) * -5  # ft/s
wind_z = jnp.zeros(N)
ds_slegers = jnp.stack(
    [wind_x, wind_y, wind_z], axis=1
)  # shape (N, 3) for jax.lax.scan
# OR
ds_slegers = jnp.zeros((N, 3))  # no wind disturbance

# Initial state (imperial units):
slegers_initial_state = {
    "x": -300.0,  # ft
    "y": 100.0,
    "z": 1000.0,
    "u": 10.0,  # ft/s
    "v": 0.1,
    "w": 5.0,
    "phi": 10 * DEG_TO_RAD,  # deg -> rad
    "theta": 2 * DEG_TO_RAD,
    "psi": 0.0 * DEG_TO_RAD,
    "p": 0.0 * DEG_TO_RAD,  # deg/s -> rad/s
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
us = us_slegers  # for plotting
ctrl_state0 = jnp.array([0.0, 0.0])  # integral, prev_error
heading_controller = controllers.TwelveStateHeadingController(kp=1, ki=0.0, kd=0.1)
dummy_controller = controllers.DummyController()
dummy_controller_state0 = None # dummy controller has no state
xs, us = simulate_with_controller(x0_slegers, slegers_discrete_dynamics,
                              heading_controller, ctrl_state0,
                              ds_slegers, ts)
end_time = time.time()
print(f"Simulation run time: {end_time - start_time:.4f} seconds")
print(f"(total simulation frames: {N})")

# convert angles to degrees for plotting
for i in range(6, 12):
    xs = xs.at[:, i].set(xs[:, i] * RAD_TO_DEG)
    
# change z-coord to altitude
xs = xs.at[:, 2].set(-xs[:, 2] + 2*xs[:,2][0])

label_idx_to_plot = {
    "x (ft)": 0,
    "y (ft)": 1,
    "z (ft)": 2,
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
plot_states(xs, labels, indices, us, ts)

# plot_jann_body(xs, ts)
