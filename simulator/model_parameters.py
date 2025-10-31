import jax.numpy as jnp

# Slegers 6DOF nonlinear parafoil model parameters
# Imperial Units
slegers_6dof_nonlinear_params = {
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

# Jann 4DOF parafoil model parameters
# SI Units
jann_4dof_params = {
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