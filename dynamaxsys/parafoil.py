import jax.numpy as jnp
from dynamaxsys.base import Dynamics
from ambiance import Atmosphere


def get_air_density_isa(altitude_meters):
    """
    Calculates the air density at a given altitude according to the
    International Standard Atmosphere (ISA) model.

    Args:
        altitude_meters (float): The geometric altitude in meters.

    Returns:
        float: The air density in kg/m^3.
    """
    atmosphere = Atmosphere(altitude_meters)
    return atmosphere.density[0]


class JannParafoil4DOF(Dynamics):
    state_dim: int = 4  # u, w, phi, psi
    control_dim: int = 2  # delta_a, delta_s

    m: float  # mass (kg)
    S: float  # parafoil area (m^2)
    C_L0: float  # baseline lift coefficient
    C_D0: float  # baseline drag coefficient
    C_L_delta_s: float  # lift coefficient per unit deflection of steering line
    C_D_delta_s: float  # drag coefficient per unit deflection of steering line
    K_phi: float  # roll model gain
    T_phi: float  # roll model time constant
    g: float = 9.81  # gravity constant (m/s^2)

    def __init__(self, params: dict):
        self.m = params["m"]
        self.S = params["S"]
        self.C_L0 = params["C_L0"]
        self.C_D0 = params["C_D0"]
        self.C_L_delta_s = params["C_L_delta_s"]
        self.C_D_delta_s = params["C_D_delta_s"]
        self.K_phi = params["K_phi"]
        self.T_phi = params["T_phi"]
        self.g = params.get("g", 9.81)

        def dynamics_func(state, control, time=0):
            u, w, phi, psi = state
            delta_a, delta_s = control

            # Aerodynamics
            C_L = self.C_L0 + self.C_L_delta_s * delta_s  # Lift coefficient
            C_D = self.C_D0 + self.C_D_delta_s * delta_s  # Drag coefficient
            V_a = jnp.sqrt(u**2 + w**2)  # Airspeed
            alpha = jnp.arctan2(w, u)  # Angle of Attack
            rho = 1.225  # Air density at sea level in kg/m^3

            L = 0.5 * rho * V_a**2 * self.S * C_L  # Lift
            D = 0.5 * rho * V_a**2 * self.S * C_D  # Drag

            # Equations of motion
            phi_dot = (self.K_phi * delta_a - phi) / self.T_phi  # Roll rate

            psi_dot = self.g / u * jnp.tan(phi) + w * phi_dot / (
                u * jnp.cos(phi)
            )  # Yaw rate

            u_dot = (
                L * jnp.sin(alpha) - D * jnp.cos(alpha)
            ) / self.m - w * self.g * jnp.sin(psi) / self.m  # fwd accel

            w_dot = (
                (-L * jnp.cos(alpha) - D * jnp.sin(alpha)) / self.m
                + self.g * jnp.cos(psi)
                + u * phi_dot * jnp.sin(phi)
            )  # down accel

            return jnp.array([u_dot, w_dot, phi_dot, psi_dot])

        # initialize super class Dynamics object
        super().__init__(dynamics_func, self.state_dim, self.control_dim)


class SlegersParafoil4DOF(Dynamics):
    state_dim: int = 12  # x, y, z, u, v, w, phi, theta, psi, p, q, r
    control_dim: int = 1  # delta_a

    m: float  # mass (kg)
    S: float  # parafoil area (m^2)
    C_L0: float  # baseline lift coefficient
    C_D0: float  # baseline drag coefficient
    C_L_delta_s: float  # lift coefficient per unit deflection of steering line
    C_D_delta_s: float  # drag coefficient per unit deflection of steering line
    K_phi: float  # roll model gain
    T_phi: float  # roll model time constant
    g: float = 9.81  # gravity constant (m/s^2)

    def __init__(self, params: dict):
        self.m = params["m"]
        self.mmoi = params["mmoi"]
        self.mmoi_inverse = jnp.linalg.inv(self.mmoi)
        self.S = params["S"]
        self.C_L0 = params["C_L0"]
        self.C_D0 = params["C_D0"]
        self.C_L_alpha = params["C_L_alpha"]
        self.C_D_alpha2 = params["C_D_alpha2"]
        self.C_L_delta_a = params["C_L_delta_a"]
        self.C_D_delta_a = params["C_D_delta_a"]
        self.K_phi = params["K_phi"]
        self.T_phi = params["T_phi"]
        self.g = params.get("g", 9.81)

        def dynamics_func(state, control, time=0):
            x, y, z, u, v, w, phi, theta, psi, p, q, r = state
            delta_a, delta_s = control

            skew_symmetric_pqr = jnp.array([[0, -r, q], [r, 0, -p], [-q, p, 0]])

            inertial_to_body = jnp.array(
                [
                    [  # row 1
                        jnp.cos(theta) * jnp.cos(psi),
                        jnp.cos(theta) * jnp.sin(psi),
                        -jnp.sin(theta),
                    ],
                    [  # row 2
                        jnp.sin(phi) * jnp.sin(theta) * jnp.cos(psi)
                        - jnp.cos(phi) * jnp.sin(psi),
                        jnp.sin(phi) * jnp.sin(theta) * jnp.sin(psi)
                        + jnp.cos(phi) * jnp.cos(psi),
                        jnp.sin(phi) * jnp.cos(theta),
                    ],
                    [  # row 3
                        jnp.cos(phi) * jnp.sin(theta) * jnp.cos(psi)
                        + jnp.sin(phi) * jnp.sin(psi),
                        jnp.cos(phi) * jnp.sin(theta) * jnp.sin(psi)
                        - jnp.sin(phi) * jnp.cos(psi),
                        jnp.cos(phi) * jnp.cos(theta),
                    ],
                ]
            )

            body_to_inertial = inertial_to_body.T  # rotation matrix is orthogonal

            # Aerodynamics
            alpha = jnp.arctan2(w, u)  # Angle of Attack

            C_L = self.C_L0 + self.C_L_alpha * alpha + self.C_L_delta_a * delta_a # Lift coefficient
            C_D = self.C_D0 + self.C_D_delta_s * delta_s  # Drag coefficient
            V_a = jnp.sqrt(u**2 + w**2)  # Airspeed

            rho = get_air_density_isa(z)  # Air density in kg/m^3

            L = 0.5 * rho * V_a**2 * self.S * C_L  # Lift
            D = 0.5 * rho * V_a**2 * self.S * C_D  # Drag

            aero_moment = (
                0.5 * rho * V_a**2 * self.S * jnp.array([0.0, 0.0, 0.0])
            )  # placeholder for now

            # Equations of motion

            # inertial frame tranlsation dynamics
            xyz_dot = body_to_inertial @ jnp.array([u, v, w])

            # body frame translation dynamics
            uvw_dot = jnp.array([0.0, 0.0, 0.0])  # u, v, w are assumed constant

            pqr_to_euler_dot = jnp.array(
                [
                    [1, jnp.sin(phi) * jnp.tan(theta), jnp.cos(phi) * jnp.tan(theta)],
                    [0, jnp.cos(phi), -jnp.sin(phi)],
                    [0, jnp.sin(phi) / jnp.cos(theta), jnp.cos(phi) / jnp.cos(theta)],
                ]
            )

            # euler angle dynamics (from body rates)
            euler_dot = pqr_to_euler_dot @ jnp.array([p, q, r])

            # body frame rotational dynamics
            pqr_dot = self.mmoi_inverse @ (
                aero_moment
                - skew_symmetric_pqr @ jnp.identity(3) @ jnp.array([p, q, r])
            )

            return jnp.stack([xyz_dot, uvw_dot, euler_dot, pqr_dot])

        # initialize super class Dynamics object
        super().__init__(dynamics_func, self.state_dim, self.control_dim)
