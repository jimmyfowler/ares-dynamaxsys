import jax.numpy as jnp
from dynamaxsys.base import Dynamics

def getInertialToBodyRotationMatrix(phi, theta, psi):
    """
    Computes the rotation matrix to transform vectors from the inertial frame to body frame
    velocity given the roll (phi), pitch (theta), and yaw (psi) angles.

    Args:
        phi (float): Roll angle in radians.
        theta (float): Pitch angle in radians.
        psi (float): Yaw angle in radians.

    Returns:
        jnp.ndarray: A 3x3 rotation matrix.
    """
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

    return inertial_to_body

class JannParafoil4DOF(Dynamics):
    state_dim: int = 4  # u, w, phi, psi
    control_dim: int = 2  # delta_a, delta_s

    m: float  # mass
    S: float  # parafoil area
    C_L0: float  # baseline lift coefficient
    C_D0: float  # baseline drag coefficient
    C_L_delta_s: float  # lift coefficient per unit deflection of steering line
    C_D_delta_s: float  # drag coefficient per unit deflection of steering line
    K_phi: float  # roll model gain
    T_phi: float  # roll model time constant
    g: float  # gravity constant

    def __init__(self, params: dict):
        self.m = params["m"]
        self.S = params["S"]
        self.C_L0 = params["C_L0"]
        self.C_D0 = params["C_D0"]
        self.C_L_delta_s = params["C_L_delta_s"]
        self.C_D_delta_s = params["C_D_delta_s"]
        self.K_phi = params["K_phi"]
        self.T_phi = params["T_phi"]
        self.g = params["g"]

        def dynamics_func(state, control, time=0):
            u, w, phi, psi = state
            delta_a, delta_s = control

            # Aerodynamics
            C_L = self.C_L0 + self.C_L_delta_s * delta_s  # Lift coefficient
            C_D = self.C_D0 + self.C_D_delta_s * delta_s  # Drag coefficient
            V_a = jnp.sqrt(u**2 + w**2)  # Airspeed
            alpha = jnp.arctan2(w, u)  # Angle of Attack
            rho = 1.225  # Air density at sea level

            L = 0.5 * rho * V_a**2 * self.S * C_L  # Lift
            D = 0.5 * rho * V_a**2 * self.S * C_D  # Drag

            # Equations of motion
            phi_dot = (self.K_phi * delta_a - phi) / self.T_phi  # Roll rate

            psi_dot = self.g / u * jnp.tan(phi) + w * phi_dot / (
                u * jnp.cos(phi)
            )  # Yaw rate

            u_dot = (
                L * jnp.sin(alpha) - D * jnp.cos(alpha) 
            ) / self.m - w * psi_dot * jnp.sin(phi) / self.m  # fwd accel

            w_dot = (
                (-L * jnp.cos(alpha) - D * jnp.sin(alpha)) / self.m
                + self.g * jnp.cos(phi)
                + u * psi_dot * jnp.sin(phi)
            )  # down accel

            return jnp.array([u_dot, w_dot, phi_dot, psi_dot])

        # initialize super class Dynamics object
        super().__init__(dynamics_func, self.state_dim, self.control_dim)

class SlegersParafoil6DOF(Dynamics):
    state_dim: int = 12  # x, y, z, u, v, w, phi, theta, psi, p, q, r
    control_dim: int = 1  # delta_a

    # Physical parameters
    m: float  # mass (kg)
    mmoi: jnp.ndarray  # moment of inertia matrix (3x3)
    mmoi_inverse: jnp.ndarray  # inverse of moment of inertia matrix (3x3)
    S: float  # parafoil area
    b: float  # span
    c: float  # chord length
    g: float  # gravity constant
    rho: float  # air density

    # Lift and drag coefficients
    C_L0: float  # baseline lift coefficient
    C_L_alpha: float  # lift coefficient per radian AoA
    C_L_delta_a: float  # lift coefficient per unit asymmetric deflection
    C_D0: float  # baseline drag coefficient
    C_D_alpha2: float  # quadratic drag coefficient per radian^2 AoA
    C_D_delta_a: float  # drag coefficient per unit asymmetric deflection

    # Aerodynamic moment coefficients
    C_lphi: float  # roll moment coefficient
    C_lp: float  # roll damping coefficient
    C_l_delta_a: float  # roll moment per unit asymmetric deflection
    C_m0: float  # pitch moment coefficient
    C_m_alpha: float  # pitch moment per radian AoA
    C_mq: float  # pitch damping coefficient
    C_n_r: float  # yaw damping coefficient
    C_n_delta_a: float  # yaw moment per unit asymmetric deflection

    def __init__(self, params: dict):
        # Physical parameters
        self.m = params["m"]
        self.mmoi = params["mmoi"]
        self.mmoi_inverse = jnp.linalg.inv(self.mmoi)
        self.S = params["S"]
        self.b = params["b"]
        self.c = params["c"]
        self.g = params["g"]
        self.rho = params["rho"]

        # Lift and drag coefficients
        self.C_L0 = params["C_L0"]
        self.C_D0 = params["C_D0"]
        self.C_L_alpha = params["C_L_alpha"]
        self.C_D_alpha2 = params["C_D_alpha2"]
        self.C_L_delta_a = params["C_L_delta_a"]
        self.C_D_delta_a = params["C_D_delta_a"]

        # Aerodynamic moment coefficients
        self.C_lphi = params["C_lphi"]
        self.C_lp = params["C_lp"]
        self.C_l_delta_a = params["C_l_delta_a"]
        self.C_m0 = params["C_m0"]
        self.C_m_alpha = params["C_m_alpha"]
        self.C_mq = params["C_mq"]
        self.C_n_r = params["C_n_r"]
        self.C_n_delta_a = params["C_n_delta_a"]

        def dynamics_func(state, control, time=0):
            x, y, z, u, v, w, phi, theta, psi, p, q, r = state
            delta_a = control

            skew_symmetric_pqr = jnp.array([[0, -r, q], [r, 0, -p], [-q, p, 0]])

            # Angle of Attack
            alpha = jnp.arctan2(w, u)  # assumes chordline along fwd-body axis

            # Lift and Drag Coefficients
            C_L = self.C_L0 + self.C_L_alpha * alpha + self.C_L_delta_a * delta_a
            C_D = self.C_D0 + self.C_D_alpha2 * alpha**2 + self.C_D_delta_a * delta_a

            V_a = jnp.sqrt(u**2 + v**2 + w**2)  # Airspeed

            inertial_to_body = getInertialToBodyRotationMatrix(phi, theta, psi)
            body_to_inertial = inertial_to_body.T  # rotation matrix is orthogonal

            L_div_Va = 0.5 * self.rho * V_a * self.S * C_L  # Lift
            D_div_Va = 0.5 * self.rho * V_a * self.S * C_D  # Drag

            # Aerodynamic Force
            aero_force = (
                L_div_Va * jnp.array([w, 0, -u]) - D_div_Va * jnp.array([u, v, w])
            )

            # Aerodynamic Moment
            aero_moment = (
                0.5
                * self.rho
                * V_a**2
                * self.S
                * jnp.array(
                    [
                        self.C_lphi * phi * self.b
                        + self.C_lp * self.b**2 * p / (2 * V_a)
                        + self.C_l_delta_a * delta_a * self.b,
                        self.C_m0 * self.c
                        + self.C_m_alpha * self.c * alpha
                        + self.C_mq * self.c**2 * q / (2 * V_a),
                        self.C_n_r * self.b**2 * r / (2 * V_a)
                        + self.C_n_delta_a * delta_a * self.b,
                    ]
                )
            )
                        

            # Weight Force
            weight_force = (
                self.m
                * self.g
                * jnp.array(
                    [
                        -jnp.sin(theta),
                        jnp.cos(theta) * jnp.sin(phi),
                        jnp.cos(theta) * jnp.cos(phi),
                    ]
                )
            )

            ## EQUATIONS OF MOTION ##

            # inertial-frame velocity
            xyz_dot = body_to_inertial @ jnp.array([u, v, w])

            # body-frame translation dynamics
            uvw_dot_force = 1 / self.m * (aero_force + weight_force)

            uvw_dot_coriolis = (
                -skew_symmetric_pqr @ jnp.array([u, v, w])
            )

            uvw_dot = uvw_dot_force + uvw_dot_coriolis

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
                - skew_symmetric_pqr @ self.mmoi @ jnp.array([p, q, r])
            )

            return jnp.concatenate([xyz_dot, uvw_dot, euler_dot, pqr_dot])

        # initialize superclass Dynamics object
        super().__init__(dynamics_func, self.state_dim, self.control_dim)
