import jax.numpy as jnp
from dynamaxsys.base import Dynamics


class JannParafoil4DOF(Dynamics):
    state_dim: int = 4 # u, w, phi, psi
    control_dim: int = 2 # delta_a, delta_s
    
    m: float
    S: float
    C_L0: float
    C_D0: float
    C_L_delta_s: float
    C_D_delta_s: float
    K_phi: float # roll model gain
    T_phi: float # roll model time constant
    g: float = 9.81

    def __init__(self, m, S, C_L0, C_D0, C_L_delta_s, C_D_delta_s, K_phi, T_phi):
        self.m = m
        self.S = S
        self.C_L0 = C_L0    
        self.C_D0 = C_D0
        self.C_L_delta_s = C_L_delta_s
        self.C_D_delta_s = C_D_delta_s
        self.K_phi = K_phi
        self.T_phi = T_phi
        
        def dynamics_func(state, control, time=0):
            u, w, phi, psi = state
            delta_a, delta_s = control

            # Aerodynamics
            C_L = self.C_L0 + self.C_L_delta_s * delta_s    # Lift coefficient
            C_D = self.C_D0 + self.C_D_delta_s * delta_s    # Drag coefficient
            V_a = jnp.sqrt(u**2 + w**2)                     # Airspeed
            alpha = jnp.arctan2(w, u)                       # Angle of Attack
            rho = 1.225                                     # Air density at sea level in kg/m^3
            
            L = 0.5 * rho * V_a**2 * S * C_L    # Lift
            D = 0.5 * rho * V_a**2 * S * C_D    # Drag
            
            # Equations of motion
            phi_dot = ( self.K_phi * delta_a - phi ) / self.T_phi                           # Roll rate
            
            psi_dot = self.g / u * jnp.tan(phi) + w * phi_dot / ( u * jnp.cos(phi) )   # Yaw rate
            
            u_dot = ( L*jnp.sin(alpha) - D*jnp.cos(alpha) )/m \
                    - w * self.g * jnp.sin(psi) / self.m                               # fwd accel
                    
            w_dot = ( -L*jnp.cos(alpha) - D*jnp.sin(alpha) )/m \
                    + self.g * jnp.cos(psi) + u * phi_dot * jnp.sin(phi)               # down accel


            return jnp.array([u_dot, w_dot, phi_dot, psi_dot])
        
        super().__init__(dynamics_func, self.state_dim, self.control_dim)