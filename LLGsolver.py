''' 
    LLG solver which utilises a simple forward difference method along with the Landau-Lifshitz-Gilbert
    equation to model magnetisation dynamics in a ferromagnetic/paramagnetic bilayer under the influence of a static
    external field and spin-orbit torque induced by the spin-Hall effect in the paramagnetic layer.
'''

# imports -----------------------------------------------------------------------------------------

import numpy as np
import numpy.linalg
from numba import njit, types
from scipy.integrate import solve_bvp
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from datetime import datetime
import progressbar

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['font.size'] = '14'


# global Variables --------------------------------------------------------------------------------

## Intrinsic Material Parameters 
rho_0 = 8.6e-8 # Platinum resistivity at R.T
thetaSH = 0.3 # spin hall angle for platinum at R.T [Lee I think?]
Gr = 5e14 # mediates spin flip scattering at the interface
Gi = 5e13j # local exchange field applies torque on spin accum at interface
G = Gr + Gi # complex spin mixing conductance 
alpha_0 = 0.03
gamma_0 = 1.76e11 #

# Extrinsic material parameters
# Cylindrical nanomagnet of radius r_F and thickness t_F
t_N = 2e-9
w_N = 30e-9
t_F = 1e-9
r_F = 15e-9
Area_F = np.pi * r_F **2
Area_N = t_N * w_N

# At Temperature T in Pt
Temp = 300 # Kelvin
e_density = 2.81e29 # (https://arxiv.org/pdf/1707.05330.pdf)
Fermi_v = 1.44e6
Fermi_k = 1.2e10
Tau_p = 8.44e-16 # momentum relaxation time (=m_e/rho*e_density*e**2)
lsf = 1.5e-9 # spin-diffusion length in platinum at R.T
Tau_sf = 1e-15 # spin flip time for platium (https://arxiv.org/pdf/1707.05330.pdf)
g_sharvin = Area_F * Fermi_k **2 / (4*np.pi)

## physical constants
elem_charge = 1.602e-19 # elementary charge
hbar = 1.0545e-34
boltz_const = 1.39e-23
G_0 = elem_charge**2 / ( hbar * 2 * np.pi )
mu_0 = 4 * np.pi * 1e-7 # N/A^2

## Field Magnitudes
theta = 0
I_max = 0e-6
h_x = 200 * 79  # A/m 
h_rf = 0.0
h_k = 500 * 79
h_Oe = 0
M_s = 1000e3 #
Nx = Ny = 0.01
Nz = 1 - Nx - Ny

t_rise = 0.5e-9
t_off = 3e-9

@njit
def Jc_in(t):
    jc_max = I_max / (Area_N)
    if t < t_off:
        j = jc_max * (1 - np.exp( - t / t_rise ))
    if t>= t_off:
        j = jc_max * (np.exp( - (t - t_off) / t_rise ))

    return j

@njit 
def Eff_Field(t, m, dt):
    """Calculates the effective field at time t for use in the dmdt LLG solver. """

    H = h_x * np.array( [ 1.0, 0.0, 0.0 ] )
    H_demag = np.array([Nx, Ny, Nz]) * M_s * m # demagnetisation field for thin film
    # anisotropy ->
    x_axis = np.array([1.0,0.0,0.0])
    y_axis = np.array([0.0,1.0,0.0])
    z_axis = np.array([0.0,0.0,1.0])
    m = np.ascontiguousarray(m)
    H_an = h_k * z_axis * ((np.dot(z_axis, m)))  # uniaxial anisotrpy
    # thermal ->
    std_dev = np.sqrt( (2 * alpha_0 * boltz_const * Temp) / (mu_0 * M_s * Area_F * t_F  )) 
    gaussian_x, gaussian_y, gaussian_z = np.random.normal(0, std_dev, 3) 
    H_t = np.array([gaussian_x, gaussian_y, gaussian_z]) # Thermal fluctuations Lengevin field
    # # when working with oscillating currents: H_Oe = J_{c,rf} * t_N /2

    return  H - H_demag + H_an + H_t


@njit
def dmdt(m, t, dt):
    """Calculates the rate of change of teh magnetisation at each time
        point according to varying fields and input charge currents. 
        The value of dmdt determines the amount of spin current injected
        into the nomral metal layer and hence the value of the effective
        gilbert damping coefficien t and effective gyromagnetic ratio. """

    H_eff = Eff_Field(t, m, dt) * mu_0
    J = Jc_in(t)
    y_hat = np.array([0.0, 1.0, 0.0])

    cj = hbar * thetaSH * J / (2 * elem_charge * M_s * t_F) 

    dmdt = - ( np.cross(m, H_eff) 
    + alpha_0 * np.cross( m, np.cross( m, H_eff)) 
    + alpha_0 * cj * np.cross(m, y_hat)
    - cj * np.cross( m, np.cross( m, y_hat) )
    ) * gamma_0 / (1+alpha_0**2) 


    return dmdt, J


def time_evolution(m_init, t_span, t_points, t_N, n_points):
    """ This takes puts together all of the calculations perfomermed
        by the other functions in this module to simulate the time
        dependendence of SMR. This uses a Heun method (second order)
        IVP solving method to calculate the rate of change at th current
        time point and estimates the gradient at the next time point to 
        develop a library of magnetisations at times t. t_span= 'length
        of time simulated', t_points='time points simulated', t_N='metal
        thickness', n_points= 'spatial points simulated along z',
        f= 'rf frequency' """
    
    dt = t_span / t_points

    #initialise arrays
    ms = np.zeros((3, t_points))
    ms[:, 0] = m_init
    ts = np.linspace(0, t_span, t_points)
    dmdt_l = np.zeros(3)
    Jc = np.zeros(t_points)
    for t in progressbar.progressbar(range(t_points-1)):

        # magnetisation dynamics
        dmdt_l, Jc[t] = dmdt(ms[:,t], ts[t], dt)
        ms[:,t+1] = ms[:,t] + (dt) * dmdt_l
        ms[:,t+1] /= np.linalg.norm(ms[:,t+1])

    return ts, ms, Jc



if __name__ == "__main__":

    m_init = np.array([0.0, 0.0, 1.0])
    m_init = m_init / np.linalg.norm(m_init)
    t_step = 5e-14
    t_span = 5e-9
    t_points = np.int(np.round(t_span / t_step))

    ts, ms, Jc = time_evolution(m_init, t_span, t_points, 0, 100 )
    ts *= 1e9 # nanoseconds

    f = plt.figure(figsize=(8,6))
    ax1 = f.add_axes([0.1,0.1,0.85,0.85])
    ax1.plot(ts, ms[0], label=r'$m_x$')
    ax1.plot(ts, ms[1], label=r'$m_y$')
    ax1.plot(ts, ms[2], label=r'$m_z$')
    ax1.set_xlabel(r'Time (ns)')
    ax1.set_ylabel(r'Norm. Magnetisation')
    ax1.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax1.set_xlim([0, t_span * 1e9])
    ax1.legend()
    

    plt.show()