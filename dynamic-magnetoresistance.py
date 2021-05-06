''' 
    Full simulation to simulate the spin-Hall magnetoresistance effect in 
    heavy metal / magnetic insulating bilayers under dynamic magnetisation
    of the magnetic layer. This incorporates spin-orbit torque from the incident spin 
    current and also spin pumping induced by magnetisation dynamics.
    Thermal effects are also accounted for. The code is initialised for rectangular Pt/YIG bilayers.
'''

# imports -----------------------------------------------------------------------------------------

import numpy as np
import numpy.linalg
from numba import njit, types
from scipy.integrate import solve_bvp
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import progressbar

# global Variables --------------------------------------------------------------------------------

## Intrinsic Material Parameters 
rho_0 = 15e-8 # Intrinsic resistivity of the NM layer
thetaSH = 0.11 # spin-Hall agnle of the NM layer
Gr = 5e14 # Real aprt of spin-mixing conductance, number of spin channels at the interface
Gi = 5e13j # Imagniary spin-mixing conductance, local exchange field applies torque on spin accum at interface
G = Gr + Gi # complex spin mixing conductance 
alpha_0 = 0.0156 # Intrinsic Gilbert damping parameter
gamma_0 = 1.76e11 # Gyromagnetic ratio of the electron

# Extrinsic material parameters
# The bilayer system is initialised here to have rectangular FM and NM layers. The model can be generalised 
# to have any FM shape as long as the thickness, t, is constant.

# Rectangular NM dimensions
t_N = 6.8e-9 # thickness (z)
w_N = 1.5e-3 # width (y)
l_N = 5e-3 # length (z)
# Rectangular FM dimensions
t_F = 12.7e-9
w_F = 1.5e-3
l_F = 5e-3

Area_F = w_F * l_F # interface area

# Electron transport parameters for the NM layer, currently for Platinum
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
G_0 = elem_charge**2 / ( hbar * 2 * np.pi ) # quantum of spin-mixing
mu_0 = 4 * np.pi * 1e-7 # vacuum permiability 

## Field parameters
theta_xy = 90 * np.pi / 180 # in plane angle to the x axis (current direction)
theta_z = 0 * np.pi / 180 # out of plane, angle between applied field and xy plane.
# Field magnitudes to be inserted in Amps / meter (Oersteds * 79)
h_x = 1.2e3 * 79 # A/m (1.2 kOe)
h_rf = 0.0 # oscillating microwave field for driving FMR
h_an = 10 # Uniaxial anisotropy
M_s = 770e3 # In A/m
# Demag factors determined by magnet geometry
Nx = 0.000005778
Nz = 0.9999746885130856
Ny = 1 - Nx - Nz


@njit
def bcs(ya, yb, m, jc_in, jc_int):
    """ Defines the boundary conditions for spin current/spin accumulation in the metal layer. 
        Used by Scipy's solve_bvp solver. """

    mus0 = np.array([ya[3], ya[4], ya[5]]) # spin accumulation 
    eJs0 = np.real(G) * np.cross(m, np.cross(m, mus0)) + np.imag(G) * np.cross(m, mus0) # spin current

    return np.array([ 
        ya[0] - eJs0[0], 
        ya[1] - eJs0[1], 
        ya[2] - eJs0[2], 
        yb[0], 
        yb[1], 
        yb[2] 
        ])

@njit
def fun(x, y, m, jc_in, js_int):
    """ Returns the coupled equations to be solved by sovle_bvp. A,B,C are the coefficients defined
        as shown in the powerpoint sent over by Aiden. Note the positive sign of B in compensated for.
        ISHso is the spin current generated by the E-field """

    jSHs0 = thetaSH * hbar * jc_in / ( 2 * elem_charge ) * np.array([0.0, 1.0, 0.0 ]) # pure spin current at the interface
    js = jSHs0 - js_int  # total spin current at the interface

    A = - 1 / (2 * rho_0)
    B = elem_charge * js
    C = lsf ** 2

    return np.vstack((
        A * y[3] / C,
        A * y[4] / C,
        A * y[5] / C, 
        (y[0] + B[0]) / A,
        (y[1] + B[1]) / A,
        (y[2] + B[2]) / A
    ))

def bvp_solver(m, n_points, t_N, jc_in, js):
    """ Calls the scipy.solve_bvp function to solve the coupled equations defined by the "fun" function
        according to the BC's defined by "bcs" funtion. The arrays zx and  are to be populated by the
        solver. In the zs, z=0 denotes the interface and z=t_N is the interface to the vacuum. y hosts the solutions to the bvp solver 
        and has 6 elements for each spatial point, index 0-2 are the spin current, 3-5 are spin accumulation. n_points in the number 
        of spatial points that js/mus are calculated at. t_N is the thickness of the metal layer. """

    zs = np.linspace(0, t_N, n_points)
    y = np.zeros((6, zs.size))
    soln = solve_bvp(lambda x, y: fun(zs, y, m, jc_in, js), lambda ya, yb: bcs(ya, yb, m, jc_in, js), zs, y)
    js = np.vstack(( soln.y[0], soln.y[1], soln.y[2])) / elem_charge
    jc = charge_current_func(js, t_N, n_points)
    jc = np.average(jc, axis=1)

    return js, jc

@njit
def charge_current_func(js, t_N, n_points):
    """ Calculates the charge current in the normal metal layer 
        that is recoupled from the reflected or pumpued spin current (density). Longitudinal
        is the x direction and transverse y. In the instance of or reflected spin current it
        is recoupling the spin current as cahrge current as a function of z. In the pumping
        mechanism it calculates the chareg current pumped as a function of time. """
    
    z_hat = np.array([0,0,1])
    jc = np.zeros((3,n_points))
    for i in range(n_points):
        jc[:,i] = thetaSH * (2 * elem_charge / hbar )  * np.cross( js[:,i], z_hat)

    return jc

@njit
def input_jc(t, t_N, jc_in_max, f_j):
    """ Alows us to define the input charge current density as a function of 
        time so that we may introduce for example a rise/fall time to 
        the DC current or apply an AC current for the driving of
        magnetisation prescession. Initialised as a DC current. """

    # AC current of frequency f_j ->
    # jc_in = jc_in_max * np.cos( 2 * np.pi * f_j * t) 

    # Current pulse of rise time t_rise and turn off time of t_off ->
    # if t < t_off:
    #     jc_in = jc_in_max * (1 - np.exp( - (t  / t_rise) )
    # else:
    #     jc_in = jc_in_max * np. exp( - (t - t_off) / t_rise )

    # DC current with negligable rise time
    jc_in = jc_in_max

    return jc_in


@njit 
def Eff_Field(t, m, dt, jc_in, f_H):
    """ Calculates the effective field at time t for use in the dmdt LLG solver. Currently initialised for 
        a static applied field along x, uniaxial anisotropy with easy axis along x, demag field dependent on the 
        demagnetisation parameters N_i, thermal fluctation field. Oersted field = 0 for DC current. """


    H = h_x * np.array( [ np.cos(theta_xy), np.sin(theta_xy), np.sin(theta_z)  ] )
    H_demag = np.array([Nx, Ny, Nz]) * m * M_s # demagnetisation field accroding to demag factors
    # Oscillating driving field ->
    H_rf = h_rf * np.cos( 2 * np.pi * f_H * t )
    # uniaxial anisotropy ->
    Ez_axis = np.array([1.0,0.0,0.0])
    m = np.ascontiguousarray(m)
    H_an = h_an * np.dot(Ez_axis, m) * Ez_axis  # uniaxial anisotrpy
    # thermal fluctation field ->
    std_dev = np.sqrt( (2 * alpha_0 * boltz_const * Temp) / (mu_0 * M_s**2 * Area_F * t_F * dt))
    gaussian_x, gaussian_y, gaussian_z = np.random.normal(0, std_dev, 3)
    H_t = np.array([gaussian_x, gaussian_y, gaussian_z]) # Thermal fluctuations Lengevin field
    # Orested ->
    H_Oe = 0 
    # when working with ac currents: H_Oe = jc_in * t_N * np.array([0.0, 1.0, 0.0]) /2 # for thin films

    return  H + H_rf - H_demag  + H_an + H_t + H_Oe

@njit
def dmdt(m, t, jc_in, jsF, dt, f_j, f_H):
    """Calculates the rate of change of teh magnetisation at each time
        point according to varying fields and input charge currents. 
        The value of dmdt determines the amount of spin current injected
        into the nomral metal layer and hence the value of the effective
        gilbert damping coefficien t and effective gyromagnetic ratio. """

    H_eff = Eff_Field(t, m, dt, f_j, f_H) * mu_0 # obtains effective field at this time

    # checks if there is spin current at the interface
    if (np.linalg.norm(jsF) == 0.0):
        jsF_mod = 0
        sigma_hat = np.array([0.0,0.0,0.0])
    else:
        jsF_mod = np.linalg.norm(jsF)
        sigma_hat = jsF / jsF_mod # spin current polarisation
    
    cj = hbar * jsF_mod / (2 * elem_charge * mu_0 * M_s * t_F * Area_F) # STT coefficient

    # decoupled LLG equation
    dmdt = - ( np.cross(m, H_eff)
        + alpha_0 * np.cross( m, np.cross( m, H_eff)) 
        - alpha_0 * cj * np.cross(m, sigma_hat )
        + cj * np.cross(m , np.cross(m, sigma_hat)) 
        ) * gamma_0 / (1+alpha_0**2) 

    return dmdt


def time_evolution(m_init, t_span, t_points, t_N, n_points, jc_in_max, f_j, f_H):
    """ This takes puts together all of the calculations performed
        by the other functions in this module to simulate the time
        dependendence of SMR. This uses a Heun method (second order)
        IVP solving method to calculate the rate of change at th current
        time point and estimates the gradient at the next time point to 
        develop a library of magnetisations at times t. t_span= 'length
        of time simulated', t_points='time points simulated', t_N='metal
        thickness', n_points= 'spatial points simulated along z',
        f_H= 'rf frequency of field oscillation', f_j= 'rf frequency of
        current oscilation' """
    
    dt = t_span / t_points

    #initialise arrays
    ms = np.zeros((3, t_points))
    ms[:, 0] = m_init
    ts = np.linspace(0, t_span, t_points)
    js_p = np.zeros((3,t_points))
    jc_long_array = np.zeros(t_points)
    jc_trans_array = np.zeros(t_points)
    V_array = np.zeros((2,t_points))
    dmdt_l = np.zeros(3)
    js = np.zeros(3)

    Rsd = ( np.pi * np.sqrt(3* Tau_sf / Tau_p) ) / ( Area_F * Fermi_k **2 ) # dimensionless resistance
    beta = Rsd / np.tanh( t_N / lsf ) # imperfect spin sink, pumped spin current reflection coefficient
    A_eff = G / ( G_0 + G * beta )

    for t in progressbar.progressbar(range(t_points-1)):
        #  Retrieves charge current density from function definition
        jc_in = input_jc(t, t_N, jc_in_max, f_j)

        # Caclulate spin current and spin accumulation
        js_SHE, jc_SHE = bvp_solver(ms[:,t], n_points, t_N, jc_in, js_p[:,t-1])
        # js_SHE incorporates the pure spin current, diffusicve current and pumped s.c

        # magnetisation dynamics 
        dmdt_l = dmdt(ms[:,t], ts[t], jc_in, js_SHE[:,0], dt, f_j, f_H)
        ms[:,t+1] = ms[:,t] + (dt) * dmdt_l
        ms[:,t+1] /= np.linalg.norm(ms[:,t+1]) # renormalise the magnetisation

        #pumped spin/charge currents (not densities) due to magnetisation dynamics
        #assuming f-layer thicker than transverse spin-coherence length
        js_p[:,t] = (hbar / 4 * np.pi) * ( np.real(A_eff) * np.cross(ms[:,t], dmdt_l) - np.imag(A_eff) * dmdt_l) 
        jc_p = thetaSH * ( 2 * elem_charge / hbar ) * np.cross( js_p[:,t], np.array([0,0,1]) )

        # total charge current densities
        jc_tot = jc_in * np.array([1,0,0]) + jc_SHE
        jc_long_array[t] = jc_tot[0]
        jc_trans_array[t] = jc_tot[1]

        # long (ac) and transverse (dc) voltage due to spin pumping 
        V_array[0, t] = jc_tot[0] * l_N * rho_0
        V_array[1, t] = jc_tot[1] * w_N * rho_0
    

    return ts, ms, jc_long_array, jc_trans_array, V_array

# Analysis and fitting functions
# def turningpoints(ys, xs):
#     """ Looks at the stationary points of a magnetisation component. Can be used to 
#         mark the maxima of magnetisation components which can then be fitted by 
#         Exponential_dec function. """
#     dydx = np.gradient(ys, xs)
#     stat_p = []
#     mzs = []
#     for i in range(len(dydx)-1):
#         if (dydx[i] * dydx[i+1] < 0): 
#             if ys[i+1] > 0:
#                 stat_p.append(xs[i+1])
#                 mzs.append(ys[i+1])
#     return np.array(stat_p), np.array(mzs)

# def Exponential_dec(t, a, tau):
#     """ Can be fitted to a component of the magnetisation under relaxation to
#         calculate the effective gilbert damping parameter. 
#         (alpha = 1 / 2pi * tau * f) where f is the frequency of precession."""
    
#     return a * np.exp( -  t / tau )



if __name__ == "__main__":
    
    # setting simulation paramters
    m_init = np.array([0.91, -0.6, 0.0])
    m_init = m_init / np.linalg.norm(m_init)
    t_step = 1.5e-13
    t_span = 1.5e-9
    t_points = np.int(np.round(t_span / t_step))
    jc_in_max = 0
    f_j = 0
    f_H = 0

    ts, ms, jc_long_array, jc_trans_array, V_array = time_evolution(m_init, t_span, t_points, t_N, 100, jc_in_max, f_j, f_H )

    ## Analysis and fitting
    # positions, mzs = turningpoints(ms[2,:], ts)
    # period_approx = (positions[-1] - positions[0]) / (len(positions) - 1)
    # freq_approx = 1 / period_approx
    # print('Freq: %.4E' % freq_approx)
    # popt, pcov = curve_fit(Exponential_dec, positions, mzs, p0=[0.5, 1e-9] )

    # data = np.column_stack((ts, ms[2]))
    # description = str("dynamic-magnetoresistance")
    # np.savetxt( str(description + '.csv'), data, delimiter=',', header="Time, mz")

    fig, ((ax1), (ax2)) = plt.subplots(nrows=2, ncols=1, sharex=False)
    ax1.plot(ts, ms[0], label=r'$m_x$')
    ax1.plot(ts, ms[1], label=r'$m_y$')
    ax1.plot(ts, ms[2], label=r'$m_z$')
    # ax1.plot(ts, Exponential_dec(ts, *popt), label=r'FIT: A=%.2f, $\tau$=%.4E' % tuple(popt)) # exponential decay fit
    # ax1.plot(positions, mzs, 'kx' ) # stationary points (maxima)
    ax1.set_xlabel('Time (ns)')
    ax1.set_ylabel('Normalised Magnetisation')
    ax1.legend()

    ax2.plot(ts, V_array[0], label="x")
    ax2.plot(ts, V_array[1], label="y")
    ax2.set_ylabel("Voltage")
    ax2.set_xlabel('Time (ns)')
    ax2.legend()

    plt.show()