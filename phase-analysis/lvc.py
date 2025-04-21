import utils
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optim
import scipy.integrate as integrate
from scipy.optimize import Bounds

def rho_z_integrate(z, A):
    rhol = A[0]
    rhog = A[1]
    z0 = A[2]
    d = A[3]
    rho = 0.5 * (rhol + rhog) - 0.5 * (rhol - rhog) * np.tanh((2 * (z - z0)) / d)
    return rho


def rho_z(A, z):
    rhol = A[0]
    rhog = A[1]
    z0 = A[2]
    d = A[3]
    rho = 0.5 * (rhol + rhog) - 0.5 * (rhol - rhog) * np.tanh((2 * (z - z0)) / d)
    return rho


def rho_z_cost(A, z, density):
    err = np.linalg.norm(density - rho_z(A, z)) ** 2
    return err

def fit_rho_z(zbin, density, A0 = np.array([800, 20, 60, 20])):

    xopt = optim.fmin(func=rho_z_cost, x0=A0, args=(zbin, density), disp=False)
    rho_l, rho_g, z0, d = xopt

    return xopt

def Tc(A, T):

    a = A[0]
    Tc = A[1]
    beta = 0.325
    diff = a * (Tc - T)**beta
    return diff
    
def Tc_cost(A, rho_l, rho_g, T):
    err = np.linalg.norm((rho_l - rho_g) - Tc(A, T))
    return err

def fit_Tc(
    rho_ls,
    rho_gs,
    temps,
    bounds = Bounds(lb = [0, 300], ub = [200, 500]),
    x0 = [100, 400],
    maxiter = 10000,
):

    xopt = optim.minimize(
        fun=Tc_cost,
        x0 = x0,
        bounds=bounds,
        args = (rho_ls, rho_gs, temps),
        options = dict(maxiter = maxiter),
        method = "Nelder-Mead",
    )
    A, Tc_fit = xopt.x

    return A, Tc_fit

def rho_c(A, T):

    B = A[0]
    rho_c_fit = A[1]
    Tc = A[2]

    return rho_c_fit + B * (Tc - T)

def rho_c_cost(A, rho_l, rho_g, T):
    err = np.linalg.norm(((rho_l + rho_g) / 2) - rho_c(A, T)) ** 2
    return err

def fit_rho_c(
    rho_ls,
    rho_gs,
    temps,
    bounds = Bounds(lb = [0, 200, 300], ub = [200, 500, 500]),
    maxiter = 10000,
    x0 = [1, 300, 400],
):

    xopt = optim.minimize(
        fun = rho_c_cost,
        x0 = x0,
        bounds = bounds,
        args = (rho_ls, rho_gs, temps),
        options = dict(maxiter=maxiter),
        method = "Nelder-Mead",
    )

    B, rho_c_fit, Tc = xopt.x

    return B, rho_c_fit, Tc

def Tc_rhoc_cost(fit_params, rho_ls, rho_gs, temps):
    
    A = fit_params[:2] # Tc -> A, Tc
    B = fit_params[2:] # rhoc -> B, rhoc
    Tc = A[1]
    B = np.concatenate((B, [Tc]))
    err_Tc = Tc_cost(A, rho_ls, rho_gs, temps)
    err_rhoc = rho_c_cost(B, rho_ls, rho_gs, temps)

    return err_Tc + err_rhoc
    
def fit_Tc_rhoc(
    rho_ls,
    rho_gs,
    temps,
    Tc_bounds = [[0, 300], [200, 500]],
    rhoc_bounds = [[0, 100], [200, 600]],
    A0 = 100,
    Tc0 = 400,
    B0 = 1,
    rhoc0 = 300,
    maxiter = 10000,
):

    lb = []
    lb.extend(Tc_bounds[0])
    lb.extend(rhoc_bounds[0])
    ub = []
    ub.extend(Tc_bounds[1])
    ub.extend(rhoc_bounds[1])
    bounds = Bounds(lb = lb, ub=ub)

    A0, Tc0 = fit_Tc(
        rho_ls,
        rho_gs,
        temps,
        bounds = Bounds(lb = Tc_bounds[0], ub = Tc_bounds[1]),
        x0 = [A0, Tc0],
        maxiter = maxiter,
    )

    rhoc_bounds[0].append(Tc_bounds[0][-1]) # Add Tc bounds
    rhoc_bounds[1].append(Tc_bounds[1][-1])
 
    B0, rhoc0, Tc0 = fit_rho_c(
        rho_ls,
        rho_gs,
        temps,
        bounds = Bounds(lb = rhoc_bounds[0], ub = rhoc_bounds[1]),
        x0=[B0, rhoc0, Tc0],
        maxiter = maxiter,
    )

    x0 = np.array([A0, Tc0, B0, rhoc0])

    xopt = optim.minimize(
        fun = Tc_rhoc_cost,
        x0 = x0,
        bounds = bounds,
        args = (rho_ls, rho_gs, temps),
        method = 'Nelder-Mead',
        options = dict(maxiter = maxiter),
    )

    A, Tc, B, rhoc = xopt.x
    return Tc, rhoc
    
    

def fit_density(
    df,
    bin_width = 4, # Angstroms
    time_avg = True,
    mode = 'molecule',
    std = True,
    absval=True,
    center = True,
    auto_range = False,
    norm = 'mass',
    actime = True,
    start = 0,
    
):

    density, zbin, mass_err = utils.density(
        df,
        bin_width = bin_width,
        time_avg=time_avg,
        mode=mode,
        std=std,
        absval=absval,
        center=center,
        auto_range=auto_range,
        norm=norm,
        actime=actime,
        start = start,
    )
    
    zmin = zbin.min()
    zmax = zbin.max()
    A0 = np.array([800, 20, 60, 20])  # rho_l, rho_g, z0, d
    
    A = fit_rho_z(zbin, density, A0)
    rho_l, rho_g, z0, d = A
    rho = rho_z(A, zbin)

    total_system_mass = df['molecule']['mass'] * df['nmolecule'] * df['AMU_TO_KG']
    L = utils.get_L(df)
    V = (zbin[1] - zbin[0]) * L[0] * L[1] * 1e-10**3 
    frac_density = density * V / total_system_mass # TODO finish fractional density

    
    # frac_density, frac_zbin, frac_err = utils.density(
    #     df,
    #     bin_width = bin_width,
    #     time_avg=True,
    #     mode = mode,
    #     std = True,
    #     absval=True,
    #     center = True,
    #     auto_range = False,
    #     norm = 'percent',
    #     actime=True,
    #     start=start,
    # )
    
    
    A0 = np.array([0.01, 0.0005, 60, 20])  # rho_l, rho_g, z0, d

    frac_A = fit_rho_z(zbin, frac_density, A0)
    frac_rho_l, frac_rho_g, frac_z0, frac_d = frac_A
    frac_rho = rho_z(frac_A, zbin)
    
    frac_liq, err_liq = integrate.quad(
        rho_z_integrate, 0, frac_z0, args=(frac_A), full_output=0
    )
    frac_gas, err_gas = integrate.quad(
        rho_z_integrate, frac_z0, zmax, args=(frac_A), full_output=0
    )
    
    total = frac_gas + frac_liq
    frac_gas /= total
    frac_liq /= total
    
    density_g, zbin_g, err_g = utils.density(
        df,
        bin_width = bin_width,
        time_avg=True,
        mode = mode,
        std = True,
        absval=True,
        center = True,
        auto_range = False,
        norm = 'mass',
        actime=True,
        phase_mask = 1,
        start=start,
    )
    
    density_l, zbin_l, err_l = utils.density(
        df,
        bin_width = bin_width,
        time_avg=True,
        mode = mode,
        std = True,
        absval=True,
        center = True,
        auto_range = True,
        norm = 'mass',
        actime=True,
        phase_mask = 0,
        start = start,
    )
    
    # frac_liq_kmeans, err_nliquid = utils.phase_frac(df, 0, actime = True, start = start)
    # frac_gas_kmeans, err_ngas = utils.phase_frac(df, 1, actime = True, start = start)

    return rho, density, zbin, rho_l, rho_g, d, z0, mass_err, frac_liq, frac_gas#, frac_liq_kmeans, frac_gas_kmeans

def plot_density_fit(
    rho,
    density,
    zbin,
    rho_l,
    rho_g,
    d,
    z0,
    mass_err,
    frac_liquid,
    frac_gas,
    # frac_liquid_kmeans,
    # frac_gas_kmeans,
    title = "",
    figsize = (8, 6),
    extras = False,
):

    zmin = zbin.min()
    zmax = zbin.max()
    interface_start = z0 - d / 2
    interface_end = z0 + d / 2
    fig, ax = plt.subplots(figsize=figsize, dpi=200)
    # ylim = (0, 1.1 * max(rho.max(), density.max()))
    
    ax.set_xlabel("Z (Å)")
    ax.set_ylabel(r"$\rho \ (kg \cdot m^{-3})$")
    # ax.set_title(f"{mol} {temp}K Density Profile")
    ax.set_title(title)
    ax.set_xlim(zmin, zmax)
    # ax.set_ylim(ylim)
    ax.plot(zbin, density, label="data", color = 'navy')
    ax.plot(zbin, rho, label="fit", color = 'goldenrod')
    ylim = ax.get_ylim()
    ax.vlines(
        z0,
        0,
        ylim[1],
        color="black",
        linestyle="--",
        label="z0",
        zorder=-1,
    )
    
    ax.hlines(
        rho_l,
        zmin,
        zmax,
        linestyle=(0, (5, 10)),
        color="black",
        zorder=0,
    )
    
    
        
    
    ax.fill_between(
        zbin,
        density - mass_err,
        density + mass_err,
        color = 'navy',
        alpha = 0.3
    )
    
    ax.text(zmin + .6 * zmax, rho_l * 1.03, r"$\rho_l$", fontsize=12)
    ax.text(zmax * 0.694, rho_g + 95, r"$\rho_g$", fontsize=12)

    if extras:
        interface_start_arg = np.argmin(np.abs(zbin - interface_start))
        interface_end_arg = np.argmin(np.abs(zbin - interface_end))
        interface_start_y = rho[interface_start_arg]
        interface_end_y = rho[interface_end_arg]
        ax.vlines(
            interface_start,
            0,
            interface_start_y,
            color="black",
            linestyle="--",
            label="d",
            zorder=0,
        )
        ax.vlines(interface_end, 0, interface_end_y * 0.95, color="black", linestyle="--", zorder=0)
    
        gas_start = np.argmin(np.abs(zbin - z0))
        
        
        ax.fill_between(
            zbin[gas_start:],
            rho[gas_start:],
            color="tab:red",
            alpha=0.3,
            label="gas",
        )
        
        
        ax.fill_between(
            zbin[:gas_start+1],
            rho[:gas_start+1],
            color="tab:blue",
            alpha=0.3,
            label="liq",
            zorder=0,
        )
        ax.text(15, 0.45 * ylim[1], f"{frac_liquid * 100:.1f}%", fontsize=12)
        ax.text(
            15,
            0.40 * ylim[1],
            f"{frac_liquid_kmeans * 100:.1f}%",
            fontsize=12,
            color="tab:green",
        )
        ax.text(
            0.62 * zmax,
            rho_g + 35,
            f"{frac_gas * 100:.1f}%",
            fontsize=12,
        )
        ax.text(
            0.73 * zmax,
            rho_g + 35,
            f"{frac_gas_kmeans * 100:.1f}%",
            fontsize=12,
            color="tab:green",
        )
    
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    return fig, ax