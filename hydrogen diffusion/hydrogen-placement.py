import numpy as np
import matplotlib
matplotlib.use('tkagg')
from matplotlib import pyplot as plt
import copy
# from copy import deepcopy
import random


def PBC(R, L):
    """
    Apply periodic boundary conditions to particle positions
    in a cubic box of side length L.

    Parameters
    ----------
    R : np.array, shape (N, N, 3)
        particle positions
    L : float
        box side length

    Returns
    -------
    np.array, shape (N, N, 3) : particle positions after PBC
    """

    return R - L * np.round(R / L)


def KE(vel, mass):
    """
    Calculate total kinetic energy.

    Parameters
    ----------
    vel : np.array, shape (N, 3)
        particle velocities
    mass : float
        particle mass

    Returns
    -------
    float
        total kinetic energy
    """
    return np.sum(np.linalg.norm(vel) ** 2 / 2 * mass)


def PE(distance_table):
    """
    Calculate total potential energy.

    Parameters
    ----------
    distance_table : np.array, shape (N, N)
        distance table

    Returns
    -------
    float
        total potential energy
    """

    n = distance_table.shape[0]
    # Create an upper triangular mask
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    r = distance_table[mask]  # Extract the upper triangular elements of rij
    sixth = 1 / r**6
    pe = np.sum(a=4 * sixth * (sixth - 1))

    return pe


def PEi(i, distance_table, rc):
    """
    Compute shifted Leonord Jones potential energy for particle i

    Args:
        i (int): particle index
        distance_table (np.array): single particle distance table, shape (N, )
        L (float): cubic box side length
        rc (float): potential cutoff radius
    Return:
        pe (float): potential energy for particle i
    """

    sixth_rc = rc**-6
    vshift = 4 * sixth_rc * (sixth_rc - 1)
    rc_mask = distance_table < rc
    i_mask = np.arange(distance_table.shape[0]) != i
    mask = rc_mask & i_mask
    sixth = distance_table[mask] ** -6
    pe = np.sum(4 * sixth * (sixth - 1) - vshift)

    return pe


def distance(R, L):
    """
    Compute distance table and rij vectors

    Parameters
    ----------
    R : np.array, shape (N, 3)
        particle positions
    L : float
        box side length

    Returns
    -------
    np.array, shape (N, N) : distance table
    np.array, shape (N, N, 3) : rij vectors
    """

    rij = R[:, np.newaxis, :] - R
    rij = PBC(rij, L)
    distance_table = np.linalg.norm(rij, axis=2)

    return distance_table, rij

def distance_alloy(R, L):
    """
    Compute distance table and rij vectors

    Parameters
    ----------
    R : np.array, shape (N, 3)
        particle positions
    L : float
        box side length

    Returns
    -------
    np.array, shape (N, N) : distance table
    np.array, shape (N, N, 3) : rij vectors
    """

    rij = R[:, np.newaxis, 0:3] - R[:, 0:3]
    rij = PBC(rij, L)
    distance_table = np.linalg.norm(rij, axis=2)

    return distance_table, rij

def distancei(i, R, L):
    """
    Compute distance table and rij vectors for a single particle

    Args:
        i (int): particle index
        R (np.array) : particle positions, shape (N, 3)
        L (float): side length of cubic box
    Returns:
        distance_table (np.array): distance table, shape (N, )
        rij (np.array): rij vectors, shape (N, 3)
    """

    rij = R[i] - R
    rij = PBC(rij, L)
    distance_table = np.linalg.norm(rij, axis=1)

    return distance_table, rij

def distancei_alloy(i, R, L):
    """
    Compute distance table and rij vectors for a single particle

    Args:
        i (int): particle index
        R (np.array) : particle positions, shape (N, 3)
        L (float): side length of cubic box
    Returns:
        distance_table (np.array): distance table, shape (N, )
        rij (np.array): rij vectors, shape (N, 3)
    """

    rij = R[i, 0:3] - R[:, 0:3]
    rij = PBC(rij, L)
    distance_table = np.linalg.norm(rij, axis=1)
    # species = R[:, 3]

    return distance_table, rij

def F(rij, distance_table):
    """
    Compute force on all particles

    Args:
        rij (np.array) : distance vectors, shape (nparticles, nparticles, 3)
        distance_table (np.array): distance table, shape (nparticles, nparticles)
    Returns:
        np.array: force on all particles, shape (nparticles, 3)
    """

    n = rij.shape[0]
    mask = ~np.eye(n, dtype=bool)
    rij = rij[mask].reshape(n, -1, 3)  # Remove self-interaction
    distance_table = distance_table[mask].reshape(n, -1)  # Remove self-interaction
    sixth = (1 / distance_table) ** 6
    factor = (1 / distance_table**2) * sixth * (2 * sixth - 1)
    factor = factor[:, :, np.newaxis]
    force = np.sum(factor * rij, axis=1)

    return 24 * force


def Fi(i, distance_table, rij, rc):
    """
    Calculate the force experienced by particle i

    Args:
        i (int): particle index
        distance_table (np.array): single particle distance table, shape (N, )
        rij (np.array): single particle rij vectors, shape (N, 3)
        rc (float): potential cutoff radius
    Return:
        force (np.array): force on particle i, shape (d,)
    """

    N = distance_table.shape[0]
    i_mask = np.arange(N) != i
    rc_mask = distance_table < rc
    mask = rc_mask & i_mask

    sixth = distance_table[mask] ** -6

    force = 24 * np.dot(distance_table[mask] ** -2 * sixth * (2 * sixth - 1), rij[mask])

    return force


def totalMomentum(vel, mass):
    """
    Calculate total kinetic momentum.

    Args:
        vel (np.array): particle velocities, shape (nparticle, 3)
        mass (float): particle mass
    Returns:
        float: total momentum
    """

    return np.sum(vel * mass, axis=0)


def pairCorrelation(distances, L, dr):
    """
    Calculate the pair correlation function g(r)

    Args:
        distances (np.array): array of distances, shape (nsteps, N, N)
        L (float): box side length
        dr (float): size of bins
    Returns:
        r (np.array): array of bin centers, shape (nbins,)
        g (np.array): pair correlation function, shape (nbins,)
    """
    N = distances.shape[1]
    nsteps = distances.shape[0]
    # Remove self-distance
    mask = ~np.eye(N, dtype=bool)
    mask = np.broadcast_to(mask, (nsteps, N, N))
    distances = distances.copy()[mask].flatten()
    # Calculate histogram
    nbins = int(L / 2 / dr)
    histogram = np.histogram(distances, bins=nbins, range=(0, nbins * dr))
    counts = histogram[0]
    r = (histogram[1] + dr / 2)[:-1]  # centers of the bins
    # Shell volume
    outer_shell_volume = 4 * np.pi / 3 * (r + dr / 2) ** 3
    inner_shell_volume = 4 * np.pi / 3 * (r - dr / 2) ** 3
    shell_volume = outer_shell_volume - inner_shell_volume
    # normalize and average
    V = L**3
    rho = N / V
    normalization = rho * shell_volume  # expected count
    g = counts / normalization
    g /= N  # Average over particles
    g /= nsteps  # Average over time

    return r, g


def legalKvecs(maxn, lbox):
    """
    Calculate k vectors commensurate with a cubic box.

    Consider only k vectors in the all-positive octant of reciprocal space.

    Args:
        maxn : maximum value for nx, ny, nz; maxn+1 is number of k-points along each axis
        lbox : side length of cubic cell

    Returns:
        array of shape (nk, ndim): collection of k vectors
    """
    n = np.arange(maxn + 1)
    n = np.vstack(np.meshgrid(n, n, n)).reshape(3, -1).T
    kvecs = 2 * np.pi / lbox * n
    tmp = np.array(kvecs[:, 0])
    kvecs[:, 0] = kvecs[:, 1]
    kvecs[:, 1] = tmp
    return kvecs


def calcRhok(kvecs, Rs):
    """
    Calculate the fourier transform of particle density.

    Args:
        kvecs (np.array): array of k-vectors, shape (nk, d)
        pos (np.array): particle positions, shape (nsteps, N, d)
    Returns:
        array of shape (nk,): fourier transformed density rho_k
    """
    return np.sum(np.exp(-1j * (Rs @ kvecs.T)), axis=1)


def Sk(maxn, L, Rs):
    """
    Calculate the structure factor S(k).

    Args:
        maxn (int): maximum value for nx, ny, nz; maxn+1 is number of k-points along each axis
        L (float): side length of cubic cell
        Rs (np.array): particle positions, shape (nsteps, N, d)
    Returns:
        unique_kmags (np.array): array of unique k magnitudes, shape (nk,)
        sk (np.array): structure factor s(k), shape (nk, )

    """
    Rs = Rs.copy()

    if Rs.ndim == 2:
        Rs = Rs[np.newaxis, :, :]

    nsteps = Rs.shape[0]
    N = Rs.shape[1]
    kvecs = legalKvecs(maxn, L)
    kmags = np.linalg.norm(kvecs, axis=1)
    unique_kmags = np.unique(kmags)
    Nunique = unique_kmags.shape[0]
    unique_sks = np.zeros((nsteps, Nunique), dtype=np.complex64)
    rhok = calcRhok(kvecs, Rs)
    sk = rhok * rhok.conj() / N

    for i, k in enumerate(unique_kmags):
        mask = kmags == k
        unique_sks[:, i] = np.mean(sk[:, mask], axis=1)

    return unique_kmags, np.mean(unique_sks, axis=0)


def Vacf0(all_vel, t):
    """
    Calculate the vacf using t_0=0.

    Args:
        all_vel (np.array): shape (nstep, natom, ndim), all particle velocities
        from an MD trajectory. For example all_vel[t, i] is the velocity
        vector of particle i at time t.
        t (int): time index at which the velocity auto-correlation is calculated.
    Returns:
        float: the velocity autocorrelation between time 0 and time t
    """
    vacf = 0
    for i in range(all_vel.shape[1]):
        vacf += np.dot(all_vel[0, i], all_vel[t, i])
    return vacf / all_vel.shape[1]


def diffusionConstant(vacf):
    """
    Calculate the diffusion constant from the
    velocity-velocity auto-correlation function (vacf).

    Args:
        vacf (np.array): shape (nt,), vacf sampled at
        nt time steps from t=0 to nt*dt. dt is set to 0.032.
    Returns:
        float: the diffusion constant calculated from the vacf.
    """
    dt = 0.032
    return np.trapz(vacf, dx=dt) / 3


def actime(E):
    """
    Calculate the autocorrelation time for a time series E

    Args:
        E (np.array): array of energies, shape (N,)
    Returns:
        (float, int): (kappa, tcutoff), the autocorrelation time and the cutoff time
    """

    mu = np.mean(E)
    std = np.std(E, ddof=1)
    N = E.shape[0]
    ts = np.arange(0, N, 1)

    a_diffs = np.zeros((N, N))

    for t in ts:
        a_diffs[t][: N - t] = (E[: N - t] - mu) * (E[t:] - mu)

    autocorrelation = np.sum(a_diffs, axis=1) / std**2 / (N - ts)
    tcutoff = (
        np.where(autocorrelation <= 0)[0][0] if (autocorrelation <= 0).any() else 0
    )
    kappa = 1 + 2 * np.sum(autocorrelation[1:tcutoff])

    return autocorrelation, kappa, tcutoff


def stderr(a):
    kappa, tcutoff = actime(a)
    N = a.shape[0]
    std = np.std(a, ddof=1)

    return std / np.sqrt(N / kappa)


def initPositionCubic(N, L):
    """
    Places Ncube^3 atoms in a cubic box

    Args:
        Ncube (int): cube root number of atoms
        L (float): box side length
    Returns:
        np.array: positions, shape (Ncube**3, 3)
    """
    import math

    Ncube = math.ceil(N ** (1 / 3))
    rs = L / Ncube
    roffset = L / 2 - rs / 2
    x = np.linspace(-roffset, roffset, Ncube)
    # Create a grid of positions using numpy broadcasting
    x, y, z = np.meshgrid(x, x, x, indexing="ij")
    # Stack the positions into a single array
    position = np.column_stack((x.ravel(), y.ravel(), z.ravel()))

    return position

def initPositionFCC(Ncell, a):
    """
    Places atoms in a fcc box

    Args:
        Ncell (int): supercell dimension in cubic-wise
        a (float): lattice constant
    Returns:
        (float, float, np.array): (N, L, position)
    """
    import math
    L = Ncell*a

    rs = L / Ncell
    roffset = L / 2
    x = np.linspace(-L/2, -L/2+a*(Ncell-1), Ncell)
    x, y, z = np.meshgrid(x, x, x, indexing="ij")
    # Stack the edge positions
    position = np.column_stack((x.ravel(), y.ravel(), z.ravel()))

    # Stack the 3 face-positions
    xx = np.column_stack((position[:,0] + a/2, position[:,1] + a/2, position[:,2]))
    xx = np.delete(xx, np.where((xx > roffset))[0], axis=0)
    yy = np.column_stack((position[:,0] + a/2, position[:,1], position[:,2] + a/2))
    yy = np.delete(yy, np.where((yy > roffset))[0], axis=0)
    zz = np.column_stack((position[:,0], position[:,1] + a/2, position[:,2] + a/2))
    zz = np.delete(zz, np.where((zz > roffset))[0], axis=0)

    position_new = np.row_stack((position, xx, yy, zz))
    N = position_new.shape[0]

    return N, L, position_new

def initVelocity(N, T0, mass=1.0, seed=1):
    """
    Initialize velocities from a Maxwell-Boltzmann distribution

    Args:
        N (int): number of particles
        T0 (float): temperature
        mass (float): particle mass
        seed (int): random seed
    Returns:
        np.array: velocities, shape (N, 3)
    """

    dim = 3
    np.random.seed(seed)
    # generate N x dim array of random numbers, and shift to be [-0.5, 0.5)
    velocity = np.random.random((N, dim)) - 0.5
    # get the average along the first axis
    meanV = np.sum(velocity, axis=0) / N
    # subtract off meanV, so there is no net momentum
    velocity -= meanV
    # calculate the total of V^2
    ke = np.sum(velocity * velocity)
    # calculate a scaling factor
    vscale = np.sqrt(dim * N * T0 / (mass * ke))
    # rescale
    velocity *= vscale

    return velocity


def LCG(m, a, c, x0, N):
    """
    Input:
    m, a, c are parameters of the LCG.
    x0: the initial pseduo-random number.
    N : number of random numbers to return

    Output:
    a list or numpy array of length N, containing the next
    N pseduo-random numbers in order (excluding x0).
    """

    output = np.zeros(N + 1)
    output[0] = x0

    for i in range(1, N + 1):
        output[i] = (a * output[i - 1] + c) % m

    return output[1:]


def gaussianRNG(m, a, c, x0, N):
    """
    Input:
    m, a, c, x0 are parameters of the LCG.
    N : number of Gaussian random numbers to return

    Output:
    a list or numpy array of length N, containing the next N
    Gaussian pseduo-random numbers in order (excluding x0).
    """

    uniform = LCG(m, a, c, x0, 2 * N) / m

    output = np.zeros(N)

    for i in range(N):
        output[i] = np.sqrt(-2 * np.log(uniform[2 * i])) * np.sin(
            2 * np.pi * uniform[2 * i + 1]
        )

    return output


def checkRandomNumbers(rand_array, NB):
    """
    Input:
    rand_array: array of size N-by-d, so (rand_array[0][0], rand_array[0][1],
                rand_array[0][2]) is the first 3D point
    NB: number of bins per dimension (for 3D we need NB*NB*NB bins in total)

    Output:
    the chi-squared value of the rand_array,
    using NB**d evenly spaced bins each in [0,1).
    """

    N = rand_array.shape[0]
    d = rand_array.shape[1]
    n = N / NB**d

    counts, edges = np.histogramdd(rand_array, bins=NB, range=[[0, 1]] * d)

    chi2 = np.sum((counts - n) ** 2 / n)

    return chi2


def computeIntegralMidpoint(func, L, N):
    """
    Input:
    func: function with a single input f(x)
    L: sets the bounds of the integral by [-L, L]
    N: number of rectangles

    Output:
    the integral using rectangle rule
    """

    dx = 2 * L / N
    x = -L + dx / 2
    s = 0

    while x <= L:
        s += func(x) * dx

        x += dx

    return s


def computeIntegralMC(func, alpha, N):
    """
    Input:
    func: a well defined function that decays fast when |x| goes to infinity
    alpha: variance of the normal distribution to be sampled
    N: length of Gaussian random numbers

    Output:
    a two-element list or numpy array, with the first element being the estimate of
    the integral, and the second being the the estimate of the variance
    """

    m = 2**32
    a = 69069
    c = 1
    x0 = 0

    # Multiply Gaussian random numbers by np.sqrt(alpha)
    # to make their variance equal to alpha
    gaussian_arrays = np.sqrt(alpha) * np.array(gaussianRNG(m, a, c, x0, N))

    # PDF of the normal distribution
    def p(x, alpha):
        return np.exp(-x * x / (2.0 * alpha)) / np.sqrt(2.0 * np.pi * alpha)

    fx = func(gaussian_arrays)
    px = p(gaussian_arrays, alpha)

    gx = fx / px
    mean = np.mean(gx)
    variance = np.var(gx)
    return mean, variance

def verletNextR(r_t, v_t, a_t, h):
    """
    Return new positions after one Verlet step

    Args:
        r_t (np.array): positions at time t, shape (nparticles, ndim)
        v_t (np.array): velocities at time t, shape (nparticles, ndim)
        a_t (np.array): accelerations at time t, shape (nparticles, ndim)
        h (float): timestep
    Returns:
        np.array: positions at time t+h, shape (nparticles, ndim)
    """

    # Note that these are vector quantities.
    # Numpy loops over the coordinates for us.
    r_t_plus_h = r_t + v_t * h + 0.5 * a_t * h * h

    return r_t_plus_h


def verletNextV(v_t, a_t, a_t_plus_h, h):
    """
    Return new velocities after one Verlet step

    Args:
        v_t (np.array): velocities at time t, shape (nparticles, ndim)
        a_t (np.array): accelerations at time t, shape (nparticles, ndim)
        a_t_plus_h (np.array): accelerations at time t+h, shape (nparticles, ndim)
        h (float): timestep

    Returns:
        np.array: velocities at time t+h, shape (nparticles, ndim)
    """

    # Note that these are vector quantities.
    # Numpy loops over the coordinates for us.
    v_t_plus_h = v_t + 0.5 * (a_t + a_t_plus_h) * h

    return v_t_plus_h

def LJi(i, distance_table, rc, self, A, B):
    """
    Compute shifted Leonord Jones potential energy for particle i

    Args:
        i (int): particle index
        distance_table (np.array): single particle distance table, shape (N, )
        L (float): cubic box side length
        rc (float): potential cutoff radius
    Return:
        pe (float): potential energy for particle i
    """

    sixth_rc = rc**-6
    vshift = A * sixth_rc**2 - B * sixth_rc
    rc_mask = distance_table < rc
    if self > 0:
        i_mask = np.arange(distance_table.shape[0]) != i
        mask = rc_mask & i_mask
    else:
        mask = rc_mask
    sixth = distance_table[mask] ** -6
    pe = np.sum(A * sixth**2 - B * sixth - vshift)

    return pe

def LJ(distance_table, rc, A, B):
    sixth_rc = rc**-6
    vshift = A * sixth_rc**2 - B * sixth_rc

    n = distance_table.shape[0]
    i_mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    rc_mask = distance_table < rc
    mask = i_mask & rc_mask
    r = distance_table[mask]
    sixth = r**-6
    v = A * sixth**2 - B * sixth

    return np.sum(v - vshift)

def LJForcei(rij, distance_table, rc, A, B):
    n = rij.shape[0]
    rc_mask = distance_table >= rc
    distance_table[rc_mask] = np.inf
    force = 12 * A * distance_table**-13 - 6 * B * distance_table**-7
    if force.ndim >1:
        force = force[:, :, np.newaxis]
        forcei = np.sum(force * rij, axis=0)
        forcej = np.sum(force * rij, axis=1)
    else:
        force = force[:, np.newaxis]
        forcei = np.sum(force * rij, axis=0)
        forcej = force * rij
    # force = np.row_stack((forcej,forcei))

    return forcej, forcei

def LJForce(rij, distance_table, rc, A, B):
    n = rij.shape[0]
    i_mask = ~np.eye(n, dtype=bool)
    rij = rij[i_mask].reshape(n, -1, 3)  # Remove self-interaction
    distance_table = distance_table[i_mask].reshape(n, -1)  # Remove self-interaction
    rc_mask = distance_table >= rc
    distance_table[rc_mask] = np.inf
    force = 12 * A * distance_table**-13 - 6 * B * distance_table**-7
    force = force[:, :, np.newaxis]
    force = np.sum(force * rij, axis=1)

    return force

def simulateMD(N, T0, L, M, steps, h, rc, thermostat=False, p=0.01, T_bath=None):
    """
    MD simulation of N particles in a cubic box of side length L

    Parameters
    ----------
    N : int
        number of particles
    T0 : float
        initial temperature
    L : float
        box side length
    M : float
        particle mass
    steps : int
        number of steps to run
    h : float
        timestep
    rc : float
        cutoff radius
    thermostat : bool, optional
        whether to use Andersen thermostat, by default False
    p : float, optional
        probability of collission, by default 0.01
    T_bath : float, optional
        temperature of bath, by default None

    Returns
    -------
    np.array
        total energy, shape (steps,)
    np.array
        total momentum, shape (steps, 3)
    np.array
        temperature, shape (steps,)
    np.array
        velocities, shape (steps, N, 3)
    np.array
        positions, shape (steps, N, 3)
    np.array
        distances, shape (steps, N, N)
    """

    np.random.seed(1)

    # Initialize inputs
    N = N
    R = initPositionCubic(N, L)
    V = initVelocity(N, T0, M)
    A = np.zeros((N, 3))

    # Initialize outputs
    E = np.zeros(steps)
    P = np.zeros((steps, 3))
    T = np.zeros(steps)
    Vs = np.zeros((steps, N, 3))
    Rs = np.zeros((steps, N, 3))
    distances = np.zeros((steps, N, N))

    # Calculate initial Forces and accelerations (N x 3 array)
    distance_table, rij = distance(R, L)
    Force = LJForce(rij, distance_table, rc)
    A = Force / M

    for t in range(0, steps):
        # Calculate energy, temperature, and momentum
        E[t] = KE(V, M)
        T[t] = 2 * E[t] / (3 * N)
        E[t] += LJ(distance_table, rc)
        P[t] = totalMomentum(V, M)

        # Save velocities, positions, and distances
        Vs[t] = V
        Rs[t] = R
        distances[t] = distance_table

        # Calculate next positions and apply PBC
        nR = verletNextR(R, V, A, h)
        nR = PBC(nR, L)
        distance_table, rij = distance(nR, L)

        # Calculate next forces, accelerations, and velocities
        nF = LJForce(rij, distance_table, rc)
        nA = nF / M
        nV = verletNextV(V, A, nA, h)

        if thermostat:
            collission_mask = np.random.rand(N) <= p
            if T_bath is None:
                nV[collission_mask] = np.random.normal(
                    0, np.sqrt(T0 / M), size=(np.sum(collission_mask), 3)
                )
            else:
                nV[collission_mask] = np.random.normal(
                    0, np.sqrt(T_bath / M), size=(np.sum(collission_mask), 3)
                )

        # update positions, velocities, accelerations and forces
        R, V, A, Force = nR, nV, nA, nF

    return E, P, T, Vs, Rs, distances

def generateEta(N, d, sigma):
    """
    Generate Gaussian random numbers for Metropolis Monte Carlo moves

    Args:
        N (int): number of particles
        d (int): dimension of space
        sigma (float): width of Gaussian random numbers
    Returns:
        np.array: array of shape (N, d) of Gaussian random numbers
    """

    return np.random.normal(loc=0, scale=sigma, size=(N, d))

def generateAcceptCheck(N):
    """
    Generate uniform random numbers for Monte Carlo acceptance checks

    Args:
        N (int): number of particles
    Returns:
        np.array: array of shape (N,) of uniform random numbers
    """
    return np.random.uniform(low=0, high=1, size=N)

def mcSwapMetropolis_FCC(R, L, rc, beta, accept_check, Ams, Bms, Asol, Bsol):
    """Perform one Monte Carlo sweep

    Args:
        R (np.array): particle positions, shape (N, d)
        L (float): cubic box side length
        rc (float): potential cutoff radius
        beta (float): inverse temperature
        eta (np.array): shape (N, d), array of Gaussian random numbers, one for each single-particle move
        accept_check (np.array): shape (natom), array of uniform random numbers, one for each single particle move
    Return:
        (int, float): (naccept, dR), the number of accepted single-particle moves, and the change in energy
    """
    naccept = 0
    dE = 0
    Rmat = np.delete(R, np.where((R[:, 3] < 0))[0], axis=0)
    Rsol = np.delete(R, np.where((R[:, 3] > 0))[0], axis=0)
    Nmat, dmat = Rmat.shape
    Nsol, dsol = Rsol.shape
    R = np.row_stack((Rmat,Rsol))
    # R = R.deepcopy()

    for i in range(Nsol):
        distance_table, rij = distancei_alloy(Nmat + i, R, L)

        PE_oldij = LJi(i, distance_table[0:Nmat], rc, -5, Ams, Bms)
        PE_oldjj = LJi(i, distance_table[Nmat:Nmat+Nsol], rc, 5, Asol, Bsol)
        PE_old = PE_oldij + PE_oldjj

        nR = copy.deepcopy(R)
        chg_ind = random.sample(range(0, Nmat), 1)
        a = copy.deepcopy(nR[Nmat + i, 0:3])
        b = copy.deepcopy(nR[chg_ind, 0:3])
        nR[Nmat + i, 0:3] = b
        nR[chg_ind, 0:3] = a
        distance_new, rij_new = distancei_alloy(Nmat + i, nR, L)
        PE_newij = LJi(i, distance_new[0:Nmat], rc, -5, Ams, Bms)
        PE_newjj = LJi(i, distance_new[Nmat:Nmat+Nsol], rc, 5, Asol, Bsol)
        PE_new = PE_newij + PE_newjj

        dEi = PE_new - PE_old
        # kb = 0.0019872041  # kcal/mol/K
        # p = -beta * dEi/kb
        p = -beta * dEi
        if dEi <= 0 or accept_check[i] <= np.exp(p):
        # if p > np.log(accept_check[i]):
            R = nR
            naccept += 1
            dE += dEi

    return naccept, dE, R

def simulateMcMetropolis_FCC(Ncell, a, T, rc, sigma, steps, Amat, Bmat, Ams, Bms, Asol, Bsol):
    beta = 1 / T
    composition = 0.2
    N, L, R = initPositionFCC(Ncell, a)
    Nsol = int(np.round(N * composition))
    list_sol = random.sample(range(0,N),Nsol)
    list_total = np.ones((N,), dtype=int)
    list_total[list_sol]=-1
    R = np.column_stack((R,list_total))
    Rs = np.zeros((steps, N, 4))
    Es = np.zeros(steps)
    distances = np.zeros((steps, N, N))
    acceptance_ratios = np.zeros(steps)

    Rmat = np.delete(R, np.where((R[:, 3] < 0))[0], axis=0)
    Rsol = np.delete(R, np.where((R[:, 3] > 0))[0], axis=0)
    Nmat, dmat = Rmat.shape
    R = np.row_stack((Rmat,Rsol))

    Rs[0] = R
    distance_table, rij = distance_alloy(R, L)
    E_ii = LJ(distance_table[0:Nmat, 0:Nmat], rc, Amat, Bmat)
    E_jj = LJ(distance_table[Nmat:Nmat + Nsol, Nmat:Nmat + Nsol], rc, Asol, Bsol)
    E_ij = LJi(1, np.ravel(distance_table[0:Nmat, Nmat:Nmat + Nsol]), rc, -5, Ams, Bms)
    E0 = E_ii + E_ij + E_jj
    Es[0] = E0

    for t in range(1,steps):
        # eta = generateEta(N, 3, sigma)
        accept_check = generateAcceptCheck(N)
        # naccept, dE, R = mcSweepMetropolis(R, L, rc, beta, eta, accept_check)
        naccept, dE, R = mcSwapMetropolis_FCC(R, L, rc, beta, accept_check, Ams, Bms, Asol, Bsol)

        Rs[t] = R

        distance_table, rij = distance_alloy(R, L)
        Es[t] = Es[t-1]+dE
        distances[t] = distance_table
        acceptance_ratios[t] = naccept / N

    return Rs, Es, distances, acceptance_ratios


def simulateMD_FCC(Ncell, T0, a, M, steps, h, rc, thermostat=False, p=0.01, T_bath=None):
    """
    MD simulation of N particles in a cubic box of side length L

    Parameters
    ----------
    Ncell : int
        number of unit cell in 1-dimension
    T0 : float
        initial temperature
    a : float
        lattice constant
    M : float
        particle mass
    steps : int
        number of steps to run
    h : float
        timestep
    rc : float
        cutoff radius
    thermostat : bool, optional
        whether to use Andersen thermostat, by default False
    p : float, optional
        probability of collission, by default 0.01
    T_bath : float, optional
        temperature of bath, by default None

    Returns
    -------
    np.array
        total energy, shape (steps,)
    np.array
        total momentum, shape (steps, 3)
    np.array
        temperature, shape (steps,)
    np.array
        velocities, shape (steps, N, 3)
    np.array
        positions, shape (steps, N, 3)
    np.array
        distances, shape (steps, N, N)
    """

    np.random.seed(1)

    # Initialize inputs
    N, L, R = initPositionFCC(Ncell, a)
    V = initVelocity(N, T0, M)
    A = np.zeros((N, 3))

    # Initialize outputs
    E = np.zeros(steps)
    P = np.zeros((steps, 3))
    T = np.zeros(steps)
    Vs = np.zeros((steps, N, 3))
    Rs = np.zeros((steps, N, 3))
    distances = np.zeros((steps, N, N))

    # Calculate initial Forces and accelerations (N x 3 array)
    distance_table, rij = distance(R, L)
    Force = LJForce(rij, distance_table, rc)
    A = Force / M

    for t in range(0, steps):
        # Calculate energy, temperature, and momentum
        E[t] = KE(V, M)
        T[t] = 2 * E[t] / (3 * N)
        E[t] += LJ(distance_table, rc)
        P[t] = totalMomentum(V, M)

        # Save velocities, positions, and distances
        Vs[t] = V
        Rs[t] = R
        distances[t] = distance_table

        # Calculate next positions and apply PBC
        nR = verletNextR(R, V, A, h)
        nR = PBC(nR, L)
        distance_table, rij = distance(nR, L)

        # Calculate next forces, accelerations, and velocities
        nF = LJForce(rij, distance_table, rc)
        nA = nF / M
        nV = verletNextV(V, A, nA, h)

        if thermostat:
            collission_mask = np.random.rand(N) <= p
            if T_bath is None:
                nV[collission_mask] = np.random.normal(
                    0, np.sqrt(T0 / M), size=(np.sum(collission_mask), 3)
                )
            else:
                nV[collission_mask] = np.random.normal(
                    0, np.sqrt(T_bath / M), size=(np.sum(collission_mask), 3)
                )

        # update positions, velocities, accelerations and forces
        R, V, A, Force = nR, nV, nA, nF

    return E, P, T, Vs, Rs, distances

def simulateMD_alloy(R, L, T0, Mmat, Msol, Amat, Bmat, Ams, Bms, Asol, Bsol, steps, h, rc, thermostat=False, p=0.01, T_bath=None):
    """
    MD simulation of N particles in a cubic box of side length L

    Parameters
    ----------
    Ncell : int
        number of unit cell in 1-dimension
    T0 : float
        initial temperature
    a : float
        lattice constant
    M : float
        particle mass
    steps : int
        number of steps to run
    h : float
        timestep
    rc : float
        cutoff radius
    thermostat : bool, optional
        whether to use Andersen thermostat, by default False
    p : float, optional
        probability of collission, by default 0.01
    T_bath : float, optional
        temperature of bath, by default None

    Returns
    -------
    np.array
        total energy, shape (steps,)
    np.array
        total momentum, shape (steps, 3)
    np.array
        temperature, shape (steps,)
    np.array
        velocities, shape (steps, N, 3)
    np.array
        positions, shape (steps, N, 3)
    np.array
        distances, shape (steps, N, N)
    """

    np.random.seed(1)

    # Initialize inputs
    Rmat = np.delete(R, np.where((R[:, 3] < 0))[0], axis=0)
    Rsol = np.delete(R, np.where((R[:, 3] > 0))[0], axis=0)
    Nmat, dmat = Rmat.shape
    Nsol, dsol = Rsol.shape

    R = np.row_stack((Rmat, Rsol))
    N, d = R.shape

    Vmat = initVelocity(Nmat, T0, Mmat)
    Vsol = initVelocity(Nsol, T0, Msol)
    V = np.row_stack((Vmat, Vsol))
    # A = np.zeros((N, 3))

    # Initialize outputs
    E = np.zeros(steps)
    P = np.zeros((steps, 3))
    T = np.zeros(steps)
    Vs = np.zeros((steps, N, 3))
    Rs = np.zeros((steps, N, 4))
    # distances = np.zeros((steps, N, N))

    # Calculate initial Forces and accelerations (N x 3 array)
    distance_table, rij = distance_alloy(R, L)
    dt_ii = distance_table[0:Nmat, 0:Nmat]
    dt_jj = distance_table[Nmat:Nmat + Nsol, Nmat:Nmat + Nsol]
    dt_ij = distance_table[0:Nmat, Nmat:Nmat + Nsol]
    rij_ii = rij[0:Nmat, 0:Nmat, :]
    rij_jj = rij[Nmat:Nmat + Nsol, Nmat:Nmat + Nsol, :]
    rij_ij = rij[0:Nmat, Nmat:Nmat + Nsol, :]

    Force_ii = LJForce(rij_ii, dt_ii, rc, Amat, Bmat)
    Force_jj = LJForce(rij_jj, dt_jj, rc, Asol, Bsol)
    Force_ij, Force_ji = LJForcei(rij_ij, dt_ij, rc, Ams, Bms)
    A = np.row_stack(((Force_ii + Force_ij) / Mmat, (Force_jj + Force_ji) / Msol))

    for t in range(0, steps):
        # Calculate energy, temperature, and momentum
        E[t] = KE(Vmat, Mmat) + KE(Vsol, Msol)
        T[t] = 2 * E[t] / (3 * N)
        E_ii = LJ(dt_ii, rc, Amat, Bmat)
        E_jj = LJ(dt_jj, rc, Asol, Bsol)
        E_ij = LJi(1, np.ravel(dt_ij), rc, -5, Ams, Bms)
        dE = E_ii + E_ij + E_jj
        E[t] += dE
        P[t] = totalMomentum(Vmat, Mmat) + totalMomentum(Vsol, Msol)

        # Save velocities, positions, and distances
        Vs[t] = V
        Rs[t] = R
        # distances[t] = distance_table

        # Calculate next positions and apply PBC
        nR = verletNextR(R[:, 0:3], V, A, h)
        nR = PBC(nR, L)
        nR = np.column_stack((nR, R[:, 3]))
        distance_table, rij = distance_alloy(nR, L)

        # Calculate next forces, accelerations, and velocities
        dt_ii = distance_table[0:Nmat, 0:Nmat]
        dt_jj = distance_table[Nmat:Nmat + Nsol, Nmat:Nmat + Nsol]
        dt_ij = distance_table[0:Nmat, Nmat:Nmat + Nsol]
        rij_ii = rij[0:Nmat, 0:Nmat, :]
        rij_jj = rij[Nmat:Nmat + Nsol, Nmat:Nmat + Nsol, :]
        rij_ij = rij[0:Nmat, Nmat:Nmat + Nsol, :]

        Force_ii = LJForce(rij_ii, dt_ii, rc, Amat, Bmat)
        Force_jj = LJForce(rij_jj, dt_jj, rc, Asol, Bsol)
        Force_ij, Force_ji = LJForcei(rij_ij, dt_ij, rc, Ams, Bms)

        nF = np.row_stack((Force_ii + Force_ij, Force_jj + Force_ji))
        nA = np.row_stack(((Force_ii + Force_ij) / Mmat, (Force_jj + Force_ji) / Msol))

        nV = verletNextV(V, A, nA, h)

        if thermostat:
            collission_mask = np.random.rand(N) <= p
            if T_bath is None:
                nV[collission_mask] = np.random.normal(
                    0, np.sqrt(T0 / (Mmat/2+Msol/2)), size=(np.sum(collission_mask), 3)
                )
            else:
                nV[collission_mask] = np.random.normal(
                    0, np.sqrt(T_bath / (Mmat/2+Msol/2)), size=(np.sum(collission_mask), 3)
                )

        # update positions, velocities, accelerations and forces
        R, V, A, Force = nR, nV, nA, nF

    return E, P, T, Vs, Rs

def simulateMD_alloy_H(R, L, T0, Mmat, Msol, Amat, Bmat, Ams, Bms, Asol, Bsol, AmH, BmH, AsH, BsH, steps, h, rc, thermostat=False, p=0.01, T_bath=None):

    np.random.seed(1)

    # Initialize inputs
    Rmat = np.delete(R, np.where((R[:, 3] < 0) | (R[:, 3] > 50))[0], axis=0)
    Rsol = np.delete(R, np.where((R[:, 3] > 0))[0], axis=0)
    RH = np.delete(R, np.where((R[:, 3] < 50))[0], axis=0)
    Nmat, dmat = Rmat.shape
    Nsol, dsol = Rsol.shape
    NH = 1

    R = np.row_stack((Rmat, Rsol, RH))
    N, d = R.shape

    Vmat = initVelocity(Nmat, T0, Mmat)
    Vsol = initVelocity(Nsol, T0, Msol)
    VH = initVelocity(NH+1, T0, 1)
    VH = VH[0,:]
    V = np.row_stack((Vmat, Vsol, VH))
    # A = np.zeros((N, 3))

    # Initialize outputs
    E = np.zeros(steps)
    P = np.zeros((steps, 3))
    T = np.zeros(steps)
    Vs = np.zeros((steps, N, 3))
    Rs = np.zeros((steps, N, 4))
    # distances = np.zeros((steps, N, N))

    # Calculate initial Forces and accelerations (N x 3 array)
    distance_table, rij = distance_alloy(R, L)
    dt_ii = distance_table[0:Nmat, 0:Nmat]
    dt_jj = distance_table[Nmat:Nmat + Nsol, Nmat:Nmat + Nsol]
    dt_ij = distance_table[0:Nmat, Nmat:Nmat + Nsol]
    dt_iH = distance_table[0:Nmat, Nmat + Nsol]
    dt_jH = distance_table[Nmat:Nmat + Nsol, Nmat + Nsol]
    rij_ii = rij[0:Nmat, 0:Nmat, :]
    rij_jj = rij[Nmat:Nmat + Nsol, Nmat:Nmat + Nsol, :]
    rij_ij = rij[0:Nmat, Nmat:Nmat + Nsol, :]
    rij_iH = rij[0:Nmat, Nmat + Nsol, :]
    rij_jH = rij[Nmat:Nmat + Nsol, Nmat + Nsol, :]

    Force_ii = LJForce(rij_ii, dt_ii, rc, Amat, Bmat)
    Force_jj = LJForce(rij_jj, dt_jj, rc, Asol, Bsol)
    Force_ij, Force_ji = LJForcei(rij_ij, dt_ij, rc, Ams, Bms)
    Force_iH, Force_Hi = LJForcei(rij_iH, dt_iH, rc, AmH, BmH)
    Force_jH, Force_Hj = LJForcei(rij_jH, dt_jH, rc, AsH, BsH)
    Force_H = Force_Hi + Force_Hj
    A = np.row_stack(((Force_ii + Force_ij + Force_iH) / Mmat, (Force_jj + Force_ji + Force_jH) / Msol, Force_H/1 ))

    for t in range(0, steps):
        # Calculate energy, temperature, and momentum
        E[t] = KE(Vmat, Mmat) + KE(Vsol, Msol)+ KE(VH, 1)
        T[t] = 2 * E[t] / (3 * N)
        E_ii = LJ(dt_ii, rc, Amat, Bmat)
        E_jj = LJ(dt_jj, rc, Asol, Bsol)
        E_ij = LJi(1, np.ravel(dt_ij), rc, -5, Ams, Bms)
        E_iH = LJi(1, np.ravel(dt_iH), rc, -5, AmH, BmH)
        E_jH = LJi(1, np.ravel(dt_jH), rc, -5, AsH, BsH)
        dE = E_ii + E_ij + E_jj + E_iH + E_jH
        E[t] += dE
        P[t] = totalMomentum(Vmat, Mmat) + totalMomentum(Vsol, Msol) + totalMomentum(VH, 1)

        # Save velocities, positions, and distances
        Vs[t] = V
        Rs[t] = R
        # distances[t] = distance_table

        # Calculate next positions and apply PBC
        nR = verletNextR(R[:, 0:3], V, A, h)
        nR = PBC(nR, L)
        nR = np.column_stack((nR, R[:, 3]))
        distance_table, rij = distance_alloy(nR, L)

        # Calculate next forces, accelerations, and velocities
        dt_ii = distance_table[0:Nmat, 0:Nmat]
        dt_jj = distance_table[Nmat:Nmat + Nsol, Nmat:Nmat + Nsol]
        dt_ij = distance_table[0:Nmat, Nmat:Nmat + Nsol]
        dt_iH = distance_table[0:Nmat, Nmat + Nsol]
        dt_jH = distance_table[Nmat:Nmat + Nsol, Nmat + Nsol]
        rij_ii = rij[0:Nmat, 0:Nmat, :]
        rij_jj = rij[Nmat:Nmat + Nsol, Nmat:Nmat + Nsol, :]
        rij_ij = rij[0:Nmat, Nmat:Nmat + Nsol, :]
        rij_iH = rij[0:Nmat, Nmat + Nsol, :]
        rij_jH = rij[Nmat:Nmat + Nsol, Nmat + Nsol, :]

        Force_ii = LJForce(rij_ii, dt_ii, rc, Amat, Bmat)
        Force_jj = LJForce(rij_jj, dt_jj, rc, Asol, Bsol)
        Force_ij, Force_ji = LJForcei(rij_ij, dt_ij, rc, Ams, Bms)
        Force_iH, Force_Hi = LJForcei(rij_iH, dt_iH, rc, AmH, BmH)
        Force_jH, Force_Hj = LJForcei(rij_jH, dt_jH, rc, AsH, BsH)
        Force_H = Force_Hi + Force_Hj

        nF = np.row_stack((Force_ii + Force_ij + Force_iH, Force_jj + Force_ji + Force_jH, Force_H))
        nA = np.row_stack(((Force_ii + Force_ij + Force_iH) / Mmat, (Force_jj + Force_ji + Force_jH) / Msol, Force_H / 1))

        nV = verletNextV(V, A, nA, h)

        if thermostat:
            collission_mask = np.random.rand(N) <= p
            if T_bath is None:

                nV[collission_mask] = np.random.normal(
                    0, np.sqrt(T0 / (Mmat/2+Msol/2)), size=(np.sum(collission_mask), 3)
                )
            else:
                nV[collission_mask] = np.random.normal(
                    0, np.sqrt(T_bath / (Mmat/2+Msol/2)), size=(np.sum(collission_mask), 3)
                )

        # update positions, velocities, accelerations and forces
        R, V, A, Force = nR, nV, nA, nF

    return E, P, T, Vs, Rs

# Cu LJ parameters
epsilon_mat = 4.72  # kcal/mol
r0_mat = 2.616  # Angstrom (also sigma)
Amat = 484800  # kcal/mol/Angstrom^12
Bmat = 3026  # kcal/mol/Angstrom^6
Mmat = 63.546  # g/mol
rc = 1.2  # nm
rc = rc * 10  # Angstrom

# Al LJ parameters
epsilon_sol = 4.02  # kcal/mol
r0_sol = 2.925  # Angstrom (also sigma)
Asol = 1577000  # kcal/mol/Angstrom^12
Bsol = 5035  # kcal/mol/Angstrom^6
Msol = 26.9815385  # g/mol
# rc = 1.2  # nm
# rc = rc * 10  # Angstrom

# H LJ parameters
AH = 7140.849  # kcal/mol/Angstrom^12
BH = 32.221  # kcal/mol/Angstrom^6
# MH = 1  # g/mol

# Lorentz-Berthelot rules
Ams = (Amat + Asol)/2
Bms = (Bmat + Bsol)/2
AmH = (Amat + AH)/2
BmH = (Bmat + BH)/2
AsH = (AH + Asol)/2
BsH = (BH + Bsol)/2

# N = int(7**3)
# targetDensity = 0.05234  # particles per Angstrom^3
# L = (N / targetDensity) ** (1 / 3)

a = 3.7643
Ncell = 3
T0 = 0.0

MCsteps = 100
steps = 1
h = 0.001
L = Ncell*a

# E, P, T, Vs, Rs, distances = simulateMD(N, T0, L, M, steps, h, rc, thermostat=True, p=0.01, T_bath=0.6)
Tmc=0.01 # 10 K
sigma = 0.8
# Rs, Es, distances, acceptance_ratios = simulateMcMetropolis_FCC(Ncell, a, Tmc, rc, sigma, MCsteps, Amat, Bmat, Ams, Bms, Asol, Bsol)
# np.save('MCpositions_CuAl20.npy', Rs)
# np.savetxt('MCenergy_CuAl20.csv', Es, delimiter=',')

Rs = np.load('MCpositions_CuAl20.npy')
SRO = 29
Nlist = Rs[SRO, :, 3]
Rs = (Rs[SRO, :, 0:3] + Ncell*3.8/2) * a/3.8 - L/2
Rs = np.column_stack((Rs, Nlist))

# E, P, T, Vs, Rs, distances = simulateMD_FCC(Ncell, T0, a, M, steps, h, rc, thermostat=False, p=0.01, T_bath=0.0)
# E, P, T, Vs, Rs = simulateMD_alloy(Rs, L, T0, Mmat, Msol, Amat, Bmat, Ams, Bms, Asol, Bsol, steps, h, rc, thermostat=True, p=0.01, T_bath=0.6)
N, ds = Rs.shape
# Hinter = np.zeros((1, 4))
Hinter = np.column_stack((PBC(Rs[:, 0:3] + a/2, L),np.ones(N,)*100))

numint = Hinter.shape[0]
Es = np.zeros(numint)
Rss = np.zeros((numint, N+1, 4))

for iii in range(numint):
    newRs = np.row_stack((Rs, Hinter[iii,:] ))
    E, P, T, Vs, Rs1 = simulateMD_alloy_H(newRs, L, T0, Mmat, Msol, Amat, Bmat, Ams, Bms, Asol, Bsol, AmH, BmH, AsH, BsH, steps, h, rc, thermostat=False, p=0.01, T_bath=0.0)
    # E, P, T, Vs, Rs1 = simulateMD_alloy_H(newRs, L, T0, Mmat, Msol, Amat, Bmat, Ams, Bms, Asol, Bsol, AmH, BmH, AsH, BsH, steps, h, rc, thermostat=False, p=0.01, T_bath=0.6)
    Rss[iii] = Rs1
    Es[iii] = E
# np.save('Hinter_positions_CuAl20_29.npy', Rss)
# np.savetxt('Hinter_energy_CuAl20_29.csv', Es, delimiter=',')

kb = 0.0019872041  # kcal/mol/K
naturalT = T / kb
print(len(naturalT))
plt.plot(naturalT)
plt.show()

# np.savetxt('E_RT_temp.csv', E, delimiter=',')
plt.plot(E)
plt.show()
print(E[0])

# animate the 3D positions

import matplotlib
from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
from IPython.core.display import HTML

fig = plt.figure(figsize=(8, 6))
ax = plt.axes(projection="3d")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("T")
ax.set_title("Positions")
ax.set_zlim(-L / 2, L / 2)
ax.set_xlim(-L / 2, L / 2)
ax.set_ylim(-L / 2, L / 2)
# surf = ax.scatter(Rs[0, :, 0], Rs[0, :, 1], Rs[0, :, 2], s=10)
# plt.show()
# plt.close()


def animate(i):
    ax.clear()
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(f"Position at t = {i:.2f}")
    ax.set_zlim(-L / 2, L / 2)
    ax.set_xlim(-L / 2, L / 2)
    ax.set_ylim(-L / 2, L / 2)
    surf = ax.scatter(Rs[i, :, 0], Rs[i, :, 1], Rs[i, :, 2], s=10)
    return surf

plt.show()
frames = np.arange(0, steps, 100)
anim = animation.FuncAnimation(fig, animate, frames=frames, interval=100)
HTML(anim.to_html5_video())