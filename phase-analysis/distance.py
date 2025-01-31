import numpy as np
import numba as nb
import gc
import utils


@nb.njit
def get_upper_tri_index(N):
    indices = np.zeros((N * (N - 1) // 2, 2), dtype=np.int32)
    index = 0
    for i in range(N):
        for j in range(i + 1, N):
            indices[index] = i, j
            index += 1
    return indices


@nb.njit
def distance_mol(dist, mol_idx, ind):
    i = ind[:, 0]
    j = ind[:, 1]
    dist_mol_idx = np.where((i == mol_idx) | (j == mol_idx))[0]
    dist_mol = dist[:, dist_mol_idx]

    return dist_mol


def generate_coord_num(
    radii,
    df,
    start=0,
    end=-1,
    method="sparse",
    to_print=True,
):
    if end == -1:
        end = df["Nt"]

    Nrad = len(radii)
    radii = np.array(radii).flatten()
    Nmol = df["Nmolecule"]

    if to_print:
        print("Generating Coordination Number...", end="")

    if method == "full":
        if "distance" not in list(df["molecule"].keys()):
            distances = distance(df, method="full")
        else:
            distances = df["molecule"]["distance"][start:end]

        coord = np.zeros((Nrad, df["Nt"], df["Nmolecule"]), dtype=np.int16)
        for idx, radius in enumerate(radii):
            coord[idx][start:end] = np.sum(distances < radius, axis=1) - 1
        del distances

    elif method == "sparse":
        if "distance" in list(df["molecule"].keys()):
            distances = df["molecule"]["distance"]
        else:
            distances = distance(df, method="sparse")
        coord = generate_coord_num_sparse(Nmol, radii, distances).astype(np.int16)

    for i, radius in enumerate(radii):
        # if f"coordination_{radius}" not in list(df["molecule"].keys()):
        if start != 0 or end != df["Nt"]:
            df["molecule"][f"coordination_{radius}"] = np.zeros(
                (df["Nt"], df["Nmolecule"]), dtype=np.int16
            )
            df["molecule"][f"coordination_{radius}"][start:end] = coord[i]

        else:
            df["molecule"][f"coordination_{radius}"] = coord[i]

    gc.collect()

    if to_print:
        print("Done")

    return None


@nb.njit(parallel=True)
def generate_coord_num_sparse(Nmol, radii, distances):
    Nt = distances.shape[0]
    Nrad = radii.shape[0]
    coord = np.zeros((Nrad, Nt, Nmol), dtype=np.int32)
    ind = get_upper_tri_index(Nmol)

    for rad_i in nb.prange(Nrad):
        radius = radii[rad_i]
        for mol_idx in range(Nmol):
            dist_mol = distance_mol(distances, mol_idx, ind)
            coord[rad_i, :, mol_idx] = np.sum(dist_mol < radius, axis=1)

    return coord


@nb.njit(parallel=True)
def distance_sparse(R, L):
    Nt, N, _ = R.shape
    Ndistances = N * (N - 1) // 2  # Number of unique distances
    D = np.zeros((Nt, Ndistances), dtype=np.float32)
    for t in nb.prange(Nt):
        index = 0
        for i in range(N):
            for j in range(i + 1, N):
                d_squared = np.sum((utils.pbc(R[t, i] - R[t, j], L)) ** 2)
                D[t, index] = np.sqrt(d_squared)
                index += 1

    return D


def distance(df, method="sparse"):
    """
    Compute distance table

    Args:
        R (np.array) : particle positions, shape (N, 3) or (Nt, N, 3)
        L (np.array): side lengths of simulation box (3, )
        method: full distance table or flat distance arrays
    Returns:
        distance_table (np.array): distance table, shape (N, N) or (Nt, N, N)
        or
        distance (np.array): shape (Nt, N * (N -1) //2)

    """
    R = utils.get_pos(df)
    L = utils.get_L(df)
    ndim = R.ndim

    if method == "sparse" and ndim == 3:
        dist = distance_sparse(R, L)

    elif method == "full":
        # One timestep
        if ndim == 2:
            dist = R[:, np.newaxis, :] - R
            dist = utils.pbc(dist, L)
            dist = np.linalg.norm(dist, axis=2)

        # All timesteps
        elif ndim == 3:
            dist = R[:, :, np.newaxis, :] - R[:, np.newaxis, :, :]
            dist = utils.pbc(dist, L)
            dist = np.linalg.norm(dist, axis=3)

    df["molecule"]["distance"] = dist
    return dist
