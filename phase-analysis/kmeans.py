import numpy as np
import numpy.linalg as la
import numba as nb
import utils
from sklearnex.cluster import KMeans as kmeans_sk

higher_vars = [
    "abs(z)",
    "abs(vz)",
    "ke",
    "lt",
    "speed",
    "dz",
    "displacement",
    "pe",
]  # highter indicates gas
lower_vars = [
    "ld",
    "coordination",
]  # lower indicates gas


@nb.njit
def init(data, k):
    idx = np.random.choice(data.shape[0], k, replace=False)

    if data.ndim == 1:
        for i in range(idx.shape[0]):
            for j in range(idx.shape[0]):
                if i == j:
                    continue

                ii = idx[i]
                jj = idx[j]

                if np.isclose(data[ii], data[jj]):
                    idx = np.random.choice(data.shape[0], k, replace=False)

        init_centroids = data[idx]

    else:
        for i in range(idx.shape[0]):
            for j in range(idx.shape[0]):
                if i == j:
                    continue
                ii = idx[i]
                jj = idx[j]

                if np.isclose(data[ii, :], data[jj, :]).any():
                    idx = np.random.choice(data.shape[0], k, replace=False)

        init_centroids = data[idx, :]

    return init_centroids


@nb.njit
def assign(data, centroids):
    N = data.shape[0]
    labels = np.empty(N, dtype=np.int8)

    for i, xi in enumerate(data):
        minimum = 1e10
        centroid_idx = 0

        for j, centroid in enumerate(centroids):
            d = la.norm(xi - centroid)

            if d < minimum:
                minimum = d
                centroid_idx = j

        labels[i] = centroid_idx

    return labels


@nb.njit
def update(data, k, labels):
    if data.ndim == 1:
        centroids = np.zeros(k)

    else:
        centroids = np.zeros((k, data.shape[1]))

    centroid_counts = np.zeros(k)

    for idx, pair in enumerate(zip(data, labels)):
        xi, label = pair
        centroids[label] += xi
        centroid_counts[label] += 1

    for idx, count in enumerate(centroid_counts):
        centroids[idx] /= count

    return centroids


@nb.njit
def run(data, k, seed=0):
    np.random.seed(seed)
    centroids = init(data, k)

    i = 0
    while True:
        assignment = assign(data, centroids)
        prev_centroids = centroids.copy()
        centroids = update(data, k, assignment)

        if np.array_equal(prev_centroids, centroids):
            break

        i += 1

        if i % 100 == 0:
            print(f"Iteration {i}")

    return assignment, centroids


def run_sk(cluster_data, k, seed=0, nstart=10, tol=1e-12):
    res = kmeans_sk(
        n_clusters=k,
        verbose=0,
        init="random",
        n_init=nstart,
        tol=tol,
        max_iter=100000,
        random_state=seed,
    ).fit(cluster_data)
    centroids = res.cluster_centers_.copy()
    molecule_phase = res.predict(cluster_data)

    return molecule_phase, centroids


def swap(cluster_vars, centroids):
    global higher_vars
    global lower_vars
    mask = np.empty(len(cluster_vars), dtype=bool)
    cluster_vars = np.array(cluster_vars)

    for i, var in enumerate(cluster_vars):
        # var_stripped = var.strip().split("(")
        # var_stripped = [item.strip(")").strip() for item in var_stripped if item][-1]
        var_stripped = utils.strip_var(var)
        if (
            var_stripped in higher_vars
            or var_stripped.split("_")[0] == "dz"
            or var_stripped.split("_")[0] == "displacement"
        ):
            if (
                var_stripped == "z" or var_stripped == "vz" and "abs" not in var
            ):  # make sure abs(z)
                mask[i] = False
            else:
                mask[i] = True

        elif var_stripped in lower_vars or var_stripped.split("_")[0] == "coordination":
            mask[i] = False

    higher_norm0 = la.norm(
        centroids[0][mask]
    )  # vars with higher gaseous norm for centroid 0
    lower_norm0 = la.norm(
        centroids[0][~mask]
    )  # vars with lower gaseous norm for centroid 0

    higher_norm1 = la.norm(
        centroids[1][mask]
    )  # vars with higher gaseous norm for centroid 1
    lower_norm1 = la.norm(
        centroids[1][~mask]
    )  # vars with lower gaseous norm for centroid 1

    # if centroid 0 has higher higher norm vars and
    # lower lower norm vars, we need to be in the gaseous state (1)
    # handle the case where we only have higher/lower norm variables
    swap_centroids = False

    if len(cluster_vars[mask]) == 0:  # only lower norm vars
        swap_centroids = lower_norm0 < lower_norm1

    elif len(cluster_vars[~mask]) == 0:  # only higher norm vars
        swap_centroids = higher_norm0 > higher_norm1

    else:
        if higher_norm0 > higher_norm1 and lower_norm0 < lower_norm1:
            swap_centroids = True
        elif higher_norm0 > higher_norm1:
            print(f"Liquid state has higher {cluster_vars[mask]} norm")
            print(
                f"Liquid state: {centroids[0][mask]}, gasesous state: {centroids[1][mask]}"
            )
        elif lower_norm0 < lower_norm1:
            print(f"Liquid state has lower {cluster_vars[~mask]} norm")
            print(
                f"Liquid state: {centroids[0][~mask]}, gasesous state: {centroids[1][~mask]}"
            )

    # # generalize to arbitrary number of centroids
    # higher_centers = []
    # lower_centers = []
    # for centroid in centroids:

    #     higher_center = la.norm(centroid[mask])
    #     higher_centers.append(higher_center)
    #     lower_center = la.norm(centroid[~mask])
    #     lower_centers.append(lower_center)

    return swap_centroids


def prep_data(df, cluster_vars, start=0, t=None):
    cluster_vars = np.array(cluster_vars)
    nvars = len(cluster_vars)

    if t is not None:
        cluster_data = np.empty((df["Nmolecule"], nvars))
        t += start

    else:
        cluster_data = np.empty((df["Nt"], df["Nmolecule"], nvars))
    max_lag = 0
    for i, var in enumerate(cluster_vars):
        feature = utils.parse(df, var)
        lag = df["Nt"] - feature.shape[0] + start

        if lag > max_lag:
            max_lag = lag

        if t is not None:
            cluster_data[:, i] = utils.normalize_arr(feature[t])

        else:
            cluster_data[lag:, :, i] = utils.normalize_arr(feature[start:])
    # max_lag += start
    if t is not None:
        if t <= max_lag:
            raise ValueError(f"t must be greater than {max_lag}")
        Nt = 1

    else:
        cluster_data = cluster_data[max_lag:].reshape(-1, nvars)
        Nt = df["Nt"] - max_lag

    return cluster_data, Nt, max_lag


def molecule_to_atom_phase(df, molecule_phase, lag=0):
    if lag > 0 and molecule_phase.shape[0] < df["Nt"]:
        molecule_phase = np.pad(
            molecule_phase, ((lag, 0), (0, 0)), "constant", constant_values=-1
        )

    lag = df["Nt"] - molecule_phase.shape[0]
    Nt = df["Nt"] - lag

    atom_phase = np.empty((Nt, df["Natom"]), dtype=np.int8)
    metal = df["metal_mask"]
    non_metal = df["non_metal"]

    for i in range(Nt):
        # apply molecule phases to atoms
        atom_phase[i][non_metal] = np.repeat(
            molecule_phase[i], df["Natom_per_molecule"]
        )
        atom_phase[i][metal] = np.repeat(2, df["Nmetal"])

    return atom_phase


def classify_phase(
    df,
    cluster_vars,
    start=0,
    t=None,
    k=2,
    external_centroids=None,
    to_print=False,
    mode="sk",
    seed=0,
    nstart=10,
    tol=1e-12,
):
    if to_print:
        print("\nClassifying...\n")

    df["cluster_vars"] = cluster_vars

    cluster_data, Nt, max_lag = prep_data(df, cluster_vars, start, t)
    if external_centroids is not None:
        molecule_phase = assign(cluster_data, external_centroids)
        centroids = external_centroids

    else:
        if mode == "self":
            molecule_phase, centroids = run(cluster_data, k, seed)
        else:
            molecule_phase, centroids = run_sk(cluster_data, k, seed, nstart, tol)

    molecule_phase = molecule_phase.reshape(Nt, df["Nmolecule"])
    swap_centroids = swap(cluster_vars, centroids)

    if swap_centroids:
        molecule_phase = 1 - molecule_phase
        tmp = centroids[1].copy()
        centroids[1] = centroids[0]
        centroids[0] = tmp

    Nfeats = len(cluster_vars)
    liquid = molecule_phase.flatten() == 0
    gas = molecule_phase.flatten() == 1
    Nliquid = liquid.sum()
    Ngas = gas.sum()
    err = {"sse": sse(cluster_data[liquid], centroids[0])}
    err["sse"] += sse(cluster_data[gas], centroids[1])
    err["mse"] = err["sse"] / df["Nmolecule"] / Nt / Nfeats
    err["mse_liquid"] = sse(cluster_data[liquid], centroids[0]) / Nliquid / Nfeats
    err["mse_gas"] = sse(cluster_data[gas], centroids[1]) / Ngas / Nfeats

    atom_phase = np.empty((Nt, df["Natom"]), dtype=np.int8)
    metal = df["metal_mask"]
    non_metal = df["non_metal"]

    for i in range(Nt):
        # apply molecule phases to atoms
        atom_phase[i][non_metal] = np.repeat(
            molecule_phase[i], df["Natom_per_molecule"]
        )
        atom_phase[i][metal] = np.repeat(k, df["Nmetal"])

    if t is not None:
        molecule_phase = molecule_phase.flatten()
        atom_phase = atom_phase.flatten()

    else:
        molecule_phase = np.pad(
            molecule_phase, ((max_lag, 0), (0, 0)), "constant", constant_values=-1
        )
        atom_phase = np.pad(
            atom_phase, ((max_lag, 0), (0, 0)), "constant", constant_values=-1
        )

    df["molecule"]["phase"] = molecule_phase
    df["atom"]["phase"] = atom_phase
    df["centroids"] = centroids
    df["err"] = err
    df["Nliquid"] = np.sum(molecule_phase == 0, axis=1)
    df["Ngas"] = np.sum(molecule_phase == 1, axis=1)

    if to_print:
        print("Classification done\n")

    utils.generate_switch_info(df)

    return molecule_phase, atom_phase, centroids, err


def sse(data, centroid):
    return np.sum(la.norm(data - centroid) ** 2)

# this is somehow slower than extra square root/square in la.norm
# def sse(data, centroid):
# return np.sum(np.sum((data - centroid) ** 2, axis=1))