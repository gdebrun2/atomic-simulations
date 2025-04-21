import numpy as np
import numpy.linalg as la
import numba as nb
import utils
from sklearnex.cluster import KMeans as kmeans_sk
import matplotlib.pyplot as plt

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
    n = data.shape[0]
    labels = np.empty(n, dtype=np.int8)

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
def assign_external(data, centroids, spread):
    n = data.shape[0]
    labels = np.empty(n, dtype=np.int8)

    for i, xi in enumerate(data):
        minimum = 1e10
        centroid_idx = 0

        for j, centroid in enumerate(centroids):
            a = (xi - centroid) / spread[j]
            d = la.norm(a)

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
                # print(f'{var_stripped} higher gas norm set true')
                mask[i] = True

        elif var_stripped in lower_vars or var_stripped.split("_")[0] == "coordination":
            # print(f'{var_stripped} higher gas norm set false')
            mask[i] = False
    # print()

    # print(f'liq centroid higher gas norm vars: {centroids[0][mask]}')
    # print(f'gas centroid higher gas norm vars: {centroids[1][mask]}')
    # print()

    # print(f'liq centroid lower gas norm vars: {centroids[0][~mask]}')
    # print(f'gas centroid lower gas norm vars: {centroids[1][~mask]}')
    # print()

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

    higher_norm0 = centroids[0][mask]
    lower_norm0 = centroids[0][~mask]
    higher_norm1 = centroids[1][mask]
    lower_norm1 = centroids[1][~mask]
    swap_centroids = False

    if (higher_norm0 > higher_norm1).all() and (lower_norm0 < lower_norm1).all():
        swap_centroids = True

    elif (higher_norm0 > higher_norm1).any():
        print(f"Liquid state has higher {cluster_vars[mask]} norm")
        print(
            f"Liquid state: {centroids[0][mask]}, gasesous state: {centroids[1][mask]}"
        )
    elif (lower_norm0 < lower_norm1).any():
        print(f"Liquid state has lower {cluster_vars[~mask]} norm")
        print(
            f"Liquid state: {centroids[0][~mask]}, gasesous state: {centroids[1][~mask]}"
        )

    return swap_centroids

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

    return swap_centroids


def prep_data(df, cluster_vars, start, nt, t=None, external_centroids=None):

    cluster_vars = np.array(cluster_vars)
    nvars = len(cluster_vars)

    if t is not None:
        cluster_data = np.empty((df["nmolecule"], nvars))
        t += start

    else:
        cluster_data = np.empty((nt, df["nmolecule"], nvars))

    for i, var in enumerate(cluster_vars):

        feature = utils.parse(df, var)[start:]

        if t is not None:
            if external_centroids is not None:
                cluster_data[:, i] = feature[t]
            else:
                x = feature[t]
                cluster_data[:, i] = (x - x.min()) / (x.max() - x.min())

        else:
            if external_centroids is not None:

                cluster_data[:, :, i] = feature

            else:

                x = feature
                cluster_data[:, :, i] = (x - x.min()) / (x.max() - x.min())  # [0,1]
                # cluster_data[:, :, i] = utils.normalize_arr(x)

    if t is not None:
        if t <= start:
            raise ValueError(f"t must be greater than {start}")

    else:
        cluster_data = cluster_data.reshape(-1, nvars)

    return cluster_data


def unnormalize_centroids(df, start=0, actime=False):

    start = utils.get_lag(df, start, actime)

    cluster_vars = df["cluster_vars"]
    molecule_phase = df["molecule"]["phase"][start:]
    liquid = molecule_phase == 0
    gas = ~liquid

    liquid_centers = []
    gas_centers = []
    for var in cluster_vars:
        feat = utils.parse(df, var)[start:]
        liquid_center = np.mean(feat[liquid])
        gas_center = np.mean(feat[gas])
        liquid_centers.append(liquid_center)
        gas_centers.append(gas_center)

    return np.array([liquid_centers, gas_centers])


def classify_phase(
    df,
    cluster_vars,
    start=0,
    actime=True,
    t=None,
    k=2,
    to_print=False,
    mode="sk",
    seed=0,
    nstart=10,
    tol=1e-12,
    external_centroids=None,
    return_norm=False,
    # external_norm = False,
    spread=None,
):
    if to_print:
        print("\nClassifying...\n")

    df["cluster_vars"] = cluster_vars
    start = utils.get_lag(df, start, actime)
    nt = df["nt"] - start

    if t is not None:
        nt = 1

    cluster_data = prep_data(df, cluster_vars, start, nt, t, external_centroids)

    if external_centroids is not None:
        assert spread is not None
        molecule_phase = assign_external(cluster_data, external_centroids, spread)
        centroids = external_centroids

    else:
        if mode == "self":
            molecule_phase, centroids = run(cluster_data, k, seed)
        else:
            molecule_phase, centroids = run_sk(cluster_data, k, seed, nstart, tol)

    molecule_phase = molecule_phase.reshape(nt, df["nmolecule"])
    swap_centroids = swap(cluster_vars, centroids)

    if swap_centroids:
        molecule_phase = 1 - molecule_phase
        tmp = centroids[1].copy()
        centroids[1] = centroids[0]
        centroids[0] = tmp

    nvars = len(cluster_vars)
    liquid = molecule_phase.flatten() == 0
    gas = molecule_phase.flatten() == 1
    nliquid = liquid.sum()
    ngas = gas.sum()
    err = {"sse": sse(cluster_data[liquid], centroids[0])}
    err["sse"] += sse(cluster_data[gas], centroids[1])
    err["mse"] = err["sse"] / df["nmolecule"] / nt / nvars
    err["mse_liquid"] = sse(cluster_data[liquid], centroids[0]) / nliquid / nvars
    err["mse_gas"] = sse(cluster_data[gas], centroids[1]) / ngas / nvars

    atom_phase = np.empty((nt, df["natom"]), dtype=np.int8)
    metal = df["metal_mask"]
    non_metal = df["non_metal"]

    for i in range(nt):
        # apply molecule phases to atoms
        atom_phase[i][non_metal] = np.repeat(
            molecule_phase[i], df["natom_per_molecule"]
        )
        atom_phase[i][metal] = np.repeat(-1, df["nmetal"])

    if t is not None:
        molecule_phase = molecule_phase.flatten()
        atom_phase = atom_phase.flatten()

    else:
        molecule_phase = np.pad(
            molecule_phase, ((start, 0), (0, 0)), "constant", constant_values=-1
        )
        atom_phase = np.pad(
            atom_phase, ((start, 0), (0, 0)), "constant", constant_values=-1
        )

    df["molecule"]["phase"] = molecule_phase
    df["atom"]["phase"] = atom_phase
    df["centroids"] = centroids
    df["err"] = err
    df["nliquid"] = np.sum(molecule_phase == 0, axis=1)
    df["ngas"] = np.sum(molecule_phase == 1, axis=1)

    if to_print:
        print("Classification done\n")

    utils.generate_switch_info(df, start=start, actime=actime)

    if return_norm:

        liquid = molecule_phase == 0
        gas = molecule_phase == 1
        spread = np.zeros((2, nvars))
        liquid = liquid[start:]
        gas = gas[start:]
        for i in range(nvars):

            x = utils.parse(df, cluster_vars[i])[start:]
            x_gas = x[gas]
            x_liq = x[liquid]
            # try:
            #     x_gas = x[gas]
            #     x_liq = x[liquid]
            # except:
            #     x_gas = x[gas[lag:]]
            #     x_liq = x[liquid[lag:]]

            q25, q75 = np.percentile(x_liq, [25, 75])
            spread[0, i] = np.std(x_liq)
            q25, q75 = np.percentile(x_gas, [25, 75])
            spread[1, i] = np.std(x_gas)

        return molecule_phase, atom_phase, centroids, err, spread

    else:
        return molecule_phase, atom_phase, centroids, err


def sse(data, centroid):
    return np.sum(la.norm(data - centroid) ** 2)


def density(
    df,
    bin_width=2,
    mode="atom",
    phase_mask=0,
    time_avg=True,
    absval=True,
    center=True,
    actime=False,
    auto_range=True,
    std=True,
    norm="mass",
    start=0,
):
    liq_density, liq_zbin, liq_err = utils.density(
        df,
        bin_width=bin_width,
        mode=mode,
        phase_mask=0,
        time_avg=time_avg,
        absval=absval,
        center=center,
        actime=actime,
        auto_range=auto_range,
        std=std,
        norm=norm,
        start=start,
    )

    gas_density, gas_zbin, gas_err = utils.density(
        df,
        bin_width=bin_width,
        mode=mode,
        phase_mask=1,
        time_avg=time_avg,
        absval=absval,
        center=center,
        actime=actime,
        auto_range=auto_range,
        std=std,
        norm=norm,
        start=start,
    )

    return liq_density, liq_zbin, liq_err, gas_density, gas_zbin, gas_err


def plot_density(
    liq_density,
    liq_zbin,
    liq_err,
    gas_density,
    gas_zbin,
    gas_err,
    ax=None,
    figsize=(4, 3),
    title="",
):

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=200)
        ax.set_xlabel("Z (Å)")
        ax.set_ylabel(r"$\rho \ (kg \cdot m^{-3})$")
        ax.set_title(title)

    ax.plot(gas_zbin, gas_density, color="darkred", label="$kmeans_{gas}$", zorder=10)
    ax.plot(liq_zbin, liq_density, color="cyan", label="$kmeans_{liq}$", linestyle="-")

    if gas_err is not None:

        ax.fill_between(
            liq_zbin,
            liq_density - liq_err,
            liq_density + liq_err,
            alpha=0.4,
            color="green",
        )
        ax.fill_between(
            gas_zbin,
            gas_density - gas_err,
            gas_density + gas_err,
            alpha=0.4,
            color="yellow",
        )

    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    return ax


def fit_density(kmeans_density, window=11, threshold=1):
    from scipy import signal

    liq_density, liq_zbin, liq_err, gas_density, gas_zbin, gas_err = kmeans_density

    x1 = liq_zbin
    y1 = liq_density
    if len(y1) < window + 1:
        y1_smooth = y1
    else:
        y1_smooth = signal.savgol_filter(y1, window_length=window, polyorder=2)
    dy1 = np.gradient(y1_smooth, x1)
    linear_regions = np.abs(dy1) < threshold
    linear_region_indices = np.where(linear_regions)[0]
    diff_indices = np.diff(linear_region_indices)
    split_points = np.where(diff_indices > 1)[0] + 1

    segments = np.split(linear_region_indices, split_points)
    rho_l = np.mean(y1[segments[0]])  # take mean of first linear region
    if rho_l < 100:  # for close to fully vaporized, no linear regime really
        rho_l = np.mean(y1[~segments[0]])

    x2 = gas_zbin
    y2 = gas_density

    if len(y2) < window + 1:
        y2_smooth = y2
    else:
        y2_smooth = signal.savgol_filter(y2, window_length=window, polyorder=2)
    dy2 = np.gradient(y2_smooth, x2)
    linear_regions = np.abs(dy2) < threshold
    linear_region_indices = np.where(linear_regions)[0]
    diff_indices = np.diff(linear_region_indices)
    split_points = np.where(diff_indices > 1)[0] + 1

    segments = np.split(linear_region_indices, split_points)
    rho_g = np.mean(y2[segments[-1]])  # take mean of last linear region

    return rho_l, rho_g


def plot_feature_distributions(
    df, temp, cluster_vars, mode="molecule", figsize=None, nbins=None, title=""
):

    nvars = len(cluster_vars)
    start = utils.get_lag(df[temp], actime=True)
    Nt = df[temp]["nt"] - start
    if figsize is None:
        figsize = (4, 2*nvars)
    fig, ax = plt.subplots(
        nrows=nvars, ncols=1, figsize=figsize, dpi=200, layout="constrained"
    )
    x = np.zeros((nvars, 2))
    x_gas = np.zeros((nvars, 2))
    x_liquid = np.zeros((nvars, 2))
    phase = df[temp][mode]["phase"][start:]
    gas = phase == 1
    liquid = ~gas
    if nvars == 1:
        ax = [ax]

    for j in range(nvars):
        try:
            x = utils.parse(df[temp], cluster_vars[j], mode=mode)[start:]
        except:

            raise KeyError(f"{var} not in {mode} data. Exiting...")

        if j == 0:
            gas_label = "gas"
            liquid_label = "liquid"
        else:
            gas_label = None
            liquid_label = None

        x_gas = x[gas]
        x_liq = x[liquid]
        bins = nbins[j] if nbins is not None else 10

        counts, bin_edges = np.histogram(x, bins=bins)
        N = counts.sum()

        counts, edges = np.histogram(x_liq, bins=bin_edges)
        counts = counts.astype(float) / N
        counts[counts == 0] = np.nan
        center = (edges[1:] + edges[:-1]) / 2
        ax[j].plot(center, counts, color="navy", label=liquid_label)

        counts, edges = np.histogram(x_gas, bins=bin_edges)
        counts = counts.astype(float) / N
        counts[counts == 0] = np.nan
        center = (edges[1:] + edges[:-1]) / 2
        ax[j].plot(center, counts, color="darkred", label=gas_label)

        ax[j].minorticks_on()
        ax[j].grid()
        ax[j].set_ylabel("Probability")

    feature_labels = [utils.parse_label(var) for var in cluster_vars]

    for j in range(nvars):
        feature = feature_labels[j]
        label = utils.parse_ylabel(cluster_vars[j])
        ax[j].set_xlabel(label)
        ax[j].set_title(feature_labels[j])

    fig.legend(bbox_to_anchor=(1.35, 0.5), loc="outside right")
    fig.suptitle(title, x=0.6)

    return fig, ax


def plot_feature_means(df, temps, cluster_vars, figsize=(6, 3)):

    nvars = len(cluster_vars)
    ntemps = len(temps)

    fig, ax = plt.subplots(
        nrows=nvars, ncols=1, figsize=figsize, dpi=200, layout="constrained"
    )
    x = np.zeros((ntemps, nvars))
    x_gas = np.zeros((ntemps, nvars))
    x_liquid = np.zeros((ntemps, nvars))

    for i, temp in enumerate(temps):

        start = utils.get_lag(df[temp], actime=True)

        for j in range(nvars):
            x[i, j] = utils.parse(df[temp], cluster_vars[j])[start:].mean()

        phase = df[temp]["molecule"]["phase"][start:]
        gas = phase == 1
        liquid = ~gas

        if gas.sum() == 0:
            for j in range(nvars):
                x_gas[i, j] = np.nan

        else:
            for j in range(nvars):
                x_gas[i, j] = utils.parse(df[temp], cluster_vars[j])[start:][gas].mean()

        if liquid.sum() == 0:
            for j in range(nvars):
                x_liquid[i, j] = np.nan

        else:
            for j in range(nvars):
                x_liquid[i, j] = utils.parse(df[temp], cluster_vars[j])[start:][
                    liquid
                ].mean()

    feature_labels = [utils.parse_label(var) for var in cluster_vars]

    for j in range(nvars):
        if j == 0:
            gas_label = "gas"
            liquid_label = "liquid"
        else:
            gas_label = None
            liquid_label = None

        ax[j].plot(temps, x[:, j], color="tab:purple", label=None)
        ax[j].set_title(feature_labels[j])

        ax[j].plot(temps, x_gas[:, j], color="tab:red", label=gas_label)
        ax[j].plot(temps, x_liquid[:, j], color="tab:blue", label=liquid_label)

    ax[-1].set_xlabel("T (K)")
    # ax[-1, 1].set_xlabel('T (K)')
    # ax[-1, 2].set_xlabel('T (K)')

    for j in range(nvars):
        feature = feature_labels[j]
        label = utils.parse_ylabel(cluster_vars[j])
        ax[j].set_ylabel(label)

    fig.legend(bbox_to_anchor=(1.35, 0.5), loc="outside right")
    fig.suptitle("Feature Mean Values vs Temperature")
    return fig, ax


# this is somehow slower than extra square root/square in la.norm
# def sse(data, centroid):
# return np.sum(np.sum((data - centroid) ** 2, axis=1))
