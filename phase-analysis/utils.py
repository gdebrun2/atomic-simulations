import numpy as np
import numba as nb
import psutil
import os
import sys


@nb.njit
def normalize_arr(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))


def generate_dz(lags, df):
    z = df["molecule"]["z"]

    for lag in lags:
        zi = np.zeros_like(z[lag:], dtype=bool)

        for mol in range(zi.shape[1]):
            zt = z[:, mol]

            # find where molecule crosses boundary
            zswitch = ((zt[lag:] < 0) & (zt[:-lag] > 0)) | (
                (zt[lag:] > 0) & (zt[:-lag] < 0)
            )
            zi[:, mol] = zswitch

        dz = np.zeros_like(zi, dtype=np.float32)
        dz[zi] = np.abs(z[lag:][zi] + z[:-lag][zi])
        dz[~zi] = np.abs(z[lag:][~zi] - z[:-lag][~zi])
        df["molecule"][f"dz_{lag}"] = dz
    return None


def generate_displacement(lags, df):
    x = df["molecule"]["x"]
    y = df["molecule"]["y"]
    z = df["molecule"]["z"]

    for lag in lags:
        xi = np.zeros_like(x[lag:], dtype=bool)
        yi = np.zeros_like(y[lag:], dtype=bool)
        zi = np.zeros_like(z[lag:], dtype=bool)

        for mol in range(xi.shape[1]):
            xt = x[:, mol]
            yt = y[:, mol]
            zt = z[:, mol]

            xswitch = ((xt[lag:] < 0) & (xt[:-lag] > 0)) | (
                (xt[lag:] > 0) & (xt[:-lag] < 0)
            )
            yswitch = ((yt[lag:] < 0) & (yt[:-lag] > 0)) | (
                (yt[lag:] > 0) & (yt[:-lag] < 0)
            )
            zswitch = ((zt[lag:] < 0) & (zt[:-lag] > 0)) | (
                (zt[lag:] > 0) & (zt[:-lag] < 0)
            )

            xi[:, mol] = xswitch
            yi[:, mol] = yswitch
            zi[:, mol] = zswitch

        dx = np.zeros_like(xi, dtype=np.float32)
        dy = np.zeros_like(yi, dtype=np.float32)
        dz = np.zeros_like(zi, dtype=np.float32)

        dx[xi] = np.abs(x[lag:][xi] + x[:-lag][xi])
        dy[yi] = np.abs(y[lag:][yi] + y[:-lag][yi])
        dz[zi] = np.abs(z[lag:][zi] + z[:-lag][zi])

        dx[~xi] = np.abs(x[lag:][~xi] - x[:-lag][~xi])
        dy[~yi] = np.abs(y[lag:][~yi] - y[:-lag][~yi])
        dz[~zi] = np.abs(z[lag:][~zi] - z[:-lag][~zi])

        net_displacement = np.sqrt(dx**2 + dy**2 + dz**2)
        df["molecule"][f"displacement_{lag}"] = net_displacement

    return None


@nb.njit
def pbc(R, L):
    """
    Apply periodic boundary conditions to particle positions
    in a cubic box of side length L.

    Args:
        R (np.array): particle positions, shape (N, 3), (N, N, 3) or (Nt, N, N, 3)
        L (np.array): side lengths of simulation box (3,)
    Returns:
        np.array: particle positions, shape (N, N, 3)
    """

    # if single pos. vector
    if R.ndim == 1:
        R[0] -= L[0] * np.round(R[0] / L[0])
        R[1] -= L[1] * np.round(R[1] / L[1])
        R[2] -= L[2] * np.round(R[2] / L[2])

    # single timestep
    elif R.ndim == 3:
        R[:, :, 0] -= L[0] * np.round(R[:, :, 0] / L[0])
        R[:, :, 1] -= L[1] * np.round(R[:, :, 1] / L[1])
        R[:, :, 2] -= L[2] * np.round(R[:, :, 2] / L[2])

    # all timesteps
    elif R.ndim == 4:
        R[:, :, :, 0] -= L[0] * np.round(R[:, :, :, 0] / L[0])
        R[:, :, :, 1] -= L[1] * np.round(R[:, :, :, 1] / L[1])
        R[:, :, :, 2] -= L[2] * np.round(R[:, :, :, 2] / L[2])

    return R


def generate_switch_info(df):
    molecule_phase = df["molecule"]["phase"]
    switch_i = []
    for i in range(df["Nmolecule"]):
        if not ((molecule_phase[:, i] == 0).all() or (molecule_phase[:, i] == 1).all()):
            switch_i.append(i)

    switch_z = df["molecule"]["z"][:, switch_i]
    switch_phase = molecule_phase[:, switch_i]
    switch_z[switch_z < 0] -= df["offset"]

    gas_liquid_z = []
    gas_liquid_i = []
    liquid_gas_z = []
    liquid_gas_i = []
    gas_liquid_t = []
    liquid_gas_t = []

    for i in range(switch_z.shape[1]):
        last_phase = switch_phase[0, i]

        for t in range(1, df["Nt"]):
            phase = switch_phase[t, i]

            if phase != last_phase:
                if last_phase == 0:
                    liquid_gas_z.append(switch_z[t, i])
                    liquid_gas_t.append(t)
                    liquid_gas_i.append(i)

                elif last_phase == 1:
                    gas_liquid_z.append(switch_z[t, i])
                    gas_liquid_t.append(t)
                    gas_liquid_i.append(i)

                last_phase = phase

    gas_liquid_z = np.array(gas_liquid_z)
    liquid_gas_z = np.array(liquid_gas_z)
    gas_liquid_t = np.array(gas_liquid_t)
    liquid_gas_t = np.array(liquid_gas_t)
    gas_liquid_i = np.array(gas_liquid_i)
    liquid_gas_i = np.array(liquid_gas_i)

    if df["metal_type"] != -1:
        switch_z = np.abs(switch_z)
        gas_liquid_z = np.abs(gas_liquid_z)
        liquid_gas_z = np.abs(liquid_gas_z)

    df["molecule"]["switch_i"] = switch_i
    df["molecule"]["switch_z"] = switch_z
    df["molecule"]["gas_liquid_z"] = gas_liquid_z
    df["molecule"]["liquid_gas_z"] = liquid_gas_z
    df["molecule"]["gas_liquid_t"] = gas_liquid_t
    df["molecule"]["liquid_gas_t"] = liquid_gas_t
    df["molecule"]["gas_liquid_i"] = gas_liquid_i
    df["molecule"]["liquid_gas_i"] = liquid_gas_i

    switch_t = np.sort(np.unique(np.concatenate([liquid_gas_t, gas_liquid_t])))
    df["molecule"]["switch_t"] = switch_t

    n_to_gas = np.zeros(df["Nt"], dtype=int)
    n_to_liquid = np.zeros(df["Nt"], dtype=int)
    nliquid = df["Nliquid"]
    ngas = df["Ngas"]
    rate_to_gas = np.zeros(df["Nt"])
    rate_to_liquid = np.zeros(df["Nt"])
    for t in range(df["Nt"]):
        n_to_gas[t] = int((liquid_gas_t == t).sum())
        n_to_liquid[t] = int((gas_liquid_t == t).sum())

        if nliquid[t] > 0:
            rate_to_gas[t] = (
                n_to_gas[t] / nliquid[t] / df["dt"] * 1000 * 1000
            )  #  per nanosecond
        else:
            rate_to_gas[t] = 0
        if ngas[t] > 0:
            rate_to_liquid[t] = (
                n_to_liquid[t] / ngas[t] / df["dt"] * 1000 * 1000
            )  #  per nanosecond
        else:
            rate_to_liquid[t] = 0

    df["molecule"]["n_to_gas"] = n_to_gas
    df["molecule"]["n_to_liquid"] = n_to_liquid
    df["molecule"]["rate_to_gas"] = rate_to_gas
    df["molecule"]["rate_to_liquid"] = rate_to_liquid

    return None


def write_switch_info(df, path):
    switch_t = df["molecule"]["switch_t"]
    gas_liquid_i = df["molecule"]["gas_liquid_i"]
    liquid_gas_i = df["molecule"]["liquid_gas_i"]
    gas_liquid_t = df["molecule"]["gas_liquid_t"]
    liquid_gas_t = df["molecule"]["liquid_gas_t"]
    n_to_gas = df["molecule"]["n_to_gas"]
    n_to_liquid = df["molecule"]["n_to_liquid"]
    nliquid = df["Nliquid"]
    ngas = df["Ngas"]
    centroids = df["centroids"]
    cluster_vars = df["cluster_vars"]
    x = df["molecule"]["x"]
    y = df["molecule"]["y"]
    z = df["molecule"]["z"]
    timesteps = df["timesteps"]
    rate_to_gas = df["molecule"]["rate_to_gas"]
    rate_to_liquid = df["molecule"]["rate_to_liquid"]

    with open(path, "w+") as f:
        header = f"HEADER {cluster_vars} mu_liq({centroids[0]}) mu_gas({centroids[-1]})"
        f.write(header + "\n")

        for t in range(df["Nt"]):
            t_header = f"TIMESTEP {timesteps[t]} nliq {nliquid[t]} ngas {ngas[t]} n_to_gas {n_to_gas[t]} n_to_liq {n_to_liquid[t]}"
            t_header += f" rate_to_gas {np.round(rate_to_gas[t], 4)} rate_to_liq {np.round(rate_to_liquid[t], 4)}"
            # t_header += " #per nanosecond"
            f.write(t_header + "\n")
            info_header = "ID X Y Z liq gas"
            f.write(info_header + "\n")
            if t not in switch_t:
                continue

            curr_gas_liquid_i = gas_liquid_i[gas_liquid_t == t]
            curr_liquid_gas_i = liquid_gas_i[liquid_gas_t == t]

            sort = np.sort(np.concatenate([curr_gas_liquid_i, curr_liquid_gas_i]))

            for id in sort:
                X = str(np.round(x[t][id], 4))
                Y = str(np.round(y[t][id], 4))
                Z = str(np.round(z[t][id], 4))
                line = f"{id} {X} {Y} {Z}"
                if id in curr_gas_liquid_i:
                    line += " 1 0\n"
                else:
                    line += " 0 1\n"
                f.write(line)

    return None


def print_memory(path):
    file_size = os.path.getsize(path)
    available_memory = psutil.virtual_memory().available
    print(f"file size: {file_size / 1024**3} GB")
    print(f"available memory: {available_memory / 1024**3} GB")

    return None


def write_phase(data_path, outfile, atom_phase):
    with open(data_path, mode="r") as fi:
        old_data = fi.readlines()
    new_fields = ["phase"]
    new_data = [atom_phase]
    with open(outfile, "w") as fi:
        data_loc = False
        t = -1
        for i, line in enumerate(old_data):
            line_data = line.split()

            try:
                if len(line_data) > 1:
                    if line_data[1] == "ATOMS":
                        line_data.extend(new_fields)
                        data_loc = True
                        fi.writelines(" ".join(line_data))
                        fi.write("\n")
                        continue

                    if line_data[1] == "TIMESTEP":
                        data_loc = False
                        t += 1

                if data_loc:
                    idx = int(line_data[0]) - 1

                    for data in new_data:
                        line_data.append(str(data[t][idx]))

                fi.writelines(" ".join(line_data))
                fi.write("\n")

            except Exception as e:
                print(e)
                print(line)
                sys.exit()


def getsize(df):
    s1 = 0

    for key in list(df["atom"].keys()):
        data = df["atom"][key]
        size = sys.getsizeof(data) / 1e9
        s1 += size

    s2 = 0

    for key in list(df["molecule"].keys()):
        data = df["molecule"][key]
        size = sys.getsizeof(data) / 1e9
        s2 += size

    s1 = np.round(s1, 3)
    s2 = np.round(s2, 3)
    s3 = np.round(s1 + s2, 3)

    print(f"df_atom: {s1} GB df_molecule: {s2} GB Total: {s3} GB")

    return None


def parse(df, expr, mode="mol"):
    if mode == "mol":
        df_mode = df["molecule"]
    else:
        df_mode = df["atom"]

    op_map = {
        "log": np.log10,
        "norm": normalize_arr,
        "abs": np.abs,
        "sqrt": np.sqrt,
    }
    ops = expr.split("(")
    ops = [op.strip(")").strip() for op in ops if op]

    var = ops[-1]
    ops = ops[:-1][::-1]
    data = df_mode[var].copy()
    if mode == "atom":
        data = data[:, df["non_metal"]]

    for op in ops:
        if op not in list(op_map.keys()):
            print("Supported operations: ", list(op_map.keys()))
            raise ValueError(f"Invalid operation: {op}")
        if op == "log":
            data = data.astype(np.float32) + 1e-6
        f = op_map[op]
        data = f(data)
    return data


def strip_var(var):
    var_stripped = var.strip().split("(")
    var_stripped = [item.strip(")").strip() for item in var_stripped if item][-1]

    return var_stripped


def parse_label(expr):
    ops = expr.split("(")
    ops = [op.strip(")").strip() for op in ops if op]

    var = ops[-1]
    ops = ops[:-1][::-1]
    if len(var.split("_")) > 1:
        var = var.split("_")
        label = "\\text{" + var[0] + "}" + "_{" + var[1] + "}"
    else:
        label = "\\text{" + var + "}"
    for op in ops:
        if op == "abs":
            label = f"\\lvert {label} \\rvert"

        elif op == "norm":
            label += "_{\\text{norm}}"

        elif op == "log":
            label = f"\\log ({label})"

        elif op == "sqrt":
            label = f"\\sqrt ({label})"

    label = "$" + label + "$"

    return label


def concat_labels(labels):
    concat = r""
    new_labels = []
    for label in labels:
        new_labels.append(parse_label(label))

    for i, label in enumerate(new_labels):
        if i == len(labels) - 1:
            concat += label.strip("$")
        else:
            concat += label.strip("$") + r", \ "

    concat = r"[" + concat + r"]"
    return concat


def format(number):
    if number == 0 or number == -1:
        return ""
    magnitude = np.floor(np.log10(number))

    if 3 <= magnitude <= 5:
        number /= 10**3
        if magnitude == 5 or magnitude == 4:
            return f"{number:.0f}K"
        else:
            return f"{number:.1f}K"
    elif magnitude == 2:
        return f"{number:.0f}"
    elif magnitude == 1:
        return f"{number:.1f}"
    elif magnitude == 0:
        return f"{number:.2f}"
    elif magnitude == -1:
        return f"{number:.2f}"
    elif magnitude == -2:
        return f"{number:.2f}"
    elif magnitude <= -3:
        number *= 10**-magnitude
        return f"{number:.0f}e{magnitude:.0f}"


def get_pos(df):
    x = df["molecule"]["x"]
    y = df["molecule"]["y"]
    z = df["molecule"]["z"]
    r = np.stack((x, y, z), axis=-1)

    return r


def get_L(df):
    L = df["bounds"][:, 1] * 2
    return L


def get_v(df):
    vx = df["molecule"]["vx"]
    vy = df["molecule"]["vy"]
    vz = df["molecule"]["vz"]
    v = np.stack((vx, vy, vz), axis=-1)
    return v


def density(z, Nbins, hist_range, time_avg=True):
    Nt = z.shape[0]

    if time_avg:
        counts, edges = np.histogram(z, bins=Nbins, range=hist_range)
        centers = (edges[:-1] + edges[1:]) / 2
        density = counts / Nt
        return density, centers

    else:
        density = np.zeros((Nt, Nbins))
        for t in range(Nt):
            counts, edges = np.histogram(z[t], bins=Nbins, range=hist_range)
            density[t] = counts
        centers = (edges[:-1] + edges[1:]) / 2
        return density, centers
