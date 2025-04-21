import numpy as np
import numba as nb
import psutil
import os
import sys


@nb.njit
def normalize_arr(x):
    # return (x - np.min(x)) / (np.max(x) - np.min(x))
    # return (x - np.mean(x)) / np.std(x)
    q75, q25 = np.percentile(x, [75 ,25])
    iqr = q75 - q25
    return (x - np.mean(x)) / iqr


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
        R (np.array): particle positions, shape (n, 3), (n, n, 3) or (nt, n, n, 3)
        L (np.array): side lengths of simulation box (3,)
    Returns:
        np.array: particle positions, shape (n, n, 3)
    """

    # if single pos. vector
    if R.ndim == 1:
        R[0] -= L[0] * np.round(R[0] / L[0])
        R[1] -= L[1] * np.round(R[1] / L[1])
        R[2] -= L[2] * np.round(R[2] / L[2])

    # z coordinate
    if R.ndim == 2:
        R[:] -= L[2] * np.round(R / L[2])

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


def generate_switch_info(df, start = 0, actime = True):
    start = get_lag(df, start, actime)
    nt = df['nt'] - start
    molecule_phase = df["molecule"]["phase"][start:]
    switch_i = []
    for i in range(df["nmolecule"]):
        if not ((molecule_phase[:, i] == 0).all() or (molecule_phase[:, i] == 1).all()):
            switch_i.append(i)

    switch_z = df["molecule"]["z"][start:, switch_i]
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

        for t in range(1, nt):
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

    n_to_gas = np.zeros(nt, dtype=int)
    n_to_liquid = np.zeros(nt, dtype=int)
    nliquid = df["nliquid"]
    ngas = df["ngas"]
    rate_to_gas = np.zeros(nt)
    rate_to_liquid = np.zeros(nt)
    for t in range(nt):
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
    nliquid = df["nliquid"]
    ngas = df["ngas"]
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

        for t in range(df["nt"]):
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
    print(f"file size: {file_size / 1024**3:.2f} GB")
    print(f"available memory: {available_memory / 1024**3:.2f} GB")

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


def parse(df, expr, mode="molecule"):

    df_mode = df[mode]

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

    if data.shape[0] < df['nt']:
        lag = df['nt'] - data.shape[0]
        # data = np.concatenate((np.zeros(df['nt'] - data.shape[0]), data))
        data = np.pad(
            data, ((lag, 0), (0, 0)), "constant", constant_values=0
        )
        
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

def parse_ylabel(expr):

    ops = expr.split("(")
    ops = [op.strip(")").strip() for op in ops if op]

    var = ops[-1]
    ops = ops[:-1][::-1]
    if len(var.split("_")) > 1:
        var = var.split("_")
        feature = var[0]
    else:
        feature = var 

    feature_units = {'coordination':'n_{molecule}', 'pe':'Kcal/mol', 'displacement':'Å', 'ke':'Kcal/mol', 'lt':'K'}
    units = feature_units[feature]
    label = units
    
    for op in ops:
        if op == "abs":
            label = f"\\lvert {units} \\rvert"

        elif op == "norm":
            label = units + "_{\\text{norm}}"

        elif op == "log":
            label = f"\\log ({units})"

        elif op == "sqrt":
            label = f"\\sqrt ({units})"

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

def get_lag(df, start = 0, actime = False, features = None):

    if features is None:
        features = df['cluster_vars']
    lag = 0
    for var in features:
        if "displacement" in var or "dz" in var:
            split = var.split('_')[-1]
            try:
                curr_lag = int(split[:2]) 
            except:
                curr_lag = int(split[:1])

            if curr_lag > lag:
                lag = curr_lag    
    if start > lag:
        lag = start

    if actime:
        diff = df['actime'] - lag
        if diff > 0:
            lag += diff
            
    return lag

@nb.njit(parallel=True)
def get_min_max(z):
    nt = z.shape[0]
    max_neg = np.zeros(nt)
    max_pos = np.zeros(nt)
    for t in nb.prange(nt):
        zt = z[t]
        z_pos = zt[zt>=0]
        z_neg = zt[zt<0]
        max_pos[t] = np.max(z_pos)
        max_neg[t] = np.max(np.abs(z_neg))
    return np.min(max_pos), np.min(max_neg)        
        

def density(
    df,
    bin_width = 4, # Angstroms
    z = None,
    mode = 'molecule',
    hist_range=None,
    time_avg=False,
    phase_mask = None,
    absval=False,
    center = False,
    offset = True,
    actime = False,
    auto_range=False,
    std = False,
    norm = 'count',
    t = None,
    start = 0,
):
   
    if z is None:
        z = parse(df, 'z', mode)
    if offset:
        lower_mask = df[mode]["lower_mask"]
        z[lower_mask] -= df["offset"]

    start = get_lag(df, start, actime)

    if center:
        com = np.mean(df['atom']['z'], axis = 1)
        z -= com[:, np.newaxis]
        L = get_L(df)
        z = pbc(z, L)
    if t is None:
        z = z[start:]
        nt = z.shape[0]
    else:
        z = z[t]
        nt = 1
    if auto_range:
        if t is None:
            if not absval:
                zmin = np.max(np.min(z, axis=1))  # lowest z bin present for all t
                zmax = np.min(np.max(z, axis=1))  # highest z bin present for all t
            else:
                zmax_pos, zmax_neg = get_min_max(z)
                zmin = np.max(np.min(np.abs(z),axis=1))
                zmax = zmax_pos if zmax_pos <= zmax_neg else zmax_neg
              
        else:
            if not absval:
                zmin = z.min()
                zmax = z.max()
            else:
                zmin = np.min(np.abs(z))
                zmax1 = np.max(np.abs(z[z<0]))
                zmax2 = np.max(z[z>0])
                zmax = zmax1 if zmax1 < zmax2 else zmax2
    if absval:
        z = np.abs(z)            
    if hist_range is None and not auto_range:
        if not absval:
            zmax = df["bounds"][-1, -1]
            zmin = df["bounds"][-1, 0]
        else:
            zmax = df["bounds"][-1, -1]
            zmin = z.min()
            
            
    if hist_range is None:
        hist_range = (zmin, zmax)
    else:
        zmin, zmax = hist_range

    nbins = int(np.round((zmax - zmin) / bin_width, 0))

    if nt == 1:
        time_avg = False

    phase = None
    if (time_avg or nt == 1) and not std and phase_mask is None and norm != 'mass':

        counts, bins = np.histogram(z, bins=nbins, range=hist_range)
        zbin = (bins[:-1] + bins[1:]) / 2
        density = counts / nt
       
        density = normalize_density(
            df,
            density,
            z,
            norm,
            phase_mask,
            mode,
            zbin,
            start,
            bins,
            nt,
            phase,
        )
        if absval:
            density /= 2
        return density, zbin

    else:
        bins = np.linspace(hist_range[0], hist_range[1], nbins+1, endpoint=True)
        zbin = (bins[1:] + bins[:-1]) / 2
        
        if phase_mask is not None:
            phase = parse(df, 'phase', mode = mode)
            if t is None:
                phase = phase[start:]
            else:
                phase = phase[t]
            mask = phase == phase_mask
            if nt == 1:
                density = density_nt_phase(z.reshape(1, -1), nt, nbins, hist_range, mask.reshape(1, -1))
            else:
                density = density_nt_phase(z, nt, nbins, hist_range, mask)
        else:
            density = density_nt(z, nt, nbins, hist_range) 
        density = normalize_density(
            df,
            density,
            z,
            norm,
            phase_mask,
            mode,
            zbin,
            start,
            bins,
            nt,
            phase,
        )
        if absval:
            density /= 2
        if nt == 1:
            density = density.flatten()
        err = None
        if std:
            err = np.std(density, axis = 0)
        if time_avg:
            density = np.mean(density, axis = 0)
        if auto_range:
            mask = density > 0
            density = density[mask]
            zbin = zbin[mask]
            if std:
                err = err[mask]
            
        return density, zbin, err

def normalize_density(
    df,
    density,
    z,
    norm,
    phase_mask,
    mode,
    zbin,
    start,
    bins,
    nt,
    phase,
):

    nt = density.shape[0]
    nbins = bins.shape[0] - 1
    if norm == 'count':
        return density
    if norm == "prob" or norm == "percent":
        if mode == "atom":
            n = df["natom"] - df["nmetal"]
        else:
            n = df["nmolecule"]
            
        density /= n

    elif norm == 'mass':
        
        L = get_L(df)
        dz = np.abs(zbin[1] - zbin[0])
        slice_volume = dz * L[0] * L[1] * 1e-10**3
        if mode  == 'atom':
        
            atom_masses = df['atom']['mass'][df['non_metal']] * df['AMU_TO_KG']
            bin_indices = np.digitize(z, bins) - 1 # (nt, natom)

            if phase_mask is not None:
                phase_filter = phase == phase_mask
            else:
                phase_filter = np.ones_like(z, dtype = np.bool)

            if nt == 1:
                bin_mass = atomic_bin_mass(nbins, nt, bin_indices.reshape(1, -1), atom_masses, phase_filter.reshape(1, -1)) 
            else:
                bin_mass = atomic_bin_mass(nbins, nt, bin_indices, atom_masses, phase_filter)
                
            density = bin_mass / slice_volume

        else:        
            mol_weight = df["molecule"]["mass"] * df['AMU_TO_KG']
            density = density * mol_weight / slice_volume
            
    return density


@nb.njit(parallel=True)
def atomic_bin_mass(nbins, nt,  bin_indices, atom_masses, phase_mask):
    bin_mass = np.zeros((nt, nbins))
    for i in nb.prange(nbins):
        bin_filter = bin_indices == i # every atom that belongs to bin i (nt, natom)
        for t in nb.prange(nt):
            bin_i_t = bin_filter[t]
            bin_mass[t][i] = np.sum(atom_masses[phase_mask[t] & bin_i_t]) # map atoms to masses and sum
    return bin_mass

@nb.njit(parallel = True)
def density_nt_phase(z, nt, nbins, hist_range, mask):
    density = np.zeros((nt, nbins))
    for t in nb.prange(nt):
        
        z_t = z[t][mask[t]]
        counts, _ = np.histogram(z_t, bins=nbins, range=hist_range)
        density[t] = counts

    return density 

@nb.njit(parallel = True)
def density_nt(z, nt, nbins, hist_range):
    density = np.zeros((nt, nbins))
    for t in nb.prange(nt):
        
        counts, _ = np.histogram(z[t], bins=nbins, range=hist_range)
        density[t] = counts

    return density 

def phase_frac(df, phase, mode = 'molecule', std = True, start = 0, actime=True, time_avg = True):
    start = get_lag(df, start, actime)
    phase_map = {0: "nliquid", 1: "ngas"}
    n_phase = phase_map[phase]
    n = df[n_phase][start :]
    if mode == 'atom':
        n *= df["natom_per_molecule"]
        frac = n / (df["natom"] - df["nmetal"])
    else:
        frac = n / df["nmolecule"]
    if std:
        err = np.std(frac)
    if time_avg:
        frac = np.mean(frac)
    if std:
        return frac, err
    else:
        return frac
        

def get_t_ns(df, t):
    timesteps = df["timesteps"]
    t0 = timesteps[0]
    dt = df["dt"]
    t_ns = (t * dt + t0) / 1e6

    return t_ns


def get_info(path):
    info = path.split("/")[-1]
    molecule_name = info.split("_")[1]
    temp = info.find("K")
    temp = int(info[temp - 3 : temp])
    return molecule_name, temp

def molecule_size(df):

    z = df['atom']['z'][200][df['reference_molecule_mask']]
    x = df['atom']['x'][200][df['reference_molecule_mask']]
    y = df['atom']['y'][200][df['reference_molecule_mask']]
    
    m = 0
    n = z.shape[0]
    for i in range(n):
        p1 = np.array([x[i], y[i], z[i]])
        for j in range(i, n):
            p2 = np.array([x[j], y[j], z[j]])
            d = np.linalg.norm(p1 - p2)
            if d > m:
                m = d
    return m
