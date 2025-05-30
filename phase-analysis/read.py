import numpy as np
import pandas as pd
import numba as nb
import mmap
import re
import os
from joblib import delayed
from ParallelTqdm import ParallelTqdm as Parallel
import gc
from tqdm import tqdm
from ovito.io import import_file
# from pqdm.processes import pqdm
import utils
import correlation

"""
Atomic mass dictionary must be in type order!
"""


def read_mmap_parallel(path, data_indices, natom, nfield):
    nt = data_indices.shape[0]
    data = np.zeros((nt, natom, nfield), dtype=np.float32)
    # get the length of the mmap
    length = np.int64(data_indices[-1, 1] - data_indices[0, 0])
    # offset the beginning of the mmap to the start of the first timestep
    offset = data_indices[0, 0]
    pagesize = mmap.ALLOCATIONGRANULARITY
    offset_pad = offset % pagesize  # pad the offset to the prior page boundary

    # if at beginning of file, offset will be 0 and length will be to last data index
    if offset < pagesize:
        offset = 0
        offset_pad = 0
        length = data_indices[-1, 1]

    # otherwise, shift back to the prior page boundary
    else:
        offset -= offset_pad

    length += offset_pad

    # convert absolute data indices to relative to the offset
    data_indices = data_indices - offset

    if length < 0:  # if last timestep in the file, read the rest of the file
        length = 0
    with open(path, mode="r+") as fi:
        with mmap.mmap(
            fi.fileno(), length=length, offset=offset, access=mmap.ACCESS_READ
        ) as fii:
            for t in range(nt):
                # if last timestep, take data through the end
                if length == 0 and t == nt - 1:
                    data_t = fii[data_indices[t][0] : -1]

                else:
                    data_t = fii[data_indices[t][0] : data_indices[t][1]]

                data_t = np.array(data_t.split(), dtype=np.float32).reshape(
                    natom, nfield
                )
                sort = np.argsort(data_t[:, 0])  # sort on atom_id
                data[t] = data_t[sort]
                del data_t
                del sort
                gc.collect()

    gc.collect()
    return data


def read_all_data(data_path, atomic_masses, nt_limit=None, to_print = False):
    length = 0  # memory map the entire file

    if to_print:
        print("\nReading Data...\n")

    with open(data_path, mode="r+") as fi:
        with mmap.mmap(fi.fileno(), length=length, access=mmap.ACCESS_WRITE) as fii:
            # Define the signatures sections of the data file
            fields_sig = b"ITEM: ATOMS"
            natom_sig = b"ITEM: NUMBER OF ATOMS"
            timestep_sig = b"ITEM: TIMESTEP\n"

            bounds_sig = b"ITEM: BOX BOUNDS"

            # natom_len = len(natom_sig)
            timestep_len = len(timestep_sig)
            # bounds_len = len(bounds_sig)

            # Read the box bounds info
            bounds_start = fii.find(bounds_sig)
            bounds_end = fii.find(b"\n", bounds_start)
            # bounds_type = fii[bounds_start + bounds_len : bounds_end].split()[0]
            bounds = np.zeros((3, 2))
            fii.seek(bounds_end + 1)

            for i in range(3):
                bounds[i] = np.array(fii.readline().split(), dtype=np.float16)

            # Read the fields
            fields_start = fii.find(fields_sig)
            fields_end = fii.find(b"\n", fields_start)
            fields = fii[fields_start + len(fields_sig) : fields_end].split()
            fields_length = fields_end - fields_start
            nfield = len(fields)

            # Read the number of atoms
            natoms_start = fii.find(natom_sig)
            natoms_end = fii.find(b"\n", natoms_start)
            fii.seek(natoms_end + 1)
            natom = np.int32(fii.readline())

            # Find all fields headers to get the data indices
            all_fields = re.compile(fields_sig).finditer(fii)
            all_fields = np.array(
                [[m.start(), m.end()] for m in all_fields], dtype=np.int64
            )
            all_fields_start = all_fields[:, 0]
            all_fields_end = all_fields[:, 1]

            # Find the timesteps
            timestep_indices_re = re.compile(timestep_sig).finditer(fii)
            timestep_indices = np.array(
                [m.end() for m in timestep_indices_re], dtype=np.int64
            )
            nt = timestep_indices.shape[0]

            # set a limit on the number of timesteps to read
            if nt_limit is not None:
                nt = nt_limit
                all_fields_start = all_fields_start[:nt]

            # read the timestep values
            timesteps = np.zeros(nt, dtype=np.int32)
            for i, t_index in enumerate(timestep_indices):
                fii.seek(t_index)
                timesteps[i] = np.int32(fii.readline())
                if i == nt - 1:
                    break

            # create the data indices for each timestep
            data_indices = np.empty((nt, 2), dtype=np.int64)
            data_indices[:, 0] = all_fields_start + fields_length
            data_indices[:-1, 1] = timestep_indices[1:nt] - timestep_len
            data_indices[-1, 1] = -1

            if nt_limit is not None:
                data_indices[-1, 1] = timestep_indices[nt] - timestep_len

    # split the nt indices as equally as possible into ncpus
    ncpus = os.cpu_count()
    data_indices = np.array_split(data_indices, ncpus)

    # process the data in parallel
    disable = False if to_print else True
    all_data = np.concatenate(
        [
            data
            for data in Parallel(n_jobs=ncpus, total_tasks=ncpus, disable_progressbar=disable)(
                delayed(read_mmap_parallel)(data_path, chunk, natom, nfield)
                for chunk in data_indices
            )
        ]
    ).astype(np.float32)

    # args = [[data_path, chunk, natom, nfield] for chunk in data_indices]
    # all_data = np.concatenate([data for data in pqdm(args, read_mmap_parallel, n_jobs=ncpus)]).astype(
    #     np.float32
    # )

    fields = [field.decode("utf-8") for field in fields]
    atomic_masses_arr = np.array(list(atomic_masses.values()))

    df = {"atom": {}, "molecule": {}, "bounds": bounds}
    df_atom = df["atom"]

    df_atom["x"] = np.empty((nt, natom), dtype=np.float32)
    df_atom["y"] = np.empty((nt, natom), dtype=np.float32)
    df_atom["z"] = np.empty((nt, natom), dtype=np.float32)
    df_atom["vx"] = np.empty((nt, natom), dtype=np.float32)
    df_atom["vy"] = np.empty((nt, natom), dtype=np.float32)
    df_atom["vz"] = np.empty((nt, natom), dtype=np.float32)
    df_atom["lt"] = np.empty((nt, natom), dtype=np.float32)
    df_atom["ke"] = np.empty((nt, natom), dtype=np.float32)
    df_atom["pe"] = np.empty((nt, natom), dtype=np.float32)
    df_atom["id"] = np.empty((natom), dtype=np.int32)
    df_atom["molecule_id"] = np.empty((natom), dtype=np.int32)
    df_atom["type"] = np.empty((natom), dtype=np.int32)

    for i, field in enumerate(fields):
        if field in ["ix", "iy", "iz"]:
            continue

        elif field == "c_ld[1]" or field == "q":
            continue

        elif field == "c_ld[2]":
            df_atom["lt"] = all_data[:, :, i].astype(np.float32)

        elif field == "c_pe":
            df_atom["pe"] = all_data[:, :, i].astype(np.float32)

        elif field == "c_ke":
            df_atom["ke"] = all_data[:, :, i].astype(np.float32)

        elif field == "mol":
            df_atom["molecule_id"] = all_data[0, :, i].astype(np.int32)

        elif field in ["id", "type"]:
            df_atom[field] = all_data[0, :, i].astype(np.int32)

        else:
            df_atom[field] = all_data[:, :, i].astype(np.float32)

    # df_atom["mass"] = (
    #     atom_map(atomic_masses_arr, df_atom["type"].flatten())
    #     .reshape(nt, natom)
    #     .astype(np.float32)
    # )

    df["timesteps"] = timesteps
    df_atom["mass"] = atom_map(atomic_masses_arr, df_atom["type"]).astype(np.float32)
    df_atom["speed"] = np.sqrt(
        df_atom["vx"] ** 2 + df_atom["vy"] ** 2 + df_atom["vz"] ** 2
    )
    df_atom["molecule_id"] -= 1
    df_atom["id"] -= 1
    molecule_name, temp = utils.get_info(data_path)
    df["molecule_name"] = molecule_name
    df["temp"] = temp

    del all_data
    del fields
    del data_indices
    del timesteps
    del all_fields_start
    del all_fields_end
    del timestep_indices
    del all_fields
    gc.collect()

    return df


def read_data(data_path, atomic_masses, nt_limit=None, to_print = False):
    length = 0  # memory map the entire file

    if to_print:
        print("\nReading Data...\n")

    with open(data_path, mode="r+") as fi:
        with mmap.mmap(fi.fileno(), length=length, access=mmap.ACCESS_WRITE) as fii:
            # Define the signatures sections of the data file

            natom_sig = b"ITEM: NUMBER OF ATOMS"
            timestep_sig = b"ITEM: TIMESTEP\n"
            bounds_sig = b"ITEM: BOX BOUNDS"

            # Read the box bounds info
            bounds_start = fii.find(bounds_sig)
            bounds_end = fii.find(b"\n", bounds_start)
            bounds = np.zeros((3, 2))
            fii.seek(bounds_end + 1)
            for i in range(3):
                bounds[i] = np.array(fii.readline().split(), dtype=np.float16)
            # Read the number of atoms
            natoms_start = fii.find(natom_sig)
            natoms_end = fii.find(b"\n", natoms_start)
            fii.seek(natoms_end + 1)
            natom = np.int32(fii.readline())
            # Find the timesteps
            timestep_indices_re = re.compile(timestep_sig).finditer(fii)
            timestep_indices = np.array(
                [m.end() for m in timestep_indices_re], dtype=np.int64
            )
            nt = timestep_indices.shape[0]
            # read the timestep values
            timesteps = np.zeros(nt, dtype=np.int32)
            for i, t_index in enumerate(timestep_indices):
                fii.seek(t_index)
                timesteps[i] = np.int32(fii.readline())
                if i == nt - 1:
                    break

    try:
        from pathlib import Path

        pipeline = import_file(Path(data_path))
    except Exception as e:
        print(e)
    atomic_masses_arr = np.array(list(atomic_masses.values()), dtype=np.float32)

    df = {"atom": {}, "molecule": {}, "bounds": bounds}
    df_atom = df["atom"]
    if nt_limit:
        nt = nt_limit
        timesteps = timesteps[:nt]
    df_atom["x"] = np.empty((nt, natom), dtype=np.float32)
    df_atom["y"] = np.empty((nt, natom), dtype=np.float32)
    df_atom["z"] = np.empty((nt, natom), dtype=np.float32)
    df_atom["vx"] = np.empty((nt, natom), dtype=np.float32)
    df_atom["vy"] = np.empty((nt, natom), dtype=np.float32)
    df_atom["vz"] = np.empty((nt, natom), dtype=np.float32)
    df_atom["lt"] = np.empty((nt, natom), dtype=np.float32)
    df_atom["ke"] = np.empty((nt, natom), dtype=np.float32)
    df_atom["pe"] = np.empty((nt, natom), dtype=np.float32)
    df_atom["q"] = np.empty((nt, natom), dtype=np.float32)
    # df_atom["id"] = np.empty((nt, natom), dtype=np.int32)
    # df_atom["molecule_id"] = np.empty((nt, natom), dtype=np.int32)
    # df_atom["type"] = np.empty((nt, natom), dtype=np.int32)
    df_atom["id"] = np.empty((natom), dtype=np.int32)
    df_atom["molecule_id"] = np.empty((natom), dtype=np.int32)
    df_atom["type"] = np.empty((natom), dtype=np.int32)
    t = 0
    disable = False if to_print else True
    for data in tqdm(pipeline.frames, total=nt, disable = disable):
        if t == nt_limit:
            break
        pos = data.particles.position[...].astype(np.float32)
        v = data.particles["Velocity"][...].astype(np.float32)
        ids = data.particles["Particle Identifier"][...]
        sort = np.argsort(ids)

        df_atom["x"][t] = pos[:, 0][sort]
        df_atom["y"][t] = pos[:, 1][sort]
        df_atom["z"][t] = pos[:, 2][sort]
        df_atom["vx"][t] = v[:, 0][sort]
        df_atom["vy"][t] = v[:, 1][sort]
        df_atom["vz"][t] = v[:, 2][sort]
        df_atom["lt"][t] = data.particles["c_ld[2]"][...][sort]
        df_atom["ke"][t] = data.particles["c_ke"][...][sort]
        df_atom["pe"][t] = data.particles["c_pe"][...][sort]
        df_atom["q"][t] = data.particles["Charge"][...][sort]
        # df_atom["type"][t] = data.particles.particle_type[...][sort]
        # df_atom["id"][t] = ids[sort]
        # df_atom["molecule_id"][t] = data.particles["Molecule Identifier"][...][sort]
        if t == 0:
            df_atom["type"] = data.particles.particle_type[...][sort]
            df_atom["id"] = ids[sort]
            df_atom["molecule_id"] = data.particles["Molecule Identifier"][...][sort]
        t += 1
        del sort

    # df_atom["mass"] = (
    #     atom_map(atomic_masses_arr, df_atom["type"].flatten())
    #     .reshape(nt, natom)
    #     .astype(np.float16)
    # )
    df_atom["mass"] = atom_map(atomic_masses_arr, df_atom["type"]).astype(np.float32)

    df_atom["speed"] = np.sqrt(
        df_atom["vx"] ** 2 + df_atom["vy"] ** 2 + df_atom["vz"] ** 2
    )
    # df_atom["timestep"] = np.repeat(timesteps, natom).reshape(nt, natom)
    df["timesteps"] = timesteps
    df["nt"] = timesteps.shape[0]
    df_atom["molecule_id"] -= 1
    df_atom["id"] -= 1
    molecule_name, temp = utils.get_info(data_path)
    df["molecule_name"] = molecule_name
    df["temp"] = temp

    # del timesteps
    del timestep_indices
    gc.collect()

    return df


@nb.njit
def atom_map(atomic_masses_arr, atom_type):
    return atomic_masses_arr[atom_type - 1]


@nb.njit(parallel=True)
def neg_map(positions, offset=0):
    new_positions = np.empty(positions.shape, dtype=np.float32)

    for j in nb.prange(len(positions)):
        position = positions[j]

        for i in nb.prange(len(position)):
            if position[i] < 0:
                new_positions[j, i] = position[i]

            else:
                new_positions[j, i] = -(position[i] - offset)

    return new_positions


@nb.njit(parallel=True)
def pos_map(positions, offset=0):
    new_positions = np.empty(positions.shape, dtype=np.float32)

    for j in nb.prange(len(positions)):
        position = positions[j]

        for i in nb.prange(len(position)):
            if position[i] >= 0:
                new_positions[j, i] = position[i]

            else:
                new_positions[j, i] = -(position[i] - offset)

    return new_positions


def weighted_average(
    values, atom_masses, nt, nmolecule, natom_per_molecule, molecule_mass
):
    A = values.reshape(
        nt, nmolecule, natom_per_molecule, -1
    )  # values grouped by molecule
    # B = atom_masses.reshape(
    #     nt, nmolecule, natom_per_molecule, -1
    # )  # masses grouped by molecule
    B = atom_masses.reshape(nmolecule, natom_per_molecule, -1)

    return np.einsum("ijkl,jkl->ij", A, B) / molecule_mass

    return np.einsum("ijkl,ijkl->ij", A, B) / molecule_mass


def weighted_average_dataframe(position, atom_masses):
    new_position = position_map(position)

    return np.average(new_position, weights=atom_masses)


def molecule_average(values, nt, nmolecule, natom_per_molecule):
    A = values.reshape(
        nt, nmolecule, natom_per_molecule, -1
    )  # values grouped by molecule

    return np.einsum("ijkl->ij", A) / natom_per_molecule


def molecule_sum(values, nt, nmolecule, natom_per_molecule):
    if values.ndim == 1:
        A = values.reshape(nmolecule, natom_per_molecule, -1)
        return np.einsum("ijk->i", A)
    elif values.ndim == 2:
        A = values.reshape(
            nt, nmolecule, natom_per_molecule, -1
        )  # values grouped by molecule

        return np.einsum("ijkl->ij", A)


def position_map(position, nt, nmolecule, natom_per_molecule):
    new_position = position.reshape(nt, nmolecule, natom_per_molecule)

    neg_mask = position < 0
    pos_mask = position >= 0

    count_neg = (neg_mask).sum()
    count_pos = (pos_mask).sum()

    tochange = np.abs(new_position.max(axis=2) - new_position.min(axis=2)) > 25

    if tochange.sum() == 0:
        return position

    pos_to_change = new_position[tochange]
    neg_mask = (pos_to_change < 0).reshape(pos_to_change.shape)
    pos_mask = (pos_to_change >= 0).reshape(pos_to_change.shape)
    count_neg = (neg_mask).sum(axis=1)
    count_pos = (pos_mask).sum(axis=1)
    pos_to_neg = count_neg > count_pos
    neg_to_pos = count_neg < count_pos
    tie = count_neg == count_pos

    new_neg_positions = neg_map(pos_to_change[neg_to_pos])
    new_pos_positions = pos_map(pos_to_change[pos_to_neg])

    pos_to_change[tie] = pos_map(pos_to_change[tie])
    pos_to_change[neg_to_pos] = new_neg_positions
    pos_to_change[pos_to_neg] = new_pos_positions

    new_position[tochange] = pos_to_change

    return new_position.flatten()


def molecule_vars(
    df,
    var,
    non_metal,
    atom_masses,
    nmolecule,
    natom_per_molecule,
    molecule_masses,
    nt,
    ntypes=1,
):
    df_atom = df["atom"]
    if ntypes > 1:
        if var in ["x", "y", "z"]:
            new_position = df_atom.apply(
                lambda dff: weighted_average_dataframe(
                    dff[var].to_numpy(), dff["mass"].to_numpy()
                ),
                include_groups=False,
            )

            return new_position

        elif var in ["ld", "lt", "timestep", "molecule_id"]:
            if var == "timestep" or var == "molecule_id":
                return df_atom.mean().astype(int)

            else:
                return df_atom.mean()

        elif var in ["ke", "pe"]:
            return df_atom.sum()

        elif var in ["vx", "vy", "vz"]:
            return df_atom.apply(
                lambda dff: np.average(
                    dff[var].to_numpy(), weights=dff["mass"].to_numpy()
                )
            )

        else:
            print(f"Variable {var} is not supported. Continuing...")

    values = df_atom[var][:, non_metal]

    if var in ["x", "y", "z"]:
        new_position = position_map(values, nt, nmolecule, natom_per_molecule)

        return weighted_average(
            new_position,
            atom_masses,
            nt,
            nmolecule,
            natom_per_molecule,
            molecule_masses,
        )

    elif var in ["ld", "lt"]:  # take the average across molecules
        return molecule_average(values, nt, nmolecule, natom_per_molecule)

    elif var in ["ke", "pe", "q"]:  # take the sum across molecules
        return molecule_sum(values, nt, nmolecule, natom_per_molecule)

    elif var in ["vx", "vy", "vz"]:
        return weighted_average(
            values,
            atom_masses,
            nt,
            nmolecule,
            natom_per_molecule,
            molecule_masses,
        )

    else:
        print(f"Variable {var} is not supported. Continuing...")


def atom_to_molecule(df, to_print = False):
    df_atom = df["atom"]
    df_molecule = df["molecule"]
    ntypes = df["ntypes"]
    nt = df["nt"]
    nmolecule = df["nmolecule"]
    non_metal = df["non_metal"]
    reference_molecule_mask = df["reference_molecule_mask"]
    natom_per_molecule = df["natom_per_molecule"]

    if to_print:
        print("\nConverting to molecule...", end="")

    if ntypes > 1:
        print(
            "More than one molecule type -- proceeding with pandas dataframe. This will be much slower"
        )
        df = pd.DataFrame(df_atom)
        df_gb = df[non_metal].groupby(["timestep", "molecule_id"])

        for var in df_atom.keys():
            if var in ["id", "speed", "mass", "type", "timestep"]:
                continue

            df_molecule[var] = molecule_vars(
                df_gb,
                var,
                -1,
                -1,
                -1,
                -1,
                -1,
                nt=nt,
                ntypes=ntypes,
            )

        molecule_speeds = np.sqrt(
            df_molecule["vx"] ** 2 + df_molecule["vy"] ** 2 + df_molecule["vz"] ** 2
        )

        df_molecule["speed"] = molecule_speeds

        return df_molecule

    else:
        atom_masses = df_atom["mass"][non_metal]
        molecule_mass = df_atom["mass"][reference_molecule_mask].sum()

        for var in df_atom.keys():
            if var in [
                "id",
                "speed",
                "mass",
                "type",
                "ix",
                "iy",
                "iz",
                "timestep",
                "molecule_id",
            ]:
                continue

            result = molecule_vars(
                df,
                var,
                non_metal,
                atom_masses,
                nmolecule,
                natom_per_molecule,
                molecule_mass,
                nt=nt,
                ntypes=ntypes,
            )

            df_molecule[var] = result
            del result

        molecule_speeds = np.sqrt(
            df_molecule["vx"] ** 2 + df_molecule["vy"] ** 2 + df_molecule["vz"] ** 2
        )

        df_molecule["speed"] = molecule_speeds
        df_molecule["mass"] = molecule_mass

        if to_print:
            print("Done")
        gc.collect()

        return df


def process_data(
    file_in,
    atomic_masses,
    nt_lim=None,
    metal_type=-1,
    reference_molecule=10,
    ntypes=1,
    mode="ovito",
    to_print = False,
):
    gc.collect()
    if mode == "ovito":
        df = read_data(file_in, atomic_masses, nt_limit=nt_lim, to_print=to_print)
    elif mode == "mmap":
        df = read_all_data(file_in, atomic_masses, nt_limit=nt_lim, to_print=to_print)

    df_atom = df["atom"]
    df_molecule = df["molecule"]
    molecule_ids = df_atom["molecule_id"]
    timesteps = df["timesteps"]
    natom = df_atom["id"].shape[0]
    df["natom"] = natom
    df["metal_type"] = metal_type
    df["reference_molecule"] = reference_molecule
    df["reference_molecule_mask"] = molecule_ids == reference_molecule
    df["natom_per_molecule"] = df["reference_molecule_mask"].sum()
    df["atomic_masses"] = atomic_masses
    df["ntypes"] = ntypes

    df_molecule["id"] = np.unique(molecule_ids)
    df["nmolecule"] = df_molecule["id"].shape[0]
    if metal_type != -1:
        df["nmolecule"] -= 1
    df["nt"] = timesteps.shape[0]
    dt = timesteps[1] - timesteps[0]
    df["dt"] = dt

    if metal_type != -1:
        df["metal_molecule"] = molecule_ids[(df_atom["type"] == df["metal_type"])][0]
        df["metal_mask"] = molecule_ids == df["metal_molecule"]
        df["non_metal"] = ~df["metal_mask"]
        df["nmetal"] = df_atom["id"][df["metal_mask"]].shape[0]

        df["lower_surface"] = df_atom["z"][0][df["metal_mask"]].min()
        df["upper_surface"] = df_atom["z"][0][df["metal_mask"]].max()
        df["offset"] = df["upper_surface"] + df["lower_surface"]

    else:
        df["metal_mask"] = np.zeros(natom, dtype=bool)
        df["non_metal"] = np.ones(natom, dtype=bool)
        df["nmetal"] = 0
        df["offset"] = 0

    df = atom_to_molecule(df, to_print = to_print)
    df_atom["lower_mask"] = df_atom["z"] < 0
    df_molecule["lower_mask"] = df_molecule["z"] < 0
    pe = np.sum(df["molecule"]["pe"], axis=1)
    _, start, _ = correlation.actime(pe)
    df['actime'] = start
    df["AMU_TO_KG"] = 1.66054e-27

    return df
