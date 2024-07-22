import numpy as np
import lmp_class as lmpc
import pandas as pd
import numba as nb
import importlib

importlib.reload(lmpc)
import lmp_class as lmpc


##### PARALLELIZE
def read_data(data_path, atomic_masses, Nt=1001, limit=None):

    if limit is not None:

        Nt = limit

    data = lmpc.obj(filename=data_path)
    i = 0
    Natom = data.nratoms
    atomic_masses_arr = np.array(list(atomic_masses.values()))

    atom_x = np.zeros((Nt, Natom))
    atom_y = np.zeros((Nt, Natom))
    atom_z = np.zeros((Nt, Natom))
    atom_vx = np.zeros((Nt, Natom))
    atom_vy = np.zeros((Nt, Natom))
    atom_vz = np.zeros((Nt, Natom))
    atom_id = np.zeros((Nt, Natom), dtype=int)

    atom_speed = np.zeros((Nt, Natom))
    molecular_id = np.zeros((Nt, Natom), dtype=int)
    atom_type = np.zeros((Nt, Natom), dtype=int)
    atom_ke = np.zeros((Nt, Natom))
    atom_pe = np.zeros((Nt, Natom))
    atom_ld = np.zeros((Nt, Natom))  # 9.5 angstrom (ave/sphere/atom) mass density
    atom_lt = np.zeros((Nt, Natom))  # local temperature

    while True:

        if data.errorflag != 0:

            break

        v = data.velocity

        pos = data.coordinate
        atom_x[i] = pos[:, 0]
        atom_y[i] = pos[:, 1]
        atom_z[i] = pos[:, 2]

        atom_vx[i] = v[:, 0]
        atom_vy[i] = v[:, 1]
        atom_vz[i] = v[:, 2]

        atom_id[i] = data.aid.flatten()

        atom_speed[i] = np.linalg.norm(v, axis=1)
        molecular_id[i] = data.molecule.flatten()
        atom_type[i] = data.type.flatten()
        atom_pe[i] = data.extras[:, 0]
        atom_ke[i] = data.extras[:, 1]
        atom_ld[i] = data.extras[:, 2]
        atom_lt[i] = data.extras[:, 3]

        i += 1
        data.next_step()

        if i == limit:

            break

    timesteps = np.repeat(np.arange(0, Nt), Natom)
    df_atom = {
        "atom_type": atom_type.flatten(),
        "atom_x": atom_x.flatten(),
        "atom_y": atom_y.flatten(),
        "atom_z": atom_z.flatten(),
        "atom_vx": atom_vx.flatten(),
        "atom_vy": atom_vy.flatten(),
        "atom_vz": atom_vz.flatten(),
        "molecule_id": molecular_id.flatten(),
        "atom_speed": atom_speed.flatten(),
        "atom_pe": atom_pe.flatten(),
        "atom_ke": atom_ke.flatten(),
        "atom_ld": atom_ld.flatten(),
        "atom_lt": atom_lt.flatten(),
        "timestep": timesteps,
        "atom_mass": atom_map(atomic_masses_arr, atom_type.flatten()),
        "atom_id": atom_id.flatten(),
    }

    return df_atom


@nb.njit(parallel=True)
def atom_map(atomic_masses_arr, atom_type):

    return atomic_masses_arr[atom_type - 1]


@nb.njit(parallel=True)
def neg_map(positions):

    new_positions = np.empty(positions.shape)
    for i in nb.prange(len(positions)):

        if positions[i] < 0:

            new_positions[i] = positions[i]

        else:

            new_positions[i] = -(positions[i] - 10)

    return new_positions


@nb.njit(parallel=True)
def pos_map(positions):

    new_positions = np.empty(positions.shape)
    for i in nb.prange(len(positions)):

        if positions[i] >= 0:

            new_positions[i] = positions[i]

        else:

            new_positions[i] = -(positions[i] - 10)

    return new_positions


@nb.njit(parallel=True)
def neg_map(positions):

    new_positions = np.empty(positions.shape)

    for j in nb.prange(len(positions)):

        position = positions[j]

        for i in nb.prange(len(position)):

            if position[i] < 0:

                new_positions[j, i] = position[i]

            else:

                new_positions[j, i] = -(position[i] - 10)

    return new_positions


@nb.njit(parallel=True)
def pos_map(positions):

    new_positions = np.empty(positions.shape)

    for j in nb.prange(len(positions)):

        position = positions[j]

        for i in nb.prange(len(position)):

            if position[i] >= 0:

                new_positions[j, i] = position[i]

            else:

                new_positions[j, i] = -(position[i] - 10)

    return new_positions


def weighted_average(
    values, atom_masses, Nt, Nmolecule, Natom_per_molecule, molecule_masses
):

    A = values.reshape(
        Nt, Nmolecule - 1, Natom_per_molecule, -1
    )  # values grouped by molecule
    B = atom_masses.reshape(
        Nt, Nmolecule - 1, Natom_per_molecule, -1
    )  # masses grouped by molecule

    return np.einsum("ijkl,ijkl->ij", A, B) / molecule_masses


def weighted_average_dataframe(position, atom_masses):

    new_position = position_map(position)

    return np.average(new_position, weights=atom_masses)


def molecule_average(values, Nt, Nmolecule, Natom_per_molecule):

    A = values.reshape(
        Nt, Nmolecule - 1, Natom_per_molecule, -1
    )  # values grouped by molecule

    return np.einsum("ijkl->ij", A) / Natom_per_molecule


def molecule_sum(values, Nt, Nmolecule, Natom_per_molecule):

    A = values.reshape(
        Nt, Nmolecule - 1, Natom_per_molecule, -1
    )  # values grouped by molecule

    return np.einsum("ijkl->ij", A)


# @nb.njit(parallel=True)
def position_map(position):

    new_position = position.reshape(1001, 101, 44)

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
    Ntie = tie.sum()

    new_neg_positions = neg_map(pos_to_change[neg_to_pos])
    new_pos_positions = pos_map(pos_to_change[pos_to_neg])

    pos_prob = np.random.rand(Ntie)
    neg_prob = 1 - pos_prob
    mask = pos_prob >= neg_prob

    new_neg = neg_map(pos_to_change[tie][~mask])
    new_pos = pos_map(pos_to_change[tie][mask])

    pos_to_change[tie][~mask] = new_neg
    pos_to_change[tie][mask] = new_pos
    pos_to_change[neg_to_pos] = new_neg_positions
    pos_to_change[pos_to_neg] = new_pos_positions

    new_position[tochange] = pos_to_change

    return new_position.flatten()


def molecule_vars(
    df_atom,
    var,
    metal_mask,
    atom_masses,
    Nmolecule,
    Natom_per_molecule,
    molecule_masses,
    Nt,
    Ntypes=1,
):

    if Ntypes > 1:

        if var in ["atom_x", "atom_y", "atom_z"]:

            new_position = df_atom.apply(
                lambda dff: weighted_average_dataframe(
                    dff[var].to_numpy(), dff["atom_mass"].to_numpy()
                ),
                include_groups=False,
            )

            return new_position

        elif var in ["atom_ld", "atom_lt", "timestep", "molecule_id"]:

            if var == "timestep" or var == "molecule_id":

                return df_atom.mean().astype(int)

            else:

                return df_atom.mean()

        elif var in ["atom_ke", "atom_pe"]:

            return df_atom.sum()

        elif var in ["atom_vx", "atom_vy", "atom_vz"]:

            return df_atom.apply(
                lambda dff: np.average(
                    dff[var].to_numpy(), weights=dff["atom_mass"].to_numpy()
                )
            )

        else:

            print(f"Variable {var} is not supported. Continuing...")

    values = df_atom[var][metal_mask]

    if var in ["atom_x", "atom_y", "atom_z"]:

        new_position = position_map(values)

        return weighted_average(
            new_position,
            atom_masses,
            Nt,
            Nmolecule,
            Natom_per_molecule,
            molecule_masses,
        )

    elif var in [
        "atom_ld",
        "atom_lt",
        "timestep",
        "molecule_id",
    ]:  # take the average across molecules

        if var == "timestep" or var == "molecule_id":

            return molecule_average(values, Nt, Nmolecule, Natom_per_molecule).astype(
                int
            )

        else:

            return molecule_average(values, Nt, Nmolecule, Natom_per_molecule)

    elif var in ["atom_ke", "atom_pe", "atom_q"]:  # take the sum across molecules

        return molecule_sum(values, Nt, Nmolecule, Natom_per_molecule)

    elif var in ["atom_vx", "atom_vy", "atom_vz"]:

        return weighted_average(
            values,
            atom_masses,
            Nt,
            Nmolecule,
            Natom_per_molecule,
            molecule_masses,
        )

    else:

        print(f"Variable {var} is not supported. Continuing...")


def atom_to_molecule(df_atom, metal_type=7, Ntypes=1, reference_molecule=1):

    Nt = np.unique(df_atom["timestep"]).shape[0]
    Nmolecule = np.unique(df_atom["molecule_id"]).shape[0]

    metal_mask = df_atom["atom_type"] != metal_type

    df_molecule = {}

    if Ntypes > 1:

        print(
            "More than one molecule type -- proceeding with pandas dataframe. This will be much slower"
        )
        df = pd.DataFrame(df_atom)
        df_gb = df[metal_mask].groupby(["timestep", "molecule_id"])

        for var in df_atom.keys():

            if var in ["atom_id", "atom_speed", "atom_mass", "atom_type"]:

                continue

            molecule_var = var

            if "atom" in var:

                molecule_var = "molecule_" + var.split("_")[-1]

            df_molecule[molecule_var] = molecule_vars(
                df_gb,
                var,
                -1,
                -1,
                -1,
                -1,
                -1,
                Nt=Nt,
                Ntypes=Ntypes,
            )

        molecule_speeds = np.sqrt(
            df_molecule["molecule_vx"] ** 2
            + df_molecule["molecule_vy"] ** 2
            + df_molecule["molecule_vz"] ** 2
        )

        df_molecule["molecule_speed"] = molecule_speeds

        return df_molecule

    else:

        reference_molecule_mask = (df_atom["timestep"] == 0) & (
            df_atom["molecule_id"] == reference_molecule
        )
        Natom_per_molecule = reference_molecule_mask.sum()
        atom_masses = df_atom["atom_mass"][metal_mask]
        molecule_masses = molecule_sum(
            df_atom["atom_mass"][metal_mask], Nt, Nmolecule, Natom_per_molecule
        )
        df_molecule["molecule_mass"] = molecule_masses

        for var in df_atom.keys():

            if var in [
                "atom_id",
                "atom_speed",
                "atom_mass",
                "atom_type",
                "atom_ix",
                "atom_iy",
                "atom_iz",
            ]:

                continue

            molecule_var = var

            if "atom" in var:

                molecule_var = "molecule_" + var.split("_")[-1]

            df_molecule[molecule_var] = molecule_vars(
                df_atom,
                var,
                metal_mask,
                atom_masses,
                Nmolecule,
                Natom_per_molecule,
                molecule_masses,
                Nt=Nt,
                Ntypes=Ntypes,
            )

        molecule_speeds = np.sqrt(
            df_molecule["molecule_vx"] ** 2
            + df_molecule["molecule_vy"] ** 2
            + df_molecule["molecule_vz"] ** 2
        )

        df_molecule["molecule_speed"] = molecule_speeds

        return df_molecule
