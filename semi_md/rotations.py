import numpy as np
import scipy


def get_rotation(v, u) -> np.ndarray:
    '''Compute an approximate rotation between two vectors.
    '''
    F = u.T @ np.array(v)  # .T turns Quantity into np.array. array @ Quantity not defined
    U, S, Vh = scipy.linalg.svd(F)
    R = Vh.T @ U.T
    return R


def get_repeated_rot(r_current, r_old, base_atom_idx, n_atoms, topology) -> (
        scipy.sparse.coo_array, scipy.sparse.coo_array, np.ndarray):
    '''Compute the rotation between the old positions r_old and the current positions r_current
    with respect to atoms bonded to the atom with base_atom_idx. Then insert that rotation block wise
    into a large rotation matrix for the entire molecule.
    '''
    RxLarge = np.zeros((n_atoms, n_atoms))
    Rx_invLarge = np.zeros((n_atoms, n_atoms))

    relevant_atoms = [base_atom_idx] + [bonded_atom.molecule_atom_index for
                                        bonded_atom in topology.atom(base_atom_idx).bonded_atoms]
    Rx = get_rotation(r_current[relevant_atoms], r_old[relevant_atoms])
    RxLarge = scipy.sparse.kron(np.eye(n_atoms), Rx, format="coo")
    Rx_invLarge = scipy.sparse.kron(np.eye(n_atoms), Rx.T, format="coo")
    return RxLarge, Rx_invLarge, Rx


def get_Rx(r_current, r_old, openff_topology) -> (scipy.sparse.csr_array, scipy.sparse.csr_array):
    RxLarge = scipy.sparse.lil_array((r_current.size, r_current.size), dtype=np.float64)
    Rx_invLarge = scipy.sparse.lil_array((r_current.size, r_current.size), dtype=np.float64)
    # for i in range(openff_topology.n_atoms):
    #     relevant_atoms = list(range(openff_topology.n_atoms))
    #     Rx = get_rotation(r_current[relevant_atoms], r_old[relevant_atoms])
    #     RxLarge[i * 3: (i + 1) * 3, i * 3: (i + 1) * 3] = Rx
    #     Rx_invLarge[i * 3: (i + 1) * 3, i * 3: (i + 1) * 3] = Rx.T
    relevant_atoms = list(range(openff_topology.n_atoms))
    Rx = scipy.sparse.csr_array(get_rotation(r_current[relevant_atoms], r_old[relevant_atoms]))
    RxLarge: scipy.sparse.csr_array = scipy.sparse.block_diag([Rx] * len(r_current), format="csr")
    Rx_invLarge = RxLarge.T
    return RxLarge.tocsr(), Rx_invLarge.tocsr()
