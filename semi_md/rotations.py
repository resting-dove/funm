import numpy as np
import scipy


def get_rotation(v, u) -> np.ndarray:
    '''Compute an approximate rotation between two vectors.
    '''
    F = u.T @ np.array(v)  # .T turns Quantity into np.array. array @ Quantity not defined
    U, S, Vh = scipy.linalg.svd(F)
    R = Vh.T @ U.T
    return R


def get_Rx_per_molecule(r_current, r_old, openff_topology) -> (scipy.sparse.csr_array, scipy.sparse.csr_array):
    RxLarge = scipy.sparse.coo_array((r_current.size, r_current.size), dtype=r_current.dtype)
    Rx_invLarge = scipy.sparse.coo_array((r_current.size, r_current.size), dtype=r_current.dtype)
    zeros = np.zeros(openff_topology.n_atoms)
    for molecule in openff_topology.molecules:
        atoms = molecule.atoms
        relevant_atoms = [openff_topology.atom_index(atom) for atom in atoms]
        RxRot = scipy.spatial.transform.Rotation.align_vectors(r_old[relevant_atoms], r_current[relevant_atoms])[0]
        zeros = 0 * zeros
        zeros[relevant_atoms] = 1
        RxLarge += scipy.sparse.kron(np.diag(zeros), RxRot.as_matrix(), format="coo")
        Rx_invLarge += scipy.sparse.kron(np.diag(zeros), RxRot.inv().as_matrix(), format="coo")
    return RxLarge.tocsr(), Rx_invLarge.tocsr()

def get_Rx(r_current, r_old, openff_topology) -> (scipy.sparse.csr_array, scipy.sparse.csr_array):
    relevant_atoms = list(range(openff_topology.n_atoms))
    # Rx = scipy.sparse.csr_array(get_rotation(r_current[relevant_atoms], r_old[relevant_atoms]))
    RxRot = scipy.spatial.transform.Rotation.align_vectors(r_old[relevant_atoms], r_current[relevant_atoms])[0]
    RxLarge: scipy.sparse.csr_array = scipy.sparse.block_diag([RxRot.as_matrix()] * len(r_current), format="csr")
    Rx_invLarge: scipy.sparse.csr_array = scipy.sparse.block_diag([RxRot.inv().as_matrix()] * len(r_current), format="csr")
    # Rx_invLarge = RxLarge.T
    return RxLarge.tocsr(), Rx_invLarge.tocsr()
    # return Rx_invLarge.tocsr(), RxLarge.tocsr()
    # return scipy.sparse.eye_array(*RxLarge.shape, format="csr"), scipy.sparse.eye_array(*RxLarge.shape, format="csr")


def get_Rx_svd(r_current, r_old, openff_topology) -> (scipy.sparse.csr_array, scipy.sparse.csr_array):
    relevant_atoms = list(range(openff_topology.n_atoms))
    Rx = scipy.sparse.csr_array(get_rotation(r_current[relevant_atoms], r_old[relevant_atoms]))
    RxLarge: scipy.sparse.csr_array = scipy.sparse.block_diag([Rx] * len(r_current), format="csr")
    Rx_invLarge: scipy.sparse.csr_array = scipy.sparse.block_diag([Rx.T] * len(r_current), format="csr")
    return RxLarge, Rx_invLarge