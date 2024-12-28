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
    # RxLarge = scipy.sparse.coo_array((r_current.size, r_current.size), dtype=r_current.dtype)
    # Rx_invLarge = scipy.sparse.coo_array((r_current.size, r_current.size), dtype=r_current.dtype)
    # zeros = np.zeros(openff_topology.n_atoms)
    # for molecule in openff_topology.molecules:
    #     atoms = molecule.atoms
    #     relevant_atoms = [openff_topology.atom_index(atom) for atom in atoms]
    #     RxRot = scipy.spatial.transform.Rotation.align_vectors(r_old[relevant_atoms], r_current[relevant_atoms])[0]
    #     # Rx = get_rotation(r_current[relevant_atoms], r_old[relevant_atoms])
    #     zeros = 0 * zeros
    #     zeros[relevant_atoms] = 1
    #     RxLarge += scipy.sparse.kron(np.diag(zeros), RxRot.as_matrix(), format="coo")
    #     Rx_invLarge += scipy.sparse.kron(np.diag(zeros), RxRot.inv().as_matrix(), format="coo")
    # Rx_invLarge = RxLarge.T
    relevant_atoms = list(range(openff_topology.n_atoms))
    # Rx = scipy.sparse.csr_array(get_rotation(r_current[relevant_atoms], r_old[relevant_atoms]))
    RxRot = scipy.spatial.transform.Rotation.align_vectors(r_old[relevant_atoms], r_current[relevant_atoms])[0]
    RxLarge: scipy.sparse.csr_array = scipy.sparse.block_diag([RxRot.as_matrix()] * len(r_current), format="csr")
    Rx_invLarge: scipy.sparse.csr_array = scipy.sparse.block_diag([RxRot.inv().as_matrix()] * len(r_current), format="csr")
    # Rx_invLarge = RxLarge.T
    return RxLarge.tocsr(), Rx_invLarge.tocsr()
    # return scipy.sparse.eye_array(*RxLarge.shape, format="csr"), scipy.sparse.eye_array(*RxLarge.shape, format="csr")