import numpy as np
import scipy.linalg


def harmonic_bond_potential_(r: float, r_0: float, k: float) -> float:
    """The potential from a single atom pair with distance r and equilibrium
    distance r_o.
    Note: r could be squared or not but r_0 has to be adjusted accordingly.
    """
    return 0.5 * k * (r - r_0) ** 2


def harmonic_bond_force_(x: np.array, y: np.array, r_0: float, k: float) -> np.array:
    """The force from a single atom pair (x,y) with equilibrium distance r_o.
    This should just be the derivative of harmonic_bond_potential_.
    """
    r = np.linalg.norm(x - y)
    return k * (r - r_0) * (x - y) / r


def get_angle(a: np.array, b: np.array, c: np.array):
    v1 = b - a
    v2 = b - c
    cross = np.cross(v1, v2)
    cross_norm = max(scipy.linalg.norm(cross), 1e-6)
    v1_norm2 = scipy.linalg.norm(v1) ** 2
    v2_norm2 = scipy.linalg.norm(v2) ** 2
    dot = np.dot(v1, v2)
    cosine = min(max(dot / np.sqrt(v1_norm2 * v2_norm2), -1), 1)
    theta = np.arccos(cosine)
    return theta, v1, v2, cross


def harmonic_angle_potential_(a: np.array, b: np.array, c: np.array, theta_0: float, k: float) -> float:
    """The potential from a single molecular angle theta and equilibrium
    angle theta_0."""
    theta, _, _, _ = get_angle(a, b, c)
    return 0.5 * k * (theta - theta_0) ** 2


def harmonic_angle_force_(a: np.array, b: np.array, c: np.array,
                          theta_0: float, k: float) -> np.array:
    """The force from a single molecular angle theta and equilibrium angle.
    Reference: https://github.com/openmm/openmm/blob/master/platforms/common/src/kernels/angleForce.cc"""
    theta, v1, v2, cross = get_angle(a, b, c)
    cross_norm = max(scipy.linalg.norm(cross), 1e-6)
    # Compute the force
    dEdAngle = k * (theta - theta_0)
    force1 = np.cross(v1, cross) * dEdAngle / (scipy.linalg.norm(
        v1) ** 2 * cross_norm)  # https://minesparis-psl.hal.science/hal-00924263/document for derivation
    force2 = np.cross(cross, v2) * dEdAngle / (scipy.linalg.norm(
        v2) ** 2 * cross_norm)  # Bernard Monasse, Frédéric Boussinot. Determination of Forces from a Potential in Molecular Dynamics. 2014. ￿hal-00924263￿
    force3 = -force1 - force2
    return np.array([force1, force2, force3])
