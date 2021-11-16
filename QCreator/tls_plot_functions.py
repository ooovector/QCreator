import numpy as np


def potential_1d(phi, e_l, e_j, alpha):
    potential = 1 / 2 * e_l[0][0] * phi ** 2
    for jj_id, e_j in enumerate(e_j):
        potential += e_j * (1 - np.cos(alpha[jj_id][0] * phi))
    return potential


def potential_2d(phi_1, phi_2, e_l, e_j, alpha):
    xx, yy = np.meshgrid(phi_1, phi_2)
    potential = 1 / 2 * e_l[0][0] * xx ** 2 + 1 / 2 * e_l[1][1] * yy ** 2
    for jj_id, e_j in enumerate(e_j):
        potential += e_j * (1 - np.cos(alpha[jj_id][0] * xx + alpha[jj_id][1] * yy))
    return xx, yy, potential
