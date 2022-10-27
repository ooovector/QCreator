import numpy as np

def find_normal_point(point1, point2, distance, reverse=False):
    from numpy.linalg import solve
    # parameters of the first line Ax + By + C = 0
    coeff_a = point1[1] - point2[1]
    coeff_b = - (point1[0] - point2[0])
    coeff_c = - point1[0] * coeff_a - point1[1] * coeff_b

    # parameters for the second line Dx + ex + F = 0
    coeff_d = coeff_b
    coeff_e = - coeff_a
    coeff_f = - point1[0] * coeff_d - point1[1] * coeff_e
    if reverse:
        sign = -1
    else:
        sign = 1
    matrix_A = np.asarray([[sign * coeff_a, sign * coeff_b], [coeff_d, coeff_e]])
    vec_b = np.asarray([- sign * coeff_c + distance * np.sqrt(coeff_a ** 2 + coeff_b ** 2), -coeff_f])

    point0 = solve(matrix_A, vec_b)
    return tuple(point0)