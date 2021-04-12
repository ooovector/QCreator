import numpy as np


def calculate_total_length(points):
    i0, j0 = points[0]
    length = 0
    for i, j in points[1:]:
        length += np.sqrt((i - i0) ** 2 + (j - j0) ** 2)
    return length


def parametric_equation_of_line(x0, y0, x1, y1, t):
    ax = x1 - x0
    ay = y1 - y0

    x = x0 + ax * t
    y = y0 + ay * t

    return x, y


def segment_points(segment):
    x0 = segment['startpoint'][0]
    y0 = segment['startpoint'][1]
    x1 = segment['endpoint'][0]
    y1 = segment['endpoint'][1]

    return x0, y0, x1, y1
