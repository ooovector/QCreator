import numpy as np
from typing import List, Tuple, Mapping, Union, Iterable, Dict
from .cpw import CPW
from copy import deepcopy
from .core import LayerConfiguration

from scipy import optimize as opt

def meander_creation(name: str, initial_position: Tuple[(float, float)], w: float, s: float, g: float,
                     orientation: float,
                     meander_length: float,
                     length_left: float, length_right: float, first_step_orientation: float,
                     meander_orientation: float, end_point: Tuple[(float, float)], end_orientation: float,
                     layer_configuration: LayerConfiguration, meander_type: str = 'round', r: float=None):

    # make small indent from the starting point
    if r is None:
        bend_radius = 40
    else:
        bend_radius = r
    points = [tuple(initial_position)]

    indent_length = bend_radius * 1.25  # (w+2*s)*2

    points.append((initial_position[0] + np.cos(orientation + np.pi) * indent_length,
                   initial_position[1] + np.sin(orientation + np.pi) * indent_length))

    if first_step_orientation =='left':
        indent_first = length_left
        points.append((points[-1][0] + np.cos(meander_orientation) * indent_first,
                       points[-1][1] + np.sin(meander_orientation) * indent_first))
    elif first_step_orientation =='right':
        indent_first = length_right
        points.append((points[-1][0] + np.cos(meander_orientation+np.pi) * indent_first,
                       points[-1][1] + np.sin(meander_orientation+np.pi) * indent_first))
    else:
        raise print('meander first step orientation is wrong')


    if end_point is not None:
        end_point_indent = [(end_point[0] + np.cos(end_orientation + np.pi) * indent_length,
                             end_point[1] + np.sin(end_orientation + np.pi) * indent_length)]
    else:
        end_point_indent = []

    rendering_meander = CPW(name=name, points=deepcopy(points + end_point_indent), w=w, s=s, g=g,
                            layer_configuration=layer_configuration, r=bend_radius,
                            corner_type=meander_type)

    if rendering_meander.length > meander_length:
        print('length is too small, change first step meander length to %f' %(meander_length-indent_length))

    # lets fill the whole rectangular
    default_bend_diameter = bend_radius * 2 + 5
    if meander_type == 'flush':
        meander_step = length_left + length_right + default_bend_diameter
    elif meander_type == 'round':
        meander_step = length_left - bend_radius + length_right - bend_radius + \
                       (default_bend_diameter - 2 * bend_radius) + np.pi * bend_radius

    N = int((meander_length - rendering_meander.length) // meander_step)

    if N == 0 and (meander_length-rendering_meander.length)==0:
       return rendering_meander
    # subtract one to connect to the end point, here i assume that the distance
    # from the end of the meander is less than the meander step
    # if end_orientation is not None:
    #     N = N - 1
    if first_step_orientation == 'right':
        N=N+1
        i=2
    else:
        i = 1
    # make a meander
    while i <= N:
        list = [(points[-1][0] + np.sin(meander_orientation) * default_bend_diameter,
                 points[-1][1] - np.cos(meander_orientation) * default_bend_diameter)]
        points.extend(list)
        list = [(points[-1][0] + np.sin(meander_orientation - np.pi / 2 + np.pi * ((i - 1) % 2)) * (
                length_left + length_right),
                 points[-1][1] - np.cos(meander_orientation - np.pi / 2 + np.pi * ((i - 1) % 2)) * (
                         length_left + length_right))]
        points.extend(list)
        i = i + 1

    # if end_orientation is not None:
    #     i = 0
    # else:
    rendering_meander = CPW(name=name, points=deepcopy(points+end_point_indent), w=w, s=s, g=g,
                            layer_configuration=layer_configuration, r=bend_radius,
                            corner_type=meander_type)

    tail = np.abs(np.floor(rendering_meander.length - meander_length))

    if tail < np.pi * bend_radius / 2:
        list = [(points[-1][0] + np.sin(meander_orientation - np.pi / 2 + np.pi * ((i) % 2)) * tail,
                 points[-1][1] - np.cos(meander_orientation - np.pi / 2 + np.pi * ((i) % 2)) * tail)]
        points.extend(list)
        rendering_meander = CPW(name=name, points=deepcopy(points+end_point_indent), w=w, s=s, g=g,
                                layer_configuration=layer_configuration, r=bend_radius,
                                corner_type=meander_type)
    elif tail < (default_bend_diameter - 2 * bend_radius) + np.pi * bend_radius + 1:
        list = [(points[-1][0] + np.sin(meander_orientation) * (bend_radius + 1),
                 points[-1][1] - np.cos(meander_orientation) * (bend_radius + 1))]
        points.extend(list)

        rendering_meander = CPW(name=name, points=deepcopy(points+end_point_indent), w=w, s=s, g=g,
                                layer_configuration=layer_configuration, r=bend_radius,
                                corner_type=meander_type)
        tail = np.abs(np.floor(rendering_meander.length - meander_length))

        list = [(points[-1][0] + np.sin(meander_orientation) * tail,
                 points[-1][1] - np.cos(meander_orientation) * tail)]
        points.extend(list)
        rendering_meander = CPW(name=name, points=deepcopy(points+end_point_indent), w=w, s=s, g=g,
                                layer_configuration=layer_configuration, r=bend_radius,
                                corner_type=meander_type)

    else:

        list = [(points[-1][0] + np.sin(meander_orientation) * default_bend_diameter,
                 points[-1][1] - np.cos(meander_orientation) * default_bend_diameter)]
        points.extend(list)
        rendering_meander = CPW(name=name, points=deepcopy(points+end_point_indent), w=w, s=s, g=g,
                                layer_configuration=layer_configuration, r=bend_radius,
                                corner_type=meander_type)

        error = np.abs(rendering_meander.length - meander_length)

        list = [(points[-1][0] + np.sin(meander_orientation - np.pi / 2 + np.pi * ((i - 1) % 2)) *
                 error,
                 points[-1][1] - np.cos(meander_orientation - np.pi / 2 + np.pi * ((i - 1) % 2)) *
                 error)]
        points.extend(list)
        rendering_meander = CPW(name=name, points=deepcopy(points+end_point_indent), w=w, s=s, g=g,
                                layer_configuration=layer_configuration, r=bend_radius,
                                corner_type=meander_type)

        error = np.abs(rendering_meander.length - meander_length)
        list = [(points[-1][0] + np.sin(meander_orientation - np.pi / 2 + np.pi * ((i - 1) % 2)) *
                 (error),
                 points[-1][1] - np.cos(meander_orientation - np.pi / 2 + np.pi * ((i - 1) % 2)) *
                 (error))]
        points.extend(list)
        rendering_meander = CPW(name=name, points=deepcopy(points+end_point_indent), w=w, s=s, g=g,
                                layer_configuration=layer_configuration, r=bend_radius, orientation2=end_orientation,
                                corner_type=meander_type)

    return rendering_meander


# class CPWMeander:
#     def __init__(self, initial_point: Tuple[(float, float)], w: float, s: float, g: float,
#                  meander_length: float, restricted_scale: float, constant_scale: float,
#                  orientation: float, connector_length: float):
#
#         """
#         Create a coplanar waveguide (CPW) meander.
#         :param initial_point: initial points for a meander
#         :param w: CPW signal conductor
#         :param s: CPW signal-g s
#         :param g:CPW finite g width
#         :param meander_length: total length of the meander
#
#         :param connector_length: length of connectors (default parameter)
#         """
#         self.initial_point = initial_point
#         self.w = w
#         self.g = g
#         self.s = s
#         self.meander_length = meander_length
#         self.constant_scale = constant_scale
#         self.restricted_scale = restricted_scale
#         # self.period = period
#         self.orientation = orientation
#         delta = self.g + self.s + self.w / 2
#         self.connector_length = connector_length
#
#         # rectangle parameters
#         meander_length_eff = self.meander_length - 2 * self.connector_length  # L
#         constant_scale_eff = self.constant_scale - 2 * self.connector_length  # a
#         restricted_scale_eff = self.restricted_scale - 2 * delta  # b
#
#         if meander_length_eff < 0 or constant_scale_eff < 0:
#             raise ValueError('Length of the meander is too short!')
#
#         number_of_curves = (meander_length_eff - constant_scale_eff) // (restricted_scale_eff - 0 * delta) + 1
#         if number_of_curves < 1.:
#             raise ValueError('Length of the meander is too short!')
#
#         scale_eff = (meander_length_eff - constant_scale_eff + 0 * delta * number_of_curves) / number_of_curves
#         x = constant_scale_eff / number_of_curves - 0 * delta
#         #if x <= 4 * delta:
#         #    raise ValueError('Length of the meander is too long or restricted parameters are to small!')
#
#         points_of_meander = [(self.initial_point[0][0], self.initial_point[0][1]),
#                              (self.initial_point[0][0] + self.connector_length * np.cos(self.orientation),
#                               self.initial_point[0][1] + self.connector_length * np.sin(self.orientation))]
#
#         check_length = self.connector_length
#
#         for i in range(int(number_of_curves)):
#             if i % 2 == 0:
#                 points_of_meander.append((points_of_meander[-1][0] - (scale_eff / 2) * np.sin(self.orientation),
#                                           points_of_meander[-1][1] + (scale_eff / 2) * np.cos(self.orientation)))
#                 check_length += scale_eff / 2
#
#                 points_of_meander.append((points_of_meander[-1][0] + x * np.cos(self.orientation),
#                                           points_of_meander[-1][1] + x * np.sin(self.orientation)))
#                 check_length += x
#
#                 points_of_meander.append((points_of_meander[-1][0] + (scale_eff / 2) * np.sin(self.orientation),
#                                           points_of_meander[-1][1] - (scale_eff / 2) * np.cos(self.orientation)))
#                 check_length += scale_eff / 2
#
#             else:
#                 points_of_meander.append((points_of_meander[-1][0] + (scale_eff / 2) * np.sin(self.orientation),
#                                           points_of_meander[-1][1] - (scale_eff / 2) * np.cos(self.orientation)))
#                 check_length += scale_eff / 2
#
#                 points_of_meander.append((points_of_meander[-1][0] + x * np.cos(self.orientation),
#                                           points_of_meander[-1][1] + x * np.sin(self.orientation)))
#                 check_length += x
#
#                 points_of_meander.append((points_of_meander[-1][0] - (scale_eff / 2) * np.sin(self.orientation),
#                                           points_of_meander[-1][1] + (scale_eff / 2) * np.cos(self.orientation)))
#                 check_length += scale_eff / 2
#
#             # points_of_meander.append((points_of_meander[-1][0], points_of_meander[-1][1] + scale_eff / 2))
#             # points_of_meander.append((points_of_meander[-1][0] + h, points_of_meander[-1][1]))
#             # points_of_meander.append((points_of_meander[-1][0], points_of_meander[-1][1] - scale_eff))
#             # points_of_meander.append((points_of_meander[-1][0] + h, points_of_meander[-1][1]))
#             # points_of_meander.append((points_of_meander[-1][0], points_of_meander[-1][1] + scale_eff / 2))
#
#         points_of_meander.append((points_of_meander[-1][0] + self.connector_length * np.cos(self.orientation),
#                                   points_of_meander[-1][1] + self.connector_length * np.sin(self.orientation)))
#
#
#         check_length += self.connector_length
#
#         #self.points = points_of_meander[1: len(points_of_meander)-1]
#         self.points = points_of_meander
#         pass
#         # print(f"""Number of curves {number_of_curves}
#         # Scale eff {scale_eff}
#         # x {x}
#         # delta {delta}
#         # check length {check_length}
#         # """)


def meander_creation2(name: str, begin_point: Tuple[(float, float)],
                    begin_orientation: float, end_point: Tuple[(float, float)], end_orientation: float,
                    meander_length: float, length_left : float, length_right : float,
                    radius_limit : Tuple[(float, float)], w: float, s: float, g: float,
                    layer_configuration: LayerConfiguration):
    """
     :param length_left: width meander limit from begin point to the left
     :param length_right: width meander limit from begin point to the right
     NB! in case of orientations that are multiples of pi in half
     these limits are determined after turning the meander by pi/2 clockwise
     """
    ## Find segments length with the model L = A+B+K(n-1)+n*pi*r, A - begin segment, B - end segment
    delta_y = end_point[1] - begin_point[1]
    delta_x = end_point[0] - begin_point[0]
    rotation = False
    if (begin_orientation%np.pi)!=(end_orientation%np.pi):
        raise ValueError('begin end end orientations are incompatible')
    ## Turn meander pi/2 clockwise if its orientations are not 0, pi, etc
    if np.abs(np.cos(begin_orientation))!=1:
        rotation = True
        begin_orientation+=-np.pi/2
        end_orientation +=-np.pi / 2
        delta_x = end_point[1] - begin_point[1]
        delta_y = -end_point[0] + begin_point[0]
        begin_point = begin_point[::-1]
        end_point = end_point[::-1]

    n_max = np.int(np.abs(delta_y)/2/radius_limit[0])
    n_min = np.int(np.abs(delta_y)/2/radius_limit[1])
    result = []
    for n in np.linspace(n_min, n_max, n_max-n_min+1):
        if np.cos(begin_orientation) == np.cos(end_orientation) and n % 2 != 0:
            pass
        elif np.cos(begin_orientation) != np.cos(end_orientation) and n % 2 == 0:
            pass
        else:
            continue
        r = np.abs(delta_y)/2/n
        A_max = length_right*(1+np.cos(begin_orientation))/2 \
                + length_left*(1-np.cos(begin_orientation))/2 - (r+w/2+s)
        B_max = (length_right-delta_x)*(1+np.cos(end_orientation))/2\
                + (length_left+delta_x)*(1-np.cos(end_orientation))/2 - (r+w/2+s)

        print(A_max, B_max)

        if np.cos(begin_orientation) != np.cos(end_orientation):
            if n!= 0:
                K = (meander_length - delta_x*np.cos(begin_orientation))/n - np.pi*r
                A = (K + delta_x*np.cos(begin_orientation)) / 2
                B = A
                print(K + delta_x*np.cos(begin_orientation))
                if (A >= A_max) or (B >= B_max):
                    AB = opt.lsq_linear([1,1], K + delta_x*np.cos(begin_orientation),
                                         bounds= ([0,0], [A_max, B_max]))
                    A = AB.x[0]
                    B = AB.x[1]

            if n == 0:
                A = meander_length
                B = 0
                K = 0
            print(A, K, B)

        if np.cos(begin_orientation) == np.cos(end_orientation):
            if n!= 1:
                K = (meander_length - n*np.pi*r)/(n + 1) + np.abs(delta_x*np.cos(begin_orientation))/(n + 1)
                if delta_x*np.cos(begin_orientation) > 0:  # A>B
                    A = K
                    B = A - delta_x*np.cos(begin_orientation)
                else:
                    B = K
                    A = B + delta_x*np.cos(begin_orientation)
            else:
                A = (meander_length - np.pi * r) / 2
                B = A
                K = 0
                if (A >= A_max) or (B >= B_max):
                    AB = opt.lsq_linear([1,1], meander_length - np.pi * r,
                                         bounds= ([0,0], [A_max, B_max]))
                    A = AB.x[0]
                    B = AB.x[1]

        if (0<=A <= A_max) and (0<=B <= B_max):
            result.append([A, K, B, n, r])

    if result==[]:
        raise ValueError('No variants were found')

    if rotation:
        begin_orientation+=np.pi/2
        end_orientation+=np.pi/2
        begin_point = begin_point[::-1]
        end_point = end_point[::-1]
        delta_y = end_point[1] - begin_point[1]
        delta_x = end_point[0] - begin_point[0]
        # rotated_points = []
        # for point in points:
        #     point = point[::-1]
        #     rotated_points.append(point)
        # points = rotated_points

    ##Transform lengths in points for CPW

    A, K, B, n, r = result[0]
    points = []
    shift = r
    points.append(begin_point)
    if n!= 0:
        # points.append((begin_point[0] + (A + shift) * np.cos(begin_orientation), begin_point[1]))
        # mid_segm_x = begin_point[0] + (A + shift) * np.cos(begin_orientation)
        points.append((begin_point[0] + (A + shift) * np.cos(begin_orientation), begin_point[1] +
                       (A + shift) * np.sin(begin_orientation)))
        mid_segm_x = begin_point[0] + (A + shift) * np.cos(begin_orientation)
        mid_segm_y = begin_point[1] + (A + shift) * np.sin(begin_orientation)

    if n > 1:
        for turn in range(1, np.int(n)):
            if turn % 2 != 0:
                # points.append((mid_segm_x, begin_point[1] + np.sign(delta_y)*2*r*turn))
                # points.append((mid_segm_x + (K + 2*shift) * np.cos(np.pi - begin_orientation),
                #                begin_point[1] + np.sign(delta_y) * 2 * r * turn))
                points.append((mid_segm_x + np.sign(delta_x)*2*r*turn*np.abs(np.sin(begin_orientation)),
                               mid_segm_y + np.sign(delta_y)*2*r*turn*np.abs(np.cos(begin_orientation))))
                points.append((mid_segm_x + (K + 2*shift) * np.cos(np.pi - begin_orientation) +
                               np.sign(delta_x)*2*r*turn*np.abs(np.sin(begin_orientation)),
                               mid_segm_y + (K + 2*shift) * np.sin(np.pi + begin_orientation) +
                               np.sign(delta_y)*2*r*turn*np.abs(np.cos(begin_orientation))))
            else:
                # points.append((mid_segm_x + (K + 2*shift) * np.cos(np.pi - begin_orientation),
                #                begin_point[1] + np.sign(delta_y) * 2 * r * turn))
                # points.append((mid_segm_x, begin_point[1] + np.sign(delta_y) * 2 * r * turn))
                points.append((mid_segm_x + (K + 2*shift) * np.cos(np.pi - begin_orientation) +
                               np.sign(delta_x)*2*r*turn*np.abs(np.sin(begin_orientation)),
                               mid_segm_y + (K + 2*shift) * np.sin(np.pi + begin_orientation) +
                               np.sign(delta_y)*2*r*turn*np.abs(np.cos(begin_orientation))))
                points.append((mid_segm_x + np.sign(delta_x)*2*r*turn*np.abs(np.sin(begin_orientation)),
                               mid_segm_y + np.sign(delta_y)*2*r*turn*np.abs(np.cos(begin_orientation))))

    last_segm_x = points[-1][0] + np.sign(delta_x)*2*r*np.abs(np.sin(begin_orientation))
    last_segm_y = points[-1][1] + np.sign(delta_y)*2*r*np.abs(np.cos(begin_orientation))
    if n!= 0:
        points.append((last_segm_x, last_segm_y))
    points.append((end_point[0], end_point[1]))

    print(points, begin_orientation, end_orientation)

    meander = CPW(name, points, w, s, g, layer_configuration, r, corner_type='round',
                  orientation1=begin_orientation, orientation2=end_orientation)


    return result, points, meander