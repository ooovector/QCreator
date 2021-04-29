import numpy as np
from typing import List, Tuple, Mapping, Union, Iterable, Dict
from .cpw import CPW
from copy import deepcopy
from .core import LayerConfiguration


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
        raise ValueError('Error, length is too small')

    # lets fill the whole rectangular
    default_bend_diameter = bend_radius * 2 + 5
    if meander_type == 'flush':
        meander_step = length_left + length_right + default_bend_diameter
    elif meander_type == 'round':
        meander_step = length_left - bend_radius + length_right - bend_radius + \
                       (default_bend_diameter - 2 * bend_radius) + np.pi * bend_radius

    N = int((meander_length - rendering_meander.length) // meander_step)

    if N == 0:
        raise ValueError('Meander is too small, N=0')
    # subtract one to connect to the end point, here i assume that the distance
    # from the end of the meander is less than the meander step
    if end_orientation is not None:
        N = N - 1
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

    if end_orientation is not None:
        i = 0
    else:
        rendering_meander = CPW(name=name, points=deepcopy(points), w=w, s=s, g=g,
                                layer_configuration=layer_configuration, r=bend_radius,
                                corner_type=meander_type)

        tail = np.abs(np.floor(rendering_meander.length - meander_length))

        if tail < np.pi * bend_radius / 2:
            list = [(points[-1][0] + np.sin(meander_orientation - np.pi / 2 + np.pi * ((i) % 2)) * tail,
                     points[-1][1] - np.cos(meander_orientation - np.pi / 2 + np.pi * ((i) % 2)) * tail)]
            points.extend(list)
            rendering_meander = CPW(name=name, points=deepcopy(points), w=w, s=s, g=g,
                                    layer_configuration=layer_configuration, r=bend_radius,
                                    corner_type=meander_type)
        elif tail < (default_bend_diameter - 2 * bend_radius) + np.pi * bend_radius + 1:
            list = [(points[-1][0] + np.sin(meander_orientation) * (bend_radius + 1),
                     points[-1][1] - np.cos(meander_orientation) * (bend_radius + 1))]
            points.extend(list)

            rendering_meander = CPW(name=name, points=deepcopy(points), w=w, s=s, g=g,
                                    layer_configuration=layer_configuration, r=bend_radius,
                                    corner_type=meander_type)
            tail = np.abs(np.floor(rendering_meander.length - meander_length))

            list = [(points[-1][0] + np.sin(meander_orientation) * tail,
                     points[-1][1] - np.cos(meander_orientation) * tail)]
            points.extend(list)
            rendering_meander = CPW(name=name, points=deepcopy(points), w=w, s=s, g=g,
                                    layer_configuration=layer_configuration, r=bend_radius,
                                    corner_type=meander_type)

        else:

            list = [(points[-1][0] + np.sin(meander_orientation) * default_bend_diameter,
                     points[-1][1] - np.cos(meander_orientation) * default_bend_diameter)]
            points.extend(list)
            rendering_meander = CPW(name=name, points=deepcopy(points), w=w, s=s, g=g,
                                    layer_configuration=layer_configuration, r=bend_radius,
                                    corner_type=meander_type)

            error = np.abs(rendering_meander.length - meander_length)

            list = [(points[-1][0] + np.sin(meander_orientation - np.pi / 2 + np.pi * ((i - 1) % 2)) *
                     error,
                     points[-1][1] - np.cos(meander_orientation - np.pi / 2 + np.pi * ((i - 1) % 2)) *
                     error)]
            points.extend(list)
            rendering_meander = CPW(name=name, points=deepcopy(points), w=w, s=s, g=g,
                                    layer_configuration=layer_configuration, r=bend_radius,
                                    corner_type=meander_type)

            error = np.abs(rendering_meander.length - meander_length)
            list = [(points[-1][0] + np.sin(meander_orientation - np.pi / 2 + np.pi * ((i - 1) % 2)) *
                     (error),
                     points[-1][1] - np.cos(meander_orientation - np.pi / 2 + np.pi * ((i - 1) % 2)) *
                     (error))]
            points.extend(list)
            rendering_meander = CPW(name=name, points=deepcopy(points), w=w, s=s, g=g,
                                    layer_configuration=layer_configuration, r=bend_radius,
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
