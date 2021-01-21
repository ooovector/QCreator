from abc import ABC

from .core import DesignElement, DesignTerminal, LayerConfiguration
from scipy.constants import c, epsilon_0
from copy import deepcopy

epsilon = 11.45
import numpy as np
import gdspy
from .. import conformal_mapping as cm
from .. import transmission_line_simulator as tlsim
from typing import List, Tuple, Mapping, Union, Iterable, Dict


class RoundResonator(DesignElement):
    def __init__(self, name: str, frequency: float, initial_position: Tuple[float, float], w: float, s: float, g: float,
                 coupler_length, open_end_length: float,
                 layer_configuration: LayerConfiguration,
                 l1, l2, l3, l4, l5, h_end, corner_type: str = 'round'
                 ):
        """
        Create a coplanar waveguide resonator which contains three parts.
        :param name: element identifier
        :param frequency:
        :param initial_position:
        :param w: CPW signal conductor
        :param s: CPW signal-g s
        :param g:CPW finite g width
        :param layer_configuration:
        :param closed_end_length: length of closed end of resonator (grounded end)
        :param coupler_length: length of coupler part of resonator
        :param open_end_length: length of open part of resonator
        :param corner_type: 'round' for circular arcs instead of sharp corners, anything else for sharp corners
        """
        super().__init__('mc-cpw', name)

        self.name = name
        self.frequency = frequency
        self.initial_position = initial_position
        self.w = w
        self.g = g
        self.s = s
        # self.closed_end_length = closed_end_length
        self.coupler_length = coupler_length
        self.open_end_length = open_end_length
        self.corner_type = corner_type
        self.layer_configuration = layer_configuration

        # What is it?
        self.l1 = l1
        self.l2 = l2
        self.l3 = l3
        self.l4 = l4
        self.l5 = l5
        self.h_end = h_end

        self.width_total, self.widths, self.offsets = widths_offsets([w], [s, s], g)
        initial_point_x, initial_point_y = self.initial_position[0], self.initial_position[1]
        epsilon_eff = (epsilon + epsilon_0) // 2
        self.L = c / (4 * np.sqrt(epsilon_eff) * self.frequency) * 1e6

        points = [(initial_point_x, initial_point_y), (initial_point_x, initial_point_y + self.open_end_length),
                  (initial_point_x - self.coupler_length / 2, initial_point_y),
                  (initial_point_x - self.coupler_length / 2, initial_point_y - self.l1),
                  (initial_point_x - self.coupler_length / 2 + self.l2, initial_point_y - self.l1),
                  (initial_point_x - self.coupler_length / 2 + self.l2, initial_point_y - self.l1 - self.l3)]

        meander_length = self.L - self.open_end_length - self.coupler_length - self.l1 - self.l2 - self.l3 - self.l4 - self.l5
        if meander_length <= 0:
            print("Error! Meander length for a resonator is less than zero!")
        meander_step = self.l4 + self.l5
        n = int(meander_length // meander_step)
        # tail = np.floor(meander_length - n * meander_step)
        meander_points = deepcopy(points)
        i = 1
        while i < n + 1:
            if i % 2 != 0:
                list1 = [
                    (meander_points[-1][0] - (i - 1) * self.l4, meander_points[-1][1]),
                    (meander_points[-1][0] - i * self.l4, meander_points[-1][1])]
                meander_points.extend(list1)

            else:
                list1 = [(meander_points[-1][0] - (i - 1) * self.l4,
                          initial_point_y + self.l1 - self.l3 - self.l5 - (self.g + self.s + self.w / 2)),
                         (meander_points[-1][0] - (i - 1) * self.l4,
                          initial_point_y + self.l1 - self.l3 - self.l5 - (self.g + self.s + self.w / 2))]
                meander_points.extend(list1)
            i = i + 1
            self.meander_points = meander_points

    # def r(self):
    #     bend_radius = self.g
    #     precision = 0.001
    #
    #     line = CPW(self.name, self.meander_points, self.w, self.s, self.g, self.layer_configuration, self.r)
    #     return {'positive': line, 'restrict': line}


def widths_offsets(w, s, g):
    width_total = g * 2 + sum(s) + sum(w)
    widths = [g] + w + [g]
    offsets = [-(width_total - g) / 2]
    for i in range(len(widths) - 1):
        offsets.append(offsets[-1] + widths[i] / 2 + s[i] + widths[i + 1] / 2)

    return width_total, widths, offsets


if __name__ == '__main__':
    print(c)
