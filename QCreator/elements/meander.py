import numpy as np
from typing import List, Tuple, Mapping, Union, Iterable, Dict


class CPWMeander:
    def __init__(self, initial_point: Tuple[(float, float)], w: float, s: float, g: float,
                 meander_length: float, restricted_scale: float, constant_scale: float,
                 orientation: float, connector_length: float):

        """
        Create a coplanar waveguide (CPW) meander.
        :param initial_point: initial points for a meander
        :param w: CPW signal conductor
        :param s: CPW signal-g s
        :param g:CPW finite g width
        :param meander_length: total length of the meander

        :param connector_length: length of connectors (default parameter)
        """
        self.initial_point = initial_point
        self.w = w
        self.g = g
        self.s = s
        self.meander_length = meander_length
        self.constant_scale = constant_scale
        self.restricted_scale = restricted_scale
        # self.period = period
        self.orientation = orientation
        delta = self.g + self.s + self.w / 2
        self.connector_length = connector_length

        # rectangle parameters
        meander_length_eff = self.meander_length - 2 * self.connector_length  # L
        constant_scale_eff = self.constant_scale - 2 * self.connector_length  # a
        restricted_scale_eff = self.restricted_scale - 2 * delta  # b

        if meander_length_eff < 0 or constant_scale_eff < 0:
            raise ValueError('Length of the meander is too short!')

        number_of_curves = (meander_length_eff - constant_scale_eff) // (restricted_scale_eff - 0 * delta) + 1
        if number_of_curves < 1.:
            raise ValueError('Length of the meander is too short!')

        scale_eff = (meander_length_eff - constant_scale_eff + 0 * delta * number_of_curves) / number_of_curves
        x = constant_scale_eff / number_of_curves - 0 * delta
        if x <= 4 * delta:
            raise ValueError('Length of the meander is too long or restricted parameters are to small!')

        points_of_meander = [(self.initial_point[0][0], self.initial_point[0][1]),
                             (self.initial_point[0][0] + self.connector_length * np.cos(self.orientation),
                              self.initial_point[0][1] + self.connector_length * np.sin(self.orientation))]


        check_length = self.connector_length

        for i in range(int(number_of_curves)):
            if i % 2 == 0:
                points_of_meander.append((points_of_meander[-1][0] - (scale_eff / 2) * np.sin(self.orientation),
                                          points_of_meander[-1][1] + (scale_eff / 2) * np.cos(self.orientation)))
                check_length += scale_eff / 2

                points_of_meander.append((points_of_meander[-1][0] + x * np.cos(self.orientation),
                                          points_of_meander[-1][1] + x * np.sin(self.orientation)))
                check_length += x

                points_of_meander.append((points_of_meander[-1][0] + (scale_eff / 2) * np.sin(self.orientation),
                                          points_of_meander[-1][1] - (scale_eff / 2) * np.cos(self.orientation)))
                check_length += scale_eff / 2

            else:
                points_of_meander.append((points_of_meander[-1][0] + (scale_eff / 2) * np.sin(self.orientation),
                                          points_of_meander[-1][1] - (scale_eff / 2) * np.cos(self.orientation)))
                check_length += scale_eff / 2

                points_of_meander.append((points_of_meander[-1][0] + x * np.cos(self.orientation),
                                          points_of_meander[-1][1] + x * np.sin(self.orientation)))
                check_length += x

                points_of_meander.append((points_of_meander[-1][0] - (scale_eff / 2) * np.sin(self.orientation),
                                          points_of_meander[-1][1] + (scale_eff / 2) * np.cos(self.orientation)))
                check_length += scale_eff / 2

            # points_of_meander.append((points_of_meander[-1][0], points_of_meander[-1][1] + scale_eff / 2))
            # points_of_meander.append((points_of_meander[-1][0] + h, points_of_meander[-1][1]))
            # points_of_meander.append((points_of_meander[-1][0], points_of_meander[-1][1] - scale_eff))
            # points_of_meander.append((points_of_meander[-1][0] + h, points_of_meander[-1][1]))
            # points_of_meander.append((points_of_meander[-1][0], points_of_meander[-1][1] + scale_eff / 2))

        points_of_meander.append((points_of_meander[-1][0] + self.connector_length * np.cos(self.orientation),
                                  points_of_meander[-1][1] + self.connector_length * np.sin(self.orientation)))


        check_length += self.connector_length

        #self.points = points_of_meander[1: len(points_of_meander)-1]
        self.points = points_of_meander
        pass
        # print(f"""Number of curves {number_of_curves}
        # Scale eff {scale_eff}
        # x {x}
        # delta {delta}
        # check length {check_length}
        # """)
