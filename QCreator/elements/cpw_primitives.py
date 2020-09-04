import gdspy
from .core import LayerConfiguration


class Stub:
    def __init__(self, w: float, s: float, g: float, length: float, layer_configuration: LayerConfiguration):
        """
        CPW without the central conductor
        :param w: signal conductor width
        :param s: s width
        :param g: finite g width
        :param length: stub length
        """
        self.w = w
        self.s = s
        self.g = g
        self.length = length
        self.layer_configuration = layer_configuration

    def render(self):
        g1 = gdspy.Rectangle((0, self.w / 2 + self.s),  (self.length, self.w / 2 + self.s + self.g),
                             layer=self.layer_configuration.total_layer)
        g2 = gdspy.Rectangle((0, -self.w / 2 - self.s), (self.length, -self.w / 2 - self.s - self.g),
                             layer=self.layer_configuration.total_layer)
        g3 = gdspy.Rectangle((self.length, -self.w / 2 - self.s - self.g),
                             (self.length+self.g, self.w / 2 + self.s + self.g),
                             layer=self.layer_configuration.total_layer)

        restrict = gdspy.Rectangle((0, -self.w/2-self.s-self.g), (self.length+self.g, self.w/2+self.s+self.g),
                                   layer=self.layer_configuration.restricted_area_layer)

        return {'positive': (g1, g2, g3), 'restrict': (restrict,)}


class Trapezoid:
    def __init__(self, w1: float, s1: float, g1: float, w2: float, s2: float, g2: float, length: float,
                 layer_configuration: LayerConfiguration):
        """
        Geometric primitive for isosceles trapezoid-form adapter from one CPW to another.
        :param w1: signal conductor width of port 1
        :param s1: signal-g s of port 1
        :param g1: finite g width of port 1
        :param w2: signal conductor width of port 2
        :param s2: signal-g s of port 2
        :param g2: finite g width of port 2
        :param layer_configuration:
        :param length: height of trapezoid
        """
        self.w1 = w1
        self.s1 = s1
        self.g1 = g1

        self.w2 = w2
        self.s2 = s2
        self.g2 = g2

        self.layer_configuration = layer_configuration
        self.length = length

    def name(self):
        """
        Returns a unique identifier of the the geometric primitive name that can be used to create a cell
        :return:
        """
        name = "Trapezoid-{w1:4.2f}-{s1:4.2f}-{g1:4.2f}-{w2:4.2f}-{s2:4.2f}-{g2:4.2f}-{length:4.2f}".format(
            w1=self.w1, s1=self.s1, g1=self.g1, w2=self.w2, s2=self.s2, g2=self.g2, length=self.length)
        return name

    def render(self):
        x_begin = - self.length/2
        x_end = self.length/2

        points_for_poly1 = [(x_begin, self.w1 / 2 + self.s1 + self.g1),
                            (x_begin, self.w1 / 2 + self.s1),
                            (x_end, self.w2 / 2 + self.s2),
                            (x_end, self.w2 / 2 + self.s2 + self.g2)]

        points_for_poly2 = [(x_begin, self.w1 / 2),
                            (x_begin, - self.w1 / 2),
                            (x_end, - self.w2 / 2),
                            (x_end, self.w2 / 2)]

        points_for_poly3 = [(x_begin, -(self.w1 / 2 + self.s1 + self.g1)),
                            (x_begin, -(self.w1 / 2 + self.s1)),
                            (x_end, -(self.w2 / 2 + self.s2)),
                            (x_end, -(self.w2 / 2 + self.s2 + self.g2))]

        points_for_restricted_area = [(x_begin, self.w1 / 2 + self.s1 + self.g1),
                                      (x_end, self.w2 / 2 + self.s2 + self.g2),
                                      (x_end, -(self.w2 / 2 + self.s2 + self.g2)),
                                      (x_begin, -(self.w1 / 2 + self.s1 + self.g1))]

        restricted_area = gdspy.Polygon(points_for_restricted_area,
                                        layer=self.layer_configuration.restricted_area_layer)

        poly1 = gdspy.Polygon(points_for_poly1, layer=self.layer_configuration.total_layer)
        poly2 = gdspy.Polygon(points_for_poly2, layer=self.layer_configuration.total_layer)
        poly3 = gdspy.Polygon(points_for_poly3, layer=self.layer_configuration.total_layer)

        result = poly1, poly2, poly3
#        polygon_to_remove = gdspy.boolean(restricted_area, result, 'not',
#                                          layer=self.layer_configuration.layer_to_remove)

        return {'positive': result, 'restrict': (restricted_area,)}#, 'remove': polygon_to_remove}

