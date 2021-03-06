from .core import DesignElement, ChipGeometry, LayerConfiguration
import gdspy
import numpy as np
from typing import List


class GridGround(DesignElement):
    def __init__(self, chip_geometry: ChipGeometry, layer_configuration: LayerConfiguration,
                 objects: List[DesignElement], width: float, period: float):
        """
        Grid for high-inductance, flux-vortex trapping g electrode on the chip.
        :param chip_geometry:
        :param layer_configuration:
        :param objects: all design objects, used to
        :param width: grid wire width (in meters)
        :param period: grid period (in meters)
        """
        super().__init__('chip_edge_ground', 'chip_edge_ground')
        self.chip_geometry = chip_geometry
        self.layer_configuration = layer_configuration
        self.objects = objects
        self.width = width
        self.period = period
        self.min_inverted_size = 4
        self.render_direct = True
        self.render_inverse = True

    def render(self):
        result_x = None
        result_y = None
        inverted_result = gdspy.Rectangle((0, 0),
                                     (self.chip_geometry.sample_horizontal_size, self.chip_geometry.sample_vertical_size),
                                     layer=self.layer_configuration.inverted)
        i = 1
        while self.period * i + self.width/2 < self.chip_geometry.sample_horizontal_size:
            rect_x = gdspy.Rectangle((self.period * i - self.width/2, 0),
                                     (self.period * i + self.width/2, self.chip_geometry.sample_vertical_size),
                                     layer=self.layer_configuration.gridline_x_layer)
            i += 1
            result_x = gdspy.boolean(rect_x, result_x, 'or')
            inverted_result = gdspy.boolean(inverted_result, rect_x, 'not', layer=self.layer_configuration.inverted)
        i = 1
        while self.period * i + self.width/2 < self.chip_geometry.sample_vertical_size:
            rect_y = gdspy.Rectangle((0, self.period * i - self.width/2),
                                     (self.chip_geometry.sample_horizontal_size, self.period * i + self.width/2),
                                     layer=self.layer_configuration.gridline_y_layer)
            i += 1
            result_y = gdspy.boolean(rect_y, result_y, 'or')
            inverted_result = gdspy.boolean(inverted_result, rect_y, 'not', layer=self.layer_configuration.inverted)
        result_x = gdspy.boolean(result_x, result_y, 'not', layer=self.layer_configuration.gridline_y_layer)

        # render rest of object in chip
        for object_ in self.objects:
            if object_ is self:
                continue
            object_polys = object_.get()
            if 'restrict' in object_polys:
                if self.render_direct:
                    result_x = gdspy.boolean(result_x, object_polys['restrict'], 'not', layer=self.layer_configuration.gridline_x_layer)
                    result_y = gdspy.boolean(result_y, object_polys['restrict'], 'not', layer=self.layer_configuration.gridline_y_layer)
                if self.render_inverse:
                    inverted_result = gdspy.boolean(inverted_result, object_polys['restrict'], 'not', layer=self.layer_configuration.inverted)
            elif 'positive' in object_polys:
                if self.render_direct:
                    result_x = gdspy.boolean(result_x, object_polys['positive'], 'not', layer=self.layer_configuration.gridline_x_layer)
                    result_y = gdspy.boolean(result_y, object_polys['positive'], 'not', layer=self.layer_configuration.gridline_y_layer)
                if self.render_inverse:
                    inverted_result = gdspy.boolean(inverted_result, object_polys['positive'], 'not',
                                                layer=self.layer_configuration.inverted)

#        if self.render_inverse:
#            inverted_result_filtered = []
#            for inverted_poly in inverted_result.polygons:
#                if gdspy.Polygon(inverted_poly).area() >= self.min_inverted_size:
#                    inverted_result_filtered.append(inverted_poly)
#            inverted_result = gdspy.PolygonSet(inverted_result_filtered)

        result = {}
        if self.render_direct:
            result['grid_x'] = result_x
            result['grid_y'] = result_y
        if self.render_inverse:
            result['inverted'] = inverted_result

        return {'grid_x': result_x, 'grid_y': result_y, 'inverted': inverted_result}

    def get_terminals(self) -> dict:
        return {}

    def add_to_tls(self, tls_instance, terminal_mapping: dict, track_changes: bool = True,
                   cutoff: float = np.inf, epsilon: float = 11.45) -> list:
        return []
