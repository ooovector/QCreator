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

    def render(self):

        result_x = None
        result_y = None
        i = 1
        while self.period * i + self.width/2 < self.chip_geometry.sample_horizontal_size:
            rect_x = gdspy.Rectangle((self.period * i - self.width/2, 0),
                                     (self.period * i + self.width/2, self.chip_geometry.sample_vertical_size),
                                     layer=self.layer_configuration.gridline_x_layer)
            i += 1
            result_x = gdspy.boolean(rect_x, result_x, 'or')
        i = 1
        while self.period * i + self.width/2 < self.chip_geometry.sample_vertical_size:
            rect_y = gdspy.Rectangle((0, self.period * i - self.width/2),
                                     (self.chip_geometry.sample_horizontal_size, self.period * i + self.width/2),
                                     layer=self.layer_configuration.gridline_y_layer)
            i += 1
            result_y = gdspy.boolean(rect_y, result_y, 'or')
        result_x = gdspy.boolean(result_x, result_y, 'not', layer=self.layer_configuration.gridline_y_layer)

        # render rest of object in chip
        for object_ in self.objects:
            if object_ is self:
                continue
            object_polys = object_.get()
            if 'restrict' in object_polys:
                result_x = gdspy.boolean(result_x, object_polys['restrict'], 'not', layer=self.layer_configuration.gridline_x_layer)
                result_y = gdspy.boolean(result_y, object_polys['restrict'], 'not', layer=self.layer_configuration.gridline_y_layer)
            elif 'positive' in object_polys:
                result_x = gdspy.boolean(result_x, object_polys['positive'], 'not', layer=self.layer_configuration.gridline_x_layer)
                result_y = gdspy.boolean(result_y, object_polys['positive'], 'not', layer=self.layer_configuration.gridline_y_layer)

        return {'grid_x': result_x, 'grid_y': result_y}

    def get_terminals(self) -> dict:
        return {}

    def add_to_tls(self, tls_instance, terminal_mapping: dict, track_changes: bool = True,
                   cutoff: float = np.inf, epsilon: float = 11.45) -> list:
        return []
