from .core import DesignElement, ChipGeometry, LayerConfiguration
from .pad import Pad
import gdspy
from .. import transmission_line_simulator as tlsim


class Pads(DesignElement):
    """
    Quasi-element that contains all contact pads of a specific chip. Used for ChipEdgeGround
    """
    def __init__(self, object_list):
        super().__init__('pads', 'pads')
        self.object_list = object_list

    def render(self):
        return {}

    def items(self):
        for object_ in self.object_list:
            if type(object_) is Pad:
                yield object_

    def get_terminals(self) -> dict:
        return {}

    def add_to_tls(self, tls_instance: tlsim.TLSystem,
                   terminal_mapping: dict, track_changes: bool = True) -> list:
        return []


class ChipEdgeGround(DesignElement):
    def __init__(self, chip_geometry: ChipGeometry, layer_configuration: LayerConfiguration, pads: Pads):
        """
        Element for creating a wide and solid g electrode around the edges of the chip to wirebond for wirebonding
        to PCB.
        :param chip_geometry:
        :param layer_configuration:
        :param pads:
        """
        super().__init__('chip_edge_ground', 'chip_edge_ground')
        self.chip_geometry = chip_geometry
        self.layer_configuration = layer_configuration
        self.pads = pads

    def render(self):
        """
        Draws edge g metallization on chip
        :return:
        """
        edge = 600 #  fundamental constant - edge length
        r1 = gdspy.Rectangle((0, 0), (self.chip_geometry.sample_horizontal_size, self.chip_geometry.sample_vertical_size))
        r2 = gdspy.Rectangle((edge, edge), (self.chip_geometry.sample_horizontal_size - edge,
                                            self.chip_geometry.sample_vertical_size - edge))
        result = gdspy.boolean(r1, r2, 'not')

        for pad in self.pads.items():
        #pads = gdspy.polygon.PolygonSet(pads)
            pad = pad.get()
            to_bool = gdspy.Rectangle(pad['positive'].get_bounding_box()[0].tolist(), pad['positive'].get_bounding_box()[1].tolist())
            result = gdspy.boolean(result, to_bool, 'not')

        return {'positive': result}

    def get_terminals(self) -> dict:
        return {}

    def add_to_tls(self, tls_instance: tlsim.TLSystem,
                   terminal_mapping: dict, track_changes: bool = True) -> list:
        return []
