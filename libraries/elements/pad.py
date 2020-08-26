from .core import DesignElement, DesignTerminal, LayerConfiguration, ChipGeometry
from .. import transmission_line_simulator as tlsim
import gdspy

class Pad(DesignElement):
    """
    Contact pad for bonding the chip to the PCB
    """
    def __init__(self, name: str, w: float, s: float, g: float, position, z0: float, orientation: float,
                 layer_configuration: LayerConfiguration, chip_geometry: ChipGeometry):
        """

        :param name: Design element name
        :param w: CPW signal conductor width
        :param s: CPW signal-ground gap
        :param g: Ground conductor width
        :param position: Position on the chip
        :param orientation: Orientation on chip in radians; 0 is along x positive direction (right-looking)
        :param z0: characteristic impedance of port for transmission line system simulation
        """
        super().__init__(self, name)
        self._z0 = z0
        self.tls_cache = []
        self.layer_configuration = layer_configuration
        self.chip_geometry = chip_geometry
        self.terminal = DesignTerminal(position=position, orientation=orientation, type='cpw', core=w, gap=s, ground=g)

    @property
    def z0(self):
        return self._z0

    @z0.setter
    def z0(self, value):
        if self._z0 != value:
            for port in self.tls_cache:
                port.Z0 = value
        self._z0 = value

    def render(self):
        coord_init_x, coord_init_y = self.terminal.position
        w = self.terminal.core
        s = self.terminal.gap
        g = self.terminal.ground
        outer_tl_width = w + 2 * (g + s)
        x, y = (coord_init_x - outer_tl_width / 2, coord_init_y)

        pad_core = 250 #  to make pad with 50 Om impedance
        pad_vacuum = 146 #  to make pad with 50 Om impedance
        pad_ground = g
        pad_length = 600
        pad_indent = 50
        edge_indent = 100
        narrowing = 160
        outer_pad_width = (pad_core + (pad_vacuum + pad_ground) * 2)
        inner_pad_width = (pad_core + pad_vacuum * 2)
        outer_width = 2 * (g + s) + w
        inner_width = 2 * s + w

        r1 = gdspy.Polygon([(x, y), (x + outer_width, y),
                            (x + (outer_width + outer_pad_width) / 2, y - narrowing),
                            (x + (outer_width + outer_pad_width) / 2, y - (narrowing + pad_length)),
                            (x - (-outer_width + outer_pad_width) / 2, y - (narrowing + pad_length)),
                            (x - (-outer_width + outer_pad_width) / 2, y - narrowing)])
        x += g
        r2 = gdspy.Polygon([(x, y), (x + inner_width, y),
                            (x + (inner_width + inner_pad_width) / 2, y - narrowing),
                            (x + (inner_width + inner_pad_width) / 2, y - (narrowing + pad_length - edge_indent)),
                            (x - (-inner_width + inner_pad_width) / 2, y - (narrowing + pad_length - edge_indent)),
                            (x - (-inner_width + inner_pad_width) / 2, y - narrowing)])
        x += s
        r3 = gdspy.Polygon([(x, y), (x + w, y),
                            (x + (pad_core + w) / 2, y - narrowing),
                            (x + (pad_core + w) / 2, y - (narrowing + pad_length - pad_indent - edge_indent)),
                            (x - (pad_core - w) / 2, y - (narrowing + pad_length - pad_indent - edge_indent)),
                            (x - (pad_core - w) / 2, y - narrowing)])
        pad, restricted_pad = gdspy.boolean(gdspy.boolean(r1, r2, 'not'), r3, 'or'), r1

        pad.rotate(self.terminal.orientation, [coord_init_x, coord_init_y])
        restricted_pad.rotate(self.terminal.orientation, [coord_init_x, coord_init_y])

        return {'positive': pad, 'restricted': restricted_pad}

    def get_terminals(self) -> dict:
        return {'port': self.terminal}

    def add_to_tls(self, tls_instance: tlsim.TLSystem, terminal_mapping: dict,
                   track_changes: bool = True) -> list:
        p = tlsim.Port(z0=self.z0)
        if track_changes:
            self.tls_cache.append(p)

        tls_instance.add_element(p, [terminal_mapping['port']])
        return [p]
