from .core import DesignElement, DesignTerminal, LayerConfiguration, ChipGeometry
from .. import transmission_line_simulator as tlsim
import gdspy
import numpy as np
from typing import Tuple, Mapping
from .cpw_primitives import Stub, Trapezoid
from .drawing import combine


def default_pad_geometry():
    return {'pad_w': 250, 'pad_s': 146, 'pad_g': 8, 'pad_length': 400, 'narrowing_length': 160, 'stub_length': 100,
            'z0': 50}


class Pad(DesignElement):
    """
    Contact pad for bonding the chip to the PCB
    """

    def __init__(self, name: str, position: Tuple[float, float], orientation: float, cpw_w: float, cpw_s: float,
                 cpw_g: float, pad_w: float, pad_s: float, pad_g: float, pad_length: float, narrowing_length: float,
                 stub_length: float, z0: float, layer_configuration: LayerConfiguration, chip_geometry: ChipGeometry):
        """
        Contact pad for bonding the chip to the PCB
        :param name: Design element name
        :param cpw_w: CPW signal conductor width
        :param cpw_s: CPW signal-g s
        :param cpw_g: Ground conductor width
        :param position: Position on the chip
        :param orientation: Orientation on chip in radians; 0 is along x positive direction (right-looking)
        :param z0: characteristic impedance of port for transmission line system simulation
        """
        super().__init__('pad', name)
        self._z0 = z0
        self.tls_cache = []
        self.layer_configuration = layer_configuration
        self.chip_geometry = chip_geometry
        self.terminal = DesignTerminal(position=position, orientation=orientation, type='cpw', w=cpw_w, s=cpw_s,
                                       g=cpw_g, disconnected='short')
        self.pad_w = pad_w
        self.pad_s = pad_s
        self.pad_g = pad_g
        self.pad_length = pad_length
        self.narrowing_length = narrowing_length
        self.stub_length = stub_length
        self.position = self.terminal.position
        self.orientation = self.terminal.orientation

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
        narrowing = Trapezoid(w1=self.terminal.w, s1=self.terminal.s, g1=self.terminal.g,
                              w2=self.pad_w, s2=self.pad_s, g2=self.pad_g,
                              length=self.narrowing_length, layer_configuration=self.layer_configuration).render()
        pad = Trapezoid(w1=self.pad_w, s1=self.pad_s, g1=self.pad_g,
                        w2=self.pad_w, s2=self.pad_s, g2=self.pad_g,
                        length=self.pad_length, layer_configuration=self.layer_configuration).render()
        stub = Stub(w=self.pad_w, s=self.pad_s, g=self.pad_g, length=self.stub_length,
                    layer_configuration=self.layer_configuration).render()

        elements = [(narrowing, (self.narrowing_length / 2, 0)),
                    (pad, (self.narrowing_length + self.pad_length / 2, 0)),
                    (stub, (self.narrowing_length + self.pad_length, 0))]
        result = combine(elements)

        for layer_name, polygons in result.items():
            result[layer_name].rotate(self.orientation, (0, 0))
            result[layer_name].translate(*self.position)
        negative=gdspy.boolean(result['restrict'],result['positive'],'not',layer=self.layer_configuration.inverted)
        result.update({'inverted': negative})
        # print(result)
        return result

    def get_terminals(self) -> Mapping[str, DesignTerminal]:
        return {'port': self.terminal}

    def add_to_tls(self, tls_instance: tlsim.TLSystem, terminal_mapping: dict, track_changes: bool = True,
                   cutoff: float = np.inf, epsilon: float = 11.45) -> list:
        p = tlsim.Port(z0=self.z0, name=self.name)
        if track_changes:
            self.tls_cache.append(p)

        tls_instance.add_element(p, [terminal_mapping['port']])
        return [p]
