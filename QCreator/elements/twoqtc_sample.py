import sys
sys.path.append('..')
sys.path.append('\\\\badwwmi\\user\\home\\di82riv\\Documents\\Desktop\\Master\\CircuitDesign\\QCreator\\QCreator')
import os
from matplotlib import pyplot as plt
from IPython.display import display, Math, Latex
import gdspy
import numpy as np
from importlib import reload
from copy import deepcopy
from QCreator import elements
from QCreator import general_sample_creator as creator
from QCreator import meshing
print(meshing.__file__)



def twoqtc_Caps(name,layer_configuration,center,Q1,Q2,TC,d1,d2,mesh_volume,save = False,plot = False):
    reload(gdspy)
    reload(creator)
    reload(elements.pp_transmon)
    reload(elements)
    reload(elements.pp_squid)
    sample = creator.Sample(name, layer_configuration)
    twoqtc = elements.twoqtc.TWOQTC(name='2qtc', center=center, layers_configuration=sample.layer_configuration,
                                    transformations={},
                                    Q1=Q1,
                                    Q2=Q2,
                                    TC=TC,
                                    d1=d1,
                                    d2=d2
                                    )

    sample.add(twoqtc)
    sample.draw_design()
    if save:
        sample.write_to_gds()

    sample.draw_cap()

    if plot:
        sample.watch()
        return 0

    Caps = sample.calculate_qubit_capacitance(cell=sample.qubit_cap_cells[0], qubit=twoqtc, mesh_volume=mesh_volume)
    return Caps