import meshpy.triangle as triangle
import numpy as np
import matplotlib.pyplot as plt

import gdspy
import subprocess
import pandas as pd
import pyclipper
import time

# module for detecting inner points
from shapely.geometry.polygon import Polygon

fastcap_paths = [r'C:\Users\User\Documents\GitHub\QCreator\designs\Mirowave_state_transfer\fastcap.exe']

class Meshing:
    def __init__(self, path, cell_name, layers):
        self.path = path
        self.cell_name = cell_name
        self.layers = layers
        self.cells = None
        self.mesh_figures_points = []
        self.mesh_figures_tris = []
        self.conductors = None
        self.fastcap_filename = None
        self.cap_filename = None


    def read_data_from_gds_file(self):
        new_cells = [gdspy.GdsLibrary() for i in range(len(self.layers))]
        cells = []
        for i, num_layer in enumerate(self.layers):
            cells.append(new_cells[i].read_gds(infile=self.path).cells[self.cell_name])
            cells[i].remove_polygons(lambda pts, layer, datatype: layer != num_layer)
        self.cells = cells
        self.prepare_for_meshing()

    def prepare_for_meshing(self):
        conductors = []

        def create_cond(cell):
            polygons = []
            data = cell.get_polygons()
            unique_polygons = [list(x) for x in set(tuple(map(tuple, x)) for x in data)]
            for fig in unique_polygons:
                polygons.append([tuple(fig[i]) for i in range(len(fig))])
            return polygons

        for i in range(len(self.layers)):
            conductors.append(create_cond(self.cells[i]))
        self.conductors = conductors

    def watch_cells(self, num):
        gdspy.LayoutViewer(cells=self.cells[num])

    def run_meshing(self, mesh_volume):
        total = 0
        for num, polygons in enumerate(self.conductors):
            mesh_points_list = []
            mesh_tris_list = []
            total_length = 0
            for conductor, volume in zip(polygons, [mesh_volume] * len(polygons)):
                data = create_mesh(conductor, max_volume=volume)
                mesh_points_list.append(data[0])
                mesh_tris_list.append(data[1])
                total_length += len(data[1])
            total += total_length
            self.mesh_figures_points.append(mesh_points_list)
            self.mesh_figures_tris.append(mesh_tris_list)
            print('For ' + str(num) + ' conductor ' + 'total length is: ', total_length)
        print("Sum :", total)

    def plot_meshing(self, colors):
        def plot_mesh(mesh_points, mesh_tris, *arg):
            plt.triplot(mesh_points[:, 0], mesh_points[:, 1], mesh_tris, arg)

        for num, color in enumerate(colors):
            for i in range(len(self.conductors[num])):
                plot_mesh(self.mesh_figures_points[num][i], self.mesh_figures_tris[num][i], color)

    def write_into_file(self,filename):
        def write_dat(points, f):
            if type(f) is str:
                f = open(f, 'w')
            counts = len(points)
            f.write('0 exported from the LayoutEditor ( www.LayoutEditor.net )\n')
            f.write('* **************************************************\n')
            f.write('* \n')
            f.write('* input file for FastCap automaticly generated by the LayoutEditor\n')
            f.write('* \n')
            f.write('* **************************************************\n')
            for i in range(counts):
                f.write('T ')
                for j in range(10):
                    f.write(str(points[i][j]))
                    f.write(' ')
                f.write('\n')
            f.write('*')
            f.close()

        list_of_conductors_to_fastcap = []
        for num in range(len(self.conductors)):
            for mesh_points, mesh_tris in zip(self.mesh_figures_points[num], self.mesh_figures_tris[num]):
                #     number+=1
                list_of_conductors_to_fastcap.append(to_fastcap(mesh_points, mesh_tris, num + 1))
        joined_data = np.concatenate(tuple(list_of_conductors_to_fastcap))
        self.fastcap_filename = filename
        write_dat(joined_data, filename)
        print('Data has been written into the file:',filename)

    def run_fastcap(self, output_file_name):
        start = time.time()
        output_file = open(output_file_name, 'w')
        for fastcap_path in fastcap_paths:
            try:
                args = [fastcap_path, self.fastcap_filename]
                self.cap_filename = output_file_name
                # ret=subprocess.call(args,stdout=output_file, shell=False,stderr=subprocess.DEVNULL)
                ret = subprocess.call(args, stdout=output_file, shell=False)
                break
            except FileNotFoundError:
                pass
        output_file.close()
        print('Time for fastcap job is: ', time.time() - start)

    def get_capacitances(self, epsilon):
        epsilon = (epsilon+1)/2
        number_of_conductors = len(self.conductors)
        with open(self.cap_filename) as file:
            text = file.readlines()
            text = [line.rstrip('\n') for line in text]
            value = text[-(number_of_conductors + 2)].split(' ')[-1]
        if value[:4] == 'pico': value = 'femtofarads'
        print('Capacitance value in: ', value)
        table = []

        def convert_pico(value):
            # epsilon = (11.9 + 1) / 2
            return round(epsilon * float(value), 2) / 1e3
        def convert_nano(value):
            # epsilon = (11.9 + 1) / 2 # TODO: NO
            return epsilon * float(value)

        if value[:5] == 'femto':
            for i in range(-(number_of_conductors), 0):
                table.append(map(convert_pico, [word for word in text[i].split(' ') if word != ''][-number_of_conductors:]))
        if value[:4] == 'nano':
            for i in range(-(number_of_conductors), 0):
                table.append(map(convert_nano, [word for word in text[i].split(' ') if word != ''][-number_of_conductors:]))

        print(table)
        self.results= pd.DataFrame(table)
        return self.results

# def simplify(points):
#     """
#     Parameters
#     ----------
#     pol_points : list(list(coord_x, coord_y))
#     list of points of WSP
#
#     Returns
#     -------
#     list(new_polygon_1, ...), where
#     new_polygon_i : list(list(coord_x, coord_y))
#     """
#     return list(np.asarray(pyclipper.SimplifyPolygon(list(np.asarray(points)*1000)))/1000)
#     # return list(np.asarray(pyclipper.SimplifyPolygon(np.asarray(points)), dtype=object))
#     # return pyclipper.SimplifyPolygon(points)

def simplify(points, adaptive=True):
    """
    Parameters
    ----------
    pol_points : list(list(coord_x, coord_y))
    list of points of WSP

    Returns
    -------
    list(new_polygon_1, ...), where
    new_polygon_i : list(list(coord_x, coord_y))
    """
    # adaptive = True
    if not adaptive:
        return pyclipper.SimplifyPolygon(points)
    else:
        return list(np.asarray(pyclipper.SimplifyPolygon(list(np.asarray(points) * 1000))) / 1000)


def round_trip_connect(start, end):
    return [(i, i + 1) for i in range(start, end)] + [(end, start)]


def inner_point(polygon):
    point = Polygon(polygon).representative_point()
    x, y = point.x, point.y
    return x, y


def create_mesh(WSP_points, max_volume=10000, min_angle=25):
    outer_polygon, *inner_polygons = simplify(WSP_points)
    inner_points_of_holes = [inner_point(pol) for pol in inner_polygons]

    points = outer_polygon

    last_point_number = len(outer_polygon) - 1
    facets = round_trip_connect(0, last_point_number)

    for pol in inner_polygons:
        points.extend(pol)
        facets.extend(round_trip_connect(last_point_number + 1, last_point_number + len(pol)))
        last_point_number = last_point_number + len(pol)

    info = triangle.MeshInfo()
    info.set_points(points)
    info.set_holes(inner_points_of_holes)
    info.set_facets(facets)

    mesh = triangle.build(info, max_volume=max_volume, min_angle=min_angle)

    mesh_points = np.array(mesh.points)
    mesh_tris = np.array(mesh.elements)
    mesh_facets = np.array(mesh.facets)

    return mesh_points, mesh_tris





def to_fastcap(mesh_points, mesh_tris, number):
    z = np.zeros((len(mesh_points[:, 0]),))
    to_output = list()
    for i in range(len(mesh_points[:, 0])):
        to_output.append([mesh_points[i, 0], mesh_points[i, 1], z[i]])
    test = []
    for index in range(len(mesh_tris)):
        test.append(np.asarray(
            [to_output[mesh_tris[index][2]], to_output[mesh_tris[index][1]], to_output[mesh_tris[index][0]]]))
    test = np.asarray(test)
    points_file = test.reshape((len(mesh_tris), 9))
    points_to_file = np.insert(points_file, 0, number, axis=1)
    return points_to_file
