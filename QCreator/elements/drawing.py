import gdspy


def combine(elements):
    result = {}
    for layers, position in elements:
        for layer_name, polygons in layers.items():
            if layer_name not in result:
                result[layer_name] = None
            for polygon in polygons:
                polygon.translate(*position)
                result[layer_name] = gdspy.boolean(result[layer_name], polygon, 'or', layer=polygon.layers[0])
    return result
