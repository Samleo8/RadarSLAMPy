from scipy import spatial
import numpy as np

# https://codegolf.stackexchange.com/questions/45611/fastest-array-by-array-nearest-neighbour-search
def nearest_neighbour(points_a, points_b):
    tree = spatial.cKDTree(points_b)
    return tree.query(points_a)[1]
