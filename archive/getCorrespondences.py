from scipy import spatial
import numpy as np

def nearest_neighbour(points_a, points_b):
    '''
    @breif
    
    @ref https://codegolf.stackexchange.com/a/45623/13033
    
    '''
    tree = spatial.cKDTree(points_b)
    return tree.query(points_a)