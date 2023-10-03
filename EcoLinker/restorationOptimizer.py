ecoscape = __import__("ecoscape-connectivity")
import numpy as np
from scgt import GeoTiff

'''
Suite of tools for finding best pixels to restore in order to maximize habitat connectivity
'''
class restorationOptimizer():
    def __init__(self, habitat_fn, terrain_fn, connectivity_fn, flow_fn, permeability_dict, pixels):
        self.habitat_geotiff = GeoTiff.from_file(habitat_fn)
        self.terrain_geotiff = GeoTiff.from_file(terrain_fn)
        self.connectivity_geotiff = GeoTiff.from_file(connectivity_fn)
        self.flow_geotiff = GeoTiff.from_file(flow_fn)
        self.permeability_dict = permeability_dict
        self.pixels = pixels

    
    def get_big_tile_reader(self, tif, width, height):
        reader = tif.get_reader(b=0, w=width, h=height)
        return reader

    def get_lowest_connectivity_tiles(self):
        min_connectivity = None
        reader = self.get_big_tile_reader(self.connectivity_geotiff, 100, 100)
        for tile in reader:
            connectivity_sum = np.sum(tile.m)
            if min_connectivity == None:
                min_connectivity = connectivity_sum
            if connectivity_sum > 0 and connectivity_sum < min_connectivity:
                min_connectivity = connectivity_sum


