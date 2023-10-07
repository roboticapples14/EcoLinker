import ecoscape_connectivity
import numpy as np
from scgt import GeoTiff, Tile, Reader

'''
Suite of tools for finding best pixels to restore in order to maximize habitat connectivity
'''
class restorationOptimizer():
    def __init__(self, habitat_fn, terrain_fn, connectivity_fn, flow_fn, permeability_dict, pixels):
        self.habitat_geotiff = GeoTiff.from_file(habitat_fn)
        self.terrain_geotiff = GeoTiff.from_file(terrain_fn)
        self.connectivity_geotiff = GeoTiff.from_file(connectivity_fn)
        self.flow_geotiff = GeoTiff.from_file(flow_fn)
        self.permeability_dict = ecoscape_connectivity.util.read_transmission_csv(permeability_dict)
        self.pixels = pixels

    
    def get_big_tile_reader(self, tif, width, height):
        reader = tif.get_reader(b=0, w=width, h=height)
        return reader

    def get_lowest_connectivity_tiles(self, width, height):
        min_connectivity = None
        min_tile = None
        reader = self.get_big_tile_reader(self.connectivity_geotiff, width, height)
        for tile in reader:
            connectivity_sum = np.sum(tile.m)
            # initalize
            if min_connectivity == None:
                min_connectivity = connectivity_sum
                min_tile = tile

            if connectivity_sum > 0 and connectivity_sum < min_connectivity:
                min_connectivity = connectivity_sum
                min_tile = tile
        return min_tile

    '''
    Sensitivity is the gradient / permeability, resulting in ∂q/∂p, representing the number of birds attempting to go through the pixel
    '''
    def get_sensitivity_of_grad(self, filename):
        raw_terrain = self.terrain_geotiff.get_all_as_tile().m
        permeability = ecoscape_connectivity.util.dict_translate(raw_terrain, self.permeability_dict)
        
        raw_flow = self.flow_geotiff.get_all_as_tile().m.astype('float64')
        sensitivity = np.divide(raw_flow, permeability, out=np.zeros_like(raw_flow), where=permeability!=0)
        norm_sensitivity = np.clip(np.log10(1. + sensitivity) * 20., 0, 255).astype(np.uint8)

        with self.connectivity_geotiff.clone_shape(filename) as sensitivity_tiff:
            tile = Tile(sensitivity_tiff.width, sensitivity_tiff.height, 0, 0, 0, 0, norm_sensitivity)
            sensitivity_tiff.set_tile(tile)
        
        tif = GeoTiff.from_file(filename)
        return tif

    '''
    death is sensitivity * (1 - permeability), which calculates the number of birds who died attempting to disperse through this pixel
    '''
    def get_death_layer(self, filename):
        raw_terrain = self.terrain_geotiff.get_all_as_tile().m
        permeability = ecoscape_connectivity.util.dict_translate(raw_terrain, self.permeability_dict)

        raw_flow = self.flow_geotiff.get_all_as_tile().m.astype('float64')
        sensitivity = np.divide(raw_flow, permeability, out=np.zeros_like(raw_flow), where=permeability!=0.0)

        inverse_permeability = np.subtract(np.ones_like(permeability), permeability)
        death = np.multiply(sensitivity, inverse_permeability)
        norm_death = np.clip(np.log10(1. + death) * 20., 0, 255).astype(np.uint8)

        with self.connectivity_geotiff.clone_shape(filename) as death_tiff:
            tile = Tile(death_tiff.width, death_tiff.height, 0, 0, 0, 0, norm_death)
            death_tiff.set_tile(tile)
        
        tif = GeoTiff.from_file(filename)
        return tif

    def get_highest_death_pixels(self, death_tif):
        death_matrix = death_tif.get_all_as_tile().m.squeeze(0)
        flat_indices = np.argpartition(death_matrix.ravel(), -self.pixels)[-self.pixels:]
        row_indices, col_indices = np.unravel_index(flat_indices, death_matrix.shape)

        min_elements = death_matrix[row_indices, col_indices]
        min_elements_order = np.argsort(min_elements)
        row_indices, col_indices = row_indices[min_elements_order], col_indices[min_elements_order]

        highest_death = {}
        for i in range(self.pixels - 1, 0, -1):
            highest_death[(col_indices[i], row_indices[i])] = death_matrix[row_indices[i]][col_indices[i]]
        
        return highest_death

    def sum_of_connectivity(self):
        raw_connectivity = self.connectivity_geotiff.get_all_as_tile().m
        return np.sum(raw_connectivity)





