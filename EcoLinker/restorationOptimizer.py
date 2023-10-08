import ecoscape_connectivity
import numpy as np
from scgt import GeoTiff, Tile, Reader

'''
Suite of tools for finding best pixels to restore in order to maximize habitat connectivity
'''
class restorationOptimizer():
    def __init__(self, habitat_fn, terrain_fn, connectivity_fn, flow_fn, permeability_dict, pixels):
        self.habitat_fn = habitat_fn
        self.terrain_fn = terrain_fn
        self.connectivity_fn = connectivity_fn
        self.flow_fn = flow_fn
        self.permeability_dict = ecoscape_connectivity.util.read_transmission_csv(permeability_dict)
        self.pixels = pixels

    
    def get_big_tile_reader(self, tif, width, height):
        reader = tif.get_reader(b=0, w=width, h=height)
        return reader

    def get_lowest_connectivity_tiles(self, width, height):
        min_connectivity = None
        min_tile = None
        with GeoTiff.from_file(self.connectivity_fn) as connectivity_geotiff:
            reader = self.get_big_tile_reader(connectivity_geotiff, width, height)
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
        with GeoTiff.from_file(self.terrain_fn) as terrain_geotiff:
            raw_terrain = terrain_geotiff.get_all_as_tile().m
        with GeoTiff.from_file(self.flow_fn) as flow_geotiff:
            raw_flow = flow_geotiff.get_all_as_tile().m.astype('float64')
    
        permeability = ecoscape_connectivity.util.dict_translate(raw_terrain, self.permeability_dict)
        sensitivity = np.divide(raw_flow, permeability, out=np.zeros_like(raw_flow), where=permeability!=0)
        norm_sensitivity = np.clip(np.log10(1. + sensitivity) * 20., 0, 255).astype(np.uint8)

        with GeoTiff.from_file(self.connectivity_fn) as connectivity_geotiff:
            connectivity_geotiff.clone_shape(filename)
        with GeoTiff.from_file(filename) as sensitivity_tiff:
            tile = Tile(sensitivity_tiff.width, sensitivity_tiff.height, 0, 0, 0, 0, norm_sensitivity)
            sensitivity_tiff.set_tile(tile)
        
        tif = GeoTiff.from_file(filename)
        return tif

    '''
    death is sensitivity * (1 - permeability), which calculates the number of birds who died attempting to disperse through this pixel
    '''
    def get_death_layer(self, filename):
        with GeoTiff.from_file(self.terrain_fn) as terrain_geotiff:
            raw_terrain = terrain_geotiff.get_all_as_tile().m
        with GeoTiff.from_file(self.flow_fn) as flow_geotiff:
            raw_flow = flow_geotiff.get_all_as_tile().m.astype('float64')
        
        permeability = ecoscape_connectivity.util.dict_translate(raw_terrain, self.permeability_dict)
        sensitivity = np.divide(raw_flow, permeability, out=np.zeros_like(raw_flow), where=permeability!=0.0)
        inverse_permeability = np.subtract(np.ones_like(permeability), permeability)
        death = np.multiply(sensitivity, inverse_permeability)
        norm_death = np.clip(np.log10(1. + death) * 20., 0, 255).astype(np.uint8)

        with GeoTiff.from_file(self.connectivity_fn) as connectivity_geotiff:
            connectivity_geotiff.clone_shape(filename)
        with GeoTiff.from_file(filename) as death_geotiff:
            tile = Tile(death_geotiff.width, death_geotiff.height, 0, 0, 0, 0, norm_death)
            death_geotiff.set_tile(tile)
        
        tif = GeoTiff.from_file(filename)
        return tif

    '''
    get the top n pixels with highest death, meaning the pixels where most birds die 
    '''
    def get_highest_death_pixels(self, death_tif, n=None):
        if (n == None):
            n = self.pixels
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

    '''
    get the sum of connectivity
    '''
    def sum_of_connectivity(self):
        with GeoTiff.from_file(self.connectivity_fn) as connectivity_geotiff:
            raw_connectivity = connectivity_geotiff.get_all_as_tile().m
        return np.sum(raw_connectivity)

    def get_most_permiable_terrain(self):
        sorted_permiability = sorted(self.permeability_dict.items(), key=lambda x:x[1], reverse=True)
        return sorted_permiability[0][0]

    def change_terrain(self, x, y, terrain_type=None):
        if terrain_type == None:
            terrain_type = self.get_most_permiable_terrain()

        with GeoTiff.from_file(self.terrain_fn) as terrain_geotiff:        
            m = np.array([[[terrain_type]]])
            tile = Tile(1, 1, 0, 0, x, y, m)
            terrain_geotiff.set_tile(tile)

