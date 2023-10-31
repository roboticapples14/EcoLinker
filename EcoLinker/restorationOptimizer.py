import ecoscape_connectivity
import numpy as np
from scgt import GeoTiff, Tile, Reader

'''
Suite of tools for finding best pixels to restore in order to maximize habitat connectivity
'''
class restorationOptimizer():
    def __init__(self, habitat_fn, terrain_fn, restored_terr_fn, connectivity_fn, flow_fn, restored_connectivity_fn, restored_flow_fn, death_fn, permeability_dict, pixels):
        self.habitat_fn = habitat_fn
        self.terrain_fn = terrain_fn
        self.restored_terr_fn = restored_terr_fn
        self.connectivity_fn = connectivity_fn
        self.flow_fn = flow_fn
        self.restored_connectivity_fn = restored_connectivity_fn
        self.restored_flow_fn = restored_flow_fn
        self.death_fn = death_fn
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
    :returns: list of pixels formatted (col,row)
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
        for i in range(n, 0, -1):
            highest_death[(col_indices[i-1], row_indices[i-1])] = death_matrix[row_indices[i-1]][col_indices[i-1]]
        
        return highest_death

    '''
    returns a cluster of n connected pixels with highest death
    :returns: list of pixels formatted (col,row)
    '''
    def get_highest_death_pixels_island(self, death_tif, n=None):
        if (n == None):
            n = self.pixels
        death_matrix = death_tif.get_all_as_tile().m.squeeze(0)

        pixels = []
        maxPixels = []
        def dfs(i,j):
            if (i < 0 or i >= len(death_matrix) or j < 0 or j >= len(death_matrix[0]) or death_matrix[i][j] == 0):
                return 0
            death = death_matrix[i][j]
            death_matrix[i][j] = 0
            pixels.append((i,j))
            return death + dfs(i+1, j) + dfs(i-1, j) + dfs(i, j+1) + dfs(i, j-1)

        maxDeath = 0
        currDeath = 0
        for i in range(len(death_matrix)):
            for j in range(len(death_matrix[0])):
                if (death_matrix[i][j] == 1):
                    currDeath = dfs(i, j)
                    if (currDeath > maxDeath):
                        print(f'currDeath: {currDeath}')
                        print(f'current pixels: {maxPixels}')
                        print(f'new max Death: {maxDeath}')
                        print(f'new max pixels: {pixels}')
                        maxDeath = currDeath
                        maxPixels = pixels
                pixels = []
        return (maxPixels, maxDeath)


        # uses DP or Kadane’s algorithm to find maximum contiguous subarray/area

        
        return highest_death

    '''
    get the sum of connectivity
    '''
    def sum_of_tif(self, tif_fn):
        with GeoTiff.from_file(tif_fn) as geotiff:
            np_tif = geotiff.get_all_as_tile().m
        return np.sum(np_tif)

    def get_most_permiable_terrain(self):
        sorted_permiability = sorted(self.permeability_dict.items(), key=lambda x:x[1], reverse=True)
        return sorted_permiability[0][0]

    def change_terrain(self, x, y, ter_fn=None, terrain_type=None, verbose=False):
        if terrain_type == None:
            terrain_type = self.get_most_permiable_terrain()
        if ter_fn==None:
            ter_fn=self.restored_terr_fn

        with GeoTiff.from_file(ter_fn) as terrain_geotiff:
            m = np.array([[[terrain_type]]])
            tile = Tile(1, 1, 0, 1, x, y, m)
            terrain_geotiff.set_tile(tile)
        
        if (verbose):
            with GeoTiff.from_file(self.terrain_fn) as terrain_geotiff:
                old_terrain = terrain_geotiff.get_pixel_value(x, y)
            print(f'Restoring pixel ({x}, {y}) from permiability {self.permeability_dict[old_terrain]} to {self.permeability_dict[terrain_type]}')

    def restore_pixels(self, n=None, verbose=False):
        current_terr_tile = GeoTiff.from_file(self.terrain_fn).get_all_as_tile()
        with GeoTiff.from_file(self.restored_terr_fn) as restored_terr:
            restored_terr.set_tile(current_terr_tile)

        death_tif = self.get_death_layer(self.death_fn)
        highest_death = self.get_highest_death_pixels(death_tif, n)

        for x, y in highest_death.keys():
            self.change_terrain(x, y, verbose=verbose)
