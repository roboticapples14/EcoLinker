import ecoscape_connectivity
import numpy as np
from scgt import GeoTiff, Tile
from ecoscape_connectivity_local import repopulation


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
        self.highest_death = None
        self.pixels = pixels
        self.permeability_dict = ecoscape_connectivity.util.read_transmission_csv(permeability_dict)

    '''
    Runs connectivity for either true or restored terrain
    '''
    def run_connectivity(self, single_tile=True, deterministic=True, restored=False):
        print(f"run regular connectivity with permiability: {self.permeability_dict}")
        if (restored):
            repopulation.compute_connectivity(self.habitat_fn, self.restored_terr_fn, self.restored_connectivity_fn, self.restored_flow_fn, self.permeability_dict, single_tile=single_tile, deterministic=deterministic)
        else:
            repopulation.compute_connectivity(self.habitat_fn, self.terrain_fn, self.connectivity_fn, self.flow_fn, self.permeability_dict, single_tile=single_tile, deterministic=deterministic)

    '''
    Get the change in connectivity before and after restoration
    '''
    def get_delta_connectivity(self):
        pre_restoration_conn = self.sum_of_tif(self.connectivity_fn)
        post_restoration_conn = self.sum_of_tif(self.restored_connectivity_fn)
        return int(post_restoration_conn) - int(pre_restoration_conn)

    '''
    Get the percent change in connectivity before and after restoration
    '''
    def get_connectivity_percent_changed(self):
        pre_restoration_conn = self.sum_of_tif(self.connectivity_fn)
        post_restoration_conn = self.sum_of_tif(self.restored_connectivity_fn)
        delta_conn = int(post_restoration_conn) - int(pre_restoration_conn)
        return delta_conn / pre_restoration_conn

    '''
    Get the efficency of restoration, measured as the ratio of gained connectivity per unit of death restored
    :param restored_pixels: list of the death of restored pixels, uses highest_death's death values if None
    '''
    def get_restoration_efficency(self, restored_pixels=None):
        if restored_pixels == None:
            restored_pixels = self.highest_death.values()
        delta_conn = self.get_delta_connectivity()
        death_sum = 0
        for i in restored_pixels:
            death_sum += i
        return delta_conn/death_sum

    '''
    Calculates death based on flow layer
    Death is dq/dp * (1 - p), which calculates the number of birds who died attempting to disperse through this pixel
    :param death_tif: filename of death geotiff
    '''
    def get_death_layer(self, death_fn=None, flow_fn=None):
        if (death_fn == None):
            death_fn = self.death_fn
        if (flow_fn == None):
            flow_fn = self.flow_fn

        with GeoTiff.from_file(self.terrain_fn) as terrain_geotiff:
            raw_terrain = terrain_geotiff.get_all_as_tile().m
        with GeoTiff.from_file(flow_fn) as flow_geotiff:
            raw_flow = flow_geotiff.get_all_as_tile().m.astype('float64')
        
        permeability = ecoscape_connectivity.util.dict_translate(raw_terrain, self.permeability_dict)
        sensitivity = np.divide(raw_flow, permeability, out=np.zeros_like(raw_flow), where=permeability!=0.0)
        inverse_permeability = np.subtract(np.ones_like(permeability), permeability)
        death = np.multiply(sensitivity, inverse_permeability)
        norm_death = np.clip(np.log10(1. + death) * 20., 0, 255).astype(np.uint8)

        with GeoTiff.from_file(self.connectivity_fn) as connectivity_geotiff:
            connectivity_geotiff.clone_shape(death_fn)
        with GeoTiff.from_file(death_fn) as death_geotiff:
            tile = Tile(death_geotiff.width, death_geotiff.height, 0, 0, 0, 0, norm_death)
            death_geotiff.set_tile(tile)
        
        tif = GeoTiff.from_file(death_fn)
        return tif

    '''
    Gets n pixels with the highest death, meaning the pixels where most birds die (either due to low permiability or high traffic there)
    :param death_tif: filename of death geotiff
    :returns: dict of highest death pixels formatted {(col,row): death}
    '''
    def get_highest_death_pixels(self, death_tif=None, n=None):
        if (n == None):
            n = self.pixels
        if (death_tif == None):
            death_tif = self.death_fn
        death_matrix = death_tif.get_all_as_tile().m.squeeze(0)
        flat_indices = np.argpartition(death_matrix.ravel(), -n)[-n:]
        row_indices, col_indices = np.unravel_index(flat_indices, death_matrix.shape)

        min_elements = death_matrix[row_indices, col_indices]
        min_elements_order = np.argsort(min_elements)
        row_indices, col_indices = row_indices[min_elements_order], col_indices[min_elements_order]

        highest_death = {}
        for i in range(n, 0, -1):
            highest_death[(col_indices[i-1], row_indices[i-1])] = death_matrix[row_indices[i-1]][col_indices[i-1]]
        
        self.highest_death = highest_death
        return highest_death

    '''
    Gets the sum of the given geotiff (for calculating the sum of connectivity to compare across restorations, for instance)
    :param tif_fn: filename of geotiff to get sum of
    '''
    def sum_of_tif(self, tif_fn):
        with GeoTiff.from_file(tif_fn) as geotiff:
            np_tif = geotiff.get_all_as_tile().m
        return np.sum(np_tif)

    '''
    Get the most permiable terrain from permeability_dict
    '''
    def get_most_permiable_terrain(self):
        sorted_permiability = sorted(self.permeability_dict.items(), key=lambda x:x[1], reverse=True)
        return sorted_permiability[0][0]

    '''
    Converts the terrain value at pixel (x, y), where x = col and y = row to terrain_type,
    or the most permiable terrain if no terrain_type is provided
    :param x: col value of pixel coordinate to change
    :param y: row value of pixel coordinate to change
    :param ter_fn: terrain file name, or self.terrain if None
    :param verbose: Prints the terrain code and permiability of the changed pixel before and after change
    '''
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

    '''
    Restores n pixels with highest death to terrain of terrain_type
    :param x: col value of pixel coordinate to change
    :param y: row value of pixel coordinate to change
    :param ter_fn: terrain file name, or self.terrain if None
    :param flow_fn: flow filename to calculate the death layer
    :param verbose: Prints the terrain code and permiability of the changed pixel before and after change
    '''
    def restore_pixels(self, n=None, terrain_type=None, flow_fn=None, verbose=False):
        current_terr_tile = GeoTiff.from_file(self.terrain_fn).get_all_as_tile()
        with GeoTiff.from_file(self.restored_terr_fn) as restored_terr:
            restored_terr.set_tile(current_terr_tile)

        death_tif = self.get_death_layer(self.death_fn, flow_fn=flow_fn)
        highest_death = self.get_highest_death_pixels(death_tif, n)

        for x, y in highest_death.keys():
            self.change_terrain(x, y, terrain_type, verbose=verbose)

    def get_big_tile_reader(self, tif, width, height):
        reader = tif.get_reader(b=0, w=width, h=height)
        return reader

    # paint the changed terrain pixels with death values
    def paint_changed_terrain_geotiff(self, changed_terrain_fn):
        with GeoTiff.from_file(self.connectivity_fn) as connectivity_tif:
            connectivity_tif.clone_shape(changed_terrain_fn)

        with GeoTiff.from_file(changed_terrain_fn) as changed_terrain_tif:
            for (x,y), death in self.highest_death.items():
                    m = np.array([[[death]]])
                    tile = Tile(1, 1, 0, 1, x, y, m)
                    changed_terrain_tif.set_tile(tile)

    # computes the difference in connectivity before and after restoration 
    def get_connectivity_difference_tif(self, connectivity_diff_fn):
        # get origional connectivity
        with GeoTiff.from_file(self.connectivity_fn) as connectivity_tif:
            connectivity_tile = connectivity_tif.get_all_as_tile()
            # create connectivity diff tif from clone of connectivity_tif
            connectivity_tif.clone_shape(connectivity_diff_fn)

        # get restored connectivity
        with GeoTiff.from_file(self.restored_connectivity_fn) as restored_connectivity_tif:
            restored_connectivity_tile = restored_connectivity_tif.get_all_as_tile()

        # get the difference of the two in a tile
        diff = Tile(connectivity_tile.w, connectivity_tile.h, connectivity_tile.b, connectivity_tile.c, connectivity_tile.x, connectivity_tile.y, (restored_connectivity_tile.m - connectivity_tile.m))

        # write the tile to connectivity diff tif
        with GeoTiff.from_file(connectivity_diff_fn) as connectivity_diff:
            connectivity_diff.set_tile(diff)

'''
Defecit based restoration
'''
class defecitRestoration(restorationOptimizer):
    '''
    Runs restore_pixels, which computes the death layer and then restores the n highest death pixels 
    :param n: number of highest death pixels to restore
    :param terrain_type: terrain type to restore to
    :param verbose: print terrain conversion of every pixel
    '''
    def restore(self, n=None, terrain_type=None, verbose=False):
        if (n==None):
            n = self.pixels
        self.restore_pixels(n=n, terrain_type=terrain_type, verbose=verbose)


'''
Noisy defecit restoration:
    1. Compute connectivity with unaltered permiability dict for baseline comparison
    2. Add noise to permiability dict
    3. Recompute connectivity with noisy permiability dict
    4. Get death layer (noise exposes where might potentially be good terrain)
    5. Perform restoration based on noisy death layer
    6. Compute connectivity with og permiability again to observe the change in connectivity before and after noisy restoration
'''
class noisyDefecitRestoration(restorationOptimizer):
    def __init__(self, habitat_fn, terrain_fn, restored_terr_fn, connectivity_fn, flow_fn, restored_connectivity_fn, restored_flow_fn, death_fn, permeability_dict, pixels, noisy_connectivity_fn, noisy_flow_fn, rand_divisor=75):
        super().__init__(habitat_fn, terrain_fn, restored_terr_fn, connectivity_fn, flow_fn, restored_connectivity_fn, restored_flow_fn, death_fn, permeability_dict, pixels)
        self.noisy_connectivity_fn = noisy_connectivity_fn
        self.noisy_flow_fn = noisy_flow_fn
        self.noisy_permeability_dict = self.get_noisy_transmission_dict(random_divisor=rand_divisor)

    '''
    :param n: number of highest death pixels to restore
    :param terrain_type: terrain type to restore to
    :param verbose: print terrain conversion of every pixel
    '''
    def restore(self, n=None, terrain_type=None, verbose=False):
        if (n==None):
            n = self.pixels
        self.restore_pixels(n, flow_fn=self.noisy_flow_fn)

    '''
    Adds noise to transmission raster then runs connectivity based on noisy terrain permiability
    '''
    def run_noisy_connectivity(self, single_tile=True, deterministic=True):
        repopulation.compute_connectivity(self.habitat_fn, self.terrain_fn, self.noisy_connectivity_fn, self.noisy_flow_fn, self.noisy_permeability_dict, single_tile=single_tile, deterministic=deterministic)
    
    '''
    Adds randomness to all low transmission values for more death exploration
    :param divisor: number to divide random value in [0,1] by. The higher the divisor the lower the random number
    '''
    def get_noisy_transmission_dict(self, random_divisor=75):
        noisy_transmission = self.permeability_dict.copy()
        with GeoTiff.from_file(self.terrain_fn) as terrain_geotiff:
            raw_terrain = terrain_geotiff.get_all_as_tile().m
            terrain_codes = np.unique(raw_terrain)
        for i in terrain_codes:
            if i not in noisy_transmission.keys() or noisy_transmission[i] == 0.0:
                noisy_transmission[i] = np.random.random() / random_divisor
        return noisy_transmission

'''
Raises all terrain uniformly to high permiability, highest flow areas are where to restore
    1. Compute connectivity with unaltered permiability dict for baseline comparison
    2. Raise all terrain uniformly to high permiability
    3. Recompute connectivity with utopia permiability dict
    4. Look for regions* with highest:
        a. f1-f0
        b. (f1-f0)^2
        * to find regions: greedy alg, low res...
    5. Restore region/corridor
    6. Compute connectivity with og permiability again to observe:
        a. change in sum of connectivity
        b. change in sum of connectivity squared
'''
class utopianRestoration(restorationOptimizer):
    def __init__(self, habitat_fn, terrain_fn, restored_terr_fn, connectivity_fn, flow_fn, restored_connectivity_fn, restored_flow_fn, death_fn, permeability_dict, pixels, utopian_connectivity_fn, utopian_flow_fn, permiability=0.9):
        super().__init__(habitat_fn, terrain_fn, restored_terr_fn, connectivity_fn, flow_fn, restored_connectivity_fn, restored_flow_fn, death_fn, permeability_dict, pixels)
        self.utopian_connectivity_fn = utopian_connectivity_fn
        self.utopian_flow_fn = utopian_flow_fn
        self.utopian_permeability_dict = self.get_utopian_transmission_dict(permiability=permiability)

    '''
    :param n: number of highest death pixels to restore
    :param terrain_type: terrain type to restore to
    :param verbose: print terrain conversion of every pixel
    '''
    def restore(self, n=None, terrain_type=None, verbose=False):
        if (n==None):
            n = self.pixels

        current_terr_tile = GeoTiff.from_file(self.terrain_fn).get_all_as_tile()
        with GeoTiff.from_file(self.restored_terr_fn) as restored_terr:
            restored_terr.set_tile(current_terr_tile)

        diff = self.get_flow_diff()


        highest_diff = self.get_highest_diff_pixels(diff, n)
        print(highest_diff)

        for x, y in highest_diff.keys():
            self.change_terrain(x, y, terrain_type, verbose=verbose)

    '''
    Adds noise to transmission raster then runs connectivity based on noisy terrain permiability
    '''
    def run_utopian_connectivity(self, single_tile=True, deterministic=True):
        print(f"run utopian connectivity with permiability: {self.utopian_permeability_dict}")

        repopulation.compute_connectivity(self.habitat_fn, self.terrain_fn, self.utopian_connectivity_fn, self.utopian_flow_fn, self.utopian_permeability_dict, single_tile=single_tile, deterministic=deterministic)
    
    '''
    :param permiability: floating point premiability [0,1] to uniformly assign to all terrain types
    '''
    def get_utopian_transmission_dict(self, permiability=0.9):
        utopian_transmission = self.permeability_dict.copy()
        with GeoTiff.from_file(self.terrain_fn) as terrain_geotiff:
            raw_terrain = terrain_geotiff.get_all_as_tile().m
            terrain_codes = np.unique(raw_terrain)
        for i in terrain_codes:
            utopian_transmission[i] = permiability
        return utopian_transmission

    '''
    Calculates f1-f0, where f1 is utopian flow, and f0 is origional flow
    :param death_tif: filename of death geotiff
    '''
    def get_flow_diff(self):
        with GeoTiff.from_file(self.flow_fn) as flow:
            raw_flow = flow.get_all_as_tile().m
        with GeoTiff.from_file(self.utopian_flow_fn) as utopian_flow:
            raw_utopian_flow = utopian_flow.get_all_as_tile().m
        
        return raw_utopian_flow - raw_flow

    '''
    Gets n pixels with the highest difference in flow, with permiability <
    :param diff: difference np matrix
    :returns: dict of highest diff pixels formatted {(col,row): death}
    '''
    def get_highest_diff_pixels(self, diff, n=None):
        if (n == None):
            n = self.pixels
        flat_indices = np.argpartition(diff.ravel(), -n)[-n:]
        diff = diff.squeeze(0)
        row_indices, col_indices = np.unravel_index(flat_indices, diff.shape)

        min_elements = diff[row_indices, col_indices]
        min_elements_order = np.argsort(min_elements)
        row_indices, col_indices = row_indices[min_elements_order], col_indices[min_elements_order]

        highest_diff = {}
        for i in range(n, 0, -1):
            highest_diff[(col_indices[i-1], row_indices[i-1])] = diff[row_indices[i-1]][col_indices[i-1]]
        
        return highest_diff

'''
Flip one permeability in every dxd square, measure increased flow
'''
class flipRestoration(restorationOptimizer):
    def flip_restoration(self):
        pass

'''
Flip one permeability in every dxd square, measure increased flow
At lower resolution
'''
class lowResFlipRestoration(restorationOptimizer):
    def flip_restoration_lower_res(self):
        pass
