import enum
import ecoscape_connectivity
import numpy as np
import math
from collections import defaultdict
from scgt import GeoTiff, Tile
from ecoscape_connectivity_local import repopulation


'''
Suite of tools for finding best pixels to restore in order to maximize habitat connectivity
'''
class restorationOptimizer():
    def __init__(self, habitat_fn, terrain_fn, restored_terr_fn, connectivity_fn, flow_fn, restored_connectivity_fn, restored_flow_fn, permeability_dict, pixels, unrestorable_matrix=None, unrestorable_terrain=[]):
        self.habitat_fn = habitat_fn
        self.terrain_fn = terrain_fn
        self.restored_terr_fn = restored_terr_fn
        self.connectivity_fn = connectivity_fn
        self.flow_fn = flow_fn
        self.restored_connectivity_fn = restored_connectivity_fn
        self.restored_flow_fn = restored_flow_fn
        self.changed_pixels = None
        self.pixels = pixels
        self.permeability_dict = ecoscape_connectivity.util.read_transmission_csv(permeability_dict)
        self.unrestorable_terrain = unrestorable_terrain
        self.unrestorable_matrix = unrestorable_matrix
        self.terrain_type = None
        self.terrain_changed = []

    '''
    Runs connectivity for either true or restored terrain
    '''
    def run_connectivity(self, single_tile=True, deterministic=True, restored=False, gap_crossing=1, num_gaps=20):
        if (restored):
            repopulation.compute_connectivity(self.habitat_fn, self.restored_terr_fn, self.restored_connectivity_fn, self.restored_flow_fn, self.permeability_dict, num_gaps=num_gaps, single_tile=single_tile, gap_crossing=gap_crossing, deterministic=deterministic)
        else:
            repopulation.compute_connectivity(self.habitat_fn, self.terrain_fn, self.connectivity_fn, self.flow_fn, self.permeability_dict, num_gaps=num_gaps, single_tile=single_tile, gap_crossing=gap_crossing, deterministic=deterministic)

    '''
    Get the change in connectivity before and after restoration
    '''
    def get_delta_connectivity(self):
        pre_restoration_conn = self.sum_of_tif(self.connectivity_fn)
        post_restoration_conn = self.sum_of_tif(self.restored_connectivity_fn)
        return int(post_restoration_conn) - int(pre_restoration_conn)

    '''
    Get the change in flow before and after restoration
    '''
    def get_delta_flow(self):
        pre_restoration_flow = self.sum_of_tif(self.flow_fn)
        post_restoration_flow = self.sum_of_tif(self.restored_flow_fn)
        return post_restoration_flow - pre_restoration_flow

    '''
    Get the percent change in connectivity before and after restoration
    '''
    def get_connectivity_percent_changed(self):
        pre_restoration_conn = self.sum_of_tif(self.connectivity_fn)
        post_restoration_conn = self.sum_of_tif(self.restored_connectivity_fn)
        delta_conn = int(post_restoration_conn) - int(pre_restoration_conn)
        return delta_conn / pre_restoration_conn

    '''
    Get the efficency of restoration, measured as the ratio of gained connectivity per unit of metric restored
    :param restored_pixels: dict of pixels restored to their metric used (i.e. death, flow), uses changed_pixels if None
    '''
    def get_restoration_efficency(self, restored_pixels=None):
        if restored_pixels == None:
            restored_pixels = self.changed_pixels.values()
        delta_conn = self.get_delta_connectivity()
        value_sum = 0
        for i in restored_pixels:
            value_sum += i
        return delta_conn/value_sum

    '''
    Gets n pixels with the highest value, meaning the pixels where most birds die (either due to low permiability or high traffic there)
    :param tif: open geotif
    :returns: dict of highest valued pixels in tif formatted {(col,row): value}
    '''
    def get_highest_pixels_from_tif(self, tif, n=None):
        if (n == None):
            n = self.pixels
        if (isinstance(tif, str)):
            tif = GeoTiff.from_file(tif)
        matrix = tif.get_all_as_tile().m
        return self.get_highest_pixels_from_matrix(matrix)

    def get_highest_pixels_from_matrix(self, mat, n=None):
        if (n == None):
            n = self.pixels
        matrix = mat.squeeze(0)
        total_px = matrix.shape[0] * matrix.shape[1]
        flat_indices = np.argpartition(matrix.ravel(), -total_px)[-total_px:]
        row_indices, col_indices = np.unravel_index(flat_indices, matrix.shape)

        min_elements = matrix[row_indices, col_indices]
        min_elements_order = np.argsort(min_elements)
        row_indices, col_indices = row_indices[min_elements_order], col_indices[min_elements_order]

        highest_px = {}
        with GeoTiff.from_file(self.terrain_fn) as terr:
            raw_terrain = terr.get_all_as_tile().m.squeeze(0)
        with GeoTiff.from_file(self.habitat_fn) as hab:
            raw_hab = hab.get_all_as_tile().m.squeeze(0)

        i = total_px
        while (len(highest_px.items()) < n and i > 0):
            terrain = raw_terrain[row_indices[i-1]][col_indices[i-1]]
            permiability = self.permeability_dict[terrain]
            if permiability < 1 and raw_hab[row_indices[i-1]][col_indices[i-1]] != 1 and terrain not in self.unrestorable_terrain:
                if self.unrestorable_matrix is None or (self.unrestorable_matrix is not None and self.unrestorable_matrix[row_indices[i-1]][col_indices[i-1]] != 1):
                    highest_px[(col_indices[i-1], row_indices[i-1])] = matrix[row_indices[i-1]][col_indices[i-1]]
            i -= 1
        return highest_px

    '''
    takes a np matrix and shrinks it to the resolution given, making each pixel the sum of the constituting pixels
    '''
    def lower_res_matrix(self, matrix, rscale, cscale, average=False):
        r, c = matrix.shape
        dividend = (rscale * cscale) if average else 1
        return matrix.reshape(r//rscale, rscale, c//cscale, cscale).sum(axis=1).sum(axis=2) / dividend

    '''
    Scales the geotiff data in tif_fn to resolution constituting of pixels of row_pixels x col_pixels, writing scaled tif to scaled_tif_fn
    param row_pixels: number of pixels to combine to 1 pixel on y axis (divisible by tif_fn's height)
    param col_pixels: number of pixels to combine to 1 pixel on x axis (divisible by tif_fn's width)
    '''
    def scale_geotiff(self, tif_fn, scaled_tif_fn, row_pixels, col_pixels, average=False):
        with GeoTiff.from_file(tif_fn) as tif:
            mat = tif.get_all_as_tile().m.squeeze(0)
            scaled_mat = self.lower_res_matrix(mat, rscale=row_pixels, cscale=col_pixels, average=average)
            scaled_tile = Tile(scaled_mat.shape[1], scaled_mat.shape[0], 0, 1, 0, 0, np.expand_dims(scaled_mat, 0))
            profile = tif.profile
            profile['width'] = scaled_mat.shape[1]
            profile['height'] = scaled_mat.shape[0]
            with tif.copy_to_new_file(scaled_tif_fn, profile) as scaled_tif:
                scaled_tif.set_tile(scaled_tile)

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
    Get the most permiable terrain from permeability_dict
    :param arr: array of permeability_dicts
    '''
    def get_most_permiable_terrain_from_mult(self, arr):
        merged = defaultdict(lambda: 0) 
        for perm_dict in arr:
            for terrain_type, perm in perm_dict.items():
                merged[terrain_type] += perm
        sorted_permiability = sorted(merged.items(), key=lambda x:x[1], reverse=True)
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
        if self.unrestorable_matrix is not None:
            if self.unrestorable_matrix[y][x] == 1:
                print(f'Cannot restore pixels ({x}, {y})')
                return False
        
        if terrain_type == None:
            terrain_type = self.get_most_permiable_terrain()
        self.terrain_type = terrain_type
        if ter_fn==None:
            ter_fn=self.restored_terr_fn

        with GeoTiff.from_file(ter_fn) as terrain_geotiff:
            old_terrain = terrain_geotiff.get_pixel_value(x, y)
            m = np.array([[[terrain_type]]])
            tile = Tile(1, 1, 0, 1, x, y, m)
            terrain_geotiff.set_tile(tile)
            self.terrain_changed.append(old_terrain)
        
        if (verbose):
            print(f'Restoring pixel ({x}, {y}) from {old_terrain} with permiability {self.permeability_dict[old_terrain]} to {terrain_type} with permiability {self.permeability_dict[terrain_type]}')

        permiability_change = self.permeability_dict[terrain_type] - self.permeability_dict[old_terrain]
        return permiability_change

    def get_big_tile_reader(self, tif, width, height):
        reader = tif.get_reader(b=0, w=width, h=height)
        return reader

    # paint the changed terrain pixels with value values
    def paint_changed_terrain_geotiff(self, changed_terrain_fn, changed_pixels=None, value=None):
        if (changed_pixels == None):
            changed_pixels = self.changed_pixels
        with GeoTiff.from_file(self.connectivity_fn) as connectivity_tif:
            connectivity_tif.clone_shape(changed_terrain_fn)

        with GeoTiff.from_file(changed_terrain_fn) as changed_terrain_tif:
            for (x,y), val in changed_pixels.items():
                    m = np.array([[[val if value == None else value]]])
                    tile = Tile(1, 1, 0, 1, x, y, m)
                    changed_terrain_tif.set_tile(tile)

    # computes the difference in connectivity before and after restoration 
    def get_connectivity_difference_tif(self, connectivity_diff_fn):
        with GeoTiff.from_file(self.connectivity_fn) as connectivity_tif:
            connectivity_tile = connectivity_tif.get_all_as_tile()
            # create connectivity diff tif from clone of connectivity_tif
            connectivity_tif.clone_shape(connectivity_diff_fn, dtype='int16')

        # get restored connectivity
        with GeoTiff.from_file(self.restored_connectivity_fn) as restored_connectivity_tif:
            restored_connectivity_tile = restored_connectivity_tif.get_all_as_tile()

        # get the difference of the two in a tile
        diff = Tile(connectivity_tile.w, connectivity_tile.h, connectivity_tile.b, connectivity_tile.c, connectivity_tile.x, connectivity_tile.y, (restored_connectivity_tile.m.astype('int16') - connectivity_tile.m.astype('int16')))

        # write the tile to connectivity diff tif
        with GeoTiff.from_file(connectivity_diff_fn) as connectivity_diff:
            connectivity_diff.set_tile(diff)

    '''
    Draw the permiability of terrain to a geotiff filename
    '''
    def get_permiability_matrix(self):
        with GeoTiff.from_file(self.terrain_fn) as ter:
            ter_tile = ter.get_all_as_tile()
            permiability_mat = np.vectorize(self.permeability_dict.get)(ter_tile.m)
            return permiability_mat

    '''
    Draw the permiability of terrain to a geotiff filename
    '''
    def draw_permiability_tiff(self, filename):
        permiability_mat = self.get_permiability_matrix()
        with GeoTiff.from_file(self.terrain_fn) as ter:
            ter_tile = ter.get_all_as_tile()
            permiability_tile = Tile(ter_tile.w, ter_tile.h, ter_tile.b, ter_tile.c, ter_tile.x, ter_tile.y, permiability_mat)
            with ter.clone_shape(filename, dtype='float32') as perm:
                perm.set_tile(permiability_tile)

    '''
    Draw the resistance of terrain to a geotiff filename
    '''
    def get_resistance_matrix(self, terrain_fn=None):
        if (terrain_fn == None):
            terrain = self.terrain_fn
        with GeoTiff.from_file(terrain) as ter:
            ter_tile = ter.get_all_as_tile()
            ter_mat = ter_tile.m
            u,inv = np.unique(ter_mat,return_inverse = True)
            return np.array([(self.permeability_dict.get(x, 0)) for x in u])[inv].reshape(ter_mat.shape)

    '''
    Draw the resistance of terrain to a geotiff filename
    '''
    def draw_resistance_tiff(self, filename):
        resistance_mat = self.get_resistance_matrix()
        with GeoTiff.from_file(self.terrain_fn) as ter:
            ter_tile = ter.get_all_as_tile()
            permiability_tile = Tile(ter_tile.w, ter_tile.h, ter_tile.b, ter_tile.c, ter_tile.x, ter_tile.y, resistance_mat)
            with ter.clone_shape(filename, dtype='float32') as perm:
                perm.set_tile(permiability_tile)

'''
Defecit based restoration
    1. Compute connectivity to get gradient, and for baseline comparison
    2. Calculate death layer from gradient
    3. Get n pixels with highest death
    4. Restore each of those pixels to higher permiability terrain
    5. Compute connectivity with restored terrain
    6. Evaluate the ratio between change in connectivity and restored permiability
'''
class defecitRestoration(restorationOptimizer):
    def __init__(self, habitat_fn, terrain_fn, restored_terr_fn, connectivity_fn, flow_fn, restored_connectivity_fn, restored_flow_fn, death_fn, permeability_dict, pixels, unrestorable_matrix=None, unrestorable_terrain=[]):
        super().__init__(habitat_fn, terrain_fn, restored_terr_fn, connectivity_fn, flow_fn, restored_connectivity_fn, restored_flow_fn, permeability_dict, pixels, unrestorable_matrix=unrestorable_matrix, unrestorable_terrain=unrestorable_terrain)
        self.death_fn = death_fn
    
    '''
    Runs restore_pixels, which computes the death layer and then restores the n highest death pixels 
    :param n: number of highest death pixels to restore
    :param terrain_type: terrain type to restore to
    :param verbose: print terrain conversion of every pixel
    '''
    def restore(self, n=None, terrain_type=None, verbose=False):
        if (n==None):
            n = self.pixels
        permiability_restored = self.restore_pixels(n=n, terrain_type=terrain_type, verbose=verbose)
        return permiability_restored

    '''
    Calculates death based on flow layer
    Death is dq/dp * (1 - p), which calculates the number of birds who died attempting to disperse through this pixel
    :param death_tif: filename of death geotiff
    '''
    def get_death_layer(self, death_fn=None, flow_fn=None, terrain_fn=None):
        if (death_fn == None):
            death_fn = self.death_fn
        if (flow_fn == None):
            flow_fn = self.flow_fn
        if (terrain_fn == None):
            terrain_fn = self.terrain_fn

        with GeoTiff.from_file(self.terrain_fn) as terrain_geotiff:
            raw_terrain = terrain_geotiff.get_all_as_tile().m
        with GeoTiff.from_file(flow_fn) as flow_geotiff:
            raw_flow = flow_geotiff.get_all_as_tile().m.astype('float64')
        
        permeability = ecoscape_connectivity.util.dict_translate(raw_terrain, self.permeability_dict)
        sensitivity = np.divide(raw_flow, permeability, out=np.zeros_like(raw_flow), where=permeability!=0.0)
        inverse_permeability = np.subtract(np.ones_like(permeability), permeability)
        death = np.multiply(sensitivity, inverse_permeability)
        norm_death = np.clip(np.log10(1. + death) * 20., 0, 255).astype(np.uint8)
        
        with GeoTiff.from_file(self.connectivity_fn) as flow_geotiff:
            flow_geotiff.clone_shape(death_fn)
        with GeoTiff.from_file(death_fn) as death_geotiff:
            tile = Tile(death_geotiff.width, death_geotiff.height, 0, 0, 0, 0, norm_death)
            death_geotiff.set_tile(tile)
        
        tif = GeoTiff.from_file(death_fn)
        return tif

    '''
    Restores n pixels with highest death to terrain of terrain_type
    :param x: col value of pixel coordinate to change
    :param y: row value of pixel coordinate to change
    :param ter_fn: terrain file name, or self.terrain if None
    :param flow_fn: flow filename to calculate the death layer
    :param verbose: Prints the terrain code and permiability of the changed pixel before and after change
    '''
    def restore_pixels(self, n=None, terrain_type=None, flow_fn=None, terrain_fn=None, restored_terrain_fn=None, verbose=False):
        terrain_fn = terrain_fn if terrain_fn else self.terrain_fn
        restored_terrain_fn = restored_terrain_fn if restored_terrain_fn else self.restored_terr_fn

        current_terr_tile = GeoTiff.from_file(terrain_fn).get_all_as_tile()
        with GeoTiff.from_file(self.restored_terr_fn) as restored_terr:
            restored_terr.set_tile(current_terr_tile)

        death_tif = self.get_death_layer(self.death_fn, flow_fn=flow_fn, terrain_fn=terrain_fn)
        highest_death = self.get_highest_pixels_from_tif(death_tif, n)
        self.changed_pixels = highest_death.copy()

        permiability_change = 0 
        for x, y in highest_death.keys():
            change = self.change_terrain(x, y, terrain_type, verbose=verbose)
            if change == False:
                self.changed_pixels.pop((x,y))
            else:
                permiability_change += change
        return permiability_change

class FlowDivPermRestoration(defecitRestoration):
    def __init__(self, habitat_fn, terrain_fn, restored_terr_fn, connectivity_fn, flow_fn, restored_connectivity_fn, restored_flow_fn, permeability_dict, pixels, unrestorable_matrix=None, unrestorable_terrain=[], flow_power=1):
        super().__init__(habitat_fn, terrain_fn, restored_terr_fn, connectivity_fn, flow_fn, restored_connectivity_fn, restored_flow_fn, permeability_dict, pixels, unrestorable_matrix=unrestorable_matrix, unrestorable_terrain=unrestorable_terrain)
        self.flow_power = flow_power

    def restore(self, n=None, terrain_type=None, verbose=False):
        if (n==None):
            n = self.pixels
        permiability_restored = self.restore_pixels(n=n, terrain_type=terrain_type, verbose=verbose)
        return permiability_restored

    '''
    Calculates death based on flow layer
    Death is dq/dp * (1 - p), which calculates the number of birds who died attempting to disperse through this pixel
    :param death_tif: filename of death geotiff
    '''
    def get_death_layer(self, death_fn=None, flow_fn=None, terrain_fn=None):
        if (death_fn == None):
            death_fn = self.death_fn
        if (flow_fn == None):
            flow_fn = self.flow_fn
        if (terrain_fn == None):
            terrain_fn = self.terrain_fn

        with GeoTiff.from_file(self.terrain_fn) as terrain_geotiff:
            raw_terrain = terrain_geotiff.get_all_as_tile().m
        with GeoTiff.from_file(flow_fn) as flow_geotiff:
            raw_flow = flow_geotiff.get_all_as_tile().m.astype('float64')
        
        permeability = ecoscape_connectivity.util.dict_translate(raw_terrain, self.permeability_dict)
        sensitivity = np.divide(raw_flow, permeability, out=np.zeros_like(raw_flow), where=permeability!=0.0)
        death = np.clip(np.log10(1. + sensitivity) * 20., 0, 255).astype(np.uint8)

        with GeoTiff.from_file(self.connectivity_fn) as flow_geotiff:
            flow_geotiff.clone_shape(death_fn)
        with GeoTiff.from_file(death_fn) as death_geotiff:
            tile = Tile(death_geotiff.width, death_geotiff.height, 0, 0, 0, 0, death)
            death_geotiff.set_tile(tile)
        
        tif = GeoTiff.from_file(death_fn)
        return tif

'''
Noisy defecit restoration:
    1. Compute connectivity with unaltered permiability dict for baseline comparison
    2. Add noise to permiability dict
    3. Recompute connectivity with noisy permiability dict
    4. Get death layer (noise exposes where might potentially be good terrain)
    5. Perform restoration based on noisy death layer
    6. Compute connectivity with og permiability again to observe the change in connectivity before and after noisy restoration
'''
class noisyDefecitRestoration(defecitRestoration):
    def __init__(self, habitat_fn, terrain_fn, restored_terr_fn, connectivity_fn, flow_fn, restored_connectivity_fn, restored_flow_fn, death_fn, permeability_dict, pixels, noisy_connectivity_fn, noisy_flow_fn, rand_divisor=75, unrestorable_matrix=None, unrestorable_terrain=[]):
        super().__init__(habitat_fn, terrain_fn, restored_terr_fn, connectivity_fn, flow_fn, restored_connectivity_fn, restored_flow_fn, death_fn, permeability_dict, pixels, unrestorable_matrix=None, unrestorable_terrain=[])
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
        permiability_restored = self.restore_pixels(n, flow_fn=self.noisy_flow_fn)
        return permiability_restored

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
    5. Restore region/corridor
    6. Compute connectivity with og permiability again to observe:
        a. change in sum of connectivity
        b. change in sum of connectivity squared
'''
class utopianRestoration(restorationOptimizer):
    def __init__(self, habitat_fn, terrain_fn, restored_terr_fn, connectivity_fn, flow_fn, restored_connectivity_fn, restored_flow_fn, permeability_dict, pixels, utopian_connectivity_fn, utopian_flow_fn, permiability=0.8, unrestorable_matrix=None, unrestorable_terrain=[]):
        super().__init__(habitat_fn, terrain_fn, restored_terr_fn, connectivity_fn, flow_fn, restored_connectivity_fn, restored_flow_fn, permeability_dict, pixels, unrestorable_matrix, unrestorable_terrain)
        self.utopian_connectivity_fn = utopian_connectivity_fn
        self.utopian_flow_fn = utopian_flow_fn
        self.utopian_permeability_dict = self.get_utopian_transmission_dict(permiability=permiability)

    '''
    :param n: number of highest death pixels to restore
    :param terrain_type: terrain type to restore to
    :param verbose: print terrain conversion of every pixel
    '''
    def restore(self, n=None, terrain_type=None, verbose=False, weighted=False):
        if (n==None):
            n = self.pixels

        current_terr_tile = GeoTiff.from_file(self.terrain_fn).get_all_as_tile()
        with GeoTiff.from_file(self.restored_terr_fn) as restored_terr:
            restored_terr.set_tile(current_terr_tile)

        diff = self.get_flow_diff()

        highest_diff = self.get_highest_diff_pixels(diff, n)
        self.changed_pixels = highest_diff.copy()

        permiability_change = 0
        for x, y in highest_diff.keys():
            change = self.change_terrain(x, y, terrain_type, verbose=verbose)
            if change == False:
                self.changed_pixels.pop((x,y))
            else:
                permiability_change += change
        return permiability_change

    '''
    Adds noise to transmission raster then runs connectivity based on noisy terrain permiability
    '''
    def run_utopian_connectivity(self, single_tile=True, deterministic=True):
        repopulation.compute_connectivity(self.habitat_fn, self.terrain_fn, self.utopian_connectivity_fn, self.utopian_flow_fn, self.utopian_permeability_dict, single_tile=single_tile, deterministic=deterministic)
    
    '''
    :param permiability: floating point premiability [0,1] to uniformly assign to all terrain types
    '''
    def get_utopian_transmission_dict(self, permiability=0.8, delta_permiability=0):
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
            raw_flow = flow.get_all_as_tile().m.astype(np.int16)
            min_val = np.min(raw_flow)
            max_val = np.max(raw_flow)
            scaled_flow = ((raw_flow - min_val) / (max_val - min_val))**2
        with GeoTiff.from_file(self.utopian_flow_fn) as utopian_flow:
            raw_utopian_flow = utopian_flow.get_all_as_tile().m.astype(np.int16)            
            min_val = np.min(raw_utopian_flow)
            max_val = np.max(raw_utopian_flow)
            scaled_utopian_flow = ((raw_utopian_flow - min_val) / (max_val - min_val))
        
        # diff = scaled_utopian_flow - scaled_flow
        diff = (scaled_utopian_flow) * scaled_flow
        return diff

    '''
    Draw the difference between utopian flow and actual flow to a geotiff filename
    '''
    def draw_flow_diff_tif(self, filename):
        with GeoTiff.from_file(self.flow_fn) as flow:
            flow_tile = flow.get_all_as_tile()
            with flow.clone_shape(filename, dtype='int32') as diff:
                diff_flow = Tile(flow_tile.w, flow_tile.h, flow_tile.b, flow_tile.c, flow_tile.x, flow_tile.y, self.get_flow_diff())
                diff.set_tile(diff_flow)

    '''
    Gets n pixels with the highest difference in flow, with permiability < 1
    :param diff: difference np matrix
    :returns: dict of highest diff pixels formatted {(col,row): death}
    '''
    def get_highest_diff_pixels(self, diff, n=None):
        if (n == None):
            n = self.pixels
        diff = diff.squeeze(0)
        total_px = diff.shape[0] * diff.shape[1]
        flat_indices = np.argpartition(diff.ravel(), -total_px)[-total_px:]
        row_indices, col_indices = np.unravel_index(flat_indices, diff.shape)

        min_elements = diff[row_indices, col_indices]
        min_elements_order = np.argsort(min_elements)
        row_indices, col_indices = row_indices[min_elements_order], col_indices[min_elements_order]

        highest_diff = {}
        with GeoTiff.from_file(self.terrain_fn) as terr:
            raw_terrain = terr.get_all_as_tile().m.squeeze(0)
        with GeoTiff.from_file(self.habitat_fn) as hab:
            raw_hab = hab.get_all_as_tile().m.squeeze(0)

        i = total_px
        while (len(highest_diff.items()) < n and i >= 0):
            terrain = raw_terrain[row_indices[i-1]][col_indices[i-1]]
            permiability = self.permeability_dict[terrain]
            if permiability < 1 and raw_hab[row_indices[i-1]][col_indices[i-1]] != 1:
                highest_diff[(col_indices[i-1], row_indices[i-1])] = diff[row_indices[i-1]][col_indices[i-1]]
            i -= 1
        return highest_diff

'''
Raises all terrain uniformly to high permiability, highest flow areas are where to restore
    1. Compute connectivity with unaltered permiability dict for baseline comparison
    2. Raise all terrain uniformly to high permiability
    3. Recompute connectivity with utopia permiability dict
    4. Look for regions* with highest:
        a. f1-f0
        b. (f1-f0)^2
    5. Restore region/corridor
    6. Compute connectivity with og permiability again to observe:
        a. change in sum of connectivity
        b. change in sum of connectivity squared
'''
class utopianBFSRestoration(restorationOptimizer):
    def __init__(self, habitat_fn, terrain_fn, restored_terr_fn, connectivity_fn, flow_fn, restored_connectivity_fn, restored_flow_fn, permeability_dict, pixels, utopian_connectivity_fn, utopian_flow_fn, permiability=0.8, num_corridors=5, corridor_hop=1, unrestorable_matrix=None, unrestorable_terrain=[]):
        super().__init__(habitat_fn, terrain_fn, restored_terr_fn, connectivity_fn, flow_fn, restored_connectivity_fn, restored_flow_fn, permeability_dict, pixels, unrestorable_matrix, unrestorable_terrain)
        self.utopian_connectivity_fn = utopian_connectivity_fn
        self.utopian_flow_fn = utopian_flow_fn
        self.utopian_permeability_dict = self.get_utopian_transmission_dict(permiability=permiability)
        self.num_corridors = num_corridors
        self.corridor_hop = corridor_hop
    '''
    :param n: number of highest death pixels to restore
    :param terrain_type: terrain type to restore to
    :param verbose: print terrain conversion of every pixel
    '''
    def restore(self, n=None, terrain_type=None, verbose=False, weighted=False):
        if (n==None):
            n = self.pixels

        current_terr_tile = GeoTiff.from_file(self.terrain_fn).get_all_as_tile()
        with GeoTiff.from_file(self.restored_terr_fn) as restored_terr:
            restored_terr.set_tile(current_terr_tile)

        diff = self.get_flow_diff()

        highest_diff = self.get_bfs_diff_pixels(diff, n)
        self.changed_pixels = highest_diff.copy()

        permiability_change = 0
        for x, y in highest_diff.keys():
            change = self.change_terrain(x, y, terrain_type, verbose=verbose)
            if change == False:
                self.changed_pixels.pop((x,y))
            else:
                permiability_change += change
        return permiability_change

    '''
    Adds noise to transmission raster then runs connectivity based on noisy terrain permiability
    '''
    def run_utopian_connectivity(self, single_tile=True, deterministic=True):
        repopulation.compute_connectivity(self.habitat_fn, self.terrain_fn, self.utopian_connectivity_fn, self.utopian_flow_fn, self.utopian_permeability_dict, single_tile=single_tile, deterministic=deterministic)
    
    '''
    :param permiability: floating point premiability [0,1] to uniformly assign to all terrain types
    '''
    def get_utopian_transmission_dict(self, permiability=0.8, delta_permiability=0):
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
            raw_flow = flow.get_all_as_tile().m.astype(np.int16)
            min_val = np.min(raw_flow)
            max_val = np.max(raw_flow)
            scaled_flow = ((raw_flow - min_val) / (max_val - min_val))**2
        with GeoTiff.from_file(self.utopian_flow_fn) as utopian_flow:
            raw_utopian_flow = utopian_flow.get_all_as_tile().m.astype(np.int16)            
            min_val = np.min(raw_utopian_flow)
            max_val = np.max(raw_utopian_flow)
            scaled_utopian_flow = ((raw_utopian_flow - min_val) / (max_val - min_val))
        
        # diff = scaled_utopian_flow - scaled_flow
        diff = (scaled_utopian_flow) * scaled_flow
        return scaled_utopian_flow

    '''
    Draw the difference between utopian flow and actual flow to a geotiff filename
    '''
    def draw_flow_diff_tif(self, filename):
        with GeoTiff.from_file(self.flow_fn) as flow:
            flow_tile = flow.get_all_as_tile()
            with flow.clone_shape(filename, dtype='int32') as diff:
                diff_flow = Tile(flow_tile.w, flow_tile.h, flow_tile.b, flow_tile.c, flow_tile.x, flow_tile.y, self.get_flow_diff())
                diff.set_tile(diff_flow)

    def get_bfs_diff_pixels(self, flow, n=None):
        if (n == None):
            n = self.pixels
        if (self.num_corridors == None):
            self.num_corridors = n / 5
        flow = flow.squeeze(0)
        total_px = flow.shape[0] * flow.shape[1]
        flat_indices = np.argpartition(flow.ravel(), -total_px)[-total_px:]
        row_indices, col_indices = np.unravel_index(flat_indices, flow.shape)

        min_elements = flow[row_indices, col_indices]
        min_elements_order = np.argsort(min_elements)
        row_indices, col_indices = row_indices[min_elements_order], col_indices[min_elements_order]

        highest_flow = {}
        with GeoTiff.from_file(self.terrain_fn) as terr:
            raw_terrain = terr.get_all_as_tile().m.squeeze(0)
        with GeoTiff.from_file(self.habitat_fn) as hab:
            raw_hab = hab.get_all_as_tile().m.squeeze(0)

        stack = []
        seen = []
        i = total_px
        while len(stack) < self.num_corridors:
            terrain = raw_terrain[row_indices[i-1]][col_indices[i-1]]
            permiability = self.permeability_dict[terrain]
            if permiability < 1 and raw_hab[row_indices[i-1]][col_indices[i-1]] != 1:
                stack.append((col_indices[i-1], row_indices[i-1]))
            i -= 1
        while (len(highest_flow.items()) < n and len(stack) > 0):
            col, row = stack.pop(0)
            seen.append((col, row))
            terrain = raw_terrain[row][col]
            permiability = self.permeability_dict[terrain]
            if permiability < 1 and raw_hab[row][col] != 1:
                highest_flow[(col, row)] = flow[row][col]
            # find highest grad of neighbors and push to stack
            max_neighbor = ()
            max_neighbor_flow = 0
            for neighbor_col, neighbor_row in [(col - self.corridor_hop, row), (col + self.corridor_hop, row), (col, row - self.corridor_hop), (col, row + self.corridor_hop), (col - self.corridor_hop, row - self.corridor_hop), (col + self.corridor_hop, row + self.corridor_hop), (col - self.corridor_hop, row + self.corridor_hop), (col + self.corridor_hop, row - self.corridor_hop)]:
                if neighbor_col >= 0 and neighbor_col < flow.shape[1] and neighbor_row >= 0 and neighbor_row < flow.shape[0]:
                    neighbor_flow = flow[neighbor_row][neighbor_col]
                    if neighbor_flow > max_neighbor_flow and (neighbor_col, neighbor_row) not in highest_flow and (neighbor_col, neighbor_row) not in seen and (neighbor_col, neighbor_row) not in stack and self.permeability_dict[raw_terrain[row][col]] < 1 and raw_hab[row][col] != 1: 
                        max_neighbor = (neighbor_col, neighbor_row)
                        max_neighbor_flow = neighbor_flow
            if max_neighbor_flow > 0:
                stack.append(max_neighbor)
            if len(stack) == 0:
                while ((col_indices[i-1], row_indices[i-1]) in seen):
                    i-=1
                stack.append((col_indices[i-1], row_indices[i-1]))
                i -= 1
        return highest_flow

'''
Raises all terrain uniformly to high permiability, multiply utopian flow by actual flow
    1. Compute connectivity with unaltered permiability dict for baseline comparison
    2. Raise all terrain uniformly to high permiability
    3. Recompute connectivity with utopia permiability dict
    4. Look for regions* with highest f1*f0
    5. Restore region/corridor
    6. Compute connectivity with og permiability again to observe:
        a. change in sum of connectivity
        b. change in sum of connectivity squared
'''
class utopianRestorationMult(utopianRestoration):
    def __init__(self, habitat_fn, terrain_fn, restored_terr_fn, connectivity_fn, flow_fn, restored_connectivity_fn, restored_flow_fn, permeability_dict, pixels, utopian_connectivity_fn, utopian_flow_fn, permiability=0.8, unrestorable_matrix=None, unrestorable_terrain=[]):
        super().__init__(habitat_fn, terrain_fn, restored_terr_fn, connectivity_fn, flow_fn, restored_connectivity_fn, restored_flow_fn, permeability_dict, pixels, utopian_connectivity_fn, utopian_flow_fn, permiability, unrestorable_matrix, unrestorable_terrain)

    '''
    :param n: number of highest death pixels to restore
    :param terrain_type: terrain type to restore to
    :param verbose: print terrain conversion of every pixel
    '''
    def restore(self, n=None, terrain_type=None, verbose=False, weighted=False):
        if (n==None):
            n = self.pixels

        current_terr_tile = GeoTiff.from_file(self.terrain_fn).get_all_as_tile()
        with GeoTiff.from_file(self.restored_terr_fn) as restored_terr:
            restored_terr.set_tile(current_terr_tile)

        mult = self.get_flow_mult()

        highest_mult = self.get_highest_pixels_from_matrix(mult, n)
        self.changed_pixels = highest_mult.copy()

        permiability_change = 0
        for x, y in highest_mult.keys():
            change = self.change_terrain(x, y, terrain_type, verbose=verbose)
            if change == False:
                self.changed_pixels.pop((x,y))
            else:
                permiability_change += change
        return permiability_change

    '''
    Calculates f1 * f0, where f1 is utopian flow, and f0 is origional flow
    :param death_tif: filename of death geotiff
    '''
    def get_flow_mult(self):
        with GeoTiff.from_file(self.flow_fn) as flow:
            raw_flow = flow.get_all_as_tile().m.astype(np.int16)
        with GeoTiff.from_file(self.utopian_flow_fn) as utopian_flow:
            raw_utopian_flow = utopian_flow.get_all_as_tile().m.astype(np.int16)            

        mult = (0.001 + raw_utopian_flow) * (0.001 + raw_flow)
        return mult

    '''
    Draw the difference between utopian flow and actual flow to a geotiff filename
    '''
    def draw_flow_mult_tif(self, filename):
        with GeoTiff.from_file(self.flow_fn) as flow:
            flow_tile = flow.get_all_as_tile()
            with flow.clone_shape(filename, dtype='float32') as diff:
                diff_flow = Tile(flow_tile.w, flow_tile.h, flow_tile.b, flow_tile.c, flow_tile.x, flow_tile.y, self.get_flow_diff())
                diff.set_tile(diff_flow)

    '''
    Gets n pixels with the highest difference in flow, with permiability < 1
    :param diff: difference np matrix
    :returns: dict of highest diff pixels formatted {(col,row): death}
    '''
    def get_highest_mult_pixels(self, diff, n=None):
        if (n == None):
            n = self.pixels
        diff = diff.squeeze(0)
        total_px = diff.shape[0] * diff.shape[1]
        flat_indices = np.argpartition(diff.ravel(), -total_px)[-total_px:]
        row_indices, col_indices = np.unravel_index(flat_indices, diff.shape)

        min_elements = diff[row_indices, col_indices]
        min_elements_order = np.argsort(min_elements)
        row_indices, col_indices = row_indices[min_elements_order], col_indices[min_elements_order]

        highest_diff = {}
        with GeoTiff.from_file(self.terrain_fn) as terr:
            raw_terrain = terr.get_all_as_tile().m.squeeze(0)
        with GeoTiff.from_file(self.habitat_fn) as hab:
            raw_hab = hab.get_all_as_tile().m.squeeze(0)

        i = total_px
        while (len(highest_diff.items()) < n and i >= 0):
            terrain = raw_terrain[row_indices[i-1]][col_indices[i-1]]
            permiability = self.permeability_dict[terrain]
            if permiability < 1 and raw_hab[row_indices[i-1]][col_indices[i-1]] != 1:
                highest_diff[(col_indices[i-1], row_indices[i-1])] = diff[row_indices[i-1]][col_indices[i-1]]
            i -= 1
        return highest_diff

'''
Flow based restoration
    1. Compute connectivity to get gradient, and for baseline comparison
    2. Get n pixels with highest gradient, or flow
    3. Restore each of those pixels to higher permiability terrain
    4. Compute connectivity with restored terrain
    5. Evaluate the ratio between change in connectivity and restored permiability
'''
class flowRestoration(restorationOptimizer):
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

        with GeoTiff.from_file(self.flow_fn) as flow_tif:
            flow = flow_tif.get_all_as_tile().m

        highest_flow = self.get_highest_flow_pixels(flow, n)
        self.changed_pixels = highest_flow.copy()

        permiability_change = 0
        for x, y in highest_flow.keys():
            change = self.change_terrain(x, y, terrain_type, verbose=verbose)
            if change == False:
                self.changed_pixels.pop((x,y))
            else:
                permiability_change += change

        return permiability_change

    '''
    Gets n pixels with the highest flow, with permiability < 1
    :param diff: difference np matrix
    :returns: dict of highest diff pixels formatted {(col,row): death}
    '''
    def get_highest_flow_pixels(self, flow, n=None):
        if (n == None):
            n = self.pixels
        flow = flow.squeeze(0)
        total_px = flow.shape[0] * flow.shape[1]
        flat_indices = np.argpartition(flow.ravel(), -total_px)[-total_px:]
        row_indices, col_indices = np.unravel_index(flat_indices, flow.shape)

        min_elements = flow[row_indices, col_indices]
        min_elements_order = np.argsort(min_elements)
        row_indices, col_indices = row_indices[min_elements_order], col_indices[min_elements_order]

        highest_flow = {}
        with GeoTiff.from_file(self.terrain_fn) as terr:
            raw_terrain = terr.get_all_as_tile().m.squeeze(0)
        with GeoTiff.from_file(self.habitat_fn) as hab:
            raw_hab = hab.get_all_as_tile().m.squeeze(0)

        i = total_px
        while (len(highest_flow.items()) < n and i > 0):
            terrain = raw_terrain[row_indices[i-1]][col_indices[i-1]]
            permiability = self.permeability_dict[terrain]
            if permiability < 1 and raw_hab[row_indices[i-1]][col_indices[i-1]] != 1 and terrain not in self.unrestorable_terrain:
                if self.unrestorable_matrix is None or (self.unrestorable_matrix is not None and self.unrestorable_matrix[row_indices[i-1]][col_indices[i-1]] != 1):
                    highest_flow[(col_indices[i-1], row_indices[i-1])] = flow[row_indices[i-1]][col_indices[i-1]]
            i -= 1
        return highest_flow

'''
Greedy Flow restoration
    1. Compute connectivity to get gradient, and for baseline comparison
    2. Perform greedy search with starting nodes of highest pixels until n pixels have been chosen
    3. Perform greedy-first search on the matrix using highest flow pixels as starting nodes
    4. Compute connectivity with restored terrain
    5. Evaluate the ratio between change in connectivity and restored permiability
'''
class greedyFlowRestoration(restorationOptimizer):
    def __init__(self, habitat_fn, terrain_fn, restored_terr_fn, connectivity_fn, flow_fn, restored_connectivity_fn, restored_flow_fn, permeability_dict, pixels, unrestorable_matrix=None, unrestorable_terrain=[], num_corridors=None, corridor_hop=1):
        super().__init__(habitat_fn, terrain_fn, restored_terr_fn, connectivity_fn, flow_fn, restored_connectivity_fn, restored_flow_fn, permeability_dict, pixels, unrestorable_matrix=unrestorable_matrix, unrestorable_terrain=unrestorable_terrain)
        self.num_corridors = num_corridors
        self.corridor_hop = corridor_hop
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

        with GeoTiff.from_file(self.flow_fn) as flow_tif:
            flow = flow_tif.get_all_as_tile().m

        highest_flow = self.get_greedy_flow_pixels(flow, n)
        self.changed_pixels = highest_flow.copy()

        permiability_change = 0
        for x, y in highest_flow.keys():
            change = self.change_terrain(x, y, terrain_type, verbose=verbose)
            if change == False:
                self.changed_pixels.pop((x,y))
            else:
                permiability_change += change

        return permiability_change

    '''
    Gets n pixels with the highest flow, with permiability < 1
    :param diff: difference np matrix
    :returns: dict of highest diff pixels formatted {(col,row): death}
    '''
    def get_greedy_flow_pixels(self, flow, n=None):
        if (n == None):
            n = self.pixels
        flow = flow.squeeze(0)
        total_px = flow.shape[0] * flow.shape[1]
        flat_indices = np.argpartition(flow.ravel(), -total_px)[-total_px:]
        row_indices, col_indices = np.unravel_index(flat_indices, flow.shape)

        min_elements = flow[row_indices, col_indices]
        min_elements_order = np.argsort(min_elements)
        row_indices, col_indices = row_indices[min_elements_order], col_indices[min_elements_order]

        highest_flow = {}
        with GeoTiff.from_file(self.terrain_fn) as terr:
            raw_terrain = terr.get_all_as_tile().m.squeeze(0)
        with GeoTiff.from_file(self.habitat_fn) as hab:
            raw_hab = hab.get_all_as_tile().m.squeeze(0)

        i = total_px
        while (len(highest_flow.items()) < n and i >= 0):
            while ((col_indices[i-1], row_indices[i-1]) in highest_flow.keys()):
                i-=1
            stack = [(col_indices[i-1], row_indices[i-1])]
            while len(stack) > 0:
                col, row = stack.pop()
                terrain = raw_terrain[row][col]
                permiability = self.permeability_dict[terrain]
                if permiability < 1 and raw_hab[row][col] != 1:
                    highest_flow[(col, row)] = flow[row][col]
                    # find highest grad of neighbors and push to stack
                    max_neighbor = ()
                    max_neighbor_flow = 0
                    for neighbor_col, neighbor_row in [(col - self.corridor_hop, row), (col + self.corridor_hop, row), (col, row - self.corridor_hop), (col, row + self.corridor_hop), (col - self.corridor_hop, row - self.corridor_hop), (col + self.corridor_hop, row + self.corridor_hop), (col - self.corridor_hop, row + self.corridor_hop), (col + self.corridor_hop, row - self.corridor_hop)]:
                        neighbor_flow = flow[neighbor_row][neighbor_col]
                        if neighbor_flow > max_neighbor_flow and (neighbor_col, neighbor_row) not in highest_flow.keys():
                            max_neighbor = (neighbor_col, neighbor_row)
                            max_neighbor_flow = neighbor_flow
                    if max_neighbor_flow > 0:
                        stack.append(max_neighbor)
                else:
                    break
            i -= 1
        return highest_flow

'''
BFS Flow restoration
    1. Compute connectivity to get gradient, and for baseline comparison
    2. Perform BFS search with starting nodes of num_corridors highest flow pixels until n eligible pixels have been chosen (if not enough eligible px found keeps adding nodes from highest flow)
    3. Restore each of those pixels to higher permiability terrain
    4. Compute connectivity with restored terrain
    5. Evaluate the ratio between change in connectivity and restored permiability
'''
class bfsFlowRestoration(restorationOptimizer):
    def __init__(self, habitat_fn, terrain_fn, restored_terr_fn, connectivity_fn, flow_fn, restored_connectivity_fn, restored_flow_fn, permeability_dict, pixels, unrestorable_matrix=None, unrestorable_terrain=[], num_corridors=None, corridor_hop=1, metric='flow'):
        super().__init__(habitat_fn, terrain_fn, restored_terr_fn, connectivity_fn, flow_fn, restored_connectivity_fn, restored_flow_fn, permeability_dict, pixels, unrestorable_matrix=unrestorable_matrix, unrestorable_terrain=unrestorable_terrain)
        self.num_corridors = num_corridors
        self.corridor_hop = corridor_hop
        self.metric = metric # can be 'flow', 'flow/perm'

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

        with GeoTiff.from_file(self.flow_fn) as flow_tif:
            flow = flow_tif.get_all_as_tile().m

        if self.metric == 'flow/perm':
            flow = self.get_flow_div_perm_layer(flow)

        highest_flow = self.get_bfs_flow_pixels(flow, n)
        self.changed_pixels = highest_flow.copy()

        permiability_change = 0
        for x, y in highest_flow.keys():
            change = self.change_terrain(x, y, terrain_type, verbose=verbose)
            if change == False:
                self.changed_pixels.pop((x,y))
            else:
                permiability_change += change

        return permiability_change

    def get_bfs_flow_pixels(self, flow, n=None):
        if (n == None):
            n = self.pixels
        if (self.num_corridors == None):
            self.num_corridors = n / 5
        flow = flow.squeeze(0)
        total_px = flow.shape[0] * flow.shape[1]
        flat_indices = np.argpartition(flow.ravel(), -total_px)[-total_px:]
        row_indices, col_indices = np.unravel_index(flat_indices, flow.shape)

        min_elements = flow[row_indices, col_indices]
        min_elements_order = np.argsort(min_elements)
        row_indices, col_indices = row_indices[min_elements_order], col_indices[min_elements_order]

        highest_flow = {}
        with GeoTiff.from_file(self.terrain_fn) as terr:
            raw_terrain = terr.get_all_as_tile().m.squeeze(0)
        with GeoTiff.from_file(self.habitat_fn) as hab:
            raw_hab = hab.get_all_as_tile().m.squeeze(0)

        stack = []
        seen = []
        i = total_px
        while len(stack) < self.num_corridors:
            terrain = raw_terrain[row_indices[i-1]][col_indices[i-1]]
            permiability = self.permeability_dict[terrain]
            if permiability < 1 and raw_hab[row_indices[i-1]][col_indices[i-1]] != 1:
                stack.append((col_indices[i-1], row_indices[i-1]))
            i -= 1
        while (len(highest_flow.items()) < n and len(stack) > 0):
            col, row = stack.pop(0)
            seen.append((col, row))
            terrain = raw_terrain[row][col]
            permiability = self.permeability_dict[terrain]
            if permiability < 1 and raw_hab[row][col] != 1:
                highest_flow[(col, row)] = flow[row][col]
            # find highest grad of neighbors and push to stack
            max_neighbor = ()
            max_neighbor_flow = 0
            for neighbor_col, neighbor_row in [(col - self.corridor_hop, row), (col + self.corridor_hop, row), (col, row - self.corridor_hop), (col, row + self.corridor_hop), (col - self.corridor_hop, row - self.corridor_hop), (col + self.corridor_hop, row + self.corridor_hop), (col - self.corridor_hop, row + self.corridor_hop), (col + self.corridor_hop, row - self.corridor_hop)]:
                if neighbor_col >= 0 and neighbor_col < flow.shape[1] and neighbor_row >= 0 and neighbor_row < flow.shape[0]:
                    neighbor_flow = flow[neighbor_row][neighbor_col]
                    if neighbor_flow > max_neighbor_flow and (neighbor_col, neighbor_row) not in highest_flow and (neighbor_col, neighbor_row) not in seen and (neighbor_col, neighbor_row) not in stack and self.permeability_dict[raw_terrain[row][col]] < 1 and raw_hab[row][col] != 1: 
                        max_neighbor = (neighbor_col, neighbor_row)
                        max_neighbor_flow = neighbor_flow
            if max_neighbor_flow > 0:
                stack.append(max_neighbor)
            if len(stack) == 0:
                while ((col_indices[i-1], row_indices[i-1]) in seen):
                    i-=1
                stack.append((col_indices[i-1], row_indices[i-1]))
                i -= 1
        return highest_flow

    '''
    Calculates death based on flow layer
    :param death_tif: filename of death geotiff
    '''
    def get_flow_div_perm_layer(self, flow_mat):
        with GeoTiff.from_file(self.terrain_fn) as terrain_geotiff:
            raw_terrain = terrain_geotiff.get_all_as_tile().m
        raw_flow = flow_mat.astype('float64')
        
        permeability = ecoscape_connectivity.util.dict_translate(raw_terrain, self.permeability_dict)
        sensitivity = np.divide(raw_flow, permeability, out=np.zeros_like(raw_flow), where=permeability!=0.0)
        flow_div_perm = np.clip(np.log10(1. + sensitivity) * 20., 0, 255).astype(np.uint8)
        return flow_div_perm

'''
Performs defecit restoration based on lower resolution terrain, scaling the pixels to squares of the area of restoration
* Attempting to focus more on corridors/clusters
    1. Compute connectivity
    2. Compute death
    3. Scale down death tif
    4. Calculate highest death on low-res death tif
    5. Restore_pixels should take the low-res high death pixels and restore that region on OG terrain
    6. Rerun connectivity w/ restored terrain
'''
class lowResFlowRestoration(restorationOptimizer):
    def __init__(self, 
                habitat_fn, 
                terrain_fn, 
                restored_terr_fn, 
                connectivity_fn, 
                restored_connectivity_fn,
                flow_fn, 
                restored_flow_fn, 
                scaled_flow_fn,
                permeability_dict, 
                rscale, cscale,
                pixels, 
                unrestorable_matrix=None,
                unrestorable_terrain=[],
                percent_impermiable=1):
        super().__init__(habitat_fn, terrain_fn, restored_terr_fn, connectivity_fn, flow_fn, restored_connectivity_fn, restored_flow_fn, permeability_dict, pixels, unrestorable_matrix, unrestorable_terrain)
        self.scaled_flow_fn = scaled_flow_fn
        self.rscale = rscale
        self.cscale = cscale
        self.percent_impermiable = percent_impermiable

    def restore(self, n=None, terrain_type=None, verbose=False):
        if (n==None):
            n = self.pixels
        permiability_restored = self.restore_pixels(n=n, terrain_type=terrain_type, flow_fn=self.flow_fn, terrain_fn=self.terrain_fn, restored_terrain_fn=self.restored_terr_fn, verbose=verbose)
        return permiability_restored

    '''
    Gets n pixels with the highest flow, with permiability < 1
    :param flow: difference np matrix
    :returns: dict of highest diff pixels formatted {(col,row): death}
    '''
    def get_highest_flow_pixels(self, flow, scale, n=None):
        if (n == None):
            n = self.pixels
        flow = flow.squeeze(0)
        total_px = flow.shape[0] * flow.shape[1]
        flat_indices = np.argpartition(flow.ravel(), -total_px)[-total_px:]
        row_indices, col_indices = np.unravel_index(flat_indices, flow.shape)

        min_elements = flow[row_indices, col_indices]
        min_elements_order = np.argsort(min_elements)
        row_indices, col_indices = row_indices[min_elements_order], col_indices[min_elements_order]

        highest_flow = []
        with GeoTiff.from_file(self.terrain_fn) as terr:
            raw_terrain = terr.get_all_as_tile().m.squeeze(0)
        with GeoTiff.from_file(self.habitat_fn) as hab:
            raw_hab = hab.get_all_as_tile().m.squeeze(0)

        k = total_px
        scale = self.rscale * self.cscale
        while (len(highest_flow) < n and k > 0):
            x = col_indices[k-1]
            y = row_indices[k-1]
            not_permiable = []
            for i in range(x * self.cscale, x * self.cscale + self.cscale):
                for j in range(y * self.rscale, y * self.rscale + self.rscale):
                    terrain = raw_terrain[j][i]
                    permiability = self.permeability_dict[terrain]
                    if permiability != 1 and raw_hab[j][i] != 1 and terrain not in self.unrestorable_terrain:
                        if self.unrestorable_matrix is None or (self.unrestorable_matrix is not None and self.unrestorable_matrix[row_indices[i-1]][col_indices[i-1]] != 1):
                            not_permiable.append((i, j))
            if len(not_permiable) / scale >= self.percent_impermiable:
                highest_flow.extend(not_permiable)
            k -= 1
        return highest_flow

    '''
    Restores n pixels
    :param x: col value of pixel coordinate to change
    :param y: row value of pixel coordinate to change
    :param ter_fn: terrain file name, or self.terrain if None
    :param flow_fn: flow filename to calculate the death layer
    :param verbose: Prints the terrain code and permiability of the changed pixel before and after change
    '''
    def restore_pixels(self, n=None, terrain_type=None, flow_fn=None, terrain_fn=None, restored_terrain_fn=None, verbose=False):
        terrain_fn = terrain_fn if terrain_fn else self.terrain_fn
        restored_terrain_fn = restored_terrain_fn if restored_terrain_fn else self.restored_terr_fn

        current_terr_tile = GeoTiff.from_file(terrain_fn).get_all_as_tile()
        with GeoTiff.from_file(self.restored_terr_fn) as restored_terr:
            restored_terr.set_tile(current_terr_tile)
        with GeoTiff.from_file(self.flow_fn) as flow:
            flow_mat = flow.get_all_as_tile().m

        self.scale_geotiff(self.flow_fn, self.scaled_flow_fn, row_pixels=self.rscale, col_pixels=self.cscale)
        with GeoTiff.from_file(self.scaled_flow_fn) as scaled_flow:
            scaled_flow_mat = scaled_flow.get_all_as_tile().m
        highest_flow = self.get_highest_flow_pixels(scaled_flow_mat, n)
        
        with GeoTiff.from_file(self.terrain_fn) as terr:
            raw_terrain = terr.get_all_as_tile().m
            raw_terrain = raw_terrain.squeeze(0)

        changed_pixels = {}
        permiability_change = 0
        for x, y in highest_flow:
            change = self.change_terrain(x,y, terrain_type, verbose=verbose)
            if change != False:
                permiability_change += change
                changed_pixels[(x,y)] = flow_mat[0][y][x]
        self.changed_pixels = changed_pixels
        return permiability_change

'''
Performs defecit restoration based on lower resolution terrain, scaling the pixels to squares of the area of restoration
* Attempting to focus more on corridors/clusters
    1. Compute connectivity
    2. Compute death
    3. Scale down death tif
    4. Calculate highest death on low-res death tif
    5. Restore_pixels should take the low-res high death pixels and restore that region on OG terrain
    6. Rerun connectivity w/ restored terrain
'''
class lowResDefecitRestoration(defecitRestoration):
    def __init__(self, 
                habitat_fn, 
                terrain_fn, 
                restored_terr_fn, 
                connectivity_fn, 
                restored_connectivity_fn,
                flow_fn, 
                restored_flow_fn, 
                death_fn, 
                scaled_death_fn,
                permeability_dict, 
                rscale, cscale,
                pixels, 
                unrestorable_matrix=None,
                unrestorable_terrain=[]):
        super().__init__(habitat_fn, terrain_fn, restored_terr_fn, connectivity_fn, flow_fn, restored_connectivity_fn, restored_flow_fn, death_fn, permeability_dict, pixels, unrestorable_matrix=unrestorable_matrix, unrestorable_terrain=unrestorable_terrain)
        self.scaled_death_fn = scaled_death_fn
        self.rscale = rscale
        self.cscale = cscale

    def restore(self, n=None, terrain_type=None, verbose=False):
        if (n==None):
            n = self.pixels
        permiability_restored = self.restore_pixels(n=n, terrain_type=terrain_type, flow_fn=self.flow_fn, terrain_fn=self.terrain_fn, restored_terrain_fn=self.restored_terr_fn, verbose=verbose)
        return permiability_restored

    '''
    Restores n pixels
    :param x: col value of pixel coordinate to change
    :param y: row value of pixel coordinate to change
    :param ter_fn: terrain file name, or self.terrain if None
    :param flow_fn: flow filename to calculate the death layer
    :param verbose: Prints the terrain code and permiability of the changed pixel before and after change
    '''
    def restore_pixels(self, n=None, terrain_type=None, flow_fn=None, terrain_fn=None, restored_terrain_fn=None, verbose=False):
        terrain_fn = terrain_fn if terrain_fn else self.terrain_fn
        restored_terrain_fn = restored_terrain_fn if restored_terrain_fn else self.restored_terr_fn

        current_terr_tile = GeoTiff.from_file(terrain_fn).get_all_as_tile()
        with GeoTiff.from_file(self.restored_terr_fn) as restored_terr:
            restored_terr.set_tile(current_terr_tile)

        death_tif = self.get_death_layer(self.death_fn, flow_fn=flow_fn, terrain_fn=terrain_fn)
        death_mat = death_tif.get_all_as_tile().m
        self.scale_geotiff(self.death_fn, self.scaled_death_fn, row_pixels=self.rscale, col_pixels=self.cscale)
        scaled_death_tif = GeoTiff.from_file(self.scaled_death_fn)
        highest_death = self.get_highest_pixels_from_tif(scaled_death_tif, math.ceil(n/(self.rscale * self.cscale)))
        changed_pixels = {}

        permiability_change = 0
        for x, y in highest_death.keys():
            for i in range(x * self.rscale, x * self.rscale + self.rscale):
                for j in range(y * self.cscale, y * self.cscale + self.cscale):
                    change = self.change_terrain(i,j, terrain_type, verbose=verbose)
                    if change is not None:
                        permiability_change += change
                        changed_pixels[(i,j)] = death_mat[0][j][i]
        self.changed_pixels = changed_pixels
        return permiability_change

'''
Performs defecit restoration based on lower resolution terrain, scaling the pixels to squares of the area of restoration
* Attempting to focus more on corridors/clusters
    1. Compute connectivity
    2. Compute death
    3. Scale down death tif
    4. Calculate highest death on low-res death tif
    5. Restore_pixels should take the low-res high death pixels and restore that region on OG terrain
    6. Rerun connectivity w/ restored terrain
'''
class lowResProbablisticDefecitRestoration(defecitRestoration):
    def __init__(self, 
                habitat_fn, 
                terrain_fn, 
                restored_terr_fn, 
                connectivity_fn, 
                restored_connectivity_fn,
                flow_fn, 
                restored_flow_fn, 
                death_fn, 
                scaled_death_fn,
                permeability_dict, 
                rscale, cscale,
                pixels,
                unrestorable_matrix=None, 
                unrestorable_terrain=[]):
        super().__init__(habitat_fn, terrain_fn, restored_terr_fn, connectivity_fn, flow_fn, restored_connectivity_fn, restored_flow_fn, death_fn, permeability_dict, pixels, unrestorable_matrix=unrestorable_matrix, unrestorable_terrain=unrestorable_terrain)
        self.scaled_death_fn = scaled_death_fn
        self.rscale = rscale
        self.cscale = cscale

    def restore(self, n=None, terrain_type=None, verbose=False):
        if (n==None):
            n = self.pixels
        permiability_restored = self.restore_pixels(n=n, terrain_type=terrain_type, flow_fn=self.flow_fn, terrain_fn=self.terrain_fn, restored_terrain_fn=self.restored_terr_fn, verbose=verbose)
        return permiability_restored

    '''
    Restores n pixels
    :param x: col value of pixel coordinate to change
    :param y: row value of pixel coordinate to change
    :param ter_fn: terrain file name, or self.terrain if None
    :param flow_fn: flow filename to calculate the death layer
    :param verbose: Prints the terrain code and permiability of the changed pixel before and after change
    '''
    def restore_pixels(self, n=None, terrain_type=None, flow_fn=None, terrain_fn=None, restored_terrain_fn=None, verbose=False):
        terrain_fn = terrain_fn if terrain_fn else self.terrain_fn
        restored_terrain_fn = restored_terrain_fn if restored_terrain_fn else self.restored_terr_fn

        current_terr_tile = GeoTiff.from_file(terrain_fn).get_all_as_tile()
        with GeoTiff.from_file(self.restored_terr_fn) as restored_terr:
            restored_terr.set_tile(current_terr_tile)

        death_tif = self.get_death_layer(self.death_fn, flow_fn=flow_fn, terrain_fn=terrain_fn)
        death_mat = death_tif.get_all_as_tile().m

        self.scale_geotiff(self.death_fn, self.scaled_death_fn, row_pixels=self.rscale, col_pixels=self.cscale)

        scaled_death_tif = GeoTiff.from_file(self.scaled_death_fn)
        scaled_death_mat = scaled_death_tif.get_all_as_tile().m
        _, rows, cols = scaled_death_mat.shape

        # cast death_mat to 1d to use numpy.random.choice with p for weights by values
        death_mat_probs = np.cbrt(scaled_death_mat.ravel())
        death_mat_probs = np.divide(death_mat_probs, np.sum(death_mat_probs))
        # sample indexes from range of probs, with no replacement and weighted by probs
        death_indices = np.random.choice(np.arange(death_mat_probs.size), size=math.ceil(n/(self.rscale * self.cscale)), replace=False, p=death_mat_probs)
        changed_pixels = {}
        permiability_change = 0 

        for index in death_indices:
            x, y = index // cols, index % cols # col, row
            for i in range(x * self.rscale, x * self.rscale + self.rscale):
                for j in range(y * self.cscale, y * self.cscale + self.cscale):

                    change = self.change_terrain(i,j, terrain_type, verbose=verbose)
                    if change is not None:
                        permiability_change += change
                        changed_pixels[(i,j)] = death_mat[0][i][j]

        self.changed_pixels = changed_pixels

        return permiability_change

'''
Flow based restoration for multiple species
    1. Compute connectivity for both species to get gradients
    2. Take average of gradients
    2. Get n pixels with highest average gradient, or flow
    3. Restore each of those pixels to the highest permiability among the species
    4. Compute connectivity for both with restored terrain
    5. Evaluate the ratio between change in connectivity and restored permiability
'''
class multiFlowRestoration(restorationOptimizer):
    '''
    :list habitat_fn: list of habitat filenames
    :string terrain_fn: one terrain filename
    :string restored_terr_fn: one restored terrain filename
    :list connectivity_fn: list of connectivity filenames
    :list flow_fn: list of flow filenames
    :list restored_connectivity_fn: list of restored connectivity filenames
    :list restored_flow_fn: list of restored flow filenames
    :list permeability_dict: list of permeability dict filenames
    :int pixels: number px to restore
    :matrix unrestorable_matrix: unrestorable matrix
    :list unrestorable_terrain: unrestorable terrain
    '''
    def __init__(self, habitat_fns, terrain_fn, restored_terr_fns, connectivity_fns, flow_fns, restored_connectivity_fns, restored_flow_fns, permeability_dicts, bird_run_params, avg_flows_fn, pixels, unrestorable_matrix=None, unrestorable_terrain=[]):
        self.habitat_fn = habitat_fns
        self.terrain_fn = terrain_fn
        self.restored_terr_fn = restored_terr_fns
        self.connectivity_fn = connectivity_fns
        self.flow_fn = flow_fns
        self.restored_connectivity_fn = restored_connectivity_fns
        self.restored_flow_fn = restored_flow_fns
        self.changed_pixels = None
        self.pixels = pixels
        self.permeability_dict = permeability_dicts
        self.unrestorable_terrain = unrestorable_terrain
        self.unrestorable_matrix = unrestorable_matrix
        self.avg_flows_fn = avg_flows_fn
        self.bird_run_params = bird_run_params
        self.restorers = []
        for i in range(len(habitat_fns)):
            print(permeability_dicts[i])
            restorer = flowRestoration(habitat_fns[i], terrain_fn, restored_terr_fns[i], connectivity_fns[i], flow_fns[i], restored_connectivity_fns[i], restored_flow_fns[i], permeability_dicts[i], pixels, unrestorable_matrix=unrestorable_matrix, unrestorable_terrain=unrestorable_terrain)
            self.restorers.append(restorer)

    '''
    Runs connectivity for either true or restored terrain
    '''
    def run_connectivity(self, single_tile=True, deterministic=True, restored=False, gap_crossing=None, num_gaps=None):
        for i, restorer in enumerate(self.restorers):
            if gap_crossing == None or num_gaps == None:
                gap_crossing, num_gaps = self.bird_run_params[i]
            if (restored):
                repopulation.compute_connectivity(restorer.habitat_fn, restorer.restored_terr_fn, restorer.restored_connectivity_fn, restorer.restored_flow_fn, restorer.permeability_dict, num_gaps=num_gaps, single_tile=single_tile, gap_crossing=gap_crossing, deterministic=deterministic)
            else:
                repopulation.compute_connectivity(restorer.habitat_fn, restorer.terrain_fn, restorer.connectivity_fn, restorer.flow_fn, restorer.permeability_dict, num_gaps=num_gaps, single_tile=single_tile, gap_crossing=gap_crossing, deterministic=deterministic)

    '''
    :param n: number of highest death pixels to restore
    :param terrain_type: terrain type to restore to
    :param verbose: print terrain conversion of every pixel
    '''
    def restore(self, n=None, terrain_type=None, verbose=False):
        if (n==None):
            n = self.pixels

        flows = []
        perm_arr = []
        for restorer in self.restorers:
            perm_arr.append(restorer.permeability_dict)
            with GeoTiff.from_file(restorer.flow_fn) as flow_tif:
                flow = flow_tif.get_all_as_tile().m
                flows.append(flow)
                flow_tif.clone_shape(self.avg_flows_fn)

        # calculate average flow
        flow_avg = flows[0]
        for flow in flows[1:]:
            flow_avg += flow
        flow_avg = flow_avg / len(self.restorers)

        with GeoTiff.from_file(self.avg_flows_fn) as avg_flow_tif:
            tile = avg_flow_tif.get_all_as_tile()
            tile.m = flow_avg
            avg_flow_tif.set_tile(tile)

        # get best terrain type for all
        if (terrain_type == None):
            terrain_type = self.get_most_permiable_terrain_from_mult(perm_arr)
        self.terrain_type = terrain_type

        # find pixels to restore
        highest_flow = self.get_highest_flow_pixels(flow_avg, n)
        permiability_changes = []
        for restorer in self.restorers:
            current_terr_tile = GeoTiff.from_file(self.terrain_fn).get_all_as_tile()
            with GeoTiff.from_file(restorer.restored_terr_fn) as restored_terr:
                restored_terr.set_tile(current_terr_tile)

            restorer.changed_pixels = highest_flow.copy()

            permiability_change = 0
            for x, y in highest_flow.keys():
                change = restorer.change_terrain(x, y, terrain_type=terrain_type, verbose=verbose)
                if change == False:
                    restorer.changed_pixels.pop((x,y))
                else:
                    permiability_change += change

            permiability_changes.append(permiability_change)

        return permiability_changes

    '''
    Gets n pixels with the highest flow, with permiability < 1
    :param diff: difference np matrix
    :returns: dict of highest diff pixels formatted {(col,row): death}
    '''
    def get_highest_flow_pixels(self, flow, n=None):
        if (n == None):
            n = self.pixels
        flow = flow.squeeze(0)
        total_px = flow.shape[0] * flow.shape[1]
        flat_indices = np.argpartition(flow.ravel(), -total_px)[-total_px:]
        row_indices, col_indices = np.unravel_index(flat_indices, flow.shape)

        min_elements = flow[row_indices, col_indices]
        min_elements_order = np.argsort(min_elements)
        row_indices, col_indices = row_indices[min_elements_order], col_indices[min_elements_order]

        highest_flow = {}
        all_habs = []
        all_ters = []
        all_perms = []
        for restorer in self.restorers:
            with GeoTiff.from_file(restorer.habitat_fn) as hab:
                raw_hab = hab.get_all_as_tile().m.squeeze(0)
                all_habs.append(raw_hab)
            with GeoTiff.from_file(restorer.terrain_fn) as terr:
                raw_terrain = terr.get_all_as_tile().m.squeeze(0)
                all_ters.append(raw_terrain)
            all_perms.append(restorer.permeability_dict)

        i = total_px
        while (len(highest_flow.items()) < n and i > 0):
            eligible_pixel = True
            for hab, ter, perm in zip(all_habs, all_ters, all_perms):
                terrain = ter[row_indices[i-1]][col_indices[i-1]]
                permiability = perm[terrain]
                if permiability >= 1 or hab[row_indices[i-1]][col_indices[i-1]] == 1 or terrain in self.unrestorable_terrain or (self.unrestorable_matrix is not None and self.unrestorable_matrix[row_indices[i-1]][col_indices[i-1]] == 1):
                    eligible_pixel = False
            if eligible_pixel:
                highest_flow[(col_indices[i-1], row_indices[i-1])] = flow[row_indices[i-1]][col_indices[i-1]]
            i -= 1
        return highest_flow

    '''
    Get the change in connectivity before and after restoration
    '''
    def get_delta_connectivity(self):
        delta_conns = []
        for restorer in self.restorers:
            pre_restoration_conn = restorer.sum_of_tif(restorer.connectivity_fn)
            post_restoration_conn = restorer.sum_of_tif(restorer.restored_connectivity_fn)
            delta_conns.append(int(post_restoration_conn) - int(pre_restoration_conn))
        return delta_conns