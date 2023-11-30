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
    def run_connectivity(self, single_tile=True, deterministic=True, restored=False, hop_length=1):
        if (restored):
            repopulation.compute_connectivity(self.habitat_fn, self.restored_terr_fn, self.restored_connectivity_fn, self.restored_flow_fn, self.permeability_dict, single_tile=single_tile, gap_crossing=hop_length, deterministic=deterministic)
        else:
            repopulation.compute_connectivity(self.habitat_fn, self.terrain_fn, self.connectivity_fn, self.flow_fn, self.permeability_dict, single_tile=single_tile, gap_crossing=hop_length, deterministic=deterministic)

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

        with GeoTiff.from_file(self.connectivity_fn) as flow_geotiff:
            flow_geotiff.clone_shape(death_fn)
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
    takes a np matrix and shrinks it to the resolution given, making each pixel the sum of the constituting pixels
    '''
    def lower_res_matrix(self, matrix, rows=None, cols=None, rscale=None, cscale=None):
        r, c = matrix.shape
        if rscale and cscale:
            return matrix.reshape(r//rscale, rscale, c//cscale, cscale).sum(axis=1).sum(axis=2)
        if rows and cols:
            return matrix.reshape(rows, matrix.shape[0]//rows, cols, matrix.shape[1]//cols).sum(axis=1).sum(axis=2)

    '''
    Scales the geotiff data in tif_fn to resolution constituting of pixels of row_pixels x col_pixels, writing scaled tif to scaled_tif_fn
    param row_pixels: number of pixels to combine to 1 pixel on y axis (divisible by tif_fn's height)
    param col_pixels: number of pixels to combine to 1 pixel on x axis (divisible by tif_fn's width)
    '''
    def scale_geotiff(self, tif_fn, scaled_tif_fn, row_pixels, col_pixels):
        with GeoTiff.from_file(tif_fn) as tif:
            mat = tif.get_all_as_tile().m.squeeze(0)
            scaled_mat = self.lower_res_matrix(mat, rscale=row_pixels, cscale=col_pixels)
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
            old_terrain = terrain_geotiff.get_pixel_value(x, y)

            m = np.array([[[terrain_type]]])
            tile = Tile(1, 1, 0, 1, x, y, m)
            terrain_geotiff.set_tile(tile)
        
        if (verbose):
            print(f'Restoring pixel ({x}, {y}) from permiability {self.permeability_dict[old_terrain]} to {self.permeability_dict[terrain_type]}')

        permiability_change = self.permeability_dict[terrain_type] - self.permeability_dict[old_terrain]
        return permiability_change

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

        permiability_change = 0 
        for x, y in highest_death.keys():
            permiability_change += self.change_terrain(x, y, terrain_type, verbose=verbose)
        return permiability_change

    def get_big_tile_reader(self, tif, width, height):
        reader = tif.get_reader(b=0, w=width, h=height)
        return reader

    # paint the changed terrain pixels with death values
    def paint_changed_terrain_geotiff(self, changed_terrain_fn, changed_pixels=None):
        if (changed_pixels == None):
            changed_pixels = self.highest_death
        with GeoTiff.from_file(self.connectivity_fn) as connectivity_tif:
            connectivity_tif.clone_shape(changed_terrain_fn)

        with GeoTiff.from_file(changed_terrain_fn) as changed_terrain_tif:
            for (x,y), death in changed_pixels.items():
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
        permiability_restored = self.restore_pixels(n=n, terrain_type=terrain_type, verbose=verbose)
        return permiability_restored


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
    def restore(self, n=None, terrain_type=None, verbose=False, weighted=False):
        if (n==None):
            n = self.pixels

        current_terr_tile = GeoTiff.from_file(self.terrain_fn).get_all_as_tile()
        with GeoTiff.from_file(self.restored_terr_fn) as restored_terr:
            restored_terr.set_tile(current_terr_tile)

        diff = self.get_flow_diff() if (not weighted) else self.get_flow_diff_weighted_by_permiability()

        highest_diff = self.get_highest_diff_pixels(diff, n)
        self.highest_death = highest_diff
        print(highest_diff)

        permiability_change = 0
        for x, y in highest_diff.keys():
            permiability_change += self.change_terrain(x, y, terrain_type, verbose=verbose)

        return permiability_change

    '''
    Adds noise to transmission raster then runs connectivity based on noisy terrain permiability
    '''
    def run_utopian_connectivity(self, single_tile=True, deterministic=True):
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
            raw_flow = flow.get_all_as_tile().m.astype(np.int16)
        with GeoTiff.from_file(self.utopian_flow_fn) as utopian_flow:
            raw_utopian_flow = utopian_flow.get_all_as_tile().m.astype(np.int16)
        
        return raw_utopian_flow - raw_flow

    '''
    Calculates difference btw utopian and regular flow, weighted by permiability inversed
    '''
    def get_flow_diff_weighted_by_permiability(self):
        with GeoTiff.from_file(self.flow_fn) as flow:
            raw_flow = flow.get_all_as_tile().m.astype(np.int16)
        with GeoTiff.from_file(self.utopian_flow_fn) as utopian_flow:
            raw_utopian_flow = utopian_flow.get_all_as_tile().m.astype(np.int16)

        diff = raw_utopian_flow - raw_flow
        permiability = self.get_permiability_matrix()
        permiability_inversed = np.ones_like(permiability) - (permiability)**2
        diff_weighted = diff * permiability_inversed
        return diff_weighted

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
        flat_indices = np.argpartition(diff.ravel(), -n)[-n:]
        diff = diff.squeeze(0)
        row_indices, col_indices = np.unravel_index(flat_indices, diff.shape)

        min_elements = diff[row_indices, col_indices]
        min_elements_order = np.argsort(min_elements)
        row_indices, col_indices = row_indices[min_elements_order], col_indices[min_elements_order]

        highest_diff = {}
        with GeoTiff.from_file(self.terrain_fn) as terr:
            raw_terrain = terr.get_all_as_tile().m
            raw_terrain = raw_terrain.squeeze(0)

        i = n
        while (len(highest_diff.items()) < n and i >= 0):
            terrain = raw_terrain[row_indices[i-1]][col_indices[i-1]]
            permiability = self.permeability_dict[terrain]
            if permiability < 1:
                highest_diff[(col_indices[i-1], row_indices[i-1])] = diff[row_indices[i-1]][col_indices[i-1]]
            i -= 1
        
        return highest_diff

'''
Flip one permeability in every dxd square, measure increased flow
'''
class flipRestoration(restorationOptimizer):
    def flip_restoration(self):
        pass

'''
Performs defecit restoration based on lower resolution terrain, scaling the pixels to squares of the area of restoration
* Attempting to focus more on corridors/clusters
    1. Compute connectivity
    2. Scale down the connectivity and flow tifs
        a. scale to pixels of size i x j, where i x j = N
        b. Size of area to restore
    3. Scale terrain geotiff:
        a. to take the value of the most frequent terrain type
        b. to be the average terrain value, creating a new permiability dict that's the average permiability of all pixels
    5. Find highest defecit pixel(s)
    5. Restore region/corridor
    6. Compute connectivity with og inputs again to observe:
        a. change in sum of connectivity
        b. change in sum of connectivity / restored permiability
'''
class lowResDefecitRestoration(restorationOptimizer):
    def flip_restoration_lower_res(self):
        def __init__(self, habitat_fn, terrain_fn, restored_terr_fn, connectivity_fn, flow_fn, restored_connectivity_fn, restored_flow_fn, death_fn, permeability_dict, pixels, low_res_connectivity_fn, low_res_flow_fn, low_res_terrain_fn, permiability=0.9):
            super().__init__(habitat_fn, terrain_fn, restored_terr_fn, connectivity_fn, flow_fn, restored_connectivity_fn, restored_flow_fn, death_fn, permeability_dict, pixels)
            self.low_res_connectivity_fn = low_res_connectivity_fn
            self.low_res_flow_fn = low_res_flow_fn
            self.low_res_terrain_fn = low_res_terrain_fn
            
    '''
    Scale terrain down to scale of all others
    Either:
    a. to take the value of the most frequent terrain type
    b. to be the average terrain value, creating a new permiability dict that's the average permiability of all pixels
    '''
    def scale_terrain_tif(self):

        pass