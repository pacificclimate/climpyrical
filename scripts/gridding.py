import numpy as np
from numba import jit
from scipy.spatial import distance
from pyproj import Proj, transform
from functools import partial

def rlat_rlon_to_ens(rlat, rlon):
    """Takes the rlat and rlon 1D arrays from the
    NetCDF files for each ensemble member, and creates
    an ordered pairing of each grid cell coordinate in
    rotated pole (rlat, rlon).

    Args:
        rlat (numpy.ndarray): 1D array containing
            the locations of the rotated latitude
            grid cells
        rlat (numpy.ndarray): 1D array containing
            the locations of the rotated longitude
            grid cells

    Return:
        lat_lon_ens (numpy.ndarray): array containing
            tuples of rlat and rlon for each grid cell
            in the ensemble shape.
    """

    rlon_ens = np.tile(rlon, rlat.shape[0])
    rlat_ens = np.repeat(rlat, rlon.shape[0])

    coord_dict = {
                'rlat': rlat_ens,
                'rlon': rlon_ens
    }

    return coord_dict

def ens_obs_distance(lat_lon_ens, coord, method='haversine'):
    '''Determines the distances between a station's coordinates
    and the grid cells in North America.
    Args:
        lat_lon_obs (numpy.ndarray): array containing
            tuple pairs of latitudes and longitudes of
            stations
        coord (tuple): lat lon location of station
        method (str): method to use for the
            distance calculations
    Returns:
        (numpy.ndarray): array containing the
            distance between each grid cell and the station
            from coord
    '''

    if method == 'haversine':
        lat1, lon1 = coord
        lat2, lon2 = zip(*lat_lon_ens)

        lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

        dlon = lon2 - lon1
        dlat = lat2 - lat1

        a = np.sin(dlat/2.0)**2+np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2.0)**2

        c = 2 * np.arcsin(np.sqrt(a))
        return c.argmin()

    if method == 'euclidean':
        return distance.cdist(
                            lat_lon_ens,
                            [coord],
                            'euclidean').argmin()
    else:
        raise ValueError("must be \'euclidean\' or \'haversine\'")

#@jit(nopython=True, parallel=True)
def dist_index(lat_lon_obs, lat_lon_ens, method='haversine'):
    """Determines the index in the ensemble shape
    of grid cells with coordinates that are the closest
    in euclidean distance to stations.
    Args:
        lat_lon_obs (numpy.ndarray): array containing
            tuple pairs of latitudes and longitudes of
            stations
        lat_lon_ens (numpy.ndarray): array containing
            typle pairs of latitudes and longitudes
            locations of the grid cells in the ensemble
        method (str): method to use for the
            distance calculations
    Returns:
        dist_list (numpy.ndarray): array containing the
            indices of grid cells in the ensemble shape
            that are closest to the station locations
    """

    lat_obs, lon_obs = zip(*lat_lon_obs)
    lat_ens, lon_ens = zip(*lat_lon_ens)

    dist_list = []

    for i, coord in enumerate(lat_lon_obs):
        print(coord)
        dist_list.append(
                        ens_obs_distance(
                                    lat_lon_ens,
                                    coord,
                                    method
                        )
        )

    return np.asarray(dist_list)

def find_nearest(array, value):

    if array.shape[0] == 1:
        return 1

    if array.shape[0] == 2:
        return (np.abs(array - value)).argmin()

    nidx = int(array.shape[0]/2)

    if array[nidx] == value:
        return nidx

    elif  value < array[nidx]:
        return find_nearest(array[:nidx], value)

    else:
        return (nidx-1)+find_nearest(array[nidx:], value)

#@jit(nopython=True, parallel=True)
def to_rotated(
    lat_obs, lon_obs, 
    proj4_str = '+proj=ob_tran +o_proj=latlon +o_lon_p=-97 +o_lat_p=42.5 +lon_0=180 +ellps=WGS84'
    ):

    rpole = Proj(proj4_str)
    crs = Proj('+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs')

    print(crs, rpole)

    transformer = partial(transform, crs, rpole)

    rlon_obs, rlat_obs = transformer(lon_obs, lat_obs)

    coord_dict = {
        'rlat_obs': rlat_obs,
        'rlon_obs': rlon_obs
    }

    return coord_dict
