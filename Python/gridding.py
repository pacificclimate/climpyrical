import numpy as np
from scipy.spatial import distance

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
        return c

    if method == 'euclidean':
        return distance.cdist(
                            lat_lon_ens,
                            [coord],
                            'euclidean')
    else:
        raise ValueError("must be \'euclidean\' or \'haversine\'")

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
        dist_list.append(
                        ens_obs_distance(lat_lon_ens,
                                       coord,
                                       method
                        ).argmin()
        )

    return np.asarray(dist_list)

def lat_lon_lookup(rlat, rlon, ds, idx):
    """Using the lat and lon grids in the CanRCM4 models,
    "look-up" the corresponding latitude and longitude at
    the intended grid cell. This effecitvely converts the
    coordinates from rotated pole to the regular lat/lon grids
    such that western longitude is negative.

    Args:
        rlat, rlon (np.ndarray): arrays containing the rotated
            latitude and longitudes of the CanRCM4 models.
        ds (xarray.DataSet): datacube containing the ensemble array
            and the lat and lon grids.
        idx (numpy.ndarray): array containing the indices in the ensemble
            that correspond to qualified grid cells after the master
            mask was applied. These are provided by the
            mask_land_and_nan_ens_index() function in preprocessing.py
    Returns:
        coord_dict (dict): dictionary of containing arrays of regular
            latitude and of longitude in the ensemble shape.

    """
    rlat_ens = rlat_rlon_to_ens(rlat, rlon)['rlat']
    rlon_ens = rlat_rlon_to_ens(rlat, rlon)['rlon']

    lats, lons = [], []

    for i in idx:
        lats.append(ds['lat'].sel(rlon=rlon_ens[i], rlat=rlat_ens[i], method='nearest'))
        lons.append(ds['lon'].sel(rlon=rlon_ens[i], rlat=rlat_ens[i], method='nearest'))

    coord_dict = {
                'lat_ens': np.array(lats),
                'lon_ens': np.array(lons)-360.0
    }

    return coord_dict
