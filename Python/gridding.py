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

def euclidean_dist_index(lat_lon_obs, lat_lon_ens):
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
    Returns:
        dist_list (numpy.ndarray): array containing the
            indices of grid cells in the ensemble shape
            that are closest to the station locations
    """

    dist_list = []

    for i, coord in enumerate(lat_lon_obs):
        dist_list.append(
                        distance.cdist(lat_lon_ens,
                                       [coord],
                                       'euclidean'
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
