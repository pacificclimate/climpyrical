import numpy as np
from scipy.spatial import distance
import sklearn
from sklearn.metrics import pairwise_distances_argmin
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

def dist_index(lat_lon_obs, lat_lon_ens):
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

    lat_obs, lon_obs = np.deg2rad(lat_obs), np.deg2rad(lon_obs)
    lat_ens, lon_ens = np.deg2rad(lat_ens), np.deg2rad(lon_ens)

    lat_lon_ens, lat_lon_obs = list(zip(lat_ens, lon_ens)), list(zip(lat_obs, lon_obs))

    with sklearn.config_context(working_memory=128):
        dist_list = sklearn.metrics.pairwise_distances_argmin(lat_lon_obs, lat_lon_ens, metric='haversine')
    return np.asarray(dist_list)


def to_rotated(
    lat_obs, lon_obs,
    to = '+proj=longlat +ellps=WGS84',
    fro = '+proj=ob_tran +o_proj=longlat +lon_0=-97 +o_lat_p=42.5 +a=1 +to_meter=0.0174532925199 +no_defs'
    ):
    """Rotates regular latlon coordinates to rotated pole
    coordinates given a proj4 string that defines
    the rotated poles. Projection string parameters are defined
    here: https://proj.org/operations/projections/ob_tran.html
    Args:
        lat_obs/lon_obs (numpy.ndarray): array containing
            latitudes and longitudes of
            stations
        proj4_str (str): proj4 string defining rotated pole
            coordinates used.
    Returns:
        coords (dict): dictionary containing the newly rotated
            coordinates
    """

    rpole = Proj(fro)
    crs = Proj(to)

    transformer = partial(transform, crs, rpole)

    rlon_obs, rlat_obs = transformer(lon_obs, lat_obs)

    coord_dict = {
        'rlat_obs': np.rad2deg(rlat_obs),
        'rlon_obs': np.rad2deg(rlon_obs)
    }

    return coord_dict

def match_coords(df, interp_dict, dv_obs_name, master_idx=None):
    """Creates modified pands dataframe containing the index
    in the flattened ensemble shape of the corresponding observation.
    Grid cells with more than one stations are averaged.
    Args:
        df (pandas.DataFrame): dataframe containing the station observations
        interp_dict (dict): dictionary containing the interpolated data
        dv_obs_name (str): name of the design value as found in the station
            observations
    Return:
        ndf (pandas.DataFrame): new dataframe containing the observations
            and the index of the closest model grid cell
    """
    coords = to_rotated(df['lat'].values, df['lon'].values)
    if master_idx is None:
        master_idx = interp_dict['idx']

    ens_coords = list(zip(interp_dict['irlat_ens'][master_idx],
                          interp_dict['irlon_ens'][master_idx])
                )
    obs_coords = list(zip(coords['rlat_obs'], coords['rlon_obs']))

    # get the nearest grids
    df['nearest_grid'] = dist_index(obs_coords, ens_coords)[1]
    df['obs_coords'] = obs_coords

    ndf = df.groupby('nearest_grid').agg({
                                        dv_obs_name: 'mean',
                                        'lat':'min',
                                        'lon':'min',
                                        'obs_coords':'min',
                                    })
    ndf['matched_idx'] = ndf.index

    return ndf
