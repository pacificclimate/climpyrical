import numpy as np
from scipy.spatial import distance
from scipy.interpolate import NearestNDInterpolator

from pyproj import Transformer, Proj
from functools import partial

def flatten_coords(x, y):
    """Takes the rlat and rlon 1D arrays from the
    NetCDF files for each ensemble member, and creates
    an ordered pairing of each grid cell coordinate in
    rotated pole (rlat, rlon).

    Args:
        x (numpy.ndarray): 1D array containing
            the locations of the rotated latitude
            grid cells
        y (numpy.ndarray): 1D array containing
            the locations of the rotated longitude
            grid cells

    Return:
        xext, yext (tuple of np.ndarrays):
            array containing tuples of rlat and
            rlon for each grid cell in the
            ensemble shape.
    """

    xext = np.tile(x, y.shape[0])
    yext = np.repeat(y, x.shape[0])

    return xext, yext


def transform_coords(
    x,
    y,
    source_crs = '+init=epsg:4326 +proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs +towgs84=0,0,0',
    target_crs = '+proj=ob_tran +o_proj=longlat +lon_0=-97 +o_lat_p=42.5 +to_meter=0.0174532925199 +a=6378137 +no_defs',
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

    p_source = Proj(source_crs)
    p_target = Proj(target_crs)
    t = Transformer.from_proj(p_source, p_target)

    return t.transform(x, y)


def find_nearest(data, val):
    lo, hi = 0, len(data) - 1
    best_ind = lo
    while lo <= hi:
        mid = int(lo + (hi - lo) / 2)
        if data[mid] < val:
            lo = mid + 1
        elif data[mid] > val:
            hi = mid - 1
        else:
            best_ind = mid
            break
        # check if data[mid] is closer to val than data[best_ind]
        if abs(data[mid] - val) < abs(data[best_ind] - val):
            best_ind = mid
    return best_ind

def find_nearest_pos(x, y, x_obs, y_obs):
    x_i = np.array([find_nearest(x, obs) for obs in x_obs])
    y_i = np.array([find_nearest(y, obs) for obs in y_obs])
    return x_i, y_i

def find_nearest_value(x, y, x_i, y_i, field, mask):
    # check field has same shape of at least (x_i, y_i)

    # find station locations over nan
    nanloc = np.isnan(field[y_i, x_i])
    if np.any(nanloc):
        xarr, yarr = np.meshgrid(x, y)

        # flatten coordinates
        xext, yext = flatten_coords(x, y)

        # arrange the pairs
        pairs = np.array(list(zip(xext, yext)))

        f = NearestNDInterpolator(pairs[mask.flatten()], field[mask])

        x_nan = xarr[y_i, x_i][nanloc]
        y_nan = yarr[y_i, x_i][nanloc]

        nan_pairs = np.array(list(zip(x_nan, y_nan)))

        field[y_i[nanloc], x_i[nanloc]] = f(nan_pairs)

    # check that final array contains no nan

    return field[y_i, x_i]
