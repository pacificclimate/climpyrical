import numpy as np
from scipy.spatial import distance
from scipy.interpolate import NearestNDInterpolator

from pyproj import Transformer, Proj
from functools import partial


def check_input_coords(x, y):
    """Checks that the input coordinates defining the CanRCM4 grid
    are the expected type, size, and range of values.
    Args:
        x, y (np.ndarray): numpy arrays of rlon, rlat respectively
            of CanRCM4 grids
    Returns:
        bool True if passed
    """
    if (not isinstance(x, np.ndarray)) or (not isinstance(y, np.ndarray)):
        raise TypeError(
            "Please provide an object of type {}".format(np.ndarray)
        )
    if (not np.isclose(x.max(), 33.8800048828125)) or (
        not np.isclose(x.min(), -33.8800048828125)
    ):
        raise ValueError("Unexpected range of values in x dimension")
    if (not np.isclose(y.max(), 28.15999984741211)) or (
        not np.isclose(y.min(), -28.59999656677246)
    ):
        raise ValueError("Unexpected range of values in y dimension")

    return True


def check_coords_are_flattened(x, y, xext, yext):
    check_input_coords(x, y)
    check_input_coords(xext, yext)

    if xext.size != yext.size:
        # bad shape
        raise ValueError("xext, and yexy must have the same shape")

    if xext.size != x.size * y.size:
        # bad shape
        raise ValueError(
            "extended arrays must be equivalent to the product of the coordinate grid original axis"
        )

    if not np.array_equal(xext[: x.size], xext[x.size : 2 * x.size]):
        # they should all be increasing tile wise
        raise ValueError(
            "Flat coords should increase np.tile-wise, i.e: 1, 2, 3, 1, 2, 3, ..."
        )

    if not np.allclose(yext[: y.size], y[0]):
        # they should all be increasing repeat wise
        raise ValueError(
            "Flat coords should increase np.repeat-wise, i.e: 1, 1, 2, 2, 3, 3, ..."
        )

    return True


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
    check_input_coords(x, y)
    xext = np.tile(x, y.shape[0])
    yext = np.repeat(y, x.shape[0])
    check_coords_are_flattened(x, y, xext, yext)

    return xext, yext


def check_transform_coords_inputs(x, y, source_crs, target_crs):
    if (not isinstance(x, np.ndarray)) or (not isinstance(y, np.ndarray)):
        raise TypeError(
            "Please provide an object of type {}".format(np.ndarray)
        )
    if (not isinstance(source_crs, dict)) or (
        not isinstance(target_crs, dict)
    ):
        raise TypeError("Please provide an object of type {}".format(dict))
    if x.shape != y.shape:
        raise ValueError("x and y must be pairwise station coordinates")

    if (np.any(x <= -139.024025)) or (np.any(x >= -53.006653)):
        raise ValueError(
            "A station location is outside of expected bounds in x dim"
        )

    if (np.any(y >= 82.511053)) or (np.any(y <= 41.631742)):
        raise ValueError(
            "A station location is outside of expected bounds in y dim"
        )

    return True


def transform_coords(
    x,
    y,
    source_crs={"init": "epsg:4326",},
    target_crs={
        "proj": "ob_tran",
        "o_proj": "longlat",
        "lon_0": -97,
        "o_lat_p": 42.5,
        "a": 6378137,
        "to_meter": 0.0174532925199,
        "no_defs": True,
    },
):
    """Rotates regular WGS84 coordinates to rotated pole
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
        x,y (tuple): tuple containing the newly rotated
            coordinates rlon, rlat
    """

    p_source = Proj(source_crs)
    p_target = Proj(target_crs)
    t = Transformer.from_proj(p_source, p_target)

    return t.transform(x, y)


def check_find_nearest_index_inputs(data, val):
    if not isinstance(data, np.ndarray):
        raise TypeError(
            "Please provide a data array of type {}".format(np.ndarray)
        )
    if np.any(np.diff(data) < 0):
        raise ValueError("Array must be monotonically increasing.")
    if data.size < 2:
        raise TypeError("Array size must be greater than 1")

    if not isinstance(val, float):
        raise TypeError("Please provide a value of type {}".format(float))

    if val > data.max() or val < data.min():
        raise ValueError("Value is not within supplied array's range.")

    return True


def find_nearest_index(data, val):
    check_find_nearest_index_inputs(data, val)
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


def check_find_element_wise_nearest_pos_inputs(x, y, x_obs, y_obs):
    is_ndarray = [
        isinstance(array, np.ndarray) for array in [x, y, x_obs, y_obs]
    ]
    if not np.any(is_ndarray):
        raise TypeError(
            "Please provide a data array of type {}".format(np.ndarray)
        )
    if x.size != y.size:
        raise ValueError(
            "To find the values in the supplied arrays, the arrays must be same shape."
        )
    if x_obs.size != y_obs.size:
        raise ValueError(
            "Array of values to find in arrays must be the same shape."
        )

    return True

def find_element_wise_nearest_pos(x, y, x_obs, y_obs):
    check_find_element_wise_nearest_pos_inputs(x, y, x_obs, y_obs)
    x_i = np.array([find_nearest_index(x, obs) for obs in x_obs])
    y_i = np.array([find_nearest_index(y, obs) for obs in y_obs])
    return x_i, y_i

def check_find_nearest_value_inputs(x, y, x_i, y_i, field, mask):
    # x y are numpy arrays consistent with rlon rlat
    if x_i.size >= x.size or y_i.size >= y.size:
        raise ValueError("More stations than grid cells. Ensure correct station array/grid cell array is provided.")
    # field same shape as xiyi
    if field.shape != (y.size, x.size):
        raise ValueError("Field provided is not consistent with coordinates provided.")
    # mask same shape as field
    if field.shape != mask.shape:
        raise ValueError("Field and mask are not the same shape.")

    return True

def check_final(x_i, y_i, final):
    if np.any(np.isnan(final)):
        raise ValueError("Final field contains unexpected NaN values.")
    if final.size != y_i.size or final.size != x_i.size:
        raise ValueError("Final field is not consistent with coordinates provided.")

    return True


def find_nearest_index_value(x, y, x_i, y_i, field, mask):
    # check field has same shape of at least (x_i, y_i)
    # find station locations over nan
    check_find_nearest_value_inputs(x, y, x_i, y_i, field, mask)
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
    final = field[y_i, x_i]
    check_final(x_i, y_i, final)

    return final

