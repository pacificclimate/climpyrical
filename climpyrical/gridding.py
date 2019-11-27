import warnings
import numpy as np
from scipy.interpolate import NearestNDInterpolator
from pyproj import Transformer, Proj


def check_ndims(data, n):
    """Checks that a provided array has n dimensions
    Args:
        data (np.ndarray): Array to check dimensions of
        n (integer): data's expected dimensions
    Raises:
        TypeError:
            If data or n are not arrays or an integer
        ValueError:
            If data's dimension is not expected
    """
    if not isinstance(data, np.ndarray):
        raise TypeError(
            "Provide an array of type {}, received {}"
            .format(np.ndarray, type(data))
        )
    if not isinstance(n, int):
        raise TypeError(
            "Provide a dimension of type {}, received {}"
            .format(int, type(n))
        )
    if data.ndim != n:
        raise ValueError(
            "Array has dimensions {}, expected {} dimensions."
            .format(data.ndim, n)
        )


def check_input_coords(x, y, ds):
    """Checks that the input coordinates defining the CanRCM4 grid
    are the expected type, size, and range of values.
    Args:
        x, y (np.ndarray): numpy arrays of rlon, rlat respectively
            of CanRCM4 grids
        ds (xarray.core.dataset.Dataset): dataset containing the ensemble for
            checking consistency with ensemble
    Raises:
        TypeError:
            If numpy array not provided
        ValueError:
            If x or y are not in expected range of values
    """

    if (not isinstance(x, np.ndarray)) or (not isinstance(y, np.ndarray)):
        raise TypeError(
            "Please provide coordinate objects of type {}".format(np.ndarray)
        )

    check_ndims(x, 1)
    check_ndims(y, 1)

    if (not np.isclose(x.max(), ds.rlon.max())) or (
        not np.isclose(x.min(), ds.rlon.min())
    ):
        raise ValueError(
            "x dimension array must have min/max values between \
            {} and {}. Array \
            provided has values between {} \
            and {}".format(
                        ds.rlon.min(),
                        ds.rlon.max(),
                        x.min(),
                        x.max()
                    )
        )

    if (not np.isclose(y.max(), ds.rlat.max())) or (
        not np.isclose(y.min(), ds.rlat.min())
    ):
        raise ValueError(
            "y dimension array must have min/max values between \
            {} and {}. Array \
            provided has values between {} \
            and {}".format(
                ds.rlat.min(), ds.rlat.max(), y.min(), y.max()
            )
        )


def check_coords_are_flattened(x, y, xext, yext, ds):
    """Checks that the coordinates provided are flattened correctly
    Args:
        x, y (np.ndarray): numpy arrays of rlon, rlat respectively
            of CanRCM4 grids
        xext, yext (np.ndarray): numpy arrays of flattened
            rlon, rlat respectively of CanRCM4 grids
        ds (xarray.core.dataset.Dataset): dataset containing the ensemble for
            checking consistency with ensemble
        Raises:
            ValueError, TypeError in check_input_coords
            TypeError:
                If input coords are not numpy arrays
            ValueError:
                If xext and yexy are not the same size
                If extended flattened size is not expected
                If flattened longitude is not increasing
                    numpy tile-wise
                If flattened latitude is not increasing
                    numpy repeat-wise
    """
    check_input_coords(x, y, ds)
    check_input_coords(xext, yext, ds)

    if xext.size != yext.size:
        # bad shape
        raise ValueError(
            "xext, and yexy must have the same size, \
            received x size {} and y size {}."
            .format(x.size, y.size)
        )

    if xext.size != x.size * y.size:
        # bad size
        raise ValueError(
            "Extended arrays must be equivalent to the product of the "
            "coordinate grid original axis. Received size {}, based on "
            "provided coordinates, expected size {}."
            .format(xext.size, x.size * y.size)
        )

    if not np.array_equal(xext[: x.size], xext[x.size: 2*x.size]):
        # they should all be increasing tile wise
        raise ValueError(
            "Flat coords should increase np.tile-wise, i.e: 1, 2, 3, 1, 2, 3,\
            ..."
        )

    if not np.allclose(yext[: y.size], y[0]):
        # they should all be increasing repeat wise
        raise ValueError(
            "Flat coords should increase np.repeat-wise, i.e: 1, 1, 2, 2, 3, 3,\
            ..."
        )


def flatten_coords(x, y, ds):
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
        ds (xarray.core.dataset.Dataset): dataset containing the ensemble for
            checking consistency with ensemble
    Return:
        xext, yext (tuple of np.ndarrays):
            array containing tuples of rlat and
            rlon for each grid cell in the
            ensemble size.
    Raises:
        ValueError, TypeError in check_coords_are_flattened and
            check)input_coords
        ITypeError:
            If input coords are not numpy arrays
        ValueError:
            If xext and yexy are not the same size
            If extended flattened size is not expected
            If flattened longitude is not increasing
                numpy tile-wise
            If flattened latitude is not increasing
                numpy repeat-wise
    """
    check_input_coords(x, y, ds)
    xext = np.tile(x, y.size)
    yext = np.repeat(y, x.size)
    check_coords_are_flattened(x, y, xext, yext, ds)

    return xext, yext


def check_transform_coords_inputs(x, y, source_crs, target_crs):
    """Checks the inputs of transform_coords(). Tests assume that
    the target and source CRS are WGS84 and rotated pole respectively.
    Args:
        x,y (numpy.ndarray): array containing
            latitudes and longitudes of
            stations in source_crs projection
        source_crs (dict): source proj4 crs
        target_crs(dict): destination proj4 crs
    Raises:
        TypeError:
                If input coords are not numpy arrays
                If crs provided are not dict
        ValueError:
                If x and y are not the same size
                If x and y ranges are outside of the CanRCM4 grid cell
                    in WGS84
    """
    if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
        raise TypeError(
            "Please provide an object of type {}".format(np.ndarray)
        )

    check_ndims(x, 1)
    check_ndims(y, 1)

    if (not isinstance(source_crs, dict)) or (
        not isinstance(target_crs, dict)
    ):
        raise TypeError("Please provide an object of type {}".format(dict))

    if x.size != y.size:
        raise ValueError(
            "x and y must be pairwise station coordinates \
            and have the same size.")


def transform_coords(
    x,
    y,
    source_crs={"init": "epsg:4326"},
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
        x,y (numpy.ndarray): array containing
            latitudes and longitudes of
            stations
        source_crs (dict): proj4 dict defining source coordinates
            coordinates used.
    Returns:
        x,y (tuple): tuple containing the newly rotated
            coordinates rlon, rlat
    Raises:
        TypeError, ValueError in check_transform_coords_inputs
        TypeError:
                If input coords are not numpy arrays
                If crs provided are not dict
        ValueError:
                If x and y are not the same shape
                If x and y ranges are outside of the CanRCM4 grid cell
                    in WGS84
    """
    check_transform_coords_inputs(x, y, source_crs, target_crs)
    p_source = Proj(source_crs)
    p_target = Proj(target_crs)
    t = Transformer.from_proj(p_source, p_target)

    return t.transform(x, y)


def check_find_nearest_index_inputs(data, val):
    """Checks the inputs for find_nearest_index() for correct
    datatypem are increasing monotonically, have a size greater than 1, and are
    located somewhere in the CanRCM4 grid cell bounds.
    Args:
        data (np.ndarray): monotonically increasing array of column or
            rowcoordinates
        val (float): location of grid cell in x (rlon) or y (rlat) coords
    Raises:
        TypeError:
                If data or val are not the correct type
        ValueError:
                If data is not monotonically increasing
                If size is not greater than 1
                If val is not within data's range of values
    """
    if not isinstance(data, np.ndarray):
        raise TypeError(
            "Please provide a data array of type {}".format(np.ndarray)
        )
    check_ndims(data, 1)
    if np.any(np.diff(data) < 0):
        raise ValueError("Array must be monotonically increasing.")
    if data.size < 2:
        raise ValueError("Array size must be greater than 1")

    if not isinstance(val, float):
        raise TypeError("Please provide a value of type {}".format(float))

    if val > data.max() or val < data.min():
        warnings.warn(
            "Value outside of array's range with domain between \
            {} and {}. A station is outside of the CanRCM4 model grid space."
            .format(data.min(), data.max())
        )


def find_nearest_index(data, val):
    """Bisect search algorithm to find a value within a monotonically
    increasing array
    Args:
        data (np.ndarray): monotonically increasing array of column or row
            coordinates
        val (float): location of grid cell in x (rlon) or y (rlat) coords
    Returns:
        best_ind (integer): index in data of closest data value to val
    Raises:
        TypeError, ValueError in check_find_nearest_index_inputs
        TypeError:
                If data or val are not the correct type
        ValueError:
                If data is not monotonically increasing
                If size is not greater than 1
                If val is not within data's range of values
    """
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
    """Checks the inputs for find_element_wise_nearest_pos()
    Args:
        x, y (np.ndarray): monotonically increasing array of column
            or rowcoordinates
        x_obs, y_obs (np.ndarray): observations full of values to find
            in x and y
    Raises:
        TypeError:
                If any arrays provided are not np.ndarray
        ValueError:
                If sizes of x and y or x_obs and y_obs are not the same
    """

    is_ndarray = [
        isinstance(array, np.ndarray) for array in [x, y, x_obs, y_obs]
    ]
    if not np.any(is_ndarray):
        raise TypeError(
            "Please provide data arrays of type {}".format(np.ndarray)
        )
    if x.size < 2 or y.size < 2:
        raise ValueError(
            "Must have x and y arrays with a size greater than 1. \
            Received {} and {} respectively.".format(x.size, y.size)
        )
    if x_obs.size != y_obs.size:
        raise ValueError(
            "Array of values to find in arrays must be the same shape. \
            Received arrays of shape {} and {}".format(x_obs.size, y_obs.size)
        )


def find_element_wise_nearest_pos(x, y, x_obs, y_obs):
    """Finds the nearest positions in x and y for each value in
    x_obs and y_obs. x and y should be the rlon and rlat arrays,
    and the x_obs and y_obs should be the station coordinates in
    rotated pole coords.
    Args:
        x, y (np.ndarray): monotonically increasing array of column
            or rowcoordinates
        x_obs, y_obs (np.ndarray): observations full of values to find
            in x and y
    Returns:
        x_i, y_i (array of indices): locations in each coordinate axis
            of locations in x and y where x_obs and y_obs are respectively
            closest
    Raises:
        TypeError, ValueError in check_find_element_wise_nearest_pos_inputs
        TypeError:
                If any arrays provided are not np.ndarray
        ValueError:
                If sizes of x and y or x_obs and y_obs are not the same
                If data is not monotonically increasing
                If val in x_obs or y_obs is not within
                    data's range of values
                If x or y are not in expected range of values
    """
    check_ndims(x, 1)
    check_ndims(y, 1)
    check_ndims(x_obs, 1)
    check_ndims(y_obs, 1)
    check_find_element_wise_nearest_pos_inputs(x, y, x_obs, y_obs)
    x_i = np.array([find_nearest_index(x, obs) for obs in x_obs])
    y_i = np.array([find_nearest_index(y, obs) for obs in y_obs])
    return x_i, y_i


def check_find_nearest_value_inputs(x, y, x_i, y_i, field, mask):
    """Checks find_nearest_value() inputs.
    Args:
        x, y (np.ndarray): monotonically increasing array of column
            or rowcoordinates
        x_i, y_i (np.ndarray): indices in the rlon and rlat arrays
            of the closest grid to stations
        field (np.ndarray): 2 dimensional field array containing
            the CanRCM4 field
        mask (np.ndarray of bool): 2 dimensional mask array matching field
            with a boolean mask of accepted values for analyses
    Raises:
        ValueError:
                If field provided is not made of x and y coordinates
                If field shape and mask shapes are different
    """
    if (not isinstance(x_i, np.ndarray)) or (not isinstance(y_i, np.ndarray)):
        raise TypeError(
            "Please provide index array of type {}.".format(np.ndarray)
        )
    if (not x_i.dtype == np.dtype("int")) or (
        not y_i.dtype == np.dtype("int")
    ):
        raise ValueError(
            "Both index array must contain integers. Received \
            {} and {}".format(x_i.dtype, y_i.dtype)
        )
    if (x_i.max() > x.size) or (y_i.max() > y.size):
        raise ValueError(
            "Indices in index arrays are larger than coordinate array size. \
            Received x, y {},{} with sizes {}, {}."
            .format(x_i.max(), y_i.max(), x.size, y.size)
        )
    # field same shape as xiyi
    if field.shape != (y.size, x.size):
        raise ValueError(
            "Field provided is not consistent with coordinates provided. \
            Recevied field shape {}, expected shape ({},{})"
            .format(field.shape, y.size, x.size)
        )
    # mask same shape as field
    if field.shape != mask.shape:
        raise ValueError(
            "Field and mask are not the same shape. Received field shape \
            {} and mask shape {}.".format(field.shape, mask.shape)
        )


def find_nearest_index_value(x, y, x_i, y_i, field, mask, ds):
    """Finds the nearest model value to a station location in the CanRCM4
    grid space
    Args:
        x, y (np.ndarray): monotonically increasing array of column
            or rowcoordinates
        x_i, y_i (np.ndarray): indices in the rlon and rlat arrays
            of the closest grid to stations
        field (np.ndarray): 2 dimensional field array containing
            the CanRCM4 field
        mask (np.ndarray of bool): 2 dimensional mask array matching field
            with a boolean mask of accepted values for analyses
        ds (xarray.core.dataset.Dataset): dataset containing the ensemble for
            checking consistency with ensemble
    Raises:
        TypeError, ValueError in check_find_nearest_value_inputs
        TypeError:
                If arrays are not of type np.ndarray
        ValueError:
                If field provided is not made of x and y coordinates
                If field shape and mask shapes are different
                If x and y arrays are not monotonically increasing
                If x and y arrays do not have expected range
                If there are indices provided in x_i or y_i outside
                    of the expected grid space
                If all values in x_i or y_i are not integers
    """
    check_find_nearest_value_inputs(x, y, x_i, y_i, field, mask)

    # find any stations that have a NaN corresponding grid cell
    nanloc = np.isnan(field[y_i, x_i])

    # combine mask and nan locations in field to
    # create a master mask of eligible grid cells
    # for interpolation
    master_mask = np.logical_and(mask, ~np.isnan(field))

    # if any NaN values found over station values,
    # perform a nearest neighbour interpolation to get
    # valid model value at that location
    if np.any(nanloc):
        # create grids of rlon and rlat
        xarr, yarr = np.meshgrid(x, y)

        # flatten coordinates
        xext, yext = flatten_coords(x, y, ds)

        # arrange the pairs
        pairs = np.array(list(zip(xext, yext)))

        # create interpolation function for every point
        # except the locations of the NaN values
        f = NearestNDInterpolator(
            pairs[master_mask.flatten()], field[master_mask]
        )

        # get the rlon and rlat locations of the NaN values
        x_nan = xarr[y_i, x_i][nanloc]
        y_nan = yarr[y_i, x_i][nanloc]

        # interpolate a value at the locations of those NaN values
        nan_pairs = np.array(list(zip(x_nan, y_nan)))

        # replace the field value at NaN locations with the
        # interpolated values
        field[y_i[nanloc], x_i[nanloc]] = f(nan_pairs)

    # provide a final array of field values at station locations
    # including any replaced NaN values if program found it neccessary
    final = field[y_i, x_i]

    return final
