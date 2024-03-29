import xarray as xr
import numpy as np
from nptyping import NDArray
from typing import Any, Union
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator


def check_valid_keys(all_keys: list, required_keys: list) -> bool:
    """A function to test that required_keys is a subset of all_keys.
    Args:
        all_keys (list): keys found in the NetCDF file
        required_keys (list): expected and required keys
            that make sense for the climpyrical analyses
    Returns:
        bool: True of passed, raises error if not.
    Raises:
        KeyError if required_keys are not a subset of the actual keys
    """
    if not set(required_keys).issubset(set(all_keys)):
        raise KeyError(
            "CanRCM4 ensemble is missing keys {}".format(
                set(required_keys) - set(all_keys)
            )
        )

    return True


def check_valid_data(ds: xr.Dataset) -> bool:
    """A function to test that the data loaded is valid and expected.
    Args:
        ds (xarray.DataSet): ensemble with all relevant data
        design_value_name (str): name of design value exactly as appears
            in the NetCDF4 file
    Returns:
        bool: True of passed, raises error if not.
    Raises:
        ValueError if loaded data is unexpected or invalid
    """

    if ("rlat" in set(ds.variables).union(set(ds.dims))) or (
        "rlon" in set(ds.variables).union(set(ds.dims))
    ):
        if (np.any(np.isnan(ds.rlat))) or (np.any(np.isnan(ds.rlon))):
            raise ValueError("Coordinate axis contain unexpected NaN values")
        if (np.any(np.diff(ds.rlat.values) < 0)) or np.any(
            (np.diff(ds.rlon.values) < 0)
        ):
            raise ValueError("Coordinate axis are not monotonically increasing")

    if bool(np.all(ds.to_array().isnull()).values):
        raise ValueError("All values are NaN. Check input.")

    return True


def gen_dataset(
    dv: str,
    field: Union[NDArray[(Any, Any), Any], NDArray[(Any, Any, Any), Any]],
    rlat: NDArray[(Any,), float],
    rlon: NDArray[(Any,), float],
    lat: NDArray[(Any, Any), float],
    lon: NDArray[(Any, Any), float],
    unit: str = "",
) -> xr.Dataset:
    """Generates standard climpyrical xarray Dataset.
    ------------------------------
    Args:
        dv (Str): key name of design value
        field (np.ndarray): 2D array of design value field
        x,y (np.ndarray, np.ndarray): coordinates along
            each axis of design value field
        z (np.ndarray or None): optional level/z coordinates
    Returns:
        ds (xarray Dataset): dataset with new keys
            and design value field
    Raises:
        From xarray.Dataset
    """

    dsarr = xr.DataArray(field, coords=[rlat, rlon], dims=["rlat", "rlon"])
    dsarr.attrs["units"] = unit
    ds = xr.Dataset(
        {dv: dsarr},
        coords={
            "lat": (["rlat", "rlon"], lat),
            "lon": (["rlat", "rlon"], lon),
            "rlon": ("rlon", rlon),
            "rlat": ("rlat", rlat),
        },
    )

    return ds


def read_data(
    data_path: str, required_keys: list = ["rlat", "rlon", "lat", "lon"]
) -> xr.Dataset:
    """Load NetCDF4 file. Default checks are for CanRCM4 model keys.

    Note that 'rlat', 'lat', 'lon', 'rlon' are all required in addition
    to a single data variable that contains a field of interest.
    -----------------------------------------------------------------
    Args:
        data_path (Str): path to folder
            containing CanRCM4 ensemble
        required_keys (list, optional): list of required keys in netCDF4
            file. Default requirements are only that it contains
            rotated lat and rotated lon coords called rlon and rlat
    Returns:
        ds (xarray Dataset): dataset of netCDF4 file
    Raises:
        FileNotFoundError: if file not found
        ValueError: if file contains unexpected or invalid data
        KeyError if NetCDF4 file is missing required keys
        TypeError if path provided is invalid
    """

    if not data_path.endswith(".nc"):
        raise TypeError("climpyrical requires a NetCDF4 file with extension .nc")

    with xr.open_dataset(data_path) as ds:
        all_keys = set(ds.variables).union(set(ds.dims))

        check_valid_keys(all_keys, required_keys)
        check_valid_data(ds)

        # keys with size > 2 are good, otherwise
        # are superfluous (time, time_bnds, rotated_pole, etc)
        extra_keys = [key for key in all_keys if ds[key].size <= 2]

        # check how many data variables with
        # size > 2 are remaining. If more than one
        # raise error as we can't distinguish from
        # the intended variable.
        dv = set(ds.data_vars) - set(extra_keys)
        if len(dv) != 1:
            raise KeyError(
                "Too many data variables detected."
                f"Found {dv}, please remove the"
                "field that is not of interest."
            )
        else:
            (dv,) = dv

        # drop an extra dimension if it snuck in
        dvfield = ds[dv].squeeze(drop=True)

        ds_new = gen_dataset(dv, dvfield, ds.rlat, ds.rlon, ds.lat, ds.lon)

        if ds.attrs:
            all_keys = set(ds_new.variables).union(set(ds_new.dims))
            for key in all_keys:
                ds_new[key].attrs = ds[key].attrs
            attr_dict = ds.attrs
            attr_dict["Climpyrical"] = (
                "CanRCM4 Reconstruction contains"
                "hybrid station and model data using"
                "Climpyrical (https://github.com/pacificclimate/climpyrical)"
            )

            ds_new.attrs = attr_dict

        return ds_new


def interpolate_dataset(
    points: NDArray[(2, Any), float],
    values: NDArray[(Any, Any), float],
    target_points: NDArray[(2, Any), float],
    method: str,
) -> NDArray[(Any,), float]:

    """Generates standard climpyrical xarray Dataset.
    ------------------------------
    Args:
        points (np.ndarray): ordered pairs of coordinates
            from current grid
        values (np.ndarray): field values at points
        target_points (np.ndarray): ordered pairs of coordinates
            from target grid
        method (str): desired method - can be either 'linear' or
            'nearest'
    Returns:
        (np.ndarray): newly predicted values at target points
    """

    if method != "linear" and method != "nearest":
        raise ValueError("Method must be linear or nearest.")

    method_dict = {
        "linear": LinearNDInterpolator(points, values),
        "nearest": NearestNDInterpolator(points, values),
    }

    f = method_dict[method]

    return f(target_points).T
