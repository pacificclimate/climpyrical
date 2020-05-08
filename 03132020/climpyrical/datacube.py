import xarray as xr
import numpy as np


def check_valid_data_path(data_path):
    """A function to test the data path for supplied file in climpyrical.
    Args:
        data_path (str): path to ensemble file
    Returns:
        bool: True of passed, raises error if not.
    Raises:
        TypeError if data_path is not valid
    """
    if not isinstance(data_path, str):
        raise TypeError("Please provide a string as a data_path")

    if not data_path.endswith(".nc"):
        raise TypeError(
            "climpyrical requires a NetCDF4 file with extension .nc"
        )

    return True


def check_valid_keys(actual_keys, required_keys):
    """A function to test that required_keys is a subset of actual_keys.
    Args:
        actual_keys (dict): dictionary with keys found in the NetCDF file
        required_keys (dict): dictionary of expected and required keys
            that make sense for the climpyrical analyses
    Returns:
        bool: True of passed, raises error if not.
    Raises:
        KeyError if required_keys are not a subset of the actual keys
    """
    if not set(required_keys).issubset(actual_keys):
        raise KeyError(
            "CanRCM4 ensemble is missing keys {}".format(
                list(set(required_keys) - set(actual_keys))
            )
        )

    return True


def check_valid_data(ds, design_value_name):
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
    if ds[design_value_name].size == 0:
        raise ValueError("Design value field is empty")
    if ("rlat" in set(ds.variables).union(set(ds.dims))) or (
        "rlon" in set(ds.variables).union(set(ds.dims))
    ):
        if (np.any(np.isnan(ds.rlat))) or (np.any(np.isnan(ds.rlon))):
            raise ValueError("Coordinate axis contain unexpected NaN values")
        if (np.any(np.diff(ds.rlat.values) < 0)) or np.any(
            (np.diff(ds.rlon.values) < 0)
        ):
            raise ValueError(
                "Coordinate axis are not monotonically increasing"
            )
    return True


def read_data(
    data_path, design_value_name, keys=None
):
    """Load an ensemble of CanRCM4
    models into a single datacube.
    ------------------------------
    Args:
        data_path (Str): path to folder
            containing CanRCM4 ensemble
        design_value_name (str): name of design value exactly as appears
            in the NetCDF4 file
        keys (list, optional): dictionary of required keys in NetCDF4
            file
    Returns:
        ds (xarray Dataset): data cube of assembled ensemble models
            into a single variable.
    Raises:
        FileNotFoundError: if file not found
        ValueError: if file contains unexpected or invalid data
        KeyError if NetCDF4 file is missing required keys
        TypeError if path provided is invalid
    """
    check_valid_data_path(data_path)
    if keys is None:
        keys = ["rlat", "rlon", "lat", "lon", "level"]
    keys.append(design_value_name)
    with xr.open_dataset(data_path) as ds:
        actual_keys = list(set(ds.variables).union(set(ds.dims)))
        check_valid_keys(actual_keys, keys)
        check_valid_data(ds, design_value_name)

        return ds
