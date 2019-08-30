import glob
import warnings

import numpy as np
import numpy.ma as ma

import xarray as xr

def check_keys(actual_keys, required_keys):
    passed = True
    if not set(required_keys).issubset(actual_keys):
        raise KeyError(
                    "CanRCM4 ensemble is missing keys {}"
                    .format(required_keys - actual_keys)
            )
        passed = False
    return passed

def check_path(data_path):
    passed = True
    if isinstance(data_path, str) is False:
        raise ValueError('Path requires str got {}'.format(type(data_path)))
        passed = False

    return passed

def read_data(data_path, design_value_name):
    """Load an ensemble of CanRCM4
    models into a single datacube.
    ------------------------------
    Args:
        data_path (Str): path to folder
            containing CanRCM4 ensemble
    Returns:
        ds (xarray Dataset): data cube of assembled ensemble models
            into a single variable.
    """
    check_path(data_path)
    ds = xr.open_dataset(data_path)
    actual_keys = set(ds.variables).union(set(ds.dims))
    check_keys(
        actual_keys,
        {'rlat', 'rlon',
        'lat', 'lon',
        'level', design_value_name}
    )

    return ds
