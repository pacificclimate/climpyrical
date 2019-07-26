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
    if len(glob.glob(data_path))==0:
        raise ValueError('Path provided has no files with \'.nc\' extension')
        passed = False
    if len(glob.glob(data_path))==1:
        raise ValueError('At least 2 ensemble members required to EOF analysis')
        passed = False
    if len(glob.glob(data_path))<=20:
        warnings.warn("Path has a low ensemble size with < 20 members")

    return passed

def check_time(len_time):
    passed = True
    if len_time != 1:
        raise ValueError('Climpyrical can not take inputs as time series. Must be uni-dimensional in the time axis.')
        passed = False
    return passed

def read_data(data_path):
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

    nc_list = np.asarray(glob.glob(data_path))
    test_file = xr.open_dataset(nc_list[0])
    actual_keys = set(test_file.variables).union(set(test_file.dims))
    check_keys(actual_keys, {'rlat', 'rlon', 'lat', 'lon'})

    xr_list = [xr.open_dataset(path) for path in nc_list]
    ds = xr.concat(xr_list, 'run')
    actual_keys = set(ds.variables).union(set(ds.dims))
    check_keys(actual_keys, {'rlat', 'rlon', 'lat', 'lon', 'run'})

    # find design value key so that it can
    # be renamed to dv for references throughout
    # climpyrical project
    for var in list(ds.variables):

        grids = ['rlon', 'rlat', 'lon', 'lat', 'time_bnds', 'bnds', 'time', 'rotated_pole']

        if var not in grids and var in ds.variables:

            ds = ds.rename({var: 'dv'})
            if 'time' in ds.keys():
                check_time(len(ds['time']))
                ds = ds.squeeze('time')

    return ds
