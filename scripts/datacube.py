import glob
import warnings

import numpy as np
import numpy.ma as ma

import xarray as xr

def check_keys(ds, required_keys):
    actual_keys = set(ds.variables).union(set(ds.dims))

    if not set(required_keys).issubset(actual_keys):
        raise KeyError(
                    "CanRCM4 ensemble is missing keys {}"
                    .format(required_keys - actual_keys)
            )

def check_path(data_path):
    if isinstance(data_path, str) is False:
        raise ValueError('Path requires str got {}'.format(type(data_path)))
    if len(glob.glob(data_path)) == 0:
        raise ValueError('Path provided has no files with \'.nc\' extension')
    if len(glob.glob(data_path)) == 1:
        raise ValueError('At least 2 ensemble members required to EOF analysis')
    if len(glob.glob(data_path)) <=20:
        warnings.warn("Path has a low ensemble size with < 20 members")

def check_time(ds):
    if len(ds['time']) != 1:
        raise ValueError('Climpyrical can not take inputs as time series. Must be uni-dimensional in the time axis.')

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
    check_keys(test_file, {'rlat', 'rlon', 'lat', 'lon'})

    xr_list = [xr.open_dataset(path) for path in nc_list]
    ds = xr.concat(xr_list, 'run')
    check_keys(ds, {'rlat', 'rlon', 'lat', 'lon', 'run'})

    # find design value key so that it can
    # be renamed to dv for references throughout
    # climpyrical project
    for var in list(ds.variables):

        grids = ['rlon', 'rlat', 'lon', 'lat', 'time_bnds', 'time', 'rotated_pole']

        if var not in grids and var in ds.variables:

            ds = ds.rename({var: 'dv'})
            if 'time' in ds.keys():
                check_time(ds)
                ds = ds.squeeze('time')

    return ds
