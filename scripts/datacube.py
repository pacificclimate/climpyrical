import glob
import warnings

import numpy as np
import numpy.ma as ma

import xarray as xr

def check_keys(ds):
    required_keys = {'rlat', 'rlon', 'run', 'dv'}
    actual_keys = set(ds.keys())
    if required_keys > actual_keys:
        raise KeyError("CanRCM4 ensembles is missing keys {}".format(required_keys - actual_keys))

def check_path(data_path):
    if isinstance(data_path, str) is False:
        raise ValueError('Path requires str got {}'.format(type(data_path)))
    if len(glob.glob(data_path)) == 0:
        raise IndexError('Path provided has no files with \'.nc\' extension')
    if len(glob.glob(data_path)) <=20:
        warnings.warn("Path has a low ensemble size with < 20 members")

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
    xr_list = [xr.open_dataset(path) for path in nc_list]

    ds = xr.concat(xr_list, 'run')

    # find design value key so that it can
    # be renamed to dv for references throughout
    # climpyrical project
    for var in list(ds.variables):

        grids = ['rlon', 'rlat', 'lon', 'lat', 'time_bnds', 'time', 'rotated_pole']

        if var not in grids and var in ds.variables:

            ds = ds.rename({var: 'dv'})
            ds = ds.squeeze('time')

        elif var not in ds.variables:
            raise KeyError("Invalid CanRCM4 model. Design value key not found.")

    check_keys(ds)
    return ds
