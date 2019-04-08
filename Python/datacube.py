import glob

import numpy as np
import xarray as xr

def check_keys(ds):
    required_keys = {'rlat', 'rlon', 'run', 'dv'}
    actual_keys = set(ds.keys())
    if required_keys > actual_keys:
        raise KeyError("CanRCM4 ensembles is missing keys {}".format(required_keys - actual_keys))

def read_data(data_path):
    """Load an ensemble of CanRCM4
    models into a single datacube.
    ------------------------------
    Args:
        data_path (Str): path to folder
            containing CanRCM4 ensemble
    Returns:
        ds (xarray Dataset): data cube of assembled ensemble models
            into a single variable
    """

    nc_list = np.asarray(glob.glob(data_path+"*.nc"))
    xr_list = [xr.open_dataset(path) for path in nc_list]

    ds = xr.concat(xr_list, 'run')

    # find design value key so that it can
    # be renamed to dv for references throughout
    # map-xtreme project
    for var in list(ds.variables):

        grids = ['rlon', 'rlat', 'lon', 'lat']

        if var not in grids and var in ds.variables:
            ds = ds.rename({var: 'dv'})

        elif var not in ds.variables:
            raise KeyError("Invalid CanRCM4 model. Design value key not found.")

    check_keys(ds)
    return ds

