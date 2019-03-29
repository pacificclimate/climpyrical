import glob

import numpy as np
import xarray as xr

def read_data(data_path, dv_key_name):
    """Load an ensemble of CanRCM4
    models into a single datacube.
    ------------------------------
    Args:
        data_path (Str): path containing CanRCM4 ensemble
        dv_key_name (Str): name of the design value as 
            in the netCDF file
    Returns:
        ds (xarray Dataset): data cube of assembled ensemble models 
            into a single variable
    """
    if not isinstance(data_path, str):
        raise TypeError("data_path must be a string, got {}"
                        .format(type(data_path)))

    if not isinstance(dv_key_name, str):
        raise TypeError("dv_key_name must be a string, got {}"
                        .format(type(dv_key_name)))

    nc_list = np.asarray(glob.glob(data_path+"*.nc"))
    xr.open_dataset(nc_list[0])

    xr_list = []
    for i, path in enumerate(nc_list):
        xr_list.append(xr.open_dataset(path))
        
    xr.concat(xr_list, 'run')

    return xr

class DataReader:
    def __init__(self, data_path, dv_key_name):
        self.data_path = data_path
        self.dv_key_name = dv_key_name
        self.read_data = read_data(data_path, dv_key_name)