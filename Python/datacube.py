import glob

import numpy as np
import xarray as xr

def read_data(data_path):
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

    nc_list = np.asarray(glob.glob(data_path+"*.nc"))
    xr.open_dataset(nc_list[0])

    xr_list = [xr.open_dataset(path) for path in nc_list]        
    xr.concat(xr_list, 'run')

    return xr

class DataReader:
    def __init__(self, data_path, dv_key_name):
        self.data_path = data_path
        self.dv_key_name = dv_key_name
        self.read_data = read_data(data_path, dv_key_name)