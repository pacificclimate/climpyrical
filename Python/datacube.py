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
        data_cube (xarray Dataset): data cube of assembled ensemble models 
            into a single variable
    """
    if not isinstance(data_path, str):
        raise TypeError("data_path must be a string, got {}"
                        .format(type(data_path)))

    nc_list = np.asarray(glob.glob(data_path+"*.nc"))
    xr_list = [xr.open_dataset(path) for path in nc_list]        
    
    data_cube = xr.concat(xr_list, 'run')

    # find design value key so that it can
    # be renamed to dv for references throughout
    # map-xtreme project
    for var in list(data_cube.variables):

        grids = ['rlon', 'rlat', 'lon', 'lat']
        
        if var not in grids and var in data_cube.variables:
            data_cube = data_cube.rename({var: 'dv'})
        
        elif var not in data_cube.variables:
            raise KeyError("Invalid CanRCM4 model. Design value key not found.")
        
    return data_cube

class DataReader:
    def __init__(self, data_path):
        self.data_path = data_path
        self.read_data = read_data(data_path)