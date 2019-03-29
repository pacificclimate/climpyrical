import glob

import numpy as np
import xarray as xr
import netCDF4 as nc

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

    # Create a list of all files in PATH
    nc_list = np.asarray(glob.glob(data_path+"*.nc"))

    # create an example NetCDF4 dataset
    inst = nc.Dataset(nc_list[0], 'r')
    data_cube = np.ones((inst['lat'].shape[0],
                         inst['lat'].shape[1],
                         nc_list.shape[0]))

    inst.close()

    # iterate through files in path
    # to create data cube
    for i, path in enumerate(nc_list):
        run = nc.Dataset(path, 'r')
        dv = run.variables[dv_key_name][:, :]
        data_cube[:, :, i] = dv

    lat = run.variables['lat'][:, :]
    lon = run.variables['lon'][:, :]

    rlat = run.variables['rlat'][:]
    rlon = run.variables['rlon'][:]

    run.close()

    # constrube datacube as an xarray dataset
    ds = xr.Dataset({'dv': (['x', 'y', 'run'], data_cube)},
                    coords={'lon': (['x', 'y'], lon),
                            'lat': (['x', 'y'], lat),
                            'rlon': rlon,
                            'rlat': rlat},
                    attrs={'dv': 'mm h-1',
                           'lon': 'degrees',
                           'lat': 'degrees',
                           'rlon': 'degrees',
                           'rlat': 'degrees'})

    return ds

class DataReader:
    def __init__(self, data_path, dv_key_name):
        self.data_path = data_path
        self.dv_key_name = dv_key_name
        self.read_data = read_data(data_path, dv_key_name)