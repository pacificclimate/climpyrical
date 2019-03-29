import warnings

import xarray as xr
import numpy as np

def check_keys(data_cube):
    rlat = 'rlat' in data_cube.variables
    rlon = 'rlon' in data_cube.variables
    run = 'run' in data_cube.variables
    if not rlat and rlon and run:
        raise KeyError("CanRCM4 ensemble is missing 'run' axis, 'rlon' \
                        and 'rlat'. Please ensure the model is properly \
                        gridded with these names.")

def cell_count(data_cube):
    check_keys(data_cube)
    n = data_cube['run'].shape[0]
    p = data_cube['rlat'].shape[0]*data_cube['rlon'].shape[0]
    return n, p

def center_data(data_cube):
    """Centers each run in data_cube
    by subtracting ensemble mean at
    a given grid cell.
    --------------------------------
    Args: 
        data_cube (xarray Dataset): datacube 
            containing an ensemble of 
            CanRCM4 models
    Returns:
        data_cube (xarray Dataset): mean centered
            datacube  
    """
    check_keys(data_cube)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean = data_cube['dv'].mean(dim='run', skipna=True)

        data_cube['dv'] = data_cube['dv'] - mean

    return data_cube

def weight_by_area(data_cube):
    """Weights each CanRCM4 grid cell 
    by fractional area.
    --------------------------------
    Args: 
        data_cube (xarray Dataset): datacube 
            containing an ensemble of 
            CanRCM4 models 
    Returns:
        (xarray Dataset): design value datacube 
            weighted by area
    """
 

    check_keys(data_cube)
    n, p = cell_count(data_cube)

    # calculate mean size of grid cell
    rlat = np.diff(data_cube['rlat']).mean()
    rlon = np.diff(data_cube['rlon']).mean()

    # calulate rectangular size of
    # grid cell 
    lat_sz = np.abs(np.sin(np.deg2rad(rlat)))
    lon_sz = np.abs(np.deg2rad(rlon))

    # calculate rectangular area on sphere.
    # note that radius wolud cancel out 
    # in fraction
    area = (lat_sz*lon_sz)
    total_area = area*p
    frac_area = area/total_area

    data_cube['dv'] = data_cube['dv']*frac_area

    return data_cube
