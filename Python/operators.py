import warnings

import xarray as xr
import numpy as np

def check_keys(data_cube):
    rlat = 'rlat' in data_cube.keys()
    rlon = 'rlon' in data_cube.keys()
    run = 'run' in data_cube.keys()
    dv = 'dv' in data_cube.keys()
    if not rlat and rlon and run and dv:
        raise KeyError("CanRCM4 ensemble is missing 'run', 'rlon', 'rlat', or 'dv'. Please ensure the model is properly \
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

def frac_grid_area(data_cube, R=6371.0):
    """Calculates the fractional area of
    each CanRCM4 cell. 
    --------------------------------
    Args: 
        data_cube (xarray Dataset): datacube 
            containing an ensemble of 
            CanRCM4 models 
    Returns:
        A (numpy.ndarray): 2-dimensional array
            containing areas of each grid cell
    """
    check_keys(data_cube)
    rlat_o = data_cube['rlat'].values
    rlon_o = data_cube['rlon'].values

    # approximate grid cell size
    gridsz_rlat = rlat_o[-1]+np.diff(rlat_o).mean()
    gridsz_rlon = rlon_o[-1]+np.diff(rlon_o).mean()

    # add an extra cell with approximated size
    rlat_o = np.append(rlat_o, gridsz_rlat) 
    rlon_o = np.append(rlon_o, gridsz_rlon) 

    # differentiate
    rlat_diff = np.diff(np.sin(np.deg2rad(rlat_o)))
    rlon_diff = np.diff(rlon_o)

    # reshape to calculate all cells
    rlat = np.repeat(rlat_diff, rlon_diff.shape[0])
    rlon = np.tile(rlon_diff, rlat_diff.shape[0])

    # convert to radians and apply sin to lat
    lat_sz = np.abs((rlat))
    lon_sz = np.abs(np.deg2rad(rlon))

    # perform area calculation
    area = (np.pi/180)*R**2*lat_sz*lon_sz

    total_area = np.trapz(area)
    # reshape into original grid
    area = np.reshape(area, (rlat_diff.shape[0], rlon_diff.shape[0]))

    return area/total_area

def weight_by_area(data_cube, R=6371.0):
    """Weights each CanRCM4 grid cell 
    by fractional area.
    --------------------------------
    Args: 
        data_cube (xarray Dataset): datacube 
            containing an ensemble of 
            CanRCM4 models 
    Returns:
        data_cube (xarray Dataset): design value datacube 
            weighted by area
    """
 
    check_keys(data_cube)
    data_cube['dv'] = data_cube['dv']*frac_grid_area(data_cube)

    return data_cube
