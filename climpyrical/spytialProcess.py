from collections import OrderedDict

from nptyping import NDArray
from typing import Any, Tuple
import numpy as np
from rpy2.robjects.packages import importr 
from rpy2.robjects import FloatVector, r, DataFrame
from rpy2 import robjects

importr('fields')

def fit(latlon: NDArray[(2, Any), float], 
        z: NDArray[(Any,), float], 
        nx: int, 
        ny: int, 
        xy: Tuple, 
        distance: str, 
        variogram_model: str) -> Tuple[
            NDArray[(Any, Any), float], 
            NDArray[(Any,), float], 
            NDArray[(Any,), float]
        ]:

    """Encapsulates the functionality of R's spatialProcess into a Python
    Args:
        latlon: grid of pairwise coordinates of observations
        z: observations
        nx: number of grid cells on interpolated grid x
        nx: number of grid cells on interpolated grid y
        xy: dimensions of interpolated grid output
        distance: distance metric to use (note, only 'geo' supported currently)
        variogram_model: choice of variogram model (note, only 'exoponential' supported)
    Returns:
        z: kriged field
        x, y: locations of kriged data

    """
    
    if latlon.shape[1] != z.size:
        raise ValueError('Different number of grid coordinates than observations')

    latlon, z = latlon.tolist(), z.tolist()

    # convert from easier to understand names
    # to something R can understand
    xy = 'c'+str(xy)

    str_args_convert = {
        'geo': "'rdist.earth'",
        'exponential': 'list(Covariance="Exponential")',

    }

    d = str_args_convert[distance]
    v_model = str_args_convert[variogram_model]

    # convert regular numeric data

    # convert latlon list into two R FloatVectors
    # list of FloatVector -> OrderedDict -> R DataFrame
    # -> numeric R data matrix
    r_lists = list(map(FloatVector, latlon))
    coords = OrderedDict(zip(map(str, range(len(r_lists))), r_lists))
    r_dataFrame = DataFrame(coords)
    r_latlon = robjects.r['data.matrix'](r_dataFrame)

    # convert observations
    r_z = FloatVector(z)    

    # forced to use %-formatting due to RRuntime interpretation
    rstring="""
        function(latlon, z, nx, ny){
            obj <- spatialProcess(latlon, z, Distance = %s, cov.args = %s)
            predictSurface(obj, grid.list = NULL, extrap = FALSE, chull.mask = NA,
                    nx = nx, ny = ny, xy = %s, verbose = FALSE, ZGrid = NULL,
                    drop.Z = FALSE, just.fixed=FALSE)
        }
    """ % (d, v_model, xy)

    rfunc = robjects.r(rstring)
    r_surface = rfunc(r_latlon, r_z, nx, ny)

    # extract data from R's interpolation
    z = np.array(list(dict(zip(r_surface.names, list(r_surface)))['z'])).reshape(nx, ny)
    x = np.array(list(dict(zip(r_surface.names, list(r_surface)))['x']))
    y = np.array(list(dict(zip(r_surface.names, list(r_surface)))['y']))

    return z, x, y