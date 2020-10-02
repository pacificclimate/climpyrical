from collections import OrderedDict
from pkg_resources import resource_string

from nptyping import NDArray
from typing import Any, Tuple
import numpy as np
from rpy2.robjects.packages import importr
from rpy2.robjects import FloatVector, DataFrame
from rpy2 import robjects

import rpy2.robjects.packages as rpackages

utils = rpackages.importr("utils")

# utils.chooseCRANmirror(ind=1) # select the first mirror in the list

# utils.install_packages(StrVector(("fields")), "../r-library")
# utils.install_packages(StrVector(("sp")), "../r-library")
# utils.install_packages(StrVector(("gstat")), "../r-library")

# print("FIELDS")
importr("fields")
# importr("sp")
# importr("gstat")


def fit(
    latlon: NDArray[(2, Any), float],
    z: NDArray[(Any,), float],
    nx: int,
    ny: int,
    extrap: bool,
) -> Tuple[
    NDArray[(Any, Any), float], NDArray[(Any,), float], NDArray[(Any,), float]
]:

    """Encapsulates the functionality of R's spatialProcess into a Python
    Args:
        latlon: grid of pairwise coordinates of observations
        z: observations
        nx: number of grid cells on interpolated grid x
        nx: number of grid cells on interpolated grid y
        xy: dimensions of interpolated grid output
        distance: distance metric to use (note, only 'geo' supported currently)
        variogram_model: choice of variogram model
          (note, only 'exoponential' supported)
    Returns:
        z: kriged field
        x, y: locations of kriged data

    """

    if not isinstance(latlon, NDArray[(2, Any), float]):
        raise TypeError(
            f"Incorrect grid shape, size, or dtype. Must be {NDArray[(2, Any), float]}"
        )

    if not isinstance(z, NDArray[(Any,), float]):
        raise TypeError(
            f"Incorrect grid shape, size, or dtype. Must be {NDArray[(Any, ), float]}"
        )

    if not isinstance(nx, int) or not isinstance(ny, int):
        raise TypeError("Provide integer grid size")

    if latlon.shape[1] != z.size:
        raise ValueError(
            "Different number of grid coordinates than observations"
        )

    latlon, z = latlon.tolist(), z.tolist()

    # convert regular numeric data

    # convert latlon list into two R FloatVectors
    # list of FloatVector -> OrderedDict -> R DataFrame
    # -> numeric R data matrix
    r_lists = list(map(FloatVector, latlon))
    coords = OrderedDict(zip(map(str, range(len(r_lists))), r_lists))
    r_dataFrame = DataFrame(coords)
    r_latlon = robjects.r["data.matrix"](r_dataFrame)

    # convert observations
    r_z = FloatVector(z)

    # use separate simple r-script in path below
    rstring = resource_string(
        "climpyrical", "tests/data/spatial_process_r.R"
    ).decode("utf-8")

    rfunc = robjects.r(rstring)
    r_surface = rfunc(r_latlon, r_z, nx, ny, extrap)

    # extract data from R's interpolation
    surface_dict = dict(zip(r_surface.names, list(r_surface)))
    # z = np.array(list(r_surface[1]))
    z = np.array(surface_dict["z"]).reshape(nx, ny)
    x = np.array(surface_dict["x"])
    y = np.array(surface_dict["y"])
    # cov = dict(zip(surface_dict["cov"].names, list(surface_dict["cov"])))
    # cov = surface_dict["cov"]

    return z, x, y
