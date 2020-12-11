import climpyrical.spytialProcess as sp
from climpyrical.gridding import find_nearest_index

from nptyping import NDArray
from typing import Any
import xarray as xr
from sklearn.neighbors import NearestNeighbors
from pykrige.ok import OrdinaryKriging
from tqdm import tqdm

from scipy.spatial import ConvexHull
import rpy2
import scipy

import numpy as np
import pandas as pd
import psutil
import dask.array as da

import warnings
from rpy2.rinterface import RRuntimeWarning

warnings.filterwarnings("ignore", category=RRuntimeWarning)
num_cpus = psutil.cpu_count(logical=False)
# ray.shutdown()
# ray.init(num_cpus=num_cpus)

def check_df(df, keys=["lat", "lon", "rlat", "rlon"]):
    contains_keys = [key not in df.columns for key in keys]
    if np.any(contains_keys):
        raise KeyError(f"Dataframe must contain {keys}")


def krigit_north(
    df: pd.DataFrame, station_dv: str, n: int, ds: xr.Dataset, extrap=True
) -> NDArray[(Any, Any), float]:
    """Krigs an extrapolated field for N nearest stations
    to the northernmost in the dataframe provided. Output is
    in the same dimensions as the dataset provided to be
    easily added to a final reconstruction
        Args:
            df: pandas dataframe containing the coordinates in
                both regular and roated, as well as the station
                data
            station_dv: name of the column containing
                station data in df
            n: number of nearest neighbors to northern
                most stations
            ds: model xarray dataset
        Returns:
            field: kriged field for the north
    """

    dataframe_keys = ["lat", "lon", "rlat", "rlon", station_dv]
    check_df(df, dataframe_keys)

    df = df[["lat", "lon", "rlat", "rlon", station_dv]]

    regular_points = np.stack([np.deg2rad(df.lat), np.deg2rad(df.lon)]).T

    # Metrics intended for two-dimensional vector spaces: Note that the
    # haversine distance metric requires data in the form
    # of [latitude, longitude] and both inputs and outputs are in units of radians.

    # the reason for using a haversine metric here on regular coordinates
    # rather than rotated coordinates, was that this particular haversine
    # implementation gives incorrect values for rotated lon and rotated lat.
    # it does give correct distances for regular lat and lon.
    nbrs = NearestNeighbors(n_neighbors=n, metric="haversine").fit(regular_points)
    dist, ind = nbrs.kneighbors(regular_points)
    imax = np.argmax(df.rlat.values)  # idxmax(axis=0, skipna=True)
    temp_df = df.iloc[ind[imax]]

    xmin, xmax = temp_df.rlon.min(), temp_df.rlon.max()
    ymin, ymax = temp_df.rlat.min(), temp_df.rlat.max()

    # Note that spytialProcess requires rlon and rlat. Different
    # from the form required for haversine distances from
    # nearest neighbour
    latlon = np.stack([temp_df.rlon, temp_df.rlat])
    stats = temp_df[station_dv]

    lw, u = (
        find_nearest_index(ds.rlat.values, ymin),
        find_nearest_index(ds.rlat.values, ymax),
    )
    l, r = (
        find_nearest_index(ds.rlon.values, xmin),
        find_nearest_index(ds.rlon.values, xmax),
    )
    ylim = u - lw
    xlim = r - l

    # krig it
    z, x, y = sp.fit(latlon, stats, xlim, ylim, extrap=extrap)

    field = np.ones((ds.rlat.size, ds.rlon.size))
    field[:, :] = np.nan
    field[lw:u, l:r] = z.T

    return field


def rkrig_py(
    df: pd.DataFrame,
    station_dv: str,
    n: int,
    ds: xr.Dataset,
    exact_values: bool = False,
) -> NDArray[(Any, Any), float]:
    """User has the option of kriging using a Python backend
    instead of using R's fields package. PyKrige has a moving
    window implementation, however, the exacts and parameterization
    is more obscured. Note that the exponentional variogram
    function for PyKrige is different from spatialProcess in R, and
    so identical results should not be expected.
        Args:
            df: pandas dataframe containing the coordinates in
                both regular and roated, as well as the station
                data
            station_dv: name of the column containing
                station data in df
            n: number of nearest neighbors to northern
                most stations
            ds: model xarray dataset
            exact_values: whether to reproduce the
                exact value of inputs
        Returns:
            field: kriged field
    """

    dataframe_keys = ["lat", "lon", "rlat", "rlon", station_dv]
    check_df(df, dataframe_keys)

    df = df[["lat", "lon", "rlat", "rlon", station_dv]]

    ok = OrdinaryKriging(
        df.rlon,
        df.rlat,
        df[station_dv],
        exact_values=exact_values,
        variogram_function="exponential",
    )
    z, ss = ok.execute(
        "grid", ds.rlon.values, ds.rlat.values, backend="C", n_closest_points=n
    )

    return z


def krig_at_field(
    ds: xr.Dataset, temp_xyr: NDArray[(Any, 4), float]
) -> NDArray[(Any, Any), float]:
    """Matches the output of spytialProcess to the dataset provided
    and returns a 2D array of the krigged field with same dimensions
    as the dataset's design value field. This produces individual
    windows in the moving window algorithm.
        Args:
            ds: model xarray dataset
            temp_xyr: subset of station ratios
                from which the kriging is calculated. This array
                must contain [longitudes, latitudes, ratios]
        Returns:
            kriged subset field
    """

    xmin, xmax = temp_xyr[:, 0].min(), temp_xyr[:, 0].max()
    ymin, ymax = temp_xyr[:, 1].min(), temp_xyr[:, 1].max()

    latlon = temp_xyr[:, :2].T

    stats = temp_xyr[:, 2]

    lw, u = (
        find_nearest_index(ds.rlat.values, ymin),
        find_nearest_index(ds.rlat.values, ymax),
    )
    l, r = (
        find_nearest_index(ds.rlon.values, xmin),
        find_nearest_index(ds.rlon.values, xmax),
    )

    ylim = u - lw
    xlim = r - l

    z, x, y = sp.fit(latlon, stats, xlim, ylim, extrap=False)

    final = np.ones((ds.rlat.size, ds.rlon.size), dtype=np.float16)
    final[:] = np.nan
    final[lw:u, l:r] = z.T

    return final


def rkrig_r(
    df: pd.DataFrame, n: int, ds: xr.Dataset, station_dv: str, min_size: int = 30
):
    """Implements climpyricals moving window method.
    Args:
        df: pandas dataframe containing the coordinates in
            both regular and roated, as well as the station
            data
        n: number of nearest neighbors to northern
            most stations
        ds: model xarray dataset
        min_size: minimum number of grid cells in target res
            to include in the reconstruction. This number is
            used to calculate an equivalent minimum area
            that is compared to the polygon produced by
            the perimeter of stations in a nearest neighbor set
    Returns:
        kriged field
    """


    dataframe_keys = ["lat", "lon", "rlat", "rlon", "ratio"]
    check_df(df, dataframe_keys)

    X_distances = np.stack([np.deg2rad(df.lat.values), np.deg2rad(df.lon.values)])
    dx = (np.amax(ds.rlon.values) - np.amin(ds.rlon.values)) / ds.rlon.size
    dy = (np.amax(ds.rlat.values) - np.amin(ds.rlat.values)) / ds.rlat.size
    dA = dx * dy

    xyr = df[["rlon", "rlat", "ratio"]].values

    # used to calculate average at end
    field = np.ones((ds.rlat.size, ds.rlon.size))
    field[:] = np.nan

    # tracks the number of summations in each grid cell
    nancount = np.zeros(field.shape)
    violate_area = []
    with tqdm(total=len(df.ratio), position=0, leave=True) as pbar:
        for i in range(df.ratio.size):
            pbar.update()
            nn = n
            if station_dv == 'RL50 (kPa)' and df.iloc[i].lat >= 60.:
                nn = 40
            elif "province" in df.columns:
                WPcond = (
                    ((station_dv == "WP10") or
                    (station_dv == "WP50")) and
                    (
                        (df.iloc[i].province == "QC") or
                        (df.iloc[i].province == "NL") or
                        (df.iloc[i].province == "NU")
                    ) and
                    (df.iloc[i].lat >= 52.0) 
                )

                if WPcond:
                    nn = 10
            else:
                nn = n

            nbrs = NearestNeighbors(n_neighbors=nn, metric="haversine").fit(
                X_distances.T
            )
            dist, ind = nbrs.kneighbors(X_distances.T)
            temp_xyr = xyr[ind[i], :]

            latlon = temp_xyr[:, :2]

            hull = ConvexHull(points=latlon)
            while hull.area < dA * min_size ** 2:
                warnings.warn("Adding stations to window!")
                nn += 1
                nbrs = NearestNeighbors(n_neighbors=nn, metric="haversine").fit(
                    X_distances.T
                )
                dist, ind = nbrs.kneighbors(X_distances.T)

                temp_xyr = xyr[ind[i], :]
                latlon = temp_xyr[:, :2]
                hull = ConvexHull(points=latlon)
            try:
                this_field = krig_at_field(ds, temp_xyr)
                field = np.nansum([field, this_field], axis=0)
                nancount[~np.isnan(this_field)] += 1

            except rpy2.rinterface_lib.embedded.RRuntimeError:
                continue

        # taking this fraction computes the mean
        return field / nancount

