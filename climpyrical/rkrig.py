import climpyrical.spytialProcess as sp
from climpyrical.gridding import find_nearest_index

from nptyping import NDArray
from typing import Any
import xarray as xr
from sklearn.neighbors import NearestNeighbors
from pykrige.ok import OrdinaryKriging
from tqdm import tqdm
import dask.array as da
from scipy.spatial import ConvexHull
import rpy2


import numpy as np
import pandas as pd

def krigit_north(
    df: pd.DataFrame,
    station_dv: str,
    n: int,
    ds: xr.Dataset
) -> NDArray[(Any, Any), float]:

    df = df[['lat', 'lon', 'rlat', 'rlon', station_dv]]

    dataframe_keys = ['lat', 'lon', 'rlat', 'rlon']
    contains_keys = [key not in df.columns for key in dataframe_keys]

    if np.any(contains_keys):
        raise ValueError(f"Dataframe must contain {dataframe_keys}")

    regular_points = np.stack([np.deg2rad(df.lat), np.deg2rad(df.lon)]).T

    # Metrics intended for two-dimensional vector spaces: Note that the
    # haversine distance metric requires data in the form 
    # of [latitude, longitude] and both inputs and outputs are in units of radians.
    nbrs = NearestNeighbors(n_neighbors=n, metric='haversine').fit(regular_points)
    dist, ind = nbrs.kneighbors(regular_points)
    imax = df.rlat.idxmax(axis=0, skipna=True)
    temp_df = df.iloc[ind[imax]]

    xmin, xmax = temp_df.rlon.min(), temp_df.rlon.max()
    ymin, ymax = temp_df.rlat.min(), temp_df.rlat.max()

    latlon = np.stack([temp_df.rlon, temp_df.rlat])
    stats = temp_df[station_dv]

    lw, u = find_nearest_index(ds.rlat.values, ymin), find_nearest_index(ds.rlat.values, ymax)
    l, r = find_nearest_index(ds.rlon.values, xmin), find_nearest_index(ds.rlon.values, xmax)
    ylim = u-lw
    xlim = r-l    

    z, x, y = sp.fit(latlon, stats, xlim, ylim, extrap=True)

    field = np.ones((ds.rlat.size, ds.rlon.size))
    field[:, :] = np.nan
    field[lw:u, l:r] = z.T

    return field

def rkrig_py(
    df: pd.DataFrame,
    station_dv: str,
    n: int,
    ds: xr.Dataset,
    exact_values: bool=False
) -> NDArray[(Any, Any), float]:

    ok = OrdinaryKriging(
        df.rlon, 
        df.rlat, 
        df[station_dv], 
        exact_values,
        variogram_function='exponential',
    )
    z, ss = ok.execute(
        "grid",
        ds.rlon.values,
        ds.rlat.values,
        backend='C',
        n_closest_points=n
    )

    return z


def rkrig_r(df, indices, n, ds, min_size = 30):

    Zl = []
    

    X_distances = np.stack([np.deg2rad(df.lat.values), np.deg2rad(df.lon.values)])
    dx = ((np.amax(ds.rlon.values)-np.amin(ds.rlon.values))/ds.rlon.size)
    dy = ((np.amax(ds.rlat.values)-np.amin(ds.rlat.values))/ds.rlat.size)
    dA = dx*dy

    xyr = df[['rlon', 'rlat', 'ratio']].values

    with tqdm(total=len(indices), position=0, leave=True) as pbar:
        for i in indices:
            pbar.update()
            nn = n

            nbrs = NearestNeighbors(n_neighbors=nn, metric='haversine').fit(X_distances.T)
            dist, ind = nbrs.kneighbors(X_distances.T)
            temp_xyr = xyr[ind[i], :]

            latlon = temp_xyr[:, :2]
            stats = temp_xyr[:, 2]
            hull = ConvexHull(points=latlon)


            while hull.area < dA*min_size**2:
                nn+=1
                nbrs = NearestNeighbors(n_neighbors=nn, metric='haversine').fit(X_distances.T)
                dist, ind = nbrs.kneighbors(X_distances.T)

                temp_xyr = xyr[ind[i], :]
                latlon = temp_xyr[:, :2]
                hull = ConvexHull(points=latlon.T)

            try:
                final = krig_at_field(ds, temp_xyr)
                final_da = da.from_array(final, chunks=(250, 250))
                Zl.append(final_da)

            except rpy2.rinterface_lib.embedded.RRuntimeError:
                continue

        return Zl

def krig_at_field(ds, temp_xyr):
        
    xmin, xmax = temp_xyr[:, 0].min(), temp_xyr[:, 0].max()
    ymin, ymax = temp_xyr[:, 1].min(), temp_xyr[:, 1].max()

    latlon = temp_xyr[:, :2].T
    stats = temp_xyr[:, 2]

    lw, u = find_nearest_index(ds.rlat.values, ymin), find_nearest_index(ds.rlat.values, ymax)
    l, r = find_nearest_index(ds.rlon.values, xmin), find_nearest_index(ds.rlon.values, xmax)

    ylim = u-lw
    xlim = r-l

    z, x, y = sp.fit(latlon, stats, xlim, ylim, extrap=False)

    final = np.ones((ds.rlat.size, ds.rlon.size), dtype=np.float16)
    final[:] = np.nan
    final[lw:u, l:r] = z.T
    
    return final