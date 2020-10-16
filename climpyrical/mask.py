from climpyrical.gridding import find_nearest_index, flatten_coords
import warnings
from typing import Union, Any
from nptyping import NDArray
import numpy as np
from shapely.geometry import Polygon
from tqdm import tqdm
import geopandas as gpd


def check_polygon_validity(p: Union[gpd.GeoSeries, gpd.GeoDataFrame]) -> bool:
    """Checks that the polygon provided is valid
    Args:
        p: polygon of type geopandas.GeoSeries
    Raises:
        ValueError: for GeoDataFrame that is empty or invalid values
    """

    if not p.size > 0:
        raise ValueError("Empty data provided in polygons")

    return True


def check_polygon_before_projection(
    p: Union[gpd.GeoSeries, gpd.GeoDataFrame]
) -> bool:
    """Raises an warning if polygon provided does not
    contain the expected WGS84 projection, but does not stop
    code from running
    Args:
        p: polygon of type geopandas.GeoSeries
    Returns:
        bool True if passed
    """
    # check polygon validity
    check_polygon_validity(p)

    warning1 = "Polygon provided is in unexpected projection. Expected epsg:4326.\
                Other transformations are experimental and have not been tested."

    warning2 = 'Neither init or datum found in proj4 data. Please provide initial \
                        reference projection with Polygon.crs = "epsg:4326"'

    # is dict or not
    if isinstance(p.crs, dict):
        if "datum" in p.crs.keys():
            if p.crs["datum"] != "WGS84":
                warnings.warn(
                    UserWarning(
                        warning1
                    )
                )
        else:
            warnings.warn(
                UserWarning(
                    warning2
                )
            )
    elif isinstance(p.crs, str):
        if "epsg:4326" in p.crs:
            warnings.warn(
                UserWarning(
                    warning1
                )
            )
        else:
            warnings.warn(
                UserWarning(
                    warning2
                )
            )
    else:
        print("CRS", p.crs)
        if "datum" in p.crs.to_dict().keys():
            if p.crs.to_dict()["datum"] != "WGS84":
                warnings.warn(
                    UserWarning(
                        warning1
                    )
                )
        else:
            warnings.warn(
                UserWarning(
                    warning2
                )
            )

    return True


def rotate_shapefile(
    p: Union[gpd.GeoSeries, gpd.GeoDataFrame],
    crs: dict = {
        "proj": "ob_tran",
        "o_proj": "longlat",
        "lon_0": -97,
        "o_lat_p": 42.5,
        "a": 6378137,
        "to_meter": 0.0174532925199,
        "no_defs": True,
    },
) -> gpd.GeoSeries:
    """Rotates a shapefile to a new crs defined by a proj4 dictionary.
    Uses geopandas crs functions.
    Args:
        p (geopandas.GeoSeries object): polygons of Canada
        crs (dict): proj4 dictionary
    Returns:
        target (geopandas.GeoSeries object): geographic polygons
            in new projection
    """
    # this checks polygon can be rotated
    check_polygon_before_projection(p)

    target = p.to_crs(crs)

    return target


def make_box(x: float, y: float, dx: float, dy: float) -> Polygon:
    """Creates a Polygon box that mimics a grid cell's geometry
    Args:
        x, y (float, float): coordinates of center of grid cell
        dx, dy (float, float): distance from center of grid cell to grid
        cell's edges
    Returns:
        (shapely Polygon object): Polygon representation of grid
    """
    test_list = [x, y, dx, dy]
    test_list_float = [isinstance(a, (int, float)) for a in test_list]
    if not np.all(test_list_float):
        raise TypeError(f"All parameters must be floats.")

    p1 = x - dx, y - dy
    p2 = x + dx, y - dy
    p3 = x + dx, y + dy
    p4 = x - dx, y + dy

    return Polygon([p1, p2, p3, p4])


def gen_raster_mask_from_vector(
    x: NDArray[(Any,), np.float],
    y: NDArray[(Any,), np.float],
    p: Union[gpd.GeoSeries, gpd.GeoDataFrame],
    progress_bar: bool = True,
) -> NDArray[(Any, Any), Any]:
    """Determines if points are contained within polygons of Canada
    Args:
        x, y (np.ndarray): Arrays containing the rlon and rlat of CanRCM4
            grids
        p (geopandas.GeoSeries object): rotated polygons of Canada
        progress_bar (bool): True, whether to display a tqdm progress bar. This
            operation can take a long time depending on the target resolution and
            the grid size/complexity of polygons provided.
    Returns:
        mask (np.ndarray): boolean 2D grid mask of CanRCM4 raster clipped
            based on polygon boundaries
    """

    # this checks the polygon input
    check_polygon_validity(p)

    cx1, cx2 = p.bounds.minx.min(), p.bounds.maxx.max()
    cy1, cy2 = p.bounds.miny.min(), p.bounds.maxy.max()

    # find the bounds of canada to begin clipping to save computation time
    icx1, icx2 = find_nearest_index(x, cx1), find_nearest_index(x, cx2)
    icy1, icy2 = find_nearest_index(y, cy1), find_nearest_index(y, cy2)

    dx = np.mean(np.diff(x)) / 2.0
    dy = np.mean(np.diff(y)) / 2.0

    # ordered pairs of relevent grid cells
    xx, yy = flatten_coords(x[icx1:icx2], y[icy1:icy2])
    xy = np.stack([xx, yy]).T
    if xy.size == 0:
        raise ValueError(
            "No matching coordinates. \
            Does polygon overlap with coordinates provided?"
        )

    # track whether or not grid cell is within polygon here
    contained = []

    if progress_bar:
        with tqdm(total=len(xy), position=0, leave=True) as pbar:
            for xcoord, ycoord in xy:
                pbar.update()
                contained.append(
                    np.any(p.intersects(make_box(xcoord, ycoord, dx, dy)))
                )
    else:
        for xcoord, ycoord in xy:
            contained.append(
                np.any(p.intersects(make_box(xcoord, ycoord, dx, dy)))
            )

    # convert back to original target size and shape
    contained = np.array(contained).reshape((icy2 - icy1, icx2 - icx1))
    mask = np.zeros((y.size, x.size))
    mask[icy1:icy2, icx1:icx2] = contained

    return mask == 1
