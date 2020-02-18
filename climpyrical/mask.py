from climpyrical.gridding import check_ndims
import numpy as np
from shapely.geometry import Point
import geopandas as gpd


def check_polygon_validity(p):
    """Checks that the polygon provided is valid
    Args:
        p: polygon of type geopandas.GeoSeries
    Raises:
        TypeError: for wrong type
        ValueError: for GeoDataFrame that is empty or invalid values
    """
    if not isinstance(p, gpd.GeoSeries):
        raise TypeError("Must be gpd.GeoSeries, not {}".format(type(p)))
    if not p.size > 0:
        raise ValueError("GeoDataFrame is is_empty")
    if not p.is_valid.values[0]:
        raise ValueError("GeoDataFrame contains invalid values.")
    return True


def check_polygon_before_projection(p):
    """Raises a warning if polygon provided does not
    contain the expected WGS84 projection, but does not stop
    code from running
    Args:
        p: polygon of type geopandas.GeoSeries
    Returns:
        bool True if passed
    """
    check_polygon_validity(p)
    crs = {"init": "epsg:4326"}
    if p.crs != crs:
        raise ValueError(
            "Polygon provided is in projection {}, expected {}".format(
                p.crs, crs
            )
        )
    return True


def check_polygon_after_projection(p):
    """Checks that the polygon after rotation is in the correct
    projection for rotated pole - a requirement of climpyrical
    Args:
        p: polygon after projection transformation
    Returns:
        bool True if passed
    """
    check_polygon_validity(p)

    crs = {
        "proj": "ob_tran",
        "o_proj": "longlat",
        "lon_0": -97,
        "o_lat_p": 42.5,
        "a": 6378137,
        "to_meter": 0.0174532925199,
        "no_defs": True,
    }

    if p.crs != crs:
        raise ValueError("{} is an incorrect projection".format(p.crs))
    return True


def check_input_grid_coords(x, y):
    """Checks that the input coordinates defining the CanRCM4 grid
    are the expected type, size, and range of values.
    Args:
        x, y (np.ndarray): numpy arrays of rlon, rlat respectively
            of CanRCM4 grids
    Returns:
        bool True if passed
    """
    if (not isinstance(x, np.ndarray)) or (not isinstance(y, np.ndarray)):
        raise TypeError(
            "Please provide an object of type {}".format(np.ndarray)
        )

    check_ndims(x, 1)
    check_ndims(y, 1)

    return True

def rotate_shapefile(
    p,
    crs={
        "proj": "ob_tran",
        "o_proj": "longlat",
        "lon_0": -97,
        "o_lat_p": 42.5,
        "a": 6378137,
        "to_meter": 0.0174532925199,
        "no_defs": True,
    },
):
    """Rotates a shapefile to a new crs defined by a proj4 dictionary.
    Args:
        p (geopandas.GeoSeries object): polygons of Canada
        crs (dict): proj4 dictionary
    Returns:
        target (geopandas.GeoSeries object): geographic polygons
            in new projection
    """
    # this checks the polyon input
    check_polygon_validity(p)
    # this checks polygon can be rotated
    check_polygon_before_projection(p)
    target = p.to_crs(crs)
    # this checks the rotation
    check_polygon_after_projection(target)

    return target


def gen_raster_mask_from_vector(x, y, p):
    """Determines if points are contained within polygons of Canada
    Args:
        x, y (np.ndarray): Arrays containing the rlon and rlat of CanRCM4
            grids
        p (geopandas.GeoSeries object): rotated polygons of Canada
    Returns:
        grid (np.ndarray): boolean 2D grid mask of CanRCM4 raster clipped
            based on polygon boundaries
    """
    # this checks the coordinate inputes
    check_input_grid_coords(x, y)
    # this checks the polygon input
    check_polygon_validity(p)
    # this checks that the polygon is in rotated form
    check_polygon_after_projection(p)
    grid = np.meshgrid(x, y)[0]
    for i, rlon in enumerate(x):
        for j, rlat in enumerate(y):
            pt = Point(rlon, rlat)
            grid[j, i] = p.contains(pt)
    return grid == 1.0
