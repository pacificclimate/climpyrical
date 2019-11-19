from climpyrical.gridding import (
    check_input_coords,
    check_coords_are_flattened,
    check_transform_coords_inputs,
    flatten_coords,
    transform_coords,
    check_find_nearest_inputs,
    find_nearest
)
import pytest
import numpy as np

# create pseudo grids with expected dimension and ranges
xi = np.linspace(-33.8800048828125, 33.8800048828125, 155)
yi = np.linspace(-28.59999656677246, 28.15999984741211, 130)
xext, yext = np.tile(xi, yi.size), np.repeat(yi, xi.shape)
xext_bad, yext_bad = np.repeat(xi, yi.size), np.tile(yi, xi.shape)


@pytest.mark.parametrize(
    "x,y,passed",
    [
        ("x", np.linspace(-24, 24, 155), False),
        (np.linspace(-24, 24, 155), "y", False),
        (
            np.linspace(0, 10, 30),
            np.linspace(-28.59999656677246, 28.15999984741211, 130),
            False,
        ),
        (
            np.linspace(-33.8800048828125, 33.8800048828125, 155),
            np.linspace(0, 10, 130),
            False,
        ),
        (xi, yi, True),
        (xext, yext, True),
    ],
)
def test_check_input_coords(x, y, passed):
    if passed:
        assert check_input_coords(x, y)
    else:
        with pytest.raises((ValueError, TypeError)):
            check_input_coords(x, y)


@pytest.mark.parametrize(
    "x,y,xext,yext,passed",
    [
        (xi, yi, xext, np.delete(yext, xi.size), False),
        (xi, np.delete(yi, 2), xext, yext, False),
        (xi, yi, xext, yext, True),
        (xi, yi, xext_bad, yext, False),
        (xi, yi, xext, yext_bad, False),
        (yi, xi, xext, yext, False),
        (yi, xi, yext, xext, False),
    ],
)
def test_check_coords_are_flattened(x, y, xext, yext, passed):
    if passed:
        assert check_coords_are_flattened(x, y, xext, yext)
    else:
        with pytest.raises((TypeError, ValueError)):
            check_coords_are_flattened(x, y, xext, yext)


@pytest.mark.parametrize("x,y,xext,yext", [(xi, yi, xext, yext)])
def test_flatten_coords(x, y, xext, yext):
    xx, yx = flatten_coords(x, y)
    assert np.array_equal(xext, xx) and np.array_equal(yext, yx)


x_station, y_station = np.linspace(-100, -80, 10), np.linspace(45, 70, 10)
x_station_bad, y_station_bad = (
    np.linspace(-10, 10, 10),
    np.linspace(-10, 10, 10),
)

source_crs = {
    "init": "epsg:4326",
    "proj": "longlat",
    "ellps": "WGS84",
    "datum": "WGS84",
    "towgs84": (0, 0, 0),
}

target_crs = {
    "proj": "ob_tran",
    "o_proj": "longlat",
    "lon_0": -97,
    "o_lat_p": 42.5,
    "a": 6378137,
    "to_meter": 0.0174532925199,
    "no_defs": True,
}


@pytest.mark.parametrize(
    "x,y,source_crs,target_crs,passed",
    [
        (4, 4, source_crs, target_crs, False),
        (x_station, 4, source_crs, target_crs, False),
        (4, x_station, source_crs, target_crs, False),
        (x_station, y_station, source_crs, target_crs, True),
        (x_station_bad, y_station, source_crs, target_crs, False),
        (x_station, y_station_bad, source_crs, target_crs, False),
        (x_station, y_station, "source_crs", target_crs, False),
        (x_station, y_station, source_crs, "target_crs", False),
        (x_station[:-1], y_station, source_crs, target_crs, False),
        (x_station, y_station[:-1], source_crs, target_crs, False),
    ],
)
def test_check_transform_coords_inputs(x, y, source_crs, target_crs, passed):
    if passed:
        assert check_transform_coords_inputs(x, y, source_crs, target_crs)
    else:
        with pytest.raises((TypeError, ValueError)):
            check_transform_coords_inputs(x, y, source_crs, target_crs)


@pytest.mark.parametrize(
    "x,y,source_crs,target_crs,expected_tuple",
    [
        (
            -123.0,
            49.0,
            source_crs,
            target_crs,
            (-16.762937096809097, 4.30869242838931),
        ),
        (
            -53.0,
            49.0,
            source_crs,
            target_crs,
            (27.50539689105958, 9.319029395345673),
        ),
    ],
)
def test_transform_coords(x, y, source_crs, target_crs, expected_tuple):
    assert np.array_equal(
        np.array(transform_coords(x, y)), np.array(expected_tuple)
    )


data = np.arange(1, 30)
bad_data = np.array([1])


@pytest.mark.parametrize(
    'data,val,passed',
    [
        ('data', 1., False),
        (data, 1., True),
        (data, '2', False),
        (bad_data, 1., False),
        (data, 30., False)
    ],
)
def test_check_find_nearest_inputs(data, val, passed):
    if passed:
        assert check_find_nearest_inputs(data, val)
    else:
        with pytest.raises((TypeError, ValueError)):
            check_find_nearest_inputs(data, val)


@pytest.mark.parametrize(
    'data,val,expected',
    [
        (data, 5., 4),
        (data, 3., 2),
        (np.linspace(-100, 100, 200), -50., 50)
    ],
)
def test_find_nearest(data, val, expected):
    assert find_nearest(data, val) == expected
