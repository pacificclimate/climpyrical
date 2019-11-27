from climpyrical.gridding import (
    check_input_coords,
    check_coords_are_flattened,
    check_transform_coords_inputs,
    flatten_coords,
    transform_coords,
    check_find_nearest_index_inputs,
    check_find_element_wise_nearest_pos_inputs,
    check_find_nearest_value_inputs,
    check_ndims,
    find_nearest_index,
    find_element_wise_nearest_pos,
    find_nearest_index_value,
)
from climpyrical.datacube import read_data
import pytest
from pkg_resources import resource_filename
import numpy as np


@pytest.mark.parametrize(
    "data,n,passed,error",
    [
        (np.linspace(0, 10), 1, True, None),
        (np.ones((10, 10)), 2, True, None),
        ("array", 3, False, TypeError),
        (np.ones((10, 10)), 'int', False, TypeError),
        (np.ones((10, 10, 10)), 4, False, ValueError)
    ],
)
def test_check_ndims(data, n, passed, error):
    if passed:
        check_ndims(data, n)
    else:
        with pytest.raises(error):
            check_ndims(data, n)


# load example ensemble dataset for testing
ds = read_data(
        resource_filename("climpyrical", "tests/data/snw_test_ensemble.nc"),
        "Rain-RL50"
    )

# read grids with expected dimension and ranges
xi, yi = ds.rlon.values, ds.rlat.values

# extend coordinates in expected way
xext_ex, yext_ex = np.tile(xi, yi.size), np.repeat(yi, xi.size)
xext_bad, yext_bad = np.repeat(xi, yi.size), np.tile(yi, xi.size)


@pytest.mark.parametrize(
    "x,y,ds,passed,error",
    [
        ("x", yi, ds, False, TypeError),
        (xi, "y", ds, False, TypeError),
        (np.linspace(0, 10, 30), yi, ds, False, ValueError),
        (xi, np.linspace(0, 10, 130), ds, False, ValueError),
        (xi, yi, ds, True, None),
        (xext_ex, yext_ex, ds, True, None),
    ],
)
def test_check_input_coords(x, y, ds, passed, error):
    if passed:
        check_input_coords(x, y, ds)
    else:
        with pytest.raises(error):
            check_input_coords(x, y, ds)


@pytest.mark.parametrize(
    "x,y,xext_ex,yext_ex,ds,passed,error",
    [
        (xi, yi, xext_ex, np.delete(yext_ex, xi.size), ds, False, ValueError),
        (xi, np.delete(yi, 2), xext_ex, yext_ex, ds, False, ValueError),
        (xi, yi, xext_ex, yext_ex, ds, True, None),
        (xi, yi, xext_bad, yext_ex, ds, False, ValueError),
        (xi, yi, xext_ex, yext_bad, ds, False, ValueError),
        (yi, xi, xext_ex, yext_ex, ds, False, ValueError),
        (yi, xi, yext_ex, xext_ex, ds, False, ValueError),
    ],
)
def test_check_coords_are_flattened(x, y, xext_ex, yext_ex, ds, passed, error):
    if passed:
        check_coords_are_flattened(x, y, xext_ex, yext_ex, ds)
    else:
        with pytest.raises(error):
            check_coords_are_flattened(x, y, xext_ex, yext_ex, ds)


@pytest.mark.parametrize(
    "xi,yi,xext_ex,yext_ex,ds",
    [(xi, yi, xext_ex, yext_ex, ds)]
)
def test_flatten_coords(xi, yi, xext_ex, yext_ex, ds):
    xext, yext = flatten_coords(xi, yi, ds)
    assert np.array_equal(xext_ex, xext) and np.array_equal(yext_ex, yext)


x_station, y_station = np.linspace(-100, -80, 10), np.linspace(45, 70, 10)
x_station_bad, y_station_bad = (
    np.linspace(-10, 10, 10),
    np.linspace(-10, 10, 10)
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
    "x,y,source_crs,target_crs,passed,error",
    [
        (4, 4, source_crs, target_crs, False, TypeError),
        (x_station, 4, source_crs, target_crs, False, TypeError),
        (4, x_station, source_crs, target_crs, False, TypeError),
        (x_station, y_station, source_crs, target_crs, True, None),
        (x_station, y_station, "source_crs", target_crs, False, TypeError),
        (x_station, y_station, source_crs, "target_crs", False, TypeError),
        (x_station[:-1], y_station, source_crs, target_crs, False, ValueError),
        (x_station, y_station[:-1], source_crs, target_crs, False, ValueError),
    ],
)
def test_check_transform_coords_inputs(
    x, y, source_crs, target_crs, passed, error
):
    if passed:
        check_transform_coords_inputs(x, y, source_crs, target_crs)
    else:
        with pytest.raises(error):
            check_transform_coords_inputs(x, y, source_crs, target_crs)


@pytest.mark.parametrize(
    "x,y,source_crs,target_crs,expected_tuple",
    [
        (
            np.array([-123.0]),
            np.array([49.0]),
            source_crs,
            target_crs,
            (-16.762937096809097, 4.30869242838931),
        ),
        (
            np.array([-60.0]),
            np.array([49.0]),
            source_crs,
            target_crs,
            (23.44545622, 7.09855438),
        ),
    ],
)
def test_transform_coords(x, y, source_crs, target_crs, expected_tuple):
    assert np.allclose(
        np.array(transform_coords(x, y)).flatten(), np.array(expected_tuple)
    )


data = np.arange(1, 30)
bad_data = np.array([1])
bad_data_a = np.linspace(30, 1, 30)


@pytest.mark.parametrize(
    "data,val,passed,raises",
    [
        ("data", 1.0, False, TypeError),
        (data, 1.0, True, None),
        (data, "2", False, TypeError),
        (bad_data, 1.0, False, ValueError),
        (bad_data_a, 1.0, False, ValueError),
    ],
)
def test_check_find_nearest_index_inputs(data, val, passed, raises):
    if passed:
        check_find_nearest_index_inputs(data, val)
    else:
        with pytest.raises(raises):
            check_find_nearest_index_inputs(data, val)


@pytest.mark.parametrize(
    "data,val,passed,warning",
    [
        (data, 25.0, True, None),
        (data, 35.0, False, UserWarning)
    ],
)
def test_check_find_nearest_index_inputs_warnings(data, val, passed, warning):
    if passed:
        check_find_nearest_index_inputs(data, val)
    with pytest.warns(warning):
        check_find_nearest_index_inputs(data, val)


@pytest.mark.parametrize(
    "data,val,expected",
    [(data, 5.0, 4), (data, 3.0, 2), (np.linspace(-100, 100, 200), -50.0, 50)],
)
def test_find_nearest_index(data, val, expected):
    assert find_nearest_index(data, val) == expected


@pytest.mark.parametrize(
    "x,y,x_obs,y_obs,passed,error",
    [
        ("x", 1, 2, 3, False, TypeError),
        (
            np.linspace(-10, 10, 1),
            np.linspace(-10, 10, 130),
            np.linspace(-10, 10, 10),
            np.linspace(-10, 10, 9),
            False,
            ValueError,
        ),
        (
            np.linspace(-10, 10, 155),
            np.linspace(-10, 10, 130),
            np.linspace(-10, 10, 10),
            np.linspace(-10, 10, 10),
            True,
            None,
        ),
    ],
)
def test_check_find_element_wise_nearest_pos_inputs(
    x, y, x_obs, y_obs, passed, error
):
    if passed:
        check_find_element_wise_nearest_pos_inputs(x, y, x_obs, y_obs)
    else:
        with pytest.raises(error):
            check_find_element_wise_nearest_pos_inputs(x, y, x_obs, y_obs)


@pytest.mark.parametrize(
    "x,y,x_obs,y_obs,expected_x,expected_y",
    [
        (
            np.linspace(-10, 10, 20),
            np.linspace(-10, 10, 20),
            np.linspace(-10, 10, 20),
            np.linspace(-10, 10, 20),
            np.array(range(20)),
            np.array(range(20))
        )
    ],
)
def test_find_element_wise_nearest_pos(
        x, y, x_obs, y_obs, expected_x, expected_y
):
    xclose, yclose = find_element_wise_nearest_pos(x, y, x_obs, y_obs)
    xclose_truth = np.allclose(xclose, expected_x)
    yclose_truth = np.allclose(yclose, expected_y)
    assert xclose_truth and yclose_truth


# simulate a field
good_field = np.ones((130, 155))
good_field_nan = good_field.copy()
bad_field = np.ones((128, 145))

mask = good_field == 1
bad_mask = mask[:-1, :]

x = np.linspace(-33.8800048828125, 33.8800048828125, 155)
y = np.linspace(-28.59999656677246, 28.15999984741211, 130)

badx = np.linspace(-33.8800048828125, 33.8800048828125, 156)
bady = np.linspace(-28.59999656677246, 28.15999984741211, 133)

idx = np.array([10, 12, 14])
bad_idx = np.array([10, 12, 200])


@pytest.mark.parametrize(
    "x,y,x_i,y_i,field,mask,passed,error",
    [
        (x, y, idx, idx, good_field, mask, True, None),
        (x, y, "x", "y", good_field, mask, False, TypeError),
        (idx, idx, x, y, good_field, bad_mask, False, ValueError),
        (x, y, idx, idx, bad_field, mask, False, ValueError),
        (x, y, idx, idx, good_field, bad_mask, False, ValueError),
        (x, y, bad_idx, bad_idx, good_field, bad_mask, False, ValueError),
        (x, y, idx, idx, good_field_nan, mask, True, None),
    ],
)
def test_check_find_nearest_value_inputs(
    x, y, x_i, y_i, field, mask, passed, error
):
    if passed:
        check_find_nearest_value_inputs(x, y, x_i, y_i, field, mask)
    else:
        with pytest.raises(error):
            check_find_nearest_value_inputs(x, y, x_i, y_i, field, mask)


x_i = np.arange(20)
y_i = np.arange(20)
idx = np.array([10, 12, 14])
good_field *= np.pi
good_field_nan = good_field.copy()
good_field_nan[idx, idx] = np.nan


@pytest.mark.parametrize(
    "x,y,x_i,y_i,field,mask,ds,expected",
    [
        (x, y, idx, idx, good_field, mask, ds, np.ones(idx.size) * np.pi),
        (x, y, idx, idx, good_field_nan, mask, ds, np.ones(idx.size) * np.pi),
    ],
)
def test_find_nearest_index_value(x, y, x_i, y_i, field, mask, ds, expected):
    final = find_nearest_index_value(x, y, x_i, y_i, field, mask, ds)
    truth = (
        np.any(np.isnan(final))
        or final.size != x_i.size
        or final.size != y_i.size
        or (np.allclose(expected, final) is False)
    )

    assert truth is False
