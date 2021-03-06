from climpyrical.gridding import (
    scale_model_obs,
    check_input_coords,
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
    regrid_ensemble,
    extend_north,
    rot2reg,
)
from climpyrical.data import read_data
import pytest
from pkg_resources import resource_filename
import numpy as np
import xarray as xr
from nptyping import NDArray
from typing import Any


@pytest.mark.parametrize(
    "model_vals, station_vals, error",
    [
        (np.linspace(1, 10, 10), np.linspace(1, 10, 10), None),
        (np.linspace(1, 10, 10), np.linspace(-5, 10, 10), "UserWarning"),
        ([1, 10, np.nan], [1, 10, 100.0], ValueError),
        ([1, 10, 100.0], [1, 10, np.nan], ValueError),
        ([1, 0.0, 100.0], [1, 10, np.nan], ValueError),
    ],
)
def test_scale_model_obs(model_vals, station_vals, error):
    if error is None:
        ratio, best_tol = scale_model_obs(model_vals, station_vals)
        assert np.all(ratio >= 0.0)
        assert np.all(ratio <= 25.0)
        assert np.all(best_tol > 0.0)
    elif isinstance(error, type(ValueError)):
        with pytest.raises(error):
            scale_model_obs(model_vals, station_vals)
    else:
        with pytest.warns(UserWarning):
            scale_model_obs(model_vals, station_vals)


@pytest.mark.parametrize(
    "data,n,error",
    [
        (np.linspace(0, 10), 1, None),
        (np.ones((10, 10)), 2, None),
        ("array", 3, TypeError),
        (np.ones((10, 10)), "int", TypeError),
        (np.ones((10, 10, 10)), 4, ValueError),
    ],
)
def test_check_ndims(data, n, error):
    if error is None:
        check_ndims(data, n)
    else:
        with pytest.raises(error):
            check_ndims(data, n)


# load example ensemble dataset for testing
# dv = "Rain-RL50"
dv = "snw"
ds = read_data(resource_filename("climpyrical", "tests/data/example2.nc"))

ds_regridded_proper = read_data(
    resource_filename("climpyrical", "tests/data/snw_target_res.nc")
)


@pytest.mark.parametrize(
    "ds,dv,n,keys,copy",
    [
        (ds, dv, 3, ["rlon", "rlat", "lon", "lat"], True),
        (ds, dv, 3, ["rlon", "rlat", "lon", "lat"], False),
    ],
)
def test_regrid_ensemble(ds, dv, n, keys, copy):
    ndim = np.ndim(ds[dv].values)
    nds = regrid_ensemble(ds, dv, n, keys, copy)
    assert isinstance(nds[dv].values, NDArray[(Any,) * ndim, Any])


# read grids with expected dimension and ranges
xi, yi = ds.rlon.values, ds.rlat.values

# extend coordinates in expected way
xext_ex, yext_ex = np.tile(xi, yi.size), np.repeat(yi, xi.size)
xext_bad, yext_bad = np.repeat(xi, yi.size), np.tile(yi, xi.size)


@pytest.mark.parametrize(
    "x,y,error",
    [
        ("x", yi, TypeError),
        (xi, "y", TypeError),
        (np.linspace(0, 10, 30), yi, ValueError),
        (xi, np.linspace(0, 10, 130), ValueError),
        (xi, yi, None),
        (xext_ex, yext_ex, None),
    ],
)
def test_check_input_coords(x, y, error):
    if error is None:
        check_input_coords(x, y, ds)
    else:
        with pytest.raises(error):
            check_input_coords(x, y, ds)


@pytest.mark.parametrize("xi,yi,xext_ex,yext_ex", [(xi, yi, xext_ex, yext_ex)])
def test_flatten_coords(xi, yi, xext_ex, yext_ex):
    xext, yext = flatten_coords(xi, yi)
    assert np.array_equal(xext_ex, xext) and np.array_equal(yext_ex, yext)


x_station = np.linspace(-100, -80, 10)
y_station = np.linspace(45, 70, 10)
x_station_bad = np.linspace(-10, 10, 10)
y_station_bad = np.linspace(-10, 10, 10)

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
    "x,y,source_crs,target_crs,error",
    [
        (4, 4, source_crs, target_crs, TypeError),
        (x_station, 4, source_crs, target_crs, TypeError),
        (4, x_station, source_crs, target_crs, TypeError),
        (x_station, y_station, source_crs, target_crs, None),
        (x_station, y_station, "source_crs", target_crs, TypeError),
        (x_station, y_station, source_crs, "target_crs", TypeError),
        (x_station[:-1], y_station, source_crs, target_crs, ValueError),
        (x_station, y_station[:-1], source_crs, target_crs, ValueError),
    ],
)
def test_check_transform_coords_inputs(x, y, source_crs, target_crs, error):
    if error is None:
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
    "data,val,error",
    [
        ("data", 1.0, TypeError),
        (data, 1.0, None),
        (data, "2", TypeError),
        (bad_data, 1.0, ValueError),
        (bad_data_a, 1.0, ValueError),
    ],
)
def test_check_find_nearest_index_inputs(data, val, error):
    if error is None:
        check_find_nearest_index_inputs(data, val)
    else:
        with pytest.raises(error):
            check_find_nearest_index_inputs(data, val)


# @pytest.mark.parametrize(
#     "data,val,warning",
#     [(data, 25.0, None), (data, 35.0, UserWarning)],
# )
# def test_check_find_nearest_index_inputs_warnings(data, val, warning):
#     if warning is None:
#         check_find_nearest_index_inputs(data, val)
#     with pytest.warns(warning):
#         check_find_nearest_index_inputs(data, val)


@pytest.mark.parametrize(
    "data,val,expected",
    [(data, 5.0, 4), (data, 3.0, 2), (np.linspace(-100, 100, 200), -50.0, 50)],
)
def test_find_nearest_index(data, val, expected):
    assert find_nearest_index(data, val) == expected


@pytest.mark.parametrize(
    "x,y,x_obs,y_obs,error",
    [
        ("x", 1, 2, 3, TypeError),
        (
            np.linspace(-10, 10, 1),
            np.linspace(-10, 10, 130),
            np.linspace(-10, 10, 10),
            np.linspace(-10, 10, 9),
            ValueError,
        ),
        (
            np.linspace(-10, 10, 155),
            np.linspace(-10, 10, 130),
            np.linspace(-10, 10, 10),
            np.linspace(-10, 10, 10),
            None,
        ),
    ],
)
def test_check_find_element_wise_nearest_pos_inputs(x, y, x_obs, y_obs, error):
    if error is None:
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
            np.array(range(20)),
        )
    ],
)
def test_find_element_wise_nearest_pos(x, y, x_obs, y_obs, expected_x, expected_y):
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
    "x,y,x_i,y_i,field,error",
    [
        (x, y, idx, idx, good_field, None),
        (x, y, "x", "y", good_field, TypeError),
        (x, y, idx, idx, bad_field, ValueError),
        (x, y, bad_idx, bad_idx, good_field, ValueError),
        (x, y, idx, idx, good_field_nan, None),
    ],
)
def test_check_find_nearest_value_inputs(x, y, x_i, y_i, field, error):
    if error is None:
        check_find_nearest_value_inputs(x, y, x_i, y_i, field)
    else:
        with pytest.raises(error):
            check_find_nearest_value_inputs(x, y, x_i, y_i, field)


x_i = np.arange(20)
y_i = np.arange(20)
idx = np.array([10, 12, 14])
good_field *= np.pi
good_field_nan = good_field.copy()
good_field_nan[idx, idx] = np.nan


@pytest.mark.parametrize(
    "x,y,x_i,y_i,field,expected",
    [
        (x, y, idx, idx, good_field, np.ones(idx.size) * np.pi),
        (x, y, idx, idx, good_field_nan, np.ones(idx.size) * np.pi),
    ],
)
def test_find_nearest_index_value(x, y, x_i, y_i, field, expected):
    final = find_nearest_index_value(x, y, x_i, y_i, field)
    truth = (
        np.any(np.isnan(final))
        or final.size != x_i.size
        or final.size != y_i.size
        or np.allclose(expected, final) is False
    )

    assert truth is False


dv = "snw"
ds_extnorth_bad = read_data(resource_filename("climpyrical", "tests/data/example2.nc"))


@pytest.mark.parametrize(
    "ds,dv,amount,fill_val,error",
    [
        (ds, dv, "200", np.nan, TypeError),
        (ds, dv, -1, np.nan, ValueError),
        (ds, dv, 20, np.nan, None),
    ],
)
def test_extend_north(ds, dv, amount, fill_val, error):
    if error is None:
        ds_ext = extend_north(ds, dv, amount, fill_val)
        assert ds_ext[dv].values.shape == (ds.rlat.size + amount, ds.rlon.size)
    else:
        with pytest.raises(error):
            extend_north(ds, dv, amount, fill_val)


ds3d = xr.open_dataset(resource_filename("climpyrical", "tests/data/snw.nc"))


@pytest.mark.parametrize("ds,error", [(ds, None), (ds3d, ValueError)])
def test_rot2reg(ds, error):
    dv = list(ds.data_vars)[0]

    if error is None:
        newds = rot2reg(ds)
        assert newds[dv].values.shape == ds[dv].values.shape
    else:
        with pytest.raises(error):
            rot2reg(ds)
