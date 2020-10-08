import pytest
import pandas as pd
import numpy as np
from nptyping import NDArray
from typing import Any

from climpyrical.rkrig import check_df, krigit_north, rkrig_py, rkrig_r
from climpyrical.data import read_data
from pkg_resources import resource_filename

df = pd.DataFrame({"x": np.ones(5), "y": np.ones(5), "z": np.ones(5)})

ds = read_data(
    resource_filename("climpyrical", "tests/data/canada_mask_rp.nc")
)

df_ = pd.read_csv(
    resource_filename("climpyrical", "tests/data/sl50_short.csv")
)


@pytest.mark.parametrize(
    "df, keys, error",
    [
        (df, ["x", "y", "z"], None),
        (df, ["x", "y", "z", "x1"], KeyError),
    ],
)
def test_check_df(df, keys, error):
    if error is None:
        check_df(df, keys)
    else:
        with pytest.raises(error):
            check_df(df, keys)


@pytest.mark.parametrize(
    "df, station_dv, n, ds",
    [
        (df_, "TJan2.5 (degC)", 10, ds),
    ],
)
def test_krigit_north(df, station_dv, n, ds):
    result = krigit_north(df, station_dv, n, ds)

    assert result.shape == (ds.rlat.size, ds.rlon.size)
    # results should not all be NaN
    assert not np.all(np.isnan(result))

    assert not np.allclose(result, 0.0)

    assert isinstance(result, NDArray[(Any, Any), float])
    # resulting area should be less than the total area of canada
    assert np.sum(~np.isnan(result)) < np.sum(~np.isnan(ds["mask"].values))


@pytest.mark.slow
@pytest.mark.parametrize(
    "df, station_dv, n, ds",
    [
        (df_, "TJan2.5 (degC)", 10, ds),
    ],
)
def test_rkrig_py(df, station_dv, n, ds):
    result = rkrig_py(df.iloc[:100], station_dv, n, ds)

    assert result.shape == (ds.rlat.size, ds.rlon.size)
    # results should not all be NaN
    assert not np.all(np.isnan(result))

    assert not np.allclose(result, 0.0)

    assert isinstance(result, NDArray[(Any, Any), float])


@pytest.mark.slow
@pytest.mark.parametrize(
    "df, n, ds, min_size",
    [(df_, 30, ds, 2), (df_, 30, ds, 200)],
)
def test_rkrig_r(df, n, ds, min_size):
    result = rkrig_r(df, n, ds, min_size)

    assert result.shape == (ds.rlat.size, ds.rlon.size)
    # results should not all be NaN
    assert not np.all(np.isnan(result))

    assert not np.allclose(result, 0.0)

    assert isinstance(result, NDArray[(Any, Any), float])
