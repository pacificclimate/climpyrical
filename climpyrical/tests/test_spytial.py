import pytest
import numpy as np
import climpyrical.spytialProcess as sp
from climpyrical.gridding import flatten_coords

N = 10
xx, yy = np.meshgrid(np.linspace(0, 50, N), np.linspace(-25, 25, N))
nxx, nyy = flatten_coords(np.linspace(0, 50, N), np.linspace(-25, 25, N))
coords = np.stack([nxx, nyy])
z = np.sin(xx ** 2 + yy ** 2) / (xx ** 2 + yy ** 2)
z = z.flatten()

new_N = 3 * N
newz, newx, newy = sp.fit(coords, z, new_N, new_N, True)


@pytest.mark.parametrize(
    "latlon, z, nx, ny, extrap, error",
    [
        (coords, z, new_N, new_N, True, None),
        (np.ones((1, 2, 3)), z, new_N, new_N, True, TypeError),
        (coords, np.ones((1, 2, 3)), new_N, new_N, True, TypeError),
        (coords, z, "blargh", new_N, True, TypeError),
        (coords, z, new_N, "blargh", True, TypeError),
        (coords, z[:-1], new_N, new_N, True, ValueError),
    ],
)
def test_fit_params(latlon, z, nx, ny, extrap, error):
    if error is None:
        sp.fit(latlon, z, nx, ny, extrap)
    else:
        with pytest.raises(error):
            sp.fit(latlon, z, nx, ny, extrap)


def test_nan():
    assert isinstance(newz, np.ndarray)
    assert not np.all(np.isnan(newz))
    assert not np.all(np.isnan(newx))
    assert not np.all(np.isnan(newy))


def test_zero():
    assert not np.all(np.isclose(newz, 0.0))
    assert not np.all(np.isclose(newx, 0.0))
    assert not np.all(np.isclose(newy, 0.0))


def test_ranges():
    # arbitrary large number
    assert not np.any(newz >= 1e7)
    assert not np.any(newx >= 1e7)
    assert not np.any(newy >= 1e7)
    assert not np.any(newz <= -1e7)
    assert not np.any(newx <= -1e7)
    assert not np.any(newy <= -1e7)


def test_mean():
    assert np.nanmean(newz)
    assert np.isclose(np.nanmean(newx), np.nanmean(nxx), rtol=10)
    assert np.isclose(np.nanmean(newy), np.nanmean(nyy), rtol=0.25)


def test_shape():
    assert newz.shape == (new_N, new_N)
    assert newx.shape == (new_N,)
    assert newy.shape == (new_N,)
