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


@pytest.mark.parametrize(
    "latlon, z, nx, ny, xy, error",
    [
        (coords, z, new_N, new_N, (1, 2), None),
        (np.ones((1, 2, 3)), z, new_N, new_N, (1, 2), TypeError),
        (coords, np.ones((1, 2, 3)), new_N, new_N, (1, 2), TypeError),
        (coords, z, "blargh", new_N, (1, 2), TypeError),
        (coords, z, new_N, "blargh", (1, 2), TypeError),
        (coords, z, new_N, new_N, "blargh", TypeError),
        (coords, z[:-1], new_N, new_N, (1, 2), ValueError)
    ],
)
def test_fit_params(latlon, z, nx, ny, xy, error):
    if error is None:
        sp.fit(latlon, z, nx, ny, xy)
    else:
        with pytest.raises(error):
            sp.fit(latlon, z, nx, ny, xy)
