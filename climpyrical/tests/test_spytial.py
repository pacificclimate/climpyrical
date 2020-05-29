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
    "latlon, z, nx, ny, xy, distance, variogram_model, error",
    [
        (coords, z, 3 * N, 3 * N, (1, 2), "geo", "exponential", None),
        ("blargh", z, 3 * N, 3 * N, (1, 2), "geo", "exponential", TypeError),
        (
            coords,
            "blargh",
            3 * N,
            3 * N,
            (1, 2),
            "geo",
            "exponential",
            TypeError,
        ),
        (coords, z, "blargh", 3 * N, (1, 2), "geo", "exponential", TypeError),
        (coords, z, 3 * N, "blargh", (1, 2), "geo", "exponential", TypeError),
        (coords, z, 3 * N, 3 * N, "blargh", "geo", "exponential", TypeError),
        (coords, z, 3 * N, 3 * N, (1, 2), "blargh", "exponential", ValueError),
        (coords, z, 3 * N, 3 * N, (1, 2), "geo", "blargh", ValueError),
        (coords[:, :-1], z, 3 * N, 3 * N, (1, 2), "geo", "blargh", ValueError),
        (coords, z, 3 * N, 3 * N, (1, 2), 4, "blargh", TypeError),
    ],
)
def test_fit_params(latlon, z, nx, ny, xy, distance, variogram_model, error):
    if error is None:
        sp.fit(latlon, z, nx, ny, xy, distance, variogram_model)
    else:
        with pytest.raises(error):
            sp.fit(latlon, z, nx, ny, xy, distance, variogram_model)
