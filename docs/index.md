# Climpyrical

API Modules:
- [`data`](https://pacificclimate.github.io/climpyrical/data.html)
- [`gridding`](https://pacificclimate.github.io/climpyrical/gridding.html)
- [`mask`](https://pacificclimate.github.io/climpyrical/mask.html)
- [`rkrig`](https://pacificclimate.github.io/climpyrical/rkrig.html)
- [`spytialProcess`](https://pacificclimate.github.io/climpyrical/spytialProcess.html)

# Getting started
A demo notebook can be found in `climpyrical/notebooks/` that demonstrates some basic functionality of the software. Additionally, the full pipeline for moving window ratio reconstruction can be found in `climpyrical/notebooks/interactive/`.

### Reading Data
Load an ensemble of climate models using `climpyrical`'s `read_data` function. `read_data` creates an `xarray` dataset containing the fields defined by `keys` and by the design value key as found in the climate model.
```python
from climpyrical.data import read_data

# necessary keys to load from .nc file

ds = read_data('/path/to/data.nc')
```

### Masking Models
To reamain domain flexible in `climpyrical`, shapefiles can be provided to mask the analysis to include only modeled values within that shape.
```python3
from climpyrical.mask rotate_shapefile, import gen_raster_mask_from_vector

world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
can_index = world[world.name == "Canada"].index
can_geom = world.loc[can_index, 'geometry']

rotated_canada = rotate_shapefile(can_geom)

mask = gen_raster_mask_from_vector(ds.rlon, ds.rlat, rotated_canada)
```

`mask` contains a 2 dimensional grid with boolean values masked based on the `rotated_canada` `GeoSeries`.

More to be found in `notebooks/dev/climpyrical_demo.ipynb`.

### Type Hinting
Climpyrical's functions have opted for type hints rather than a strict checking of each data input and output. It is also meant to help aid in the documentation and understanding of how to use its functions. In general, array like object inputs are assumed to be `numpy.ndarray`s. These arrays should be of a specific shape (sometimes datatype), in order to play well with the functions. Incorrect array types are *usually* not strictly enforced within `climpyrical` itself, and errors about the nature of array inputs are raised mainly in external packages. However, we have included [nptyping](https://github.com/ramonhagenaars/nptyping) - a third party type hinting package designed specifically for numpy arrays. We are currently working to have climpyrical's usage (starting with testing) be with `mypy` which reports errors when the type hinting is not met (aka enforced type hinting).

Examples of understanding numpy typing:
```python3
def flatten_coords(
    x: NDArray[(Any, ), float], y: NDArray[(Any, ), float]
) -> Tuple[NDArray[(Any, ), float], NDArray[(Any, ), float]]:
    """Takes the rlat and rlon 1D arrays from the
    NetCDF files for each ensemble member, and creates
    an ordered pairing of each grid cell coordinate in
    rotated pole (rlat, rlon).
    ...
```

In this simple example above, `x` and `y` are `numpy` `NDArray` objects of 1 dimension and any length of type `float`.

An example of 2 dimensional arrays is below:

```python3
def interpolate_dataset(
    points: NDArray[(Any, Any), np.float],
    values: NDArray[(Any, Any), np.float],
    target_points: NDArray[(Any, Any), np.float],
    method: str,
) -> NDArray[(Any,), np.float]:

    ...
```

This function, for example, takes `points` a 2 dimensional `NDArray` of any size (in either dimension) and of type `float`. It returns a 1 dimensional `NDArray` of any size and type `float`.  

The description of what these variables mean are still found in the docstring of the function itself, however, the intended type of `NDArray` provided can be found by reading the type hints throughout `climpyrical`. 
