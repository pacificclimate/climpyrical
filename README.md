![Python CI](https://github.com/pacificclimate/climpyrical/workflows/Python%20CI/badge.svg)
# climpyrical

`climpyrical` is a Python tool for reconstructing design value fields using meteorological station observations 

and ensembles of design value fields provided in CanRCM4 models.

# Setup

`climpyrical` is still in development and is not registered. To package `climpyrical`, run
```bash
$ pip install .
```

# Requirements
To install all of the dependencies used by climpyrical, install from requirements file found in `requirements.txt`

via 

```bash
$ pip install -r requirements.txt
```

`climpyrical` also requires a version of `R` be installed with the `fields` package. To do this on a local machine, use

```bash
apt install r-base 
bash r_install.sh
```

which installs all `R` requirements. If you receive an error when installing R packages, check that the latest version is in `r_requirements.txt`.

# Getting started
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
Climpyrical's functions have opted for type hints rather than a strict checking of each data input and output. It is also meant to help aid in the documentation and understanding of how to use its functions. In general, arrays that are supplied to functions in climpyrical are `numpy.ndarray`s. These arrays should be of a specific shape (sometimes datatype), in order to play well with the functions. Incorrect array types are *usually* not strictly enforced within `climpyrical` itself, and errors about the nature of array inputs are raised mainly in external packages. However, we have included [nptyping](https://github.com/ramonhagenaars/nptyping) - a third party type hinting package designed specifically for numpy arrays. We are currently working to have climpyrical's usage (starting with testing) be with `mypy` which reports errors when the type hinting is not met (aka enforced type hinting).

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

### Ratio kriging reconstruction
There are a series of notebooks in `notebooks` that describe how to achieve a ratio reconstruction as demanded by the ratio kriging reconstruction method.

They are meant to be completed in the following order, to acheive results. Please note that this method has only been demonstrated on CanRCM4 gridded models in rotated pole projection, although this software has been adopted to be as flexible as possible to other regions with different projections. 

1.) `notebooks/mask.ipynb` demonstrates constructing a raster mask based on arbitrarily polygons provided. This improves the native CanRCM4 coarse coastlines
2.) `notebooks/process_model.ipynb` demonstrates the necessary preprocessing steps on the model and writes it to file to be used later on in the method.
3.) `notebooks/stations.ipynb` demonstrates how to process the station data corresponding to your CanRCM4 model and writes to file
4.) `notebooks/ratio_kriging.ipynb` performs the ratio kriging method 

Additional steps have not been factored out further due to the ever-changing requirements of various design value fields. 

## Authors
* **Nic Annau** - [Pacific Climate Impacts Consortium](https://www.pacificclimate.org/)
