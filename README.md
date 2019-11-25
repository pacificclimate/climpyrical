![logo.png](https://images.zenhubusercontent.com/5bc02597fcc72f27390ed1f9/c2cf2ba4-edb1-4b47-856e-20338712d4a7)
# climpyrical
[![Build Status](https://travis-ci.org/pacificclimate/climpyrical.svg?branch=master)](https://travis-ci.org/pacificclimate/climpyrical)

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

# Getting started
### Reading Data
Load an ensemble of climate models using `climpyrical`'s `read_data` function. `read_data` creates an `xarray` dataset containing the fields defined by `keys` and by the design value key as found in the climate model.
```python
from climpyrical.datacube import read_data

# necessary keys to load from .nc file
keys = {'rlat', 'rlon', 'lat', 'lon', 'level'}
ds = read_data('/path/to/data.nc', 'snow', keys)
```

### Masking Models
To reamain domain flexible in `climpyrical`, shapefiles can be provided to mask the analysis to include only modeled values within that shape.

```python3
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
can_index = world[world.name == "Canada"].index
can_geom = world.loc[can_index, 'geometry']

rotated_canada = rotate_shapefile(can_geom)

mask = gen_raster_mask_from_vector(ds.rlon, ds.rlat, rotated_canada)
```

`mask` contains a 2 dimensional grid with boolean values masked based on the `rotated_canada` `GeoSeries`.

## Authors
* **Nic Annau** - [Pacific Climate Impacts Consortium](https://www.pacificclimate.org/)
