# climpyrical
---
A Python tool for spatially downscaling and reconstructing design value fields using meteorological station observations and CanRCM4 models.

# Build status
---
![Python CI](https://github.com/pacificclimate/climpyrical/workflows/Python%20CI/badge.svg)
![Documentation Check](https://github.com/pacificclimate/climpyrical/workflows/Documentation%20Check/badge.svg)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# View Notebooks
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/pacificclimate/climpyrical/HEAD?filepath=notebooks%2Fdemo%2F)
![Logs of Recent Runs](https://nbviewer.jupyter.org/github/pacificclimate/climpyrical_dv_log/tree/main/)

# Setup
```bash
git clone https://github.com/pacificclimate/climpyrical/
```

To install, run
```bash
$ pip install climpyrical/
```

### Requirements
To install all of the dependencies used by climpyrical, install from requirements file found in `requirements.txt`

via 

```bash
$ pip install -r climpyrical/requirements.txt
```

`climpyrical` also requires a version of `R` be installed with the `fields` package. To do this, and install R dependencies on a local machine, use

```bash
apt install r-base 
Rscript install_pkgs.R r_requirements.txt
```

# Getting started
The first step to running the pipeline is configuring it. Various configurations also need to be added to a configuration yaml file. These contain design value specific information, such as paths to input station and model files, plotting parameters, and output filenames. See `config_example.yml` as a full configuration for running the software on all of the design values. 

The recipe is the following, using RL50 as an example:

```yaml
# Which notebooks to use in the pipeline
steps: [
"preprocess_model.ipynb", 
"stations.ipynb", 
"MWOrK.ipynb", 
"plots.ipynb", 
"nbcc_stations.ipynb", 
"combine_tables.ipynb"
]

# To be placed in climpyrical/
paths:
    output_notebook_path: /intermediate/notebook/logs/path
    preprocessed_model_path: /path/to/folder/
    preprocessed_stations_path: /path/to/folder/
    output_reconstruction_path: /path/to/folder/
    output_tables_path: /path/to/folder/
    output_figure_path: /path/to/folder/
    mask_path: data/masks/canada_mask_rp.nc
    north_mask_path: data/masks/canada_mask_north_rp.nc
    nbcc_loc_path: data/station_inputs/NBCC_2020_new_coords.xlsm

# whether to apply median correction from NBCC 2015
nbcc_median_correction: True
dvs:
    RL50:
        station_dv: "RL50 (kPa)" # Column header for the design value
        station_path: 'data/station_inputs/Interim_snow_rain_load_LR_composite_stations_tbd_v4.csv' 
        input_model_path: 'data/model_inputs/snw_rain_CanRCM4-LE_ens35_1951-2016_max_rl50_load_ensmean.nc'
        medians: 
            value: 0.4
            action: "multiply"
        fill_glaciers: True
```

### Option 1: Interactive (recommended)
[Jupyter Notebooks](https://jupyter.org/) have been paramaterized using [Papermill](https://github.com/nteract/papermill), so in addition to running them in [Jupyter Lab](https://jupyterlab.readthedocs.io/en/stable/getting_started/overview.html), they can be executed from the terminal. For a tutorial on using Jupyter Lab, you can [read their docs](https://jupyterlab.readthedocs.io/en/stable/getting_started/overview.html).

To reconstruct a design value field, users need a CanRCM4 `netCDF` design value field as well as an accompanying station data file in the form of a `.csv`. The user also needs to know the column name of the design value field in the `.csv` file. These will be configured in the configuration yaml.

The processing notebooks can be found in the following directory:
```bash
├── climpyrical
├── notebooks
│   ├── README.ipynb
│   ├── climpyrical_demo.ipynb
...
```

Open `README.ipynb` with Jupyter to view detailed instructions on how to reproduce. The notebooks generate a series of files (including intermediate logs) that are laid out in detail in the aforementioned `README.ipynb` and in `pipeline.ipynb`.

### Option 2: CLI (Command Line Interface) Using Papermill

Since the notebooks are parameterized, they can be run from the command line with Papermill. Papermill produces a log of the notebook once it has been executed. You can select which design values you'd like to run, or which steps you'd like to run from the pipeline in the configuration yaml. [Read more about executing notebooks using papermill here](https://papermill.readthedocs.io/en/latest/usage-execute.html).

You supply the configuration yaml using the `-f` argument.

```bash
$[climpyrical/notebooks/] papermill -f config.yml pipeline.ipynb pipeline_log.ipynb
```

If one wants to run only a segment (or segments) of the pipeline, edit the `config.yml` file to only include the steps of interest.

i.e.

```yaml
# Which notebooks to use in the pipeline
steps: [
"preprocess_model.ipynb", 
"stations.ipynb"
]

# To be placed in climpyrical/
paths:
    output_notebook_path: /intermediate/notebook/logs/path
    preprocessed_model_path: /path/to/folder/
    preprocessed_stations_path: /path/to/folder/
    output_reconstruction_path: /path/to/folder/
    output_tables_path: /path/to/folder/
    output_figure_path: /path/to/folder/
    mask_path: data/masks/canada_mask_rp.nc
    north_mask_path: data/masks/canada_mask_north_rp.nc
    nbcc_loc_path: data/station_inputs/NBCC_2020_new_coords.xlsm

# whether to apply median correction from NBCC 2015
nbcc_median_correction: True
dvs:
    RL50:
        station_dv: "RL50 (kPa)" # Column header for the design value
        station_path: 'data/station_inputs/Interim_snow_rain_load_LR_composite_stations_tbd_v4.csv' 
        input_model_path: 'data/model_inputs/snw_rain_CanRCM4-LE_ens35_1951-2016_max_rl50_load_ensmean.nc'
        medians: 
            value: 0.4
            action: "multiply"
        fill_glaciers: True
```

Then simply run the command as before:

```bash
$[climpyrical/notebooks/] papermill -f config.yml pipeline.ipynb pipeline_log.ipynb
```

### Reading Data --> Put into API documentation
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

# Setting up `climpyrical` for use with rot2reg on Lynx or Leopard
One consistent problem encountered while working on this project, was working to understand the polar stereographic projection that CanRCM4 models are in. This offers several advantages, but can be difficult to understand. A technical overview of this is beyond the scope of this README, however, a guide is included below on how to use `climpyrical` to perform a transformation from polar stereographic to regular `EPSG:4326`/`WGS84` projection.

This guide is tailored to PCIC internal servers `lynx` or `leopard` which has some specific installation features that may or may not be encountered on other machines.

Start in a fresh directory and clone `climpyrical` into it:
```bash
git clone https://github.com/pacificclimate/climpyrical.git 
```

Create a `python3` virtual environment and activate it:
```bash
python3 -m venv myvenv
source myvenv/bin/activate
```

Load the `R` module into the environment. This step and the next `PYTHONPATH` steps may be unnecessary for your own machine. 
```bash
module load R
module load proj/7.2.0
```

Loading the `R` module also loads gdal/2.2.3 which sets the PYTHONPATH env variable interfering with our virtual python environment. So it's important to unset `PYTHONPATH` before proceeding:
```bash
unset PYTHONPATH
```

Verify you are still in your virtual environment:
```bash
which python
> ...myvenv/bin/python
```

Next install `climpyrical`'s requirements with:
```bash
pip install cython
pip install -r climpyrical/requirements.txt
```

Then finally install `climpyrical` with:
```bash
pip install -e climpyrical/
```

Now you can use `climpyrical`. To unrotate a CanRCM4 file and write it to a new `netCDF4` file, simply:
```bash
python climpyrical/cmd/rot2reg.py "path/to/input_CanRCM4.nc" "path/to/output_CanRCM4.nc"
```

## Authors
* **Nic Annau** - [Pacific Climate Impacts Consortium](https://www.pacificclimate.org/)
