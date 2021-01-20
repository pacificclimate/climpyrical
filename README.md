# climpyrical
A Python tool for spatially downscaling and correcting CanRCM4 derived design value fields using meteorological station observations and CanRCM4 models.

# Build status
![Python CI](https://github.com/pacificclimate/climpyrical/workflows/Python%20CI/badge.svg)
![Documentation Check](https://github.com/pacificclimate/climpyrical/workflows/Documentation%20Check/badge.svg)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# View Notebooks
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/pacificclimate/climpyrical/HEAD?filepath=notebooks%2Fdemo%2F)
[Logs of Recent Runs](https://nbviewer.jupyter.org/github/pacificclimate/climpyrical_dv_log/tree/main/)

# Setup
The following instructions are general for unix-type systems with Python installed.

Start by cloning this repository to your local machine. 

```bash
git clone https://github.com/pacificclimate/climpyrical/
```

## Python
Climpyrical is designed for Python 3.

## Dependencies
For Python version < 3.8, [install proj v >= 7.2.0](https://proj.org/install.html). It is safe to proceed to the next steps without installing proj if using Python 3.8. Geopandas for Python < 3.8 does not require proj v 7.2.0.

If you intend on using the Moving Window Ordinary ratio Kriging scripts, R must be installed as well as specific R packages. This repo contains a custom requirements script that should handle this automatically.

```bash
apt install r-base 
Rscript install_pkgs.R r_requirements.txt
```
## Virtual Environment
Please use a [virtual environment](https://docs.python.org/3/tutorial/venv.html) to handle python packages and your climpyrical project. A quick start on this is to use
```bash
python3 -m venv path/to/myvenv
```

and activate with
```bash
source path/to/myvenv/bin/activate
```

Don't forget to reactivate this environment with the above command between sessions.

## Requirements
To install all of the dependencies used by climpyrical, install from requirements file found in `requirements.txt`

via 

```bash
pip install -r climpyrical/requirements.txt
```
To install climpyrical, run
```bash
pip install -e climpyrical/
```

Jupyter is included in the requirements. If running in jupyter mode, to properly render progress bars, install the nbwidget extension
```bash
jupyter nbextension enable --py widgetsnbextension
jupyter labextension install @jupyter-widgets/jupyterlab-manager
```
And restart the jupyter server.

# Input Data
Climpyrical assumes input models and stations. The following tables summarize the requirements. The model data must be in [NetCDF4](https://www.unidata.ucar.edu/software/netcdf/docs/index.html) format, and the station files must be in `.csv` format.

| NetCDF4 (.nc)   | 2D Data Field Variable: e.g. "Rain-RL50"                     | Coordinates: lat(rlon, rlat), lon(rlon, rlat), rlon, rlat | Put in: climpyrical/data/model_inputs/   |
|-----------------|--------------------------------------------------------------|-----------------------------------------------------------|------------------------------------------|
| Stations (.csv) | List of station values with column header: e.g. "RL50 (kPa)" | Coordinates: lat, lon (rlon, rlat optional)               | Put in: climpyrical/data/station_inputs/ |

## Output data tree
```bash
climpyrical/data/results
├── netcdf
│   └── 
├── figures
│   ├── 
├── intermediate
│   ├── notebooks (only if running with papermill)
│   │   ├── model_log_{design value}.ipynb
│   │   ├── plotting_log_{design value}.ipynb
│   │   ├── MWOrK_log_{design value}.ipynb
│   │   ├── station_log_{design value}.ipynb
│   ├── preprocessed_netcdf
│   │   ├── {design value}.nc
│   └── preprocessed_stations
│       └── {design value}.csv
└── TableC2
     └── {design_vale}_TableC2.csv
```

# Getting started

## The pipeline
The pipeline runs in order and downstream notebooks require outputs from upstream notebooks. That is, don't remove steps from the configuration that are upstream of the step you want to run without previously ensuring its outputs are placed in the proper directories.

1.) Preprocess the model. This downscales by bilinear interpolation, and masks to a finer grid scale of the Canadian coastline. It also fills in glacier points supplied in a default mask.
2.) Preprocess the stations. This converts coordinates and matches and aggregates stations that fall into a single grid cell, as well as locating their indices in the climate model.
3.) Moving Window Ordinary ratio Kriging uses the results of the previous steps to generate a reconstructed NetCDF4 file
4.) Generates useful figures for each design value provided
5.) Finds reconstructed values at NBCC locations nd generates a csv
6.) Combines all of the NBCC table into a final version

The first step to running the pipeline is configuring it. Various configurations also need to be added to a configuration yaml file. These contain design value specific information, such as paths to input station and model files, plotting parameters, and output filenames. See `/climpyrical/notebooks/interactive/config_example.yml` as a full configuration for running the software on all of the design values. 

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
    output_notebook_path: /data/results/intermediate/notebooks/ # logs of notebooks
    preprocessed_model_path: /data/results/intermediate/preprocessed_netcdf/ # preprocessed model netcdf files (not reconstructed!)
    preprocessed_stations_path: /data/results/intermediate/preprocessed_stations/ # preprocessed station output files
    output_reconstruction_path: /data/results/netcdf/ # reconstruction outputs
    output_tables_path: /data/results/TableC2/ # table c2 csv file outputs
    output_figure_path: /data/results/figures/ # all figures
    mask_path: data/masks/canada_mask_rp.nc # rotated pole canada only mask
    north_mask_path: data/masks/canada_mask_north_rp.nc # rotated pole upper arctic archepelago mask
    nbcc_loc_path: data/station_inputs/NBCC_2020_new_coords.xlsm # NBCC locations

# whether to apply median correction from NBCC 2015
nbcc_median_correction: True

# put all design values configured here
dvs:
    # name of the design value you'd like to use (no spaces)
    RL50:
        # name of design value column header in station csv
        station_dv: "RL50 (kPa)" # Column header for the design value
        # path to station data
        station_path: 'data/station_inputs/Interim_snow_rain_load_LR_composite_stations_tbd_v4.csv' 
        # where to find this particular model
        input_model_path: 'data/model_inputs/snw_rain_CanRCM4-LE_ens35_1951-2016_max_rl50_load_ensmean.nc'
        # to be used only if nbcc_median_correction is True. See MWOrK.ipynb for specific use
        medians: 
            value: 0.4
            action: "multiply"
        # whether to interpolate over glaciers in glacier path
        fill_glaciers: True
```

## Running the pipeline
Once the pipeline is configured, there are two options for running the pipeline (a third option with a simple python interface is currently under development). 

## Option 1: Interactively with Jupyter (recommended)
For a tutorial on using Jupyter Lab, you can [read their docs](https://jupyterlab.readthedocs.io/en/stable/getting_started/overview.html).

Start `jupyter` within the `/notebooks` directory

```bash
jupyter lab
```

Open `README.ipynb` with Jupyter to view detailed instructions on how to reproduce. The notebooks generate a series of files (including intermediate logs) that are put in detail in `README.ipynb` and in `pipeline.ipynb`.

To run the pipeline, open `pipeline.ipynb` and execute the cells. You should see progress bars update which step the pipeline is currently running. 

### Option 2: CLI (Command Line Interface) Using Papermill

Since the notebooks are parameterized, they can be run from the command line with Papermill. Papermill produces a log of the notebook once it has been executed. [Read more about executing notebooks using papermill here](https://papermill.readthedocs.io/en/latest/usage-execute.html).

Within the `notebooks/` directory, run

```bash
papermill -p config_yml "path/to/config.yml" pipeline.ipynb pipeline_log.ipynb
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
    ...
```

Then simply run the command as before:

```bash
$[climpyrical/notebooks/] papermill -p config_yml "path/to/config.yml" pipeline.ipynb pipeline_log.ipynb
```

# Setting up `climpyrical` for use on PCIC compute nodes

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

Refer to above directions for further instructions.

## Authors
* **Nic Annau** - [Pacific Climate Impacts Consortium](https://www.pacificclimate.org/)

Please fork and open a pull request to contribute.
