"""
quick usage of climpyrical find_matched_model_vals.py
usage:
python find_matched_model_vals.py -m model.nc -s stations.csv -o output.csv
This script takes a model with rlon and rlat, as well as station locations
and finds their closest model counterpart in the model grid. It
generates a .csv file identical to the input csv, but with the added
model_values column
"""

from climpyrical.data import read_data
from climpyrical.gridding import (
    find_element_wise_nearest_pos,
    find_nearest_index_value,
    transform_coords,
)

import click
import logging

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


def add_model_values(model_path, stations_path=None, df=None, model_dv="model_values", log_level="INFO"):
    """Locates the model value that's spatially closest to a station
    Args:
        model_path, stations_path (strings): directories of NetCDF4 file
            input and station file. Must give filename with extension
            .nc and .csv respectively
        out_path (str): directory of output csv file name. must include
            extension
        log_level (str): Default INFO
    Returns:
        Creates a .csv file with corresponding model values.
    """
    logging.basicConfig(level=log_level)

    ds = read_data(model_path)
    (dv,) = ds.data_vars
    unit = ds[dv].attrs["units"]

    rlon, rlat = np.meshgrid(ds.rlon, ds.rlat)
    mean = ds[dv].values

    accepted_units = ["kPa", "Pa", "degC", "mm", "unitless", "%"]

    logging.info(f"Detect units: {unit}")
    if unit not in accepted_units:
        warnings.warn(
            f"{unit} not recognized from list of accepted units: {accepted_units}"
        )

    if unit == "degC":
        kelvin = 273.15  # K
        logging.info("Temperature field detected. Converting to Kelvin.")
        mean += kelvin
        ds[dv].attrs["units"] = "K"

    if stations_path is not None:
        if stations_path.endswith(".csv"):
            df = pd.read_csv(stations_path)
        else:
            df = pd.read_excel(stations_path)

    if stations_path is None and df is None:
        raise ValueError("Must provide either stations_path or pandas.Dataframe")

    keys = ["lat", "lon"]
    contains_keys = [key not in df.columns for key in keys]
    if np.any(contains_keys):
        raise KeyError(f"Dataframe must contain {keys}")

    rkeys = ["rlat", "rlon"]
    contains_rkeys = [key not in df.columns for key in rkeys]
    if np.any(contains_rkeys):
        logging.info(
            "rlat or rlon not detected in input file."
            "converting assumes WGS84 coords to rotated pole"
        )
        nx, ny = transform_coords(df.lon.values, df.lat.values)
        df = df.assign(rlat=ny, rlon=nx)

    logging.info("Matching coordinates now")
    ix, iy = find_element_wise_nearest_pos(
        ds.rlon.values, ds.rlat.values, df.rlon.values, df.rlat.values
    )

    logging.info(
        "Locating corresponding model values"
        "Interpolating to nearest if matched model value is NaN"
    )
    model_vals = find_nearest_index_value(
        ds.rlon.values, ds.rlat.values, ix, iy, ds[dv].values
    )

    if np.any(np.isnan(model_vals)):
        raise ValueError("NaN detected as matching output. Critical error.")

    df_new = df.assign(irlat=iy, irlon=ix)
    df_new[model_dv] = model_vals

    return df_new


@click.command()
@click.option("-m", "--model-path", help="Input CanRCM4 file", required=True)
@click.option("-s", "--stations-path", help="Input csv file to match", required=True)
@click.option(
    "-o", "--out-path", help="Output csv file with matched vals", required=True
)
@click.option(
    "-dv", "--station-dv", help="Output csv file with matched vals", required=True
)
@click.option(
    "-l",
    "--log-level",
    help="Logging level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]),
    default="INFO",
)
def write_to_file(
    model_path,
    stations_path,
    out_path,
    model_dv,
    log_level):
    df = add_model_values(model_path, stations_path, model_dv, log_level)
    df.to_csv(out_path)


if __name__ == "__main__":
    write_to_file()