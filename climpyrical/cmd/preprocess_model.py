"""
quick usage of climpyrical preprocess_mode.py
usage:
python preprocess_model.py -i input.nc -o output.nc -m True

This script takes a CanRCM4 model at the native resolution and
downscales from 50 km to  5 km and fills in missing land values
using external masks.
"""

from climpyrical.data import read_data, gen_dataset, interpolate_dataset
from climpyrical.gridding import regrid_ensemble, extend_north

import click
from pkg_resources import resource_filename
import logging

import warnings

import numpy as np

warnings.filterwarnings("ignore")


@click.command()
@click.option("-i", "--in-path", help="Input CanRCM4 file", required=True)
@click.option("-o", "--out-path", help="Output file", required=True)
@click.option(
    "-m", "--fill-glaciers", help="Refill glacier points", default=True
)
@click.option(
    "-l",
    "--log-level",
    help="Logging level",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]),
    default="INFO",
)
def run_processing(in_path, out_path, fill_glaciers, log_level):
    """Completes the preprocessing of the model required
    for the NRC project.
    Args:
        in_path, out_path (strings): directories of NetCDF4 file
            input and output. Must give filename, too with extension
            .nc. Overwites files with same name in same directory.
        fill_glaciers (bool): whether to fill spurious glacier
            points with preprocessed mask. Default is True.
        quiet (bool): whether to log quietly or not
    Returns:
        Creates a NetCDF4 file at out_path at target resolution
    """
    logging.basicConfig(level=log_level)

    ds = read_data(in_path)
    dv = list(ds.data_vars)[0]

    rlon, rlat = np.meshgrid(ds.rlon, ds.rlat)
    mean = ds[dv].values
    kelvin = 273.15  # K

    logging.info("Detect unit conversions")
    if dv in ["twb", "tas"]:
        print("Temperature field detected. Converting to Kelvin.")
        mean += kelvin

    path_mask = resource_filename(
        "climpyrical", "nrc_data/land_mask_CanRCM4_sftlf.nc"
    )

    path_glacier_mask = resource_filename(
        "climpyrical", "nrc_data/glacier_mask.nc"
    )

    logging.info("Load and regrid file to target resolution")
    mask = read_data(path_mask)
    mask = regrid_ensemble(mask, "sftlf", 10, copy=True)
    mask = mask["sftlf"] >= 1.0

    logging.info("Load original reoslution mask for reference")
    mask_og = read_data(path_mask)["sftlf"].values != 0.0

    glaciermask = read_data(path_glacier_mask)["mask"].values != 0.0


    logging.info(
        "Insert NaN values into glacier points to fill"
        "and interpolate if fill_galciers is set"
    )
    if fill_glaciers:
        logging.info("Filling spurious glacier points.")
        mean[glaciermask] = np.nan

    nanmask = ~np.isnan(mean)
    points = np.stack([rlon[nanmask], rlat[nanmask]]).T
    target_values = mean[nanmask]
    target_points = np.stack([rlon[glaciermask], rlat[glaciermask]]).T

    mean[glaciermask] = interpolate_dataset(
        points, target_values, target_points, "linear"
    )

    ds = gen_dataset(dv, mean, ds.rlat, ds.rlon, ds.lat, ds.lon)
    ds = ds.assign({dv: (["rlat", "rlon"], mean)})

    logging.info("Remove water cells at original resolution")
    ds[dv].values[~mask_og] = np.nan
    nanmask = ~np.isnan(ds[dv].values)

    ds10 = regrid_ensemble(ds, dv, 10, copy=True)
    ds10[dv].values[~mask] = np.nan
    nrlon, nrlat = np.meshgrid(ds10.rlon, ds10.rlat)
    nanmask10 = ~np.isnan(ds10[dv].values)

    logging.info("Interpolating full remaining grid")
    points = np.stack([rlon[nanmask], rlat[nanmask]]).T
    target_points = np.stack([nrlon[nanmask10], nrlat[nanmask10]]).T
    values = ds[dv].values[nanmask]
    ds10[dv].values[nanmask10] = interpolate_dataset(
        points, values, target_points, "linear"
    )

    logging.info("Add northern domain to model")
    ds10 = extend_north(ds10, dv, 210, fill_val=np.nan)

    nanmask10 = ~np.isnan(ds10[dv].values)

    canada_mask_path = resource_filename(
        "climpyrical", "/tests/data/canada_mask_rp.nc"
    )

    with read_data(canada_mask_path) as ds_canada:
        ca_mask = extend_north(ds_canada, "mask", 210, fill_val=np.nan)
        ca_mask = ds_canada["mask"].values

    # select NaN values within new mask
    ca_mask_or = ~np.logical_or(~ca_mask, nanmask10)

    logging.info(
        "Fill remaining missing points using closest neighbour."
    )
    nrlon, nrlat = np.meshgrid(ds10.rlon.values, ds10.rlat.values)

    temp_field = ds10[dv].values

    points = np.stack([nrlon[nanmask10], nrlat[nanmask10]]).T
    target_points = np.stack([nrlon[ca_mask_or], nrlat[ca_mask_or]]).T
    target_values = ds10[dv].values[nanmask10]
    temp_field[~ca_mask] = np.nan

    temp_field[ca_mask_or] = interpolate_dataset(
        points, target_values, target_points, "nearest"
    )

    logging.info("Remove the processed northern region.")
    uaa_mask_path = resource_filename(
        "climpyrical", "tests/data/canada_mask_north_rp.nc"
    )
    uaa_mask = read_data(uaa_mask_path)["mask"]
    temp_field[uaa_mask] = np.nan

    ds_processed = gen_dataset(
        dv, temp_field, ds10.rlat, ds10.rlon, ds10.lat, ds10.lon
    )

    logging.info("Dataset generated and writing to file.")

    ds_processed.to_netcdf(out_path, "w")

    logging.info("Completed!")


if __name__ == "__main__":
    run_processing()
