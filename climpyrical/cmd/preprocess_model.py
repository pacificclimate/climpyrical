from climpyrical.data import read_data, gen_dataset, interpolate_dataset
from climpyrical.gridding import regrid_ensemble, extend_north

import os
from pkg_resources import resource_filename
from argparse import ArgumentParser

import numpy as np
import warnings

warnings.filterwarnings("ignore")


"""
quick usage of climpyrical.rot2reg
usage:
python rot2reg input.nc output.nc
"""


def main():
    desc = globals()["__doc__"]
    parser = ArgumentParser(description=desc)
    parser.add_argument(
        "-i",
        "--input_file",
        help=(
            "Path to intput CanRCM4 file "
            "Note: must have lat, lon"
            "rlat, rlon and a data variable."
        ),
    )
    parser.add_argument(
        "-o",
        "--output_file",
        help=(
            "Path to store results."
            "Caution! Overwrites existing"
            "file in the directory."
        ),
    )
    parser.add_argument(
        "-m",
        "--fill_glaciers",
        default=True,
        help=("Boolean, removes spurrious glacier" "points within model"),
    )

    args = parser.parse_args()

    if args.fill_glaciers is not None:
        if args.fill_glaciers:
            fill_glaciers = True
        else:
            fill_glaciers = False

    if not args.input_file.endswith(".nc") or not args.output_file.endswith(
        ".nc"
    ):
        raise IOError("Please provide a .nc file.")

    run_processing(args.input_file, args.output_file, fill_glaciers)


def run_processing(IN_PATH, OUT_PATH, fill_glacieres):
    """Completes the preprocessing of the model required
    for the NRC project.
    Args:
        IN_PATH, OUT_PATH (strings): directories of NetCDF4 file
            input and output. Must give filename, too with extension
            .nc. Overwites files with same name in same directory.
        fill_glaciers (bool): whether to fill spurious glacier
            points with preprocessed mask. Default is True.
    Returns:
        Creates a NetCDF4 file at OUT_PATH at target resolution
            for
    """

    ds = read_data(IN_PATH)
    dv = list(ds.data_vars)[0]

    # create handy 2D grids
    rlon, rlat = np.meshgrid(ds.rlon, ds.rlat)
    # extract data field
    mean = ds[dv].values

    # Detect unit conversions
    temp_dv_names = ["twb", "tas", "heating_degree_days_per_time_period"]
    if dv in temp_dv_names:
        print("Temperature field detected. Converting to Kelvin.")
        mean = ds[dv].values + 273.15
    else:
        mean = ds[dv].values

    # import path masks
    # surface to land mask
    PATH_MASK = resource_filename(
        "climpyrical", "nrc_data/land_mask_CanRCM4_sftlf.nc"
    )

    # glacier point mask developed from SL50
    PATH_GLACIER_MASK = resource_filename(
        "climpyrical", "nrc_data/glacier_mask.nc"
    )

    # load mask data
    mask = read_data(PATH_MASK)

    # regrid dataset to target resolution
    mask = regrid_ensemble(mask, "sftlf", 10, copy=True)

    # convert to boolean mask
    mask = mask["sftlf"] >= 1.0

    # keep original mask, convert to boolean
    mask_og = read_data(PATH_MASK)["sftlf"].values != 0.0
    # load and convert glacier mask to boolean
    glaciermask = read_data(PATH_GLACIER_MASK)["mask"].values != 0.0

    # insert NaN values into glacier points to fill
    fill_glaciers = True
    if fill_glaciers:
        print("Filling spurious glacier points.")
        mean[glaciermask] = np.nan

    nanmask = ~np.isnan(mean)
    points = np.stack([rlon[nanmask], rlat[nanmask]]).T
    target_values = mean[nanmask]
    target_points = np.stack([rlon[glaciermask], rlat[glaciermask]]).T

    # interpolate NaN values using bilinear interpolation
    mean[glaciermask] = interpolate_dataset(
        points, target_values, target_points, "linear"
    )
    ds = gen_dataset(dv, mean, ds.rlat, ds.rlon, ds.lat, ds.lon)

    # nanmask = ~np.isnan(mean)
    ds = ds.assign({dv: (["rlat", "rlon"], mean)})

    # make all water values NaN at original resolution
    ds[dv].values[~mask_og] = np.nan
    nanmask = ~np.isnan(ds[dv].values)

    # copy newly masked dv field to target resolution
    ds10 = regrid_ensemble(ds, dv, 10, copy=True)

    # Mask out ocean values
    ds10[dv].values[~mask] = np.nan
    nrlon, nrlat = np.meshgrid(ds10.rlon, ds10.rlat)
    nanmask10 = ~np.isnan(ds10[dv].values)

    print("Interpolating full grid.")

    # bilinearly interpolate over non NaN grids
    points = np.stack([rlon[nanmask], rlat[nanmask]]).T
    target_points = np.stack([nrlon[nanmask10], nrlat[nanmask10]]).T
    values = ds[dv].values[nanmask]
    ds10[dv].values[nanmask10] = interpolate_dataset(
        points, values, target_points, "linear"
    )

    # extend the northern region
    ds10 = extend_north(ds10, dv, 210, fill_val=np.nan)

    # create nanmask for interpolation
    nanmask10 = ~np.isnan(ds10[dv].values)

    # load processed canada-only land mask
    canada_mask_path = resource_filename(
        "climpyrical", "/tests/data/canada_mask_rp.nc"
    )

    with read_data(canada_mask_path) as ds_canada:
        ca_mask = extend_north(ds_canada, "mask", 210, fill_val=np.nan)
        ca_mask = ds_canada["mask"].values

    # select NaN values within new mask
    ca_mask_or = ~np.logical_or(~ca_mask, nanmask10)

    # Fill inconsistent points using closest neighbour.
    # Not that the northern section will be filled later on
    nrlon, nrlat = np.meshgrid(ds10.rlon.values, ds10.rlat.values)

    temp_field = ds10[dv].values

    points = np.stack([nrlon[nanmask10], nrlat[nanmask10]]).T
    target_points = np.stack([nrlon[ca_mask_or], nrlat[ca_mask_or]]).T
    target_values = ds10[dv].values[nanmask10]
    temp_field[~ca_mask] = np.nan

    print("Detecting missing points from canada mask and filling.")

    temp_field[ca_mask_or] = interpolate_dataset(
        points, target_values, target_points, "nearest"
    )

    # UAA "unfill"
    uaa_mask_path = resource_filename(
        "climpyrical",
        "tests/data/canada_mask_north_rp.nc"
    )

    uaa_mask = read_data(uaa_mask_path)['mask']
    temp_field[uaa_mask] = np.nan

    # create final dataset
    ds_processed = gen_dataset(
        dv, temp_field, ds10.rlat, ds10.rlon, ds10.lat, ds10.lon
    )

    print("Dataset generated and writing to file.")

    if os.path.exists(OUT_PATH):
        os.remove(OUT_PATH)

    ds_processed.to_netcdf(OUT_PATH, "w")

    print("Completed!")


if __name__ == "__main__":
    main()
