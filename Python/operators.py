import warnings

import numpy as np
import numpy.ma as ma

def ens_mean(ens_array):
    """Centers each run in data_cube
    by subtracting ensemble mean at
    a given grid cell.
    --------------------------------
    Args:
        dv_field (xarray.DataArray): datacube
            containing design values of from
            an ensemble of CanRCM4 models
    Returns:
        data_cube (xarray Dataset): mean centered
            datacube
    """

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean = ens_array.mean(axis=1)

    return mean

def frac_grid_area(dv_field, R=6371.0):
    """Calculates the fractional area of
    each CanRCM4 cell.
    --------------------------------
    Args:
        dv_field (xarray.DataArray): datacube
            containing design values of from
            an ensemble of CanRCM4 models
        R (float): radius of Earth in km
    Returns:
        A (numpy.ndarray): 2-dimensional array
            containing areas of each grid cell
    """
    rlat_o = dv_field['rlat'].values
    rlon_o = dv_field['rlon'].values

    # approximate grid cell size
    gridsz_rlat = rlat_o[-1]+np.diff(rlat_o).mean()
    gridsz_rlon = rlon_o[-1]+np.diff(rlon_o).mean()

    # add an extra cell with approximated size
    rlat_o = np.append(rlat_o, gridsz_rlat)
    rlon_o = np.append(rlon_o, gridsz_rlon)

    # differentiate
    rlat_diff = np.diff(np.sin(np.deg2rad(rlat_o)))
    rlon_diff = np.diff(rlon_o)

    # reshape to calculate all cells
    rlat = np.repeat(rlat_diff, rlon_diff.shape[0])
    rlon = np.tile(rlon_diff, rlat_diff.shape[0])

    # convert to radians and apply sin to lat
    lat_sz = np.abs((rlat))
    lon_sz = np.abs(np.deg2rad(rlon))

    # perform area calculation
    area = (np.pi/180)*R**2*lat_sz*lon_sz

    total_area = np.trapz(area)
    # reshape into original grid
    area = np.reshape(area, (rlat_diff.shape[0], rlon_diff.shape[0]))

    return area/total_area

def mask_land_and_nan(dv_field, mask_land):
    mask = np.apply_over_axes(np.logical_or.reduce, mask_land, (0))
    mask = np.broadcast_to(mask, dv_field.shape)
    masked_arr = ma.MaskedArray(dv_field, mask)

    ens = masked_arr.reshape((masked_arr.shape[0], masked_arr.shape[1]*masked_arr.shape[2])).T
    ens_land_masked = ens.data[~ens.mask[:,0], :]

    indx = np.arange(0, ens.shape[0])
    indx = indx[~ens.mask[:,0]]

    nan_mask = np.apply_over_axes(
                            np.logical_or.reduce,
                            np.isnan(ens_land_masked),
                            (1)
                        )
    nan_mask = np.broadcast_to(nan_mask, ens_land_masked.shape)

    ens_masked = ens_land_masked[~nan_mask[:, 0], :]

    new_indx = indx[~nan_mask[:, 0]]

    mask_dict = {
            'ens': ens_masked,
            'idx': new_indx,
    }

    return mask_dict
