import numpy as np
import numpy.ma as ma
import xarray as xr

def load_land_mask(data_path):
    """Loads land fraction file from
    http://climate-modelling.canada.ca/climatemodeldata/canrcm/CanRCM4/NAM-44_ECMWF-ERAINT_evaluation/fx/atmos/sftlf/index.shtml
    and creates a mask for cells that are 0% land.
    ----------------------------------------------
    Args:
        data_path (str): path to the land fraction mask file
    Returns:
        mask (numpy.masked_array): masked array based on file
    """
    ds_mask = xr.open_dataset(data_path)
    mask = ma.masked_greater(ds_mask['sftlf'].values, 0.0).mask
    return mask


def mask_land_and_nan(dv_field, mask_land):
    """Creates a master mask by masking any cell in the ensemble
    that has a NaN value. If a NaN value is found in the ensemble,
    that grid cell is masked for every ensemble member, thus
    disqualifying that grid cell from further analysis. It is then
    combined with the land mask, to create a master mask of cells
    to use in the analysis.
    ------------------------------------------------------------
    Args:
        dv_field (numpy.ndarray): datacube containing the ensemble members
        mask_land (numpy.masked_array): land mask created by load_land_mask()

    Returns:
        mask_master (numpy.ndarray): a CanRCM4 field shaped array containing
            a boolean master mask
    """
    nan_mask = np.apply_over_axes(
                            np.logical_or.reduce,
                            np.isnan(dv_field),
                            (0)
    )
    mask_master = ~np.logical_or(~mask_land, nan_mask)

    return mask_master


def mask_land_and_nan_ens_index(mask_master):
    """Reshapes the master mask created in mask_master into
    the ensemble shape, that is, (number of grid cells) x (number of
    ensemble members). Each row in this new shape represents a grid cell
    from the ensemble members.

    Args:
        mask_master (numpy.ndarray): boolean mask containing qualified grid cells
    Returns:
        idx (numpy.ndarray): boolean mask containing qualified grid cells
            in the ensemble shape (number of grid cells) x (number of
            ensemble members)
    """
    mask_ens_master = mask_master.reshape(
                        mask_master.shape[0],
                        mask_master.shape[1]*mask_master.shape[2]
    )

    idx = np.where(mask_ens_master[0]==True)[0]

    return idx


def ens_flat(dv_field):
    """Flattens data cube into ensemble
    size x number of cells.
    --------------------------------
    Args:
        dv_field (xarray.DataArray): datacube
            containing design values of from
            an ensemble of CanRCM4 models
    Returns:
        ens_dv_field (numpy.ndarray): reshaped datacube into
            (number of grid cells) x (number of ensemble members)
    """

    ens_sz = dv_field.shape[0]
    n_grid_cells = dv_field.shape[1]*dv_field.shape[2]

    ens_dv_field = dv_field.reshape((ens_sz, n_grid_cells))

    return ens_dv_field

def generate_pseudo_obs(ens_arr, frac):
    """Randomly sample a range of integers.
    --------------------------------
    Args:
        y_obs (numpy.ndarray): array to be sampled
        frac (float): fractional size of y_obs
            to sample
    Returns:
        (numpy.ndarray): array containing indices
            (in the ensemble shape) of randomly
            sampled values
    """

    ens_sz = ens_arr.shape[0]

    # randomly select an ensemble member
    # to sample
    i = np.random.randint(0, ens_sz-1)
    y_obs = ens_arr[i, :]

    n_grid_cells = y_obs.shape[0]
    index = np.random.choice(np.arange(n_grid_cells),
                             int(frac*n_grid_cells))

    return index
