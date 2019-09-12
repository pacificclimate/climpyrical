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
    mask = ma.masked_greater(ds_mask['sftlf'].values, 10.).mask
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

def flatten_mask(mask_master):
    """Reshapes the master mask created in mask_master into
    the flattened ensemble shape, that is, (number of grid cells) x (number of
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

    idx = np.where(mask_ens_master==True)

    return idx[1]

def interpolate_land_mask(dv_field, mask, factor=10):
    """Interpolates the final mask into a interpolated final mask.

    Args:
        dv_field (numpy.ndarray): design value field in cubic shape
        mask (numpy.ndarray): boolean mask containing qualified grid cells
        factor (int): factor to increase spatial resolution
    Returns:
        mask_master (numpy.ndarray): interpolated boolean mask containing
            qualified grid cells in the cubic shape.
    """
    mask_master = mask_land_and_nan(dv_field, mask)
    mask_master = np.repeat(
                        np.repeat(mask_master, factor, axis=1),
                        factor
                    )

    return mask_master

def flatten_interpolated_land_mask(mask_master, coordict):
    """Interpolates the final mask into a interpolated final mask.

    Args:
        mask_master (numpy.ndarray): interpolated boolean mask containing
            qualified grid cells in the cubic shape
        coordict (numpy.ndarray): coordinates of interpolated lat and lon in
            the ensemble shape
    Returns
        imask (numpy.ndarray): boolean array containing indices of qualified
            grid cells in the flattened ensemble shape
    """
    imask = np.expand_dims(
                        np.reshape(
                                mask_master,
                                (coordict['irlat'].shape[0],
                                 coordict['irlon'].shape[0])
                        ),
                        axis=0
            )

    return imask

def mask(mask_path, dv_field):
    """Loads mask information into a dictionary for processing
    Args:
        mask_path (str): path to mask file
        dv_field (numpy.ndarray): design value field in cubic shape
    Returns:
        mask_dict (dict): containes the land mask, master mask, and indices
            to be masked in the flattened ensemble shape.
    """
    mask_land = load_land_mask(mask_path)
    mask_master = mask_land_and_nan(dv_field, mask_land)
    idx = flatten_mask(mask_master)

    mask_dict = {
        'land': mask_land,
        'master': mask_master,
        'index': idx
    }

    return mask_dict

def interpolated_mask(mask_path, dv_field, coordict, factor=10):
    """Creates the interpolated mask from start to finish
    Args:
        mask_path (str): path to mask file
        dv_field (numpy.ndarray): design value field in cubic shape
        coordict (dict): coordinates of interpolated and original data
        factor (int): factor to increase spatial resolution
    Returns:
        mask_dict (dict): containes the interpolated land mask, master mask, and indices
            to be masked in the flattened ensemble shape.
    """
    mask_land = load_land_mask(mask_path)
    imask_master = interpolate_land_mask(dv_field, mask_land, factor)
    imask = flatten_interpolated_land_mask(imask_master, coordict)
    iidx = flatten_mask(imask)

    mask_dict = {
        'land': mask_land,
        'master': imask_master,
        'master_interp': imask,
        'index': iidx
    }

    return mask_dict