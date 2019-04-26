import numpy as np
import numpy.ma as ma
import pandas as pd
from sklearn.decomposition import pca
from sklearn import linear_model
from sklearn.preprocessing import Imputer
import warnings
from operators import ens_mean, frac_grid_area

def mask_flat(mask):
    """Flattens mask into ensemble shape
    for compatibility.
    --------------------------------
    Args:
        dv_field (xarray.DataArray): datacube
            containing design values of from
            an ensemble of CanRCM4 models
        mask (numpy.ma): masked array for run in dv_field
    Returns:
        mask (nump.ma): masked array in ensemble shape,
        ensemble_size x number of grid cells
    """
    n_grid_cells = mask.shape[1]*mask.shape[2]
    mask = np.reshape(mask, n_grid_cells)

    return mask

def mask_land(ens, mask):
    """Gets only land values from flattened ensemble
    of CanRCM4 models. Interpolates remaining
    invalid NA values if necessary. Returns masked
    flattened ensemble.
    --------------------------------
    Args:
        ens (numpy.array): flattened array
            containing design values of from
            an ensemble of CanRCM4 models
        mask (numpy.ma): flattened ocean mask array
            for CanRCM4 grids
    Returns:
        masked (nump.ma): masked array in ensemble shape,
        ensemble_size x number of grid cells
    """
    # apply mask along grid dimension
    masked_flat_ens = ens[:, mask].T
    return masked_flat_ens

def mask_invalid(ens):
    print(ens.shape)
    imputer = Imputer(axis=1)
    ens = imputer.fit_transform(ens)
    # check if there are any non-finite, or nan values in ensemble
    #if np.any(np.isnan(ens)) or np.all(np.isfinite(ens)) is False:
    #    # convert to dataframe to find NaN rows and interpolate over them
    #    ens = pd.DataFrame(ens)
    #    diff = ens.shape[0] - ens.dropna().shape[0]
    #    ens = ens.fillna(ens.mean(axis=1)).values

        # warn user that NaN land values were found and will be interpolated
    #    warnings.warn("Removed {} land grid cells containing NaN values found in ensemble".format(diff))

    return ens

def ens_flat(dv_field):
    """Flattens data cube into ensemble
    size x number of cells.
    --------------------------------
    Args:
        dv_field (xarray.DataArray): datacube
            containing design values of from
            an ensemble of CanRCM4 models
    Returns:
        dv_field (numpy.ndarray): reshaped 2-d array
            into number of cells x ensemble size
    """

    ens_sz = dv_field.shape[0]
    n_grid_cells = dv_field.shape[1]*dv_field.shape[2]

    dv_field = dv_field.values.reshape((ens_sz, n_grid_cells))

    return dv_field


def rand_sample_index(y_obs, frac):
    """Randomly sample a range of integers.
    --------------------------------
    Args:
        y_obs (numpy.ndarray): array to be sampled
        frac (float): fractional size of y_obs
            to sample
    Returns:
        (numpy.ndarray): array containing indices of
            sampled values
    """
    n_grid_cells = y_obs.shape[0]
    index = np.random.choice(np.arange(n_grid_cells),
                             int(frac*n_grid_cells))
    return index


def get_obs(ens_arr):
    """Randomly sample a data cube to generate
    pseudo observations from ensemble.
    --------------------------------
    Args:
        dv_field (xarray Dataset): datacube
            containing an ensemble of
            CanRCM4 models
    Returns:
        y_sample (numpy.ndarray): fraction
            of random spatially sampled design
            value grid cells
        index (numpy.ndarray): index locations
            of y_sample from original ensemble size
    """
    ens_sz = ens_arr.shape[0]

    # randomly select an ensemble member
    # to sample
    i = np.random.randint(0, ens_sz-1)
    y_obs = ens_arr[i, :]

    return y_obs


def ens_to_eof(ens_arr, explained_variance=0.95):
    """Perform EOF/PCA dimensionality reduction
    on ensemble array
    --------------------------------
    Args:
        ens_arr (numpy.ndarray): reshaped 2-d array
            into number of cells x ensemble size
    Returns:
        eofs (numpy.ndarray): transformed EOFs of ens_arr
    """

    skpca = pca.PCA(explained_variance)
    eofs = skpca.fit_transform(ens_arr)

    return eofs


def regress_eof(eofs, obs):
    """Perform a linear regression between
    EOFs and observations
    --------------------------------
    Args:
        eofs (numpy.ndarray): EOF transformed
            ens_arr containing same grid cells
            sample as observations
        obs (numpy.ndarray): Gridded observations
            either from pseudo or real observations
    Returns:
        model (scikit-learn model obj): fitted
            model object to eofs and obs
    """
    lm = linear_model.LinearRegression()
    eofs = eofs.reshape(-1, 1)
    model = lm.fit(eofs, obs)
    print("Model score:", model.score(eofs, obs))
    return model


def predict_dv(model, eofs_of_model):
    """Reconstruct design value field from
    full EOF sample
    --------------------------------
    Args:
        model (scikit-learn model): model of regressed obs
            and models
        eofs_of_model (numpy.ndarray): EOF transformed
            ens_arr
    Returns:
        (numpy.ndarray): predicted design
            value field
    """
    eofs_of_model = eofs_of_model.reshape(-1, 1)
    return model.predict(eofs_of_model)


def pred_to_grid(dv_field, pred, mask):
    """Reshape from flattened ensemble size x
    number of cells to rlat x rlon grid space
    --------------------------------
    Args:
        dv_field (xarray.DataArray): datacube
            containing design values of from
            an ensemble of CanRCM4 models
        pred (numpy.ndarray): reconstructed
            predictions from EOF regression
        mask (numpy.ma): mask used to mask
            values not considered
    Returns:
        dv_field (xarray.DataArray): reconstructed design
            value field
    """

    dv_field = dv_field[0, :, :]
    print(mask.shape, dv_field.shape, pred.shape)
    dv_field.values[mask[0, :, :]] = pred

    return dv_field


def eof_pseudo_full(dv_field, mask):
    """Perform all steps to reconstruct a
    design value field from pseudo observations
    ---------------
    Args:
        dv_field (xarray.DataArray): datacube
            containing design values of from
            an ensemble of CanRCM4 models
    Returns:
        pred (xarray.DataArray): datacube
            containing an ens of CanRCM4 models
            with added eofs variable
    """
    maskflat = mask_flat(mask)
    mean = ens_mean(dv_field)
    area = frac_grid_area(dv_field)

    # area weighted ens to get obs
    ens_obs = ens_flat(dv_field*area)

    # mask flattened ensemble
    ens_obs = mask_land(ens_obs, maskflat)
    ens_obs = mask_invalid(ens_obs)
    #ens_obs = ens_obs.data[~ens_obs.mask]
    print(ens_obs)
    # get random ens to generate
    # the pseudo obs
    obs = get_obs(ens_obs)
    obs_idx = rand_sample_index(obs, 0.02)
    obs_sample = obs[obs_idx]

    ens = ens_flat(dv_field*area - mean)
    ens = mask_land(ens, maskflat)
    ens = mask_invalid(ens)
    ens = ens_to_eof(ens.T)[:, 0]

    model = regress_eof(ens[obs_idx], obs_sample)
    pred = predict_dv(model, ens)

    pred = pred_to_grid(dv_field, pred, mask)
    pred = (pred/area + mean)

    return pred
