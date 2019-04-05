import numpy as np
import numpy.ma as ma
from sklearn.decomposition import pca 
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm

from operators import check_keys, cell_count, ens_means, frac_grid_area

def mask_nan(ds):
    """Gets indices of nan values to 
    identify where land or ocean is. 
    --------------------------------
    Args: 
        ensemble_arr (numpy.ndarray): reshaped 2-d array
            into number of cells x ensemble size
    Returns:
        (numpy.ndarray): 2-d array containing
            the indices of nan values
    """
    dv_field = ds['dv'].values
    mask = ma.masked_invalid(ds['dv'][0,:,:].values)
    return mask

def mask_flat(ds, mask):
    dv_field = ds['dv'].values
    n_grid_cells = dv_field.shape[1]*dv_field.shape[2]
    mask = np.reshape(mask, n_grid_cells)
    return mask

def ens_flat(ds):
    """Reshapes data cube into array shape 
    for eof calculations. 
    --------------------------------
    Args: 
        ds (xarray Dataset): datacube 
            containing an ensemble of 
            CanRCM4 models
        mask (numpy.ma.masked_invalid): masked array
            of a single ensemble member grid. This mask
            will usually be either an ocean or land mask. 
    Returns:
        dv_field (numpy.ndarray): reshaped 2-d array
            into number of cells x ensemble size
    """
    check_keys(ds)

    dv_field = ds['dv'].values

    ensemble_sz = dv_field.shape[0]
    n_grid_cells = dv_field.shape[1]*dv_field.shape[2]

    dv_field = np.reshape(dv_field, (ensemble_sz, n_grid_cells))

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

def get_obs(ensemble_arr):
    """Randomly sample a data cube to generate
    pseudo observations from ensemble. 
    --------------------------------
    Args:  
        ds (xarray Dataset): datacube 
            containing an ensemble of 
            CanRCM4 models 
    Returns:
        y_sample (numpy.ndarray): fraction
            of random spatially sampled design
            value grid cells
        index (numpy.ndarray): index locations 
            of y_sample from original ensemble size
    """
    ensemble_sz = ensemble_arr.shape[0]

    # randomly select an ensemble member
    # to sample
    i = np.random.randint(0, ensemble_sz-1)
    y_obs = ensemble_arr[i, :]

    return y_obs

def ensemble_to_eof(ensemble_arr):
    """Perform EOF/PCA dimensionality reduction
    on ensemble array 
    --------------------------------
    Args:  
        ensemble_arr (numpy.ndarray): reshaped 2-d array
            into number of cells x ensemble size
    Returns:
        eofs (numpy.ndarray): EOF transformed
            ensemble_arr sorted by max variance 
            explained
    """
    
    skpca = pca.PCA(0.95)
    eofs = skpca.fit_transform(ensemble_arr)
    
    return eofs

def regress_eof(eofs, obs):
    """Perform a linear regression between
    EOFs and observations 
    --------------------------------
    Args:  
        eofs (numpy.ndarray): EOF transformed
            ensemble_arr containing same grid cells 
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
        model (scikit-learn model obj): fitted 
            model object to eofs and obs
        eofs_of_model (numpy.ndarray): EOF transformed
            ensemble_arr containing all grid
            cells in design value field
    Returns:
        (numpy.ndarray): predicted design
            value field
    """
    eofs_of_model = eofs_of_model.reshape(-1, 1)
    return model.predict(eofs_of_model)

def pred_to_grid(ds, pred, mask):

    ds['eof'] = ds['dv'][0,:,:]
    ds['eof'].values[~mask.mask] = pred

    return ds['eof']

def eof_pseudo_full(ds, mask=None):
    """Perform all steps to reconstruct a 
    design value field from pseudo observations
    ---------------
    Args: 
        ds (xarray Dataset): datacube 
            containing an ensemble of CanRCM4 models 
    Returns:
        ds (xarray Dataset): datacube 
            containing an ensemble of CanRCM4 models
            with added eofs variable
    """

    mean = ens_means(ds)
    area = frac_grid_area(ds)
    if mask is None:
        mask = mask_flat(ds, mask_nan(ds))

    # area weighted ensemble to get obs
    ens_obs = ens_flat(ds*area)

    # mask 
    ens_obs = ens_obs[:, ~mask.mask]

    # get random ensemble to generate
    # the pseudo obs 
    obs = get_obs(ens_obs)
    obs_idx = rand_sample_index(obs, 0.02)
    obs_sample = obs[obs_idx]

    ens = ens_flat(ds*area - mean)
    ens = ens[:, ~mask.mask]
    ens = ensemble_to_eof(ens.T)[:, 0]

    model = regress_eof(ens[obs_idx], obs_sample)
    pred = predict_dv(model, ens)

    pred = pred_to_grid(ds, pred, mask_nan(ds))
    pred = (pred/area + mean)

    return pred