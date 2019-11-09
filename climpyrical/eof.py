import numpy as np
import numpy.ma as ma
import pandas as pd
from sklearn.decomposition import pca
from sklearn import linear_model
from sklearn.preprocessing import Imputer
from sklearn.metrics import r2_score
import warnings
from operators import frac_grid_area


def ens_to_eof(ens_arr, n_components, explained_variance=0.95):
    """Perform EOF/PCA dimensionality reduction
    on ensemble array
    --------------------------------
    Args:
        ens_arr (numpy.ndarray): reshaped 2-d array
            into number of cells x ensemble size
        explained_variance (float): fraction of total
            variance that the reductions account for
    Returns:
        eofs (sklearn.decomposition.pca.PCA): fitted
            principal compoenent sklearn object
    """

    skpca = pca.PCA(n_components, explained_variance)
    eofs = skpca.fit(ens_arr)

    return eofs

def fit_eof(eofs, obs, sample_weight=None):
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
    from sklearn import ensemble


    # lm = ensemble.GradientBoostingRegressor(loss='lad',learning_rate=0.015, n_estimators=250)
    # lm = ensemble.RandomForestRegressor()
    # lm = linear_model.Ridge(alpha = 0.001, normalize=True)
    # lm = linear_model.RidgeCV(alphas=np.logspace(-10, 10, 10))
    # lm = linear_model.LogisticRegression(random_state=0, solver='lbfgs',multi_class='multinomial')
    lm = linear_model.LinearRegression()
    model = lm.fit(eofs, obs)
    # print(
        # "Regressed model score:",
        # r2_score(obs, model.predict(eofs)),
        # lm.score(eofs, obs)
    # )
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


    return model.predict(eofs_of_model)
