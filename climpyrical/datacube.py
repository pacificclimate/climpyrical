import xarray as xr


def check_valid_keys(actual_keys, required_keys):
    '''A function to test that required_keys is a subset of actual_keys.
    Args:
        actual_keys (dict): dictionary with keys found in the NetCDF file
        required_keys (dict): dictionary of expected and required keys
            that make sense for the climpyrical analyses
    Returns:
        bool: True of passed, raises error if not.
    Raises:
        KeyError if required_keys are not a subset of the actual keys
    '''
    if not set(required_keys).issubset(actual_keys):
        raise KeyError(
            "CanRCM4 ensemble is missing keys {}".format(
                required_keys - actual_keys
            )
        )
    return True


def read_data(
    data_path, design_value_name, keys={"rlat", "rlon", "lat", "lon", "level"}
):
    """Load an ensemble of CanRCM4
    models into a single datacube.
    ------------------------------
    Args:
        data_path (Str): path to folder
            containing CanRCM4 ensemble
        design_value_name (str): name of design value as found in the
            NetCDF4 file
        keys (dict, optional): dictionary of required keys in NetCDF4
            file

    Returns:
        ds (xarray Dataset): data cube of assembled ensemble models
            into a single variable.
    """

    ds = xr.open_dataset(data_path)
    actual_keys = set(ds.variables).union(set(ds.dims))
    keys.add(design_value_name)
    check_valid_keys(actual_keys, keys)

    return ds
