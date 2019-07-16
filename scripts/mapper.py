import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature 

res = '110m'

def color_pallette():
    """Assembles a custom color temperature color scheme

    Returns:
        cmap (Matplotlib ListedColormap): Color pallette used
        for plotting
    """ 
    cmap = mpl.colors.ListedColormap(['#e11900', '#ff7d00', '#ff9f00',
                                      '#ffff01', '#c8ff32', '#64ff01',
                                      '#00c834', '#009695', '#0065ff',
                                      '#3232c8', '#dc00dc', '#ae00b1'])
    return cmap

def ocean_mask():
    """Gets coastlines for North america at resolution set in self
    Returns:
    ocean (Cartopy NaturalEarthFeature): polygon containing
    coastlines
    """
    ocean = cartopy.feature.NaturalEarthFeature('physical', 'ocean', res, edgecolor='k', facecolor='white')
    return ocean


def rp():
    """Defines projected coordinates used for plotting
    Returns:
        rp (Cartopy crs RotatedPole): projection used for
        plotting coordinates in rotated field.
    """
    rp = ccrs.RotatedPole(pole_longitude=-97-180,
                          pole_latitude=42.5)
    return rp

def plot_reference(data_cube, title, save_fig=False):
    """
    Plots the mean value along the run axis of CanRCM4 simulations

    Parameters
    ----------
    xarray dict : Data cube with geospatial and field
    data for ensemble of CanRCM4 data

    Returns
    -------
    out : matplotlib axis object

    """
    rp_obj = rp()
    ocean = ocean_mask()
    cmap = color_pallette()

    plt.figure(figsize=(15, 15))

    rlon = data_cube['rlon']
    rlat = data_cube['rlat']
    field = data_cube['dv'][0, :, :]

    # define projections
    ax = plt.axes(projection=rp_obj)
    ax.set_title(title, fontsize=30, verticalalignment='bottom')
    ax.add_feature(ocean, zorder=2)

    # plot design values with custom colormap
    colorplot = ax.scatter(rlon, rlat, c=field,
                           marker='s', cmap=cmap,
                           vmin=1., vmax=13.)

    cbar = plt.colorbar(colorplot, ax=ax,
                        orientation="horizontal",
                        fraction=0.07, pad=0.025)

    cbar.ax.tick_params(labelsize=25)

    # constrain to data
    plt.xlim(rlon.min(), rlon.max())
    plt.ylim(rlat.min(), rlat.max())

    if save_fig:
        plt.savefig(run)

    return ax