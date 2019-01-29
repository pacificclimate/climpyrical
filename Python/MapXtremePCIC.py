import glob
import warnings

import numpy as np
import numpy.ma as ma
import pandas as pd
import xarray as xr
import netCDF4 as nc
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature


class MapXtremePCIC:

    """ 
    MapXtremePCIC maps design values over North America
    ====================================================
    Arguments
     CanRCM4.lens : information data list for CanRCM4 modelled design values over North America
     obs : data array of observed design values over North America, [lon, lat, data] three columns 
     res : resolution (in km) of the map
     method : whether EOF or SOM-based method is employed for mapping
     
    Value
     rlon : vector of longitude coordinates of the map
     rlat : vector of latitude coordinates of the map
     xtreme : data array of the mapped design values
     sp.basis : data array of the spatial basis functions estimated from CanRCM4 modelled data
     obs.grid : data array of the gridded observations 
    
    Note: the coordinate system is in polar rotated projection for all involved arrays. The projection
     is "+proj=ob_tran +o_proj=longlat +lon_0=-97 +o_lat_p=42.5 +a=1 +to_meter=0.0174532925199 +no_defs"
     
    Author : Nic Annau at PCIC, University of Victoria, nannau@uvic.ca
    """

    def __init__(self, res, method, data_path, variable,  R = 6371.0):
        #self.obs = obs
        self.res = res
        self.method = method
        self.data_path = data_path
        self.variable = variable
        self.R = R

        # Check inputs
        if (type(data_path) != type('string')):
            raise ValueError('Method argument requires {} got {}'.format(type('string'), type(data_path)))    
        
        if (type(method) != type('string')):
            raise ValueError('Method argument requires {} got {}'.format(type('string'), type(method)))
        if (method != 'eof' and method != 'som'):
            raise Exception('MapXtremePCIC requires specified \'som\' or \'eof\'. Got {}'.format(method))
        
        #if (obs.shape[0] < 100):
        #    raise Exception('Observed design values sample size of {} is too small (<100).'.format(obs.shape[0]))
        
        # Set default res if not specified
        if (res == None):
            print("Res not specified. Setting default res = 50.")
            self.res = 50

        # Check map resolution type    
        if (type(res) != type('a')):
            raise ValueError('Mapping resolution requires {}, got {}'.format(type('a'), type(res)))

    def read_data(self):
        """
        Arguments
          Path to data directory with netcdf files
        Value
          Data cube with lat and lon keys
        """

        PATH = self.data_path
        variable = self.variable

        # Create a list of all files in PATH
        nc_list = np.asarray(glob.glob(PATH+"*.nc"))

        for path in nc_list:
            if path.endswith('.nc') == False:
                raise IOError('{} is not a supported file type. Data directory must have only .nc files'.format(path))

        # Create list with datasets as entries
        dataset_list = np.empty(nc_list.shape) 

        inst = nc.Dataset(nc_list[0], 'r')

        data_cube = np.ones((inst['lat'].shape[0], inst['lat'].shape[1], nc_list.shape[0]))

        inst.close()

        for i, path in enumerate(nc_list):
            run = nc.Dataset(path, 'r')
            obs = run.variables[variable][:, :]
            data_cube[:, :, i] = obs

        lat = run.variables['lat'][:, :]
        lon = run.variables['lon'][:, :]

        rlat = run.variables['rlat'][:]
        rlon = run.variables['rlon'][:]
        
        run.close()
        
        ds = xr.Dataset({'obs': (['x', 'y', 'run'], data_cube)},
                        coords = {'lon': (['x', 'y'], lon),
                                  'lat': (['x', 'y'], lat),
                                  'rlon': rlon,
                                  'rlat': rlat},
                        attrs = {'obs': 'mm h-1',
                                'lon': 'degrees',
                                'lat': 'degrees',
                                'rlon': 'degrees',
                                'rlat': 'degrees'})

        return ds
        
        
    def color_pallette(self):
        # custom colormap
        cmap = mpl.colors.ListedColormap(['#e11900', '#ff7d00', '#ff9f00', 
                                          '#ffff01', '#c8ff32', '#64ff01', 
                                          '#00c834', '#009695', '#0065ff', 
                                          '#3232c8', '#dc00dc', '#ae00b1'])
        return cmap

    def ocean_mask(self):
        ocean = cartopy.feature.NaturalEarthFeature('physical', 'ocean', self.res,
                                        edgecolor='k',
                                        facecolor='white')
        return ocean
    
    def rp(self):
        rp = ccrs.RotatedPole(pole_longitude=-97.45 + 180,
                              pole_latitude=42.66)
        return rp

    def grid_area(self):
        """
        Returns array of areas for grid space, calculated from rectangular section
        on the sphere.

        Parameters 
        ----------
        xarray : data cube from which to pull areas
        float : radius of earth
        """
        ds = self.read_data()
        R = self.R

        lat = np.deg2rad(ds['lat'].values[:-1, :-1])

        p = lat.shape[0]*lat.shape[1]

        # calculate the differences along each axis to get grid
        # cell size - shape is 1 smaller because the end point doesn't have a grid area. 
        lat_sz = np.abs(np.diff(np.sin(np.deg2rad(ds['lat'].values)), axis=0)[:, :-1])
        lon_sz = np.abs(np.deg2rad(np.diff(ds['lon'].values, axis=1)[:-1, :]))

        # calculate rectangular area on sphere. 
        area = (lat_sz * lon_sz) * R**2

        # reshape to be flat
        grid_area_flat = np.reshape(area, p)

        return grid_area_flat, area, p 

    def get_arr(self, df = None):
        """
        Converts pandas dataframe containing runs as columns organized from dataset 
        to n x p array

        Parameters
        ----------
        pandas dataframe : full dataframe with each run labelled as in MapXtremePCIC.get_df

        Returns
        -------
        out : n x p array containing design values
        """
        if df is None:
            df = self.get_df()

        n = 0

        for column in df.columns:
            if 'run' in column:
                n += 1 

        p = df.shape[0]

        run_list = ['run{}'.format(i) for i in range(n)]

        X = np.empty((n, p))

        for i, run in enumerate(run_list):
            run_obs = df[run].values
            X[i, :] = run_obs
        
        return n, p, X
            
    def get_df(self, ds = None):
        """
        Converts xarray Dataset to pandas dataframe

        Parameters
        ----------
        xarray Dataset : Data cube with geospatial and field data for ensemble of CanRCM4 data

        Returns
        -------
        out : pandas dataframe in rotated polar coordinates
        """
        if ds is None:
            ds = self.read_data()

        # reshape to be flat
        grid_area_flat, area, p = self.grid_area()

        # set up repeating values of rlat, rlon times
        rlat = np.repeat(ds['rlat'][:-1].values, ds['rlon'][:-1].shape[0])# + lat_grid_sz.values/2.
        # set up repeating sequence of rlon, rlat times
        rlon = np.tile(ds['rlon'][:-1].values, ds['rlat'][:-1].shape[0])# + lon_grid_sz.values/2.

        # sized 1 less in each dimension because of diff
        obs_n = ds['obs'][:-1, :-1, :].values

        # reshape the obs grids
        obs = np.reshape(obs_n,  (p, ds['obs'].shape[2]))

        d = dict(('run{}'.format(i), obs[:, i]) for i in range(obs[0, :].shape[0]))

        # create a dictionary from arrays
        pd_dict = {'rlon': rlon, 'rlat': rlat, 'areas': grid_area_flat}

        # create dataframe from dict
        df = pd.DataFrame(pd_dict)

        # set a column to each run in simulation
        for i in range(obs[0, :].shape[0]):
            df['run{}'.format(i)] = obs[:, i]

        return df


    def weight_matrix(self):
        """
        Returns fractional weighting array using fractional grid cell areas

        Parameters
        ----------
        xarray dict : Data cube with geospatial and field data for ensemble of CanRCM4 data

        Returns
        -------
        pandas Series : n x p weighted spatial array
        """
        grid_areas_flat, area, p = self.grid_area()

        total_area = np.sum(grid_areas_flat)

        # divide by total area of grids
        fractional_areas = (1.0/total_area)*grid_areas_flat

        # reshape from p1 x p2 to 1 x p
        f = pd.Series(np.reshape(fractional_areas, p))

        return f

    def ensemble_mean(self, df = None, reverse = False):
        """
        Returns ensemble mean of data region for all runs

        Parameters
        ----------
        pandas dataframe : Data cube with geospatial and field data for ensemble of CanRCM4 data

        Returns
        -------
        out : n x p array of ensemble mean of design values
        """

        np.seterr(invalid = 'ignore')

        # Check df input is defined, if not, get it from object
        df_no_mean_corr = self.get_df()

        if df is None:
            if reverse == True:
                warnings.warn("Incorrect mean if you do not provide a previously corrected dataframe")

            df = self.get_df()

        df_copy = df.copy()

        # apply the mean correction if false
        for column in df.columns:
            if 'run' in column:
                mean = df_no_mean_corr[column].mean()
                if reverse == False:
                    df_copy[column] =  df[column].subtract(mean)
                # undo the mean correction if true
                if reverse == True:
                    df_copy[column] =  df[column].add(mean)

        np.seterr(invalid = 'warn')

        return df_copy

    def standard_matrix(self, df = None, reverse = False, mean_run = None):
        """
        Returns weighted and standardized array using fractional grid cell areas

        Returns
        -------
        ndarray : n x p weighted amd standardized spatial array
        """
        
        if df is None:

            if reverse == True:
                warnings.warn("Incorrect mean if you do not provide a previously corrected dataframe")

            df = self.ensemble_mean(reverse = False)

        df_copy = df.copy()
        df_return = df.copy()
        df_no_mean_corr = self.get_df()

        #df = self.ensemble_mean(df, reverse = reverse)

        f = self.weight_matrix()
        for column in df_no_mean_corr.columns:
            if 'run' in column:
                mean = df_no_mean_corr[column].mean()
                if reverse == False:
                    df_copy[column] = df[column].multiply(f, axis = 0)
                    df_return[column] = df_copy[column]

                if reverse == True:
                    df_copy[column] = df[column].div(f, axis = 0)
                    df_return[column] = df_copy[column].add(mean)
        
        if mean_run is not None:
            if reverse == True:
                mean = df_no_mean_corr['run{}'.format(mean_run)].mean()
                df_copy['run_eof'] = df_copy['run_eof'].div(f, axis = 0)
                df_return['run_eof'] = df_copy['run_eof'].add(mean)

        return df_return

    def plot_reference(self, run = 0):
        """
        Plots the mean value along the run axis of CanRCM4 simulations

        Parameters
        ----------
        xarray dict : Data cube with geospatial and field data for ensemble of CanRCM4 data

        Returns
        -------
        out : matplotlib axis object

        """
        # Ignore NaN calculations(for better plotting, for any calculations)
        np.seterr(invalid = 'ignore')

        data_cube = self.read_data()
        res = self.res

        # take mean of all simulation runs
        N = data_cube['obs'][:, :, run]
        
        rlat, rlon = data_cube['rlat'], data_cube['rlon']

        # custom colormap
        cmap = self.color_pallette()

        # ocean mask
        ocean = self.ocean_mask(res)

        # custom ax object with projection
        rp = self.rp()

        plt.figure(figsize = (15, 15))

        # define projections
        ax = plt.axes(projection = rp)
        ax.set_title('50-year daily precipitation [mm/h]', fontsize=30, verticalalignment='bottom')
        ax.add_feature(ocean, zorder=2)
        
        # plot design values with custom colormap
        colorplot = plt.pcolormesh(rlon, rlat, N, transform=rp, cmap=cmap, vmin=1., vmax=13.)
        cbar = plt.colorbar(colorplot, ax=ax, orientation="horizontal", fraction=0.07, pad=0.025)
        cbar.ax.tick_params(labelsize=25)

        # constrain to data
        plt.xlim(rlon.min(), rlon.max())
        plt.ylim(rlat.min(), rlat.max())
      
        # return/undo the supress invalid warning
        np.seterr(invalid = 'warn')

        return ax

    def sample(self, df = None, frac = 0.02, seed = None, dropna = True):
        """
        Returns randomly sampled land data from an average of CanRCM4 runs

        Parameters
        ----------
        float : fraction of data to return
        int   : random state which to use. Use the same integer to preserve sample that is randomly
        sampled between calls to the function. If None, returns a new random sample each call to the function.  
        
        Returns
        -------
        out : sampled pandas df
        """
        if df is None:
            df = self.get_df()
        
        if dropna == True:
            df = df.dropna()

        df = df.sample(frac=frac, random_state = seed)

        return df

    def pseudo_obs(self, df = None, run = 0, frac = 0.02, seed = None):

        if df is None:
            df = self.get_df()




    def plot_scatter(MapXtreme, frac, run = 0, seed = True):

        if seed == True:
            seed = np.random.randint(0, 100)

        res = MapXtreme.res

        
        df = MapXtremePCIC.sample(MapXtreme, frac = frac, seed = seed)[1]

        # get observations from run number
        run_obs = np.asarray([df['obs'].iloc[i][run] for i in range(len(df))])
        
        np.seterr(invalid = 'ignore')

        cmap = MapXtremePCIC.color_pallette()

        # ocean mask
        ocean = MapXtremePCIC.ocean_mask(res)

        # custom ax object with projection
        rp = MapXtremePCIC.rp()

        plt.figure(figsize = (15, 15))

        # define projections
        ax = plt.axes(projection = rp)
        ax.set_title('50-year daily precipitation [mm/h]', fontsize=30, verticalalignment='bottom')
        ax.add_feature(ocean, zorder=2)

        # plot sampled design values with custom colormap
        colorplot = ax.scatter(df['rlon'], df['rlat'], c = run_obs, cmap = cmap, transform = rp, vmin=1., vmax=13.)
        
        # make colorbar object
        cbar = plt.colorbar(colorplot, ax=ax, orientation="horizontal", fraction=0.07, pad=0.025)

        cbar.ax.tick_params(labelsize=25)

        # constrain to data
        plt.xlim(df['rlon'].min(), df['rlon'].max())
        plt.ylim(df['rlat'].min(), df['rlat'].max())
        np.seterr(invalid = 'warn')
        
        return ax