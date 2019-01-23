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
    def __init__(self, res, method, data_path, variable):
        #self.obs = obs
        self.res = res
        self.method = method
        self.data_path = data_path
        self.variable = variable

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

        def read_data(PATH = data_path):
            """
            Arguments
              Path to data directory with netcdf files
            Value
              Data cube with lat and lon keys
            """

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

        self.load_data = read_data(data_path)
        
        
    def color_pallette():
        # custom colormap
        cmap = mpl.colors.ListedColormap(['#e11900', '#ff7d00', '#ff9f00', 
                                          '#ffff01', '#c8ff32', '#64ff01', 
                                          '#00c834', '#009695', '#0065ff', 
                                          '#3232c8', '#dc00dc', '#ae00b1'])
        return cmap

    def ocean_mask(res):
        ocean = cartopy.feature.NaturalEarthFeature('physical', 'ocean', res,
                                        edgecolor='k',
                                        facecolor='white')
        return ocean
    
    def rp():
        rp = ccrs.RotatedPole(pole_longitude=-97.45 + 180,
                              pole_latitude=42.66)
        return rp

    def get_arr(MapXtreme, df):

        ds = MapXtreme.load_data

        n = ds['obs'].shape[2]
        p = df.shape[0]

        run_list = ['run{}'.format(i) for i in range(n)]

        X = np.empty((n, p))

        for i, run in enumerate(run_list):
            run_obs = df[run].values
            X[i, :] = run_obs
        
        return n, p, X
            
    def get_df(MapXtreme, run = 0):

        ds = MapXtreme.load_data

        # calculate the differences along each axis to get grid
        # cell size - shape is 1 smaller because the 
        lat_sz = np.abs(np.diff(np.sin(np.deg2rad(ds['lat'].values)), axis=0)[:, :-1])
        lon_sz = np.abs(np.deg2rad(np.diff(ds['lon'].values, axis=1)[:-1, :]))

        lat = np.deg2rad(ds['lat'].values[:-1, :-1])

        R = 6371.
        p = lat.shape[0]*lat.shape[1]

        # calculate rectangular area on sphere
        area = (lat_sz * lon_sz) * R**2

        # reshape to be flat
        grid_area_flat = np.reshape(area, p)

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

    def ensemble_mean(MapXtreme, df):
        """
        Returns ensemble mean of data region

        Parameters
        ----------
        xarray dict : Data cube with geospatial and field data for ensemble of CanRCM4 data

        Returns
        -------
        out : n x p array of ensemble mean of design values
        """

        n, p, X = MapXtremePCIC.get_arr(MapXtreme, df)

        # n x n identity matrix
        I_n = np.eye(n)

        # all ones n x n matrix
        one_n = np.ones((n, n))

        # n x p ensemble mean
        X_prime = np.dot((I_n - (1.0/n)*one_n), X)
            
        return X_prime

    def inv_ensemble_mean(MapXtreme, df):
        """
        Returns ensemble mean of data region

        Parameters
        ----------
        xarray dict : Data cube with geospatial and field data for ensemble of CanRCM4 data

        Returns
        -------
        out : n x p array of ensemble mean of design values
        """

        n, p, X = MapXtremePCIC.get_arr(MapXtreme, df)

        # n x n identity matrix
        I_n = np.eye(n)

        # all ones n x n matrix
        one_n = np.ones((n, n))

        # n x p ensemble mean
        X_prime = np.dot(X, np.linalg.inv(I_n - (1.0/n)*one_n))
            
        return X_prime

    def weight_matrix(MapXtreme, frac = 0.02):
        """
        Returns weighted array using fractional grid cell areas

        Parameters
        ----------
        xarray dict : Data cube with geospatial and field data for ensemble of CanRCM4 data

        Returns
        -------
        out : n x p weighted spatial array
        """

        # calculate differences between array entires to get approximate grid sizes
        ds = MapXtremePCIC.load_data(MapXtreme)

        # calculate differences between array entires to get grid sizes
        lat = np.diff(ds['lat'].values, axis=0)[:, 1:]
        lon = np.diff(ds['lon'].values, axis=1)[1:, :]

        # define size of p
        p = (lat.shape[0])*(lon.shape[1])

        # multiply each grid size by each other and sum 
        grid_areas = np.multiply(lon, lat)
        total_area = np.sum(grid_areas)

        # divide by total area of grids
        fractional_areas = (1.0/total_area)*grid_areas

        # reshape from p1 x p2 to 1 x p
        f = np.reshape(fractional_areas, p)

        # diagonalize reshapes fractional areas vector
        diag_f = np.diag(f)

        # get the ensemble means but ignore the grids at the edges of the fields
        # since the area cannot be determined 
        X_prime = MapXtremePCIC.ensemble_mean(MapXtreme, frac)[:, p:]

        # apply fractional areas to get weighted array
        X_w = np.dot(X_prime, diag_f)

        return X_w
        
    def plot_reference(MapXtreme, run = 0):
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

        data_cube = MapXtreme.load_data
        res = MapXtreme.res

        # take mean of all simulation runs
        N = data_cube['obs'][:, :, run]
        
        rlat, rlon = data_cube['rlat'], data_cube['rlon']

        # custom colormap
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

    def sample(MapXtreme, frac, run = 0, seed = True, dropna = True):
        """
        Returns randomly sampled land data from an average of CanRCM4 runs

        Parameters
        ----------
        xarray dict : Data cube with geospatial and field data for ensemble of CanRCM4 data
        float : fraction of dataset desired

        Returns
        -------
        out : xarray Dataset

        """

        # check seed, generate random int if necessary
        if seed == True:
            seed = np.random.randint(0, 100)

        ds = MapXtreme.load_data

        lat = np.deg2rad(ds['lat'].values[:-1, :-1])

        R = 6371.
        p = lat.shape[0]*lat.shape[1]

        # calculate the differences along each axis to get grid
        # cell size - shape is 1 smaller because the 
        lat_sz = np.abs(np.diff(np.sin(np.deg2rad(ds['lat'].values)), axis=0)[:, :-1])
        lon_sz = np.abs(np.deg2rad(np.diff(ds['lon'].values, axis=1)[:-1, :]))

        # calculate rectangular area on sphere
        area = (lat_sz * lon_sz) * R**2

        # reshape to be flat
        grid_area_flat = np.reshape(area, p)

        # set up repeating values of rlat, rlon times
        rlat = np.repeat(ds['rlat'][:-1].values, ds['rlon'][:-1].shape[0])# + lat_grid_sz.values/2.
        # set up repeating sequence of rlon, rlat times
        rlon = np.tile(ds['rlon'][:-1].values, ds['rlat'][:-1].shape[0])# + lon_grid_sz.values/2.

        # create a dictionary from arrays
        pd_dict = {'rlon': rlon, 'rlat': rlat, 'areas': grid_area_flat}

        # create dataframe from dict
        df = pd.DataFrame(pd_dict)

        # sized 1 less in each dimension because of diff
        obs_n = ds['obs'][:-1, :-1, :].values

        # reshape the obs grids
        obs = np.reshape(obs_n,  (p, ds['obs'].shape[2]))

        # set a column to each run in simulation
        for i in range(obs[0, :].shape[0]):
            df['run{}'.format(i)] = obs[:, i]
        
        if dropna == True:
            df = df.dropna()
        df = df.sample(frac=frac, random_state = seed)

        return df

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
        #plt.savefig('north_america_scatter')
        np.seterr(invalid = 'warn')

        
        return ax