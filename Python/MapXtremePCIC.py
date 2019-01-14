import glob

import numpy as np
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
    def __init__(self, res, method, data_path):
        #self.obs = obs
        self.res = res
        self.method = method
        self.data_path = data_path
        
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
        if (type(res) != type(1)):
            raise ValueError('Mapping resolution requires {}, got {}'.format(type(1), type(res)))

        def read_data(PATH = data_path):
            """
            Arguments
              Path to data directory with netcdf files
            Value
              Data cube with lat and lon keys
            """

            # Create a list of all files in PATH
            nc_list = np.asarray(glob.glob(PATH+"*"))

            for path in nc_list:
                if path.endswith('.nc') == False:
                    raise IOError('{} is not a supported file type.'.format(path))

            # Create list with datasets as entries
            dataset_list = np.empty(nc_list.shape) 

            inst = nc.Dataset(nc_list[0], 'r')

            data_cube = np.empty((inst['lat'].shape[0], inst['lat'].shape[1], nc_list.shape[0]))

            inst.close()

            for i, path in enumerate(nc_list):
                run = nc.Dataset(path, 'r')
                pr = run.variables['pr'][:, :]
                data_cube[:, :, i] = pr

            lat = run.variables['lat'][:, :]
            lon = run.variables['lon'][:, :]

            rlat = run.variables['rlat'][:]
            rlon = run.variables['rlon'][:]

            ds = xr.Dataset({'pr': (['x', 'y', 'run'], data_cube)},
                            coords = {'lon': (['x', 'y'], lon),
                                      'lat': (['x', 'y'], lat),
                                      'rlon': rlon,
                                      'rlat': rlat},
                            attrs = {'pr': 'mm h-1',
                                    'lon': 'degrees',
                                    'lat': 'degrees',
                                    'rlon': 'degrees',
                                    'rlat': 'degrees'})

            return ds

        self.load_data = read_data(data_path)
        
        
    def ensemble_mean(self, data_cube):
        """
        Returns ensemble mean of data region

        Parameters
        ----------
        xarray dict : Data cube with geospatial and field data for ensemble of CanRCM4 data

        Returns
        -------
        out : n x p array of ensemble mean of design values
        """

        # number of simulation runs
        n = data_cube['run'].values.shape[0]

        # number of grids
        p = data_cube['lat'].shape[0]*data_cube['lat'].shape[1]

        # n x n identity matrix
        I_n = np.eye(n)

        # all ones n x n matrix
        one_n = np.ones((n, n))

        # n x p reshaped data
        X = np.reshape(data_cube['pr'].values, (data_cube['run'].shape[0], p))

        # change nan values to zero for proper mean
        X = np.nan_to_num(X, 0.0)

        # n x p ensemble mean
        X_prime = np.dot((I_n - (1.0/n)*one_n), X)

        return X_prime
    
    
    def weight_matrix(self, data_cube):
        """
        Returns weighted array using fractional grid cell areas

        Parameters
        ----------
        xarray dict : Data cube with geospatial and field data for ensemble of CanRCM4 data

        Returns
        -------
        out : n x p weighted spatial array
        """
        # calculate differences between array entires to get grid sizes
        lat = np.diff(data_cube['lat'].values, axis=0)[:, :-1]
        lon = np.diff(data_cube['lon'].values, axis=1)[:-1, :]

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
        X_prime = MapXtremePCIC.ensemble_mean(self, data_cube)[:, 0:p]

        # apply fractional areas to get weighted array
        X_w = np.dot(X_prime, diag_f)

        return X_w
    
    def plot_reference(self, data_cube):
            """
            Plots the mean value along the run axis of CanRCM4 simulations

            Parameters
            ----------
            xarray dict : Data cube with geospatial and field data for ensemble of CanRCM4 data

            Returns
            -------
            out : matplotlib axis object

            """
            # take mean of all simulation runs
            N = data_cube['pr'].mean(axis=2)
            # take away all 0.0 values for ocean
            N = N.where(N != 0.0)

            
            rlat, rlon = data_cube['rlat'], data_cube['rlon']
            
            # defined projection from pre-defined proj4 params
            rp = ccrs.RotatedPole(pole_longitude=-97.45 + 180,
                                  pole_latitude=42.66)
            # custom colormap
            
            cdict = {'#e11900':1, '#ff7d00':2, '#ff9f00':3, 
                     '#ffc801':4, '#ffff01':5, '#c8ff32':6, 
                     '#64ff01':7, '#00c834':8, '#009695':9, 
                     '#0065ff':10, '#3232c8':11, '#dc00dc':12, '#ae00b1':12}
            
            cmap = mpl.colors.ListedColormap(cdict)
            
            #cmap = mpl.colors.ListedColormap(['#e11900', '#ff7d00', '#ff9f00', 
            #                                 '#ffc801', '#ffff01', '#c8ff32', 
            #                                 '#64ff01', '#00c834', '#009695', 
            #                                 '#0065ff', '#3232c8', '#dc00dc', 
            #                                 '#ae00b1'])
            
            plt.figure(figsize=(15, 15))
            
            # define projections
            ax = plt.axes(projection=rp)
            ax.coastlines('110m', linewidth=2.)
            ax.set_title('50-year daily precipitation [mm/h]', fontsize=30, verticalalignment='bottom')
            
            # plot design values with custom colormap
            colorplot = plt.pcolormesh(rlon, rlat, N, transform=rp, cmap=cmap)
            
            # make colorbar object
            cbar = plt.colorbar(colorplot, ax=ax, orientation="horizontal", fraction=0.07, pad=0.025)
            
            cbar.ax.tick_params(labelsize=25)
            
            # constrain to data
            plt.xlim(rlon.min(), rlon.max())
            plt.ylim(rlat.min(), rlat.max())
            plt.savefig('north_america_simulation_mean')

            return ax