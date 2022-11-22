# Standard packages
import os
import sys
import time
import glob
# Numerical packages
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
# Local scripts
import geo_tools

#
# Field types
#

class Equidistant_Lon_Lat_Field(xr.DataArray):
    
    '''
    DataArrays that are defined on a two-dimensional equidistant longitude ('lon') and latitude ('lat') grid with optional time dimension.
    '''
    
    def global_mean(self):
        return geo_tools.global_mean(self)

#
# Main class 
#
    
class Scenario:
    
    def __init__(self,name,cases):
        '''
        Initialize a scenario
        
        name (string):
        cases (dictionary string: dictionary: case names (keys) with metadata specification:
            The value dictionary needs to have the 'directory' keyword with a filepath to the Snellius archive
            directionry of the case, e.g. '/gpfs/work4/0/uuesm2/archive/b.e21.BSSP585cmip6.f09_g17.control.01'.
            The value for the keyword 'years' must be a list of years for which to select data, e.g. 'np.arange(2020,2031)'
            to select data from 2020 to 2030. If it is 'None', all data is selected. The 'years' keyword is important
            when selecting data from multiple cases where there might be an overlap in time intervals.
        '''
        
        self.name = name
        self.cases = cases
        
        # Initialize an empty dictionary that is later used to store xarray Datasets
        self.var = {}

    def __get_attr__(self,var):
        return self.var[var]

    #
    # Loading netcdfs for the Scenario
    #
    
    def _check_var(self,var,component,get=True):
        '''
        Check if variable(s) was already fetched. If not, get the variable(s)
        
        Parameters:
        ---
        var (string or list of string): Variable names
        component (string or list of string): Respective components
        get (Boolean): get variable 
        
        Returns:
        ---
        Boolean: True if variables are now stored in the scenario. If get=True this will always be the case
        '''

        # Turn single-valued arguments into lists
        if not hasattr(var,'__iter__'):
            var = [var]
            
        if not hasattr(component, '__iter__'):
            component = [component]
            
        assert len(var)==len(component), 'var and component need to have same length'
        
        fetched = True
        for v,c in zip(var,component):
            if not v in self.var.keys():
                fetched = False
                if get:
                    self._get_monthly_var(v,c)
                    fetched = True
        
        return fetched
    
    def _get_monthly_var(self,var,component):
        '''
        Load monthly output variable from a specific component, e.g. 'ocn'
        
        Parameters:
        ---
        var (string): name of variable
        component (string): generic component name 'ocn', 'atm' etc.
        '''
        # components: their names (e.g. 'cam') and the tag for the monthly output stream (e.g. 'h0')
        comp_names = {'ocn': {'name': 'pop', 'stream': 'h'},
                    'atm': {'name': 'cam', 'stream': 'h0'},
                    'ice': {'name': 'cice', 'stream': 'h'}}
        
        # container in which to load separate xr DataArray
        da_container = []

         # iterate over different case directories
        # append data of different case files
        for case_name, case_metadata in self.cases.items():
            
            # directory in which to find the time series
            load_parent_path = os.path.join(case_metadata['directory'],f'{component}/proc/tseries/month_1')
            # prefix of the data files/usually includes the CESM component, e.g. 'cam' 
            # and a specification for the frequency e.g. 'h0' for monthly
            # remember the trailing '.' after the variable name
            # this is important to avoid confusion between variables like 'VVEL' and 'VVEL2'
            load_prefix = f"{case_name}.{comp_names[component]['name']}.{comp_names[component]['stream']}.{var}."
            # full load path
            load_name = load_parent_path+'/'+load_prefix+'*' # add wildcard '*' for use in open_mfdataset 
            
            # get all file names that fit the pattern in 'load_name'
            files_case = glob.glob(load_name)
            # ensure that time series are concatenated appropriately
            files_case.sort() # in-place!
            
            # iterate over these files
            da_case_container = []
            for file_path in files_case:
                da_case_container.append(xr.open_dataset(file_path,chunks={'time': 12})[var]) 
            
            # Concatenate and select years
            # This selection is done because a case can in principle have faulty data in some years
            # (e.g. if an error occured in later years and the simulation is then branched off)
            year_slice = slice(case_metadata['years'][0].astype(str), case_metadata['years'][-1].astype(str))
            da_container.append(xr.concat(da_case_container,dim='time').sel(time=year_slice))
            
        # concatenate over all cases
        #
        da = xr.concat(da_container,dim='time') 
        self.var[var] = da
        return da
        
    def get_atm_var(self,var):
        '''
        Get monthly atmospheric component variable
        '''
        self._get_monthly_var(var,'atm')

    def get_ocn_var(self,var):
        '''
        Get monthly ocean component variable
        '''
        self._get_monthly_var(var,'ocn')
        
    def get_ice_var(self,var):
        '''
        Get monthly ice component variable
        '''
        self._get_monthly_var(var,'ice')
            
    # Hassle-free computation of derived variables:
    # use stored time series of CESM output to compute variables of interest, 
    # e.g. freshwater import/export due to overturning at southern Atlantic boundary or global mean surface temperature
    
    def compute_T0(self):
        '''
        Compute the global mean surface temperature
        '''
        self._check_var('TREFHT','atm')
       
        self.var['T0'] = geo_tools.global_mean(self.var['TREFHT'])
        return self.var['T0']
    
    def M_ov(self,S_0=34.5):
        
        ###
        ### auxilliary functions
        ###
        def sel_cs(da):
            '''
            Select a cross-section at 34deg south in the Atlantic starting from a global DataArray defined on the POP2 grid. 
            This is the cross-section through which the salinity import/export is measured.
            '''
            # 0-54: mid-Atlantic to African west coast
            # 300-320: South American east coast to mid-Atlantic
            atl_lon = np.append(np.arange(0,54),np.arange(300,320))
            nlat_cs = 84
            return da.sel(nlat=nlat_cs).sel(nlon=atl_lon).fillna(0)
        
        def zonal_avg(da,top=None):
            '''
            Perform (topography-aware) zonal averaging at -34deg south

            Need to provide topographical mask for the normalization to work properly.
            If it is not provided the function attempts to 

            Units:
            [Output] = [Input]
            '''
            if top is None:
                top = np.clip(np.abs(da),0,1)

            # we can perform sums rather than coordinate-aware integrals 
            # as we divide out any units/scalings of the longitudinal axi      
            norm = top.sum(dim='nlon') # this counts the grid cells not filled by topography

            # treat the singular case which can happen at low z_t levels 
            # note that in this case the sum over nlon of da is zero 
            # this means we have a zero value in total no matter what the value of 'norm' is there
            norm = norm.fillna(1) 
            return da.sum(dim='nlon')/norm
        
        def zonal_int(da):
            '''
            Integrate a field zonally

            Field must be in nlon and z_t dimensions at lat=34degS

            Topography does not need to be specified as we assume
            da=0 (not nan) on non-ocean grid points. If this is not the case
            pre-process the data.

            Units:
            [Output] = m/s * [Input]
            '''
            R_earth = 6.371*10**6 # Earth mean radius [m]
            # perform zonal integral at 34deg south
            dlon = np.deg2rad(1.0) # Zonal grid spacing angle [dim.less]
            dx = np.cos(np.deg2rad(-34.0)) * R_earth * dlon # Zonal grid length [m]
            return da.sum(dim='nlon') * dx
        ###
        ###
        ###
        
        self._check_var(['VVEL','SALT'],['ocn','ocn'])
        
        vvel = sel_cs(self.var['VVEL'])
        salt = sel_cs(self.var['SALT'])

        pre_factor = -1/S_0
        integrand = zonal_int(vvel) * (zonal_avg(salt)-S_0)
        # converting cm/s * m to m/s * m = m^2/s
        integrand = integrand * 0.01
        # some terms in the integrand might be nan:
        # at lowest z level there are no ocean cells and zonal_avg returns 0
        # set the integrand to be 0
        integrand = integrand.fillna(0)
        integral = integrand.integrate(coord=['z_t'])
        # converting cm * m^2/s to to m^3/s
        integral = 0.01 * integral
        
        return pre_factor * integral/10**6 # convert to Sverdrup
        