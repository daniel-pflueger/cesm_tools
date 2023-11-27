#
# Tools to analyse the meridional stream function (MSF) output produced by Michal Kliphuis' script.
#
# Created 3.10.22 by Daniel Pflueger, d.pfluger@uu.nl
#

import xarray as xr
import numpy as np
import os

# I/0

def _get_timestamp_from_filename(path):
    
    '''Extract a timestamp from file name and return it as an xarray-compatible index'''
    
    split = path.split('.')
    time_str = split[-2]
    year, month = time_str.split('-')
    
    t_string = f"{year}-{month.rjust(2,'0')}"
    ts = xr.date_range(start=t_string,end=t_string,use_cftime=True)
    
    return ts

def open_msf(path):
    
    '''
    Open a MSF netcdf in xarray and add the timestamp implied by the filename.
    '''
    
    da = xr.open_dataset(path)
    ts = _get_timestamp_from_filename(path)
    da = da.expand_dims({'time': ts})
    
    return da

# AMOC index computation functions

def msf_max_index(ds):
    
    '''
    Obtain the maximum of the AMOC MSF (below the surface layer and northwards of 28 deg). This prescription is consistent with Tilmes et al.
    '''
    
    ds_sel = ds.TMTA.sel(depth_w=slice(500,6000)).sel(lat_mht=slice(28,90))
    index = float(ds_sel.max())
    return index
    
def lat_max_index(ds,lat):
    
    '''
    Obtain the MSF maximum at a given latitude.
    '''
    
    ds_sel = ds.TMTA.sel(depth_w=slice(500,6000)).sel(lat_mht=lat,method='nearest')
    index = float(ds_sel.max())
    return index

def max_heat(ds):
    '''
    Obtain maximum heat transport
    '''

    index = float(ds.MHTA.max())
    return index
    
def depth_index(ds):
    
    '''
    Get the depth position of the MSF maximum obtained by 'msf_max_index'
    '''
    
    ds_sel = ds.TMTA.sel(depth_w=slice(500,6000)).sel(lat_mht=slice(28,90))
    max_pos = np.unravel_index(ds_sel.argmax(),ds_sel.shape)
    index = ds_sel.depth_w.data[max_pos[0]]
    return index

# Wrapper function

index_functions_std = {'msf_max': msf_max_index,
                      'depth': depth_index,
                      'msf_30S_max': lambda da: lat_max_index(da,-30),
                      'msf_30N_max': lambda da: lat_max_index(da,30),
                      'msf_45N_max':lambda da: lat_max_index(da,45),
                      'msf_60N_max':lambda da: lat_max_index(da,60),
                      'max_heat': max_heat}


def amoc_indices(msf_dir,index_functions=index_functions_std,get_msf=True):
    
    '''
    Extract AMOC indices from a MSF data directory
    
    The index_functions dictionary determines which AMOC indices are computed. 
    It consists of index names (keys) and functions of DataArrays (elements) that compute the respective index..
    '''
    
    files = os.listdir(msf_dir)
    
    # construct a time axis for the Dataset
    # first get all the timestamps from the individual files, 
    # then concatenate and sort the time indices
    single_indices = list(map(_get_timestamp_from_filename,files))
    indices = single_indices[0]
    for t_index in single_indices[1:]:
        indices = indices.append(t_index)
    indices = indices.sort_values()
    ts_len = len(indices)
    
    #
    # now iterate over the data points and extract all the different AMOC indices
    #
    dummy_da = xr.DataArray(data=np.zeros(ts_len),
                            dims=['time'],
                           coords={'time': indices}) # empty DataArray/will be copied to fill a Dataset
    # Initialise the AMOC indices Dataset that is to be returned later
    # Its data will be modified in place in the innermost loop of this function.
    ds = xr.Dataset(
        data_vars = {name:dummy_da.copy() for name in index_functions.keys()},
                   coords={'time':indices})
    for msf_file, time_index in zip(files,single_indices):
    
        # this dictionary will hold all the index data
        # the keys are the index names
        #index_dict = {None for index_name in index_functions.keys()}
        ds_in = xr.open_dataset(os.path.join(msf_dir,msf_file))
        
        # iterate over all indices
        for index_name, index_func in index_functions.items():
            ds[index_name].loc[time_index] = index_func(ds_in)
            
    if get_msf:
        first_file=True
        for msf_file, time_index in zip(files,single_indices):
            da_in = xr.open_dataset(os.path.join(msf_dir,msf_file)).TMTA
            lats = da_in.lat_mht
            lat_len = len(lats)
            zs = da_in.depth_w * 100 # m to cm (in agreement with CMIP6 data)
            z_len = len(zs)
            if first_file:
                da_msf = xr.DataArray(data=np.zeros((ts_len,z_len,lat_len)),
                                       dims=['time','lev','lat'],
                                       coords={'time': indices, 'lev': zs.data, 'lat': lats.data})
                first_file=False
            da_msf.loc[time_index,:,:] = da_in.data
            da_msf = da_msf.rename('TMTA')
                
        ds = xr.merge([ds,da_msf])
    
    return ds
    
    