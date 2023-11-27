import numpy as np
import xarray as xr

# Code snippet taken from Matt Long
# https://gist.github.com/matt-long/159523f52ac8834ae88f2837e9e4b4fe

def pop_add_cyclic(ds,grid='T'):
    
    if grid=='T':
        lat_name = 'TLAT'
        lon_name = 'TLONG'
    elif grid=='U':
        lat_name = 'ULAT'
        lon_name = 'ULONG'
    
    nj = ds[lat_name].shape[0] # size of POP grid
    ni = ds[lon_name].shape[1]

    xL = int(ni/2 - 1)
    xR = int(xL + ni)

    tlon = ds[lon_name].data
    tlat = ds[lat_name].data
    
    tlon = np.where(np.greater_equal(tlon, min(tlon[:,0])), tlon-360., tlon) #make monotoncially increasing
    lon  = np.concatenate((tlon, tlon + 360.), 1) # concatenate to make larger array
    lon = lon[:, xL:xR] #restrict to middle rane

    if ni == 320: # this is the x1 POP grid
        lon[367:-3, 0] = lon[367:-3, 0] + 360.        
    lon = lon - 360.
    
    lon = np.hstack((lon, lon[:, 0:1] + 360.)) # add in cyclic point
    if ni == 320:
        lon[367:, -1] = lon[367:, -1] - 360.

    #-- trick cartopy into doing the right thing:
    #   it gets confused when the cyclic coords are identical
    lon[:, 0] = lon[:, 0] - 1e-8

    #-- periodicity
    lat = np.concatenate((tlat, tlat), 1)
    lat = lat[:, xL:xR]
    lat = np.hstack((lat, lat[:,0:1]))

    LAT = xr.DataArray(lat, dims=('nlat', 'nlon'))
    LONG = xr.DataArray(lon, dims=('nlat', 'nlon'))
    
    dso = xr.Dataset({lat_name: LAT, lon_name: LONG})

    # copy vars
    varlist = [v for v in ds.data_vars if v not in ['TLAT', 'TLONG']]
    for v in varlist:
        v_dims = ds[v].dims
        if not ('nlat' in v_dims and 'nlon' in v_dims):
            dso[v] = ds[v]
        else:
            # determine and sort other dimensions
            other_dims = set(v_dims) - {'nlat', 'nlon'}
            other_dims = tuple([d for d in v_dims if d in other_dims])
            lon_dim = ds[v].dims.index('nlon')
            field = ds[v].data
            field = np.concatenate((field, field), lon_dim)
            field = field[..., :, xL:xR]
            field = np.concatenate((field, field[..., :, 0:1]), lon_dim)       
            dso[v] = xr.DataArray(field, dims=other_dims+('nlat', 'nlon'), 
                                  attrs=ds[v].attrs)


    # copy coords
    for v, da in ds.coords.items():
        if not ('nlat' in da.dims and 'nlon' in da.dims):
            dso = dso.assign_coords(**{v: da})
                
    # add name of main data variable
    if len(varlist)==1:
        dso.attrs['name'] = varlist[0]
    return dso


def add_cyclic_coords(ds,grid='T'):
    
    '''
    Add the coordinates to the final dataset (rather than having different value fields)
    '''
    
    if grid=='T':
        lat_name = 'TLAT'
        lon_name = 'TLONG'
    elif grid=='U':
        lat_name = 'ULAT'
        lon_name = 'ULONG'
    
    nj = ds[lat_name].shape[0] # size of POP grid
    ni = ds[lon_name].shape[1]

    xL = int(ni/2 - 1)
    xR = int(xL + ni)

    tlon = ds[lon_name].data
    tlat = ds[lat_name].data
    
    tlon = np.where(np.greater_equal(tlon, min(tlon[:,0])), tlon-360., tlon) #make monotoncially increasing
    lon  = np.concatenate((tlon, tlon + 360.), 1) # concatenate to make larger array
    lon = lon[:, xL:xR] #restrict to middle rane

    if ni == 320: # this is the x1 POP grid
        lon[367:-3, 0] = lon[367:-3, 0] + 360.        
    lon = lon - 360.
    
    lon = np.hstack((lon, lon[:, 0:1] + 360.)) # add in cyclic point
    if ni == 320:
        lon[367:, -1] = lon[367:, -1] - 360.

    #-- trick cartopy into doing the right thing:
    #   it gets confused when the cyclic coords are identical
    lon[:, 0] = lon[:, 0] - 1e-8

    #-- periodicity
    lat = np.concatenate((tlat, tlat), 1)
    lat = lat[:, xL:xR]
    lat = np.hstack((lat, lat[:,0:1]))

    LAT = xr.DataArray(lat, dims=('nlat', 'nlon'))
    LONG = xr.DataArray(lon, dims=('nlat', 'nlon'))
    
    dso = xr.Dataset({lat_name: LAT, lon_name: LONG})

    # copy vars
    varlist = [v for v in ds.data_vars if v not in ['TLAT', 'TLONG']]
    for v in varlist:
        v_dims = ds[v].dims
        if not ('nlat' in v_dims and 'nlon' in v_dims):
            dso[v] = ds[v]
        else:
            # determine and sort other dimensions
            other_dims = set(v_dims) - {'nlat', 'nlon'}
            other_dims = tuple([d for d in v_dims if d in other_dims])
            lon_dim = ds[v].dims.index('nlon')
            field = ds[v].data
            field = np.concatenate((field, field), lon_dim)
            field = field[..., :, xL:xR]
            field = np.concatenate((field, field[..., :, 0:1]), lon_dim)       
            dso[v] = xr.DataArray(field, dims=other_dims+('nlat', 'nlon'), 
                                  attrs=ds[v].attrs)


    # copy coords
    for v, da in ds.coords.items():
        if not ('nlat' in da.dims and 'nlon' in da.dims):
            dso = dso.assign_coords(**{v: da})

    for dim_name in [lon_name,lat_name]:
        dso.coords[lon_name] = dso[lon_name]
        dso.coords[lat_name] = dso[lat_name]

    
    return dso