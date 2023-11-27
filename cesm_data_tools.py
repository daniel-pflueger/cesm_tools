# Helpful methods to handle CESM2 data

# Extract variables from CESM netcdf output
# This only requires NCO 

import os
import numpy as np
import xarray as xr
import cmocean
import cartopy
import cartopy.crs as ccrs
import pop_plot
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.ticker as mticker
from matplotlib.animation import FuncAnimation

# Extracting variables

def extract_and_concat(var,years,in_dir,prefix,out_dir):
    
    '''
    Extracts a single variable from CESM2 data and concats it a single netcdf file.
    Output is written to a specified directory.
    
    Parameters:

    ---
    var (string): name of variable to be extracted
    years (list of int): years for which to extract the variable
    in_dir (string): directory with relevant CESM output
    prefix (string): prefix of file names, 
        e.g. use 'b.e21.BSSP585cmip6.f09_g17.control.01.pop.h' when the file names are
        of the form 'b.e21.BSSP585cmip6.f09_g17.control.01.pop.h.2080-07.nc'
    '''
    
    # example parameters for a function call
    #var = 'PREC_F'
    #years = np.arange(2020,2100)
    #in_dir = '~/links/leo_archive/b.e21.BSSP585cmip6.f09_g17.control.01/ocn/hist'
    #prefix = 'b.e21.BSSP585cmip6.f09_g17.control.01.pop.h'
    #out_dir = '~/data/b.e21.BSSP585cmip6.f09_g17.control.01/prec_f'
    
    # create output directory in case it does not exist
    if not os.path.exists(out_dir):
        print(f'Creating output directory {out_dir}')
        os.makedirs(out_dir)
    else:
        print(f'Directory {out_dir} already exists. Deleting content before extracting.')
        for f in os.listdir(out_dir):
            os.remove(os.path.join(out_dir,f))
    
    for year in years:
        for month in range(1,13):
            month_str = str(month).rjust(2,'0')
            date = f'{year}-{month_str}'
            in_path = os.path.join(in_dir,f'{prefix}.{date}.nc')
            out_path = os.path.join(out_dir,f'{var}.{date}.nc')
            os.system(f'ncks -v {var} {in_path} {out_path}')
            # Debug in case ncks fails
            #print('---')
            #print(f'ncks -v {var} {in_path} {out_path}')
            #print('---')
            
    out_dir_m = out_dir.rstrip('/')# remove trailing slash if it is there
    os.system(f'ncrcat {out_dir_m}/*.nc {out_dir_m}/{var}.nc') # concatenate file

#
# Plot functions
#
    
def plot_2d_map(da,plot_params={},
                mode='std',save_path=None,ax=None):
    '''
    Plot a 2D map of a given data set and save the figure
    '''
    
    assert mode in ['std','pop'], 'Plot mode should be \'std\' or \'pop\''
    assert isinstance(da,xr.DataArray)
    
    if save_path is None:
        save_path = './out.png'
    
    if mode == 'std':
        fig, plot_out = plot_std_map(da,plot_params,ax=ax)
    elif mode == 'pop':
        fig, plot_out = plot_pop_map(da,plot_params,ax=ax) 
    
    fig.savefig(save_path)
    
    return fig, plot_out

# Standard parameters for surface 2D plots
fig_params_std = {'dpi': 300}
pc_params_std = {}
ax_params_std = {'title': None}
proj_params_std = {'central_longitude': 330, 'extent': None}     
proj_params_na = {'central_longitude': 330, 'extent': [20, 280, 30, 70]}     
cb_params_std = {'shrink': 0.4}

def plot_pop_map_na(da,plot_params={},grid='T'):
    
    plot_params['proj_params'] = proj_params_na
    fig, plot_out = plot_pop_map(da, plot_params, grid='T')
    
    return (fig, plot_out)

def plot_pop_sp(ds, 
               pc_params = {},
               fig_params = {}):

    # make circular boundary for polar stereographic circular plots
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center) 

    # Plot with pcolormesh
    vmax_in = 1000*100
    vmin_in = 0
    cmap_in = cmocean.cm.dense

    # create figure
    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.SouthPolarStereo()}, **fig_params)

    # plot the region as subplots - note it's nrow x ncol x index (starting upper left)
    #ax.add_feature(cfeature.LAND,zorder=100,edgecolor='k')
    ax.set_boundary(circle, transform=ax.transAxes)
    ax.set_extent([0.005, 320, -90, -45], crs=ccrs.PlateCarree())
    pc = ax.pcolormesh(ds.TLONG, ds.TLAT,
        ds[ds.attrs['name']],
        transform=ccrs.PlateCarree(),
        **pc_params)
    fig.colorbar(pc,orientation='horizontal',label=ds.attrs['name'],fraction=0.03,pad=0.05) 
    
    print('here')
    plot_out = {'ax': ax, 'pc': pc}
    print(plot_out)
    return fig, plot_out

def plot_pop_map(da,plot_params={},grid='T'):
    
    '''
    
    '''
    
    assert grid in ['T','U'], 'Grid must either be T or U'
    
    # Extract plot params
    fig_params = plot_params.get('fig_params',fig_params_std)
    pc_params = plot_params.get('pc_params',pc_params_std)
    proj_params = plot_params.get('proj_params',proj_params_std)
    ax_params = plot_params.get('ax_params',ax_params_std)
    cb_params = plot_params.get('cb_params',cb_params_std)
    
    #
    #
    #
    if grid=='T':
        lat_name = 'TLAT'
        lon_name = 'TLONG'
    elif grid=='U':
        lat_name = 'ULAT'
        lon_name = 'ULONG'
    
    #
    # Preparation for pop_add_cyclic
    # This step is necessary to plot ocean maps without discontinuities
    #
    
    # promote DataArray to Dataset/this is needed to work with pop_add_cyclic
    ds = da.to_dataset(name='var') # 'var' is just a dummy name
    ds = pop_plot.pop_add_cyclic(ds,grid=grid)
       
    # Define figure and projection
    
    fig = plt.figure(dpi=300)
    ax = plt.axes(projection=ccrs.Robinson(central_longitude=proj_params['central_longitude']),**ax_params)
    if 'extent' in proj_params.keys():
        if not proj_params['extent'] is None:
            ax.set_extent(proj_params['extent'])
    
    # Adding features to map
    land = ax.add_feature(
        cartopy.feature.NaturalEarthFeature('physical', 'land', '110m',
                                            linewidth=0.5,
                                            edgecolor='black',
                                            facecolor='darkgray'))
    
    
    # Perform plot (color mesh)
    pc = ax.pcolormesh(ds[lon_name], ds[lat_name], ds['var'].squeeze(),transform=ccrs.PlateCarree(),**pc_params) 

    # color bar
    cb = plt.colorbar(pc, **cb_params)
    
    ax.set_title(ax_params.get('title',None))
    
    # Return plot output: the figure itself as well as the elements
    # The actual plot can be saved with the parent function plot_2d_map
    plot_out = {'ax': ax, 'pc': pc, 'cb': cb}  

    return (fig,plot_out)   


def plot_pop_sp(ds, 
               pc_params = {},
               fig_params = {}):

    # make circular boundary for polar stereographic circular plots
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center) 

    # Plot with pcolormesh
    vmax_in = 1000*100
    vmin_in = 0
    cmap_in = cmocean.cm.dense

    # create figure
    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.SouthPolarStereo()}, **fig_params)

    # plot the region as subplots - note it's nrow x ncol x index (starting upper left)
    #ax.add_feature(cfeature.LAND,zorder=100,edgecolor='k')
    ax.set_boundary(circle, transform=ax.transAxes)
    ax.set_extent([0.005, 320, -90, -45], crs=ccrs.PlateCarree())
    pc = ax.pcolormesh(ds.TLONG, ds.TLAT,
        ds[ds.attrs['name']],
        transform=ccrs.PlateCarree(),
        **pc_params)
    fig.colorbar(pc,orientation='horizontal',label=ds.attrs['name'],fraction=0.03,pad=0.05) 
    
    print('here')
    plot_out = {'ax': ax, 'pc': pc}
    print(plot_out)
    return fig, plot_out

def plot_pop_np(ds, 
               pc_params = {},
               fig_params = {}):

    # make circular boundary for polar stereographic circular plots
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center) 

    # Plot with pcolormesh
    vmax_in = 1000*100
    vmin_in = 0
    cmap_in = cmocean.cm.dense

    # create figure
    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.NorthPolarStereo()}, **fig_params)

    # plot the region as subplots - note it's nrow x ncol x index (starting upper left)
    #ax.add_feature(cfeature.LAND,zorder=100,edgecolor='k')
    ax.set_boundary(circle, transform=ax.transAxes)
    ax.set_extent([0.005, 320, 45, 85], crs=ccrs.PlateCarree())
    pc = ax.pcolormesh(ds.TLONG, ds.TLAT,
        ds[ds.attrs['name']],
        transform=ccrs.PlateCarree(),
        **pc_params)
    fig.colorbar(pc,orientation='horizontal',label='fraction',fraction=0.03,pad=0.05) 
    
    print('here')
    plot_out = {'ax': ax, 'pc': pc}
    print(plot_out)
    return fig, plot_out

def plot_pop_na_conic(da,
                     pc_params={'cmap': cmocean.cm.amp},
                     fig_params={'figsize': (10,10)}):
    
    # Define projections
    dataproj = ccrs.PlateCarree()
    #mapproj  = ccrs.Stereographic(central_longitude=-30)
    mapproj = ccrs.LambertConformal(
        central_longitude=-20, central_latitude=50, standard_parallels=(30, 60))
    # Plot figure
    fig, ax = plt.subplots(subplot_kw={'projection': mapproj}, **fig_params)

    # Add some features

    #ax.add_feature(cartopy.feature.OCEAN)
    ###
    # I want to have kind of a conic shape. I manage to do this by adapting some code from
    # https://stackoverflow.com/a/65690841/9970523
    #
    # Lon and Lat Boundaries
    xlim = [-70, 30]
    ylim = [40, 80]
    lower_space = 20
    rect = mpath.Path([[xlim[0], ylim[0]],
                       [xlim[1], ylim[0]],
                       [xlim[1], ylim[1]],
                       [xlim[0], ylim[1]],
                       [xlim[0], ylim[0]],
                       ]).interpolated(20)
    proj_to_data   = ccrs.PlateCarree()._as_mpl_transform(ax) - ax.transData
    rect_in_target = proj_to_data.transform_path(rect)
    ax.set_boundary(rect_in_target)
    ax.set_extent([xlim[0], xlim[1], ylim[0] - lower_space, ylim[1]])
    #
    # End of https://stackoverflow.com/a/65690841/9970523
    ###
    # lats/lons labels and ticks
    pc = ax.pcolormesh(da.TLONG,da.TLAT,da[da.attrs['name']],transform = ccrs.PlateCarree(),
                      **pc_params)

    gl = ax.gridlines(draw_labels=True,x_inline=False,y_inline=False, crs=ccrs.PlateCarree())

    ax.coastlines()
    #ax.add_feature(cartopy.feature.LAND,facecolor='grey')
    #ax.gridlines(draw_labels=True)
    gl.xlocator = mticker.FixedLocator([-60,-40,-20,0,20])
    gl.ylocator = mticker.FixedLocator([30,40,50, 60,70])
    gl.bottom_labels = True
    gl.left_labels   = True
    gl.top_labels    = False
    gl.right_labels  = False

    fig.colorbar(pc, orientation='horizontal',shrink=0.5,pad=-0.05)

    plot_out = {'pc': pc, 'ax': ax}
    
    return fig, plot_out

def plot_std_map(da,plot_params={}):
    
    '''
    
    '''
    
    # Extract plot params
    fig_params = plot_params.get('fig_params',fig_params_std)
    pc_params = plot_params.get('pc_params',pc_params_std)
    proj_params = plot_params.get('proj_params',proj_params_std)
    ax_params = plot_params.get('ax_params',ax_params_std)
    cb_params = plot_params.get('cb_params',cb_params_std)
    
       
    # Define figure and projection
    
    fig = plt.figure(dpi=300)
    ax = plt.axes(projection=ccrs.Robinson(central_longitude=proj_params['central_longitude']),**ax_params)
    if 'extent' in proj_params.keys():
        if not proj_params['extent'] is None:
            ax.set_extent(proj_params['extent'])

    # Perform plot (color mesh)
    pc = da.plot(ax=ax,cbar_kwargs=cb_params,transform=ccrs.PlateCarree(),**pc_params) 
    ax.set_title(ax_params.get('title',None))
    
    # Adding features to map
    #land = ax.add_feature(
    #    cartopy.feature.NaturalEarthFeature('physical', 'land', '110m',
    #                                        linewidth=0.5,
    #                                        edgecolor='black',
    #                                        facecolor='darkgray'))
    ax.coastlines(resolution='110m',linewidth=0.5)
    # Return plot output: the figure itself as well as the elements
    # The actual plot can be saved with the parent function plot_2d_map
    plot_out = {'ax': ax, 'pc': pc}  
    return (fig,plot_out)  

### Animations

# Animation function for North Atlantic POP2 surface data
# Example usage:
# 
# pop_na_anim(ml_depth_mean, plot_params={'pc_params': pc_params})
# where ml_depth_mean is an annual mean mixed layer field

def pop_na_anim(da,time_label='year',
                interval=100,
                grid='T',
                plot_params={}):
    
    assert grid in ['T','U'], 'Grid must either be T or U. Default is T'
    
    # Setting up initial figure
    # These objects are later modified
    fig, plot_out = plot_pop_map_na(da[0,:,:],grid=grid,
                               plot_params=plot_params)
    pc = plot_out['pc']
    ax = plot_out['ax']
    ax.set_title('')
    
    ds_mod = pop_plot.pop_add_cyclic(da.to_dataset())
    def update_mesh(t):
        ax.set_title(f'{time_label}: {t}')
        pc.set_array(ds_mod[da.name][t,:,:].data.ravel())

    ts = np.arange(0,np.size(ds_mod[time_label].data))
    ani = FuncAnimation(fig, update_mesh, frames=ts,
                    interval=interval)
    
    return ani

def pop_anim(ds, plot_name, time_label='time', interval=100,
            fig_params={},pc_params={}
            ):
    
    plot_types = {
        'North Atlantic': plot_pop_na_conic,
        'North Pole': plot_pop_np,
        'South Pole': plot_pop_sp
    }
    plot_func = plot_types[plot_name]
    
    #assert grid in ['T','U'], 'Grid must either be T or U. Default is T'
    
    # Setting up initial figure
    # These objects are later modified
    var_name = ds.attrs['name']
    ds_tsel = ds.isel({time_label: 0})
    fig, plot_out = plot_func(ds_tsel,
                               pc_params=pc_params,fig_params=fig_params)
    pc = plot_out['pc']
    ax = plot_out['ax']
    ax.set_title('')
    
    def update_mesh(t):
        ax.set_title(f'{time_label}: {t}')
        pc.set_array(ds.isel({time_label: t})[var_name].data.ravel())

    ts = np.arange(0,np.size(ds[time_label].data))
    ani = FuncAnimation(fig, update_mesh, frames=ts,
                    interval=interval)

    return ani
