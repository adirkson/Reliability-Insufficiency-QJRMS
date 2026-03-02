#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 13:07:48 2024

@author: ard000
"""

import numpy as np
import matplotlib as mpl
import cartopy.crs as ccrs
import cartopy.feature as cfeature

class FigureFunctions:
    def add_map_subplot(self, fig, gs, bbox):
        """Create the base map with static features."""
        
        transform = ccrs.Orthographic(central_longitude=-70, central_latitude=55)
        ax = fig.add_subplot(gs, projection=transform)
        ax.set_extent(bbox, crs=ccrs.PlateCarree())        
        # ax.set_extent([-120, -50, 25, 75], crs=ccrs.PlateCarree())
        # Add geographic features (static)
        ax.add_feature(cfeature.BORDERS, color='k', linewidth=0.5, zorder=3)
        ax.coastlines(resolution='110m',linewidth=0.5, color='k', zorder=3)  

        return ax, transform          

    def truncate_div_cmap(self, cmap, frac_subtract=None, frac_subtract_low=None, frac_subtract_high=None):
        if frac_subtract is not None:
            interval = np.hstack([np.linspace(0, 0.5-frac_subtract), np.linspace(0.5+frac_subtract, 1)])
        elif  frac_subtract_low is not None and frac_subtract_high is not None:
            interval = np.hstack([np.linspace(0, 0.5-frac_subtract_low), np.linspace(0.5+frac_subtract_high, 1)])
        else:
            raise(ValueError)
            
        colors = cmap(interval)
        cmap_new = mpl.colors.LinearSegmentedColormap.from_list('name', colors)
        return cmap_new
    
    # cuts off the ends of cmap colors at minval and maxval
    def truncate_colormap(self, cmap, minval=0.0, maxval=1.0, n=100):
        # Use the cmap's name if it has one, otherwise use a default name
        cmap_name = getattr(cmap, 'name', 'custom_cmap')
    
        new_cmap = mpl.colors.LinearSegmentedColormap.from_list(
            'trunc({},{:.2f},{:.2f})'.format(cmap_name, minval, maxval),
            cmap(np.linspace(minval, maxval, n))
        )
        
        return new_cmap

    def shiftedColorMap(self, cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
        '''
        Function to offset the "center" of a colormap. Useful for
        data with a negative min and positive max and you want the
        middle of the colormap's dynamic range to be at zero
    
        Input
        -----
          cmap : The matplotlib colormap to be altered
          start : Offset from lowest point in the colormap's range.
              Defaults to 0.0 (no lower ofset). Should be between
              0.0 and `midpoint`.
          midpoint : The new center of the colormap. Defaults to 
              0.5 (no shift). Should be between 0.0 and 1.0. In
              general, this should be  1 - vmax/(vmax + abs(vmin))
              For example if your data range from -15.0 to +5.0 and
              you want the center of the colormap at 0.0, `midpoint`
              should be set to  1 - 5/(5 + 15)) or 0.75
          stop : Offset from highets point in the colormap's range.
              Defaults to 1.0 (no upper ofset). Should be between
              `midpoint` and 1.0.
        '''
        cdict = {
            'red': [],
            'green': [],
            'blue': [],
            'alpha': []
        }
    
        # regular index to compute the colors
        reg_index = np.linspace(start, stop, 257)
    
        # shifted index to match the data
        shift_index = np.hstack([
            np.linspace(0.0, midpoint, 128, endpoint=False), 
            np.linspace(midpoint, 1.0, 129, endpoint=True)
        ])
    
        for ri, si in zip(reg_index, shift_index):
            r, g, b, a = cmap(ri)
    
            cdict['red'].append((si, r, r))
            cdict['green'].append((si, g, g))
            cdict['blue'].append((si, b, b))
            cdict['alpha'].append((si, a, a))
    
        newcmap = mpl.colors.LinearSegmentedColormap(name, cdict)
    
        return newcmap
