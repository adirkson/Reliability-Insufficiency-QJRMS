#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 10 18:13:51 2023

@author: ard000
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import chisquare, chi2
from matplotlib.gridspec import GridSpec as GS
import matplotlib.cm as cm

from figure_functions import FigureFunctions

# fixed parameters
R_all = [0.9, 0.6, 0.2]
sigma_y = 1.0
n=50

def delta_func(Delta_rho, sigma_ratio, R):
    """
    delta = Delta_sigma^2 - 2*DeltaSigma
    """
    r = R + Delta_rho
    sigma_x = sigma_ratio * sigma_y
    sigx2 = sigma_x**2
    sigy2 = sigma_y**2
    Delta_sigma2 = sigx2 - sigy2
    Delta_Sigma = sigx2 * r - sigma_x * sigma_y * R
    delta = Delta_sigma2 - 2.0 * Delta_Sigma
    return delta

def alpha_func(Delta_rho, sigma_ratio, R):
    r = R + Delta_rho
    sigma_x = sigma_ratio * sigma_y
    sigx2 = sigma_x**2
    sigma_e2 = sigx2 * (1.0 - r)
    sigx2_mean = r * sigx2 +  sigma_e2/n
    sigma_x_mean = np.sqrt(sigx2_mean)
    cov_yxmean = R * sigma_x * sigma_y
    rho_mean = cov_yxmean /( sigma_x_mean * sigma_y)
    
    alpha = sigma_y/sigma_x_mean * ((n-1)*rho_mean + np.sqrt((n-1)**2.0 * rho_mean**2.0 + 4*n))/(2*n)    
    return alpha

def beta2_func(Delta_rho, sigma_ratio, R):
    r = R + Delta_rho
    sigma_x = sigma_ratio * sigma_y
    sigx2 = sigma_x**2
    sigy2 = sigma_y**2
    sigma_e2 = sigx2 * (1.0 - r)
    sigx2_mean = r * sigx2 +  sigma_e2/n
    sigma_x_mean = np.sqrt(sigx2_mean)
    cov_yxmean = R * sigma_x * sigma_y
    rho_mean = cov_yxmean /( sigma_x_mean * sigma_y)
    
    alpha = sigma_y/sigma_x_mean * ((n-1)*rho_mean + np.sqrt((n-1)**2.0 * rho_mean**2.0 + 4*n))/(2*n)

    beta2 = (sigy2 - alpha * rho_mean*sigma_x_mean*sigma_y)/sigma_e2
    return beta2

sigma_ratio = np.concatenate((np.arange(1/8, 1.0, 0.01),np.arange(1.0, 8.0+0.01,0.01)))

ff = FigureFunctions()

fig = plt.figure(num=2, figsize=(3, 5))
plt.clf()
gs = GS(3, 1, left=0.19, right=0.96, bottom=0.08, top=0.98, hspace=0.2)

for count_R, R in enumerate(R_all):
    # stddevrat_all = np.concatenate((np.arange(0.25, 0.95+0.01, 0.05),np.arange(1.0,8.0+0.2,0.25)))
    
    if R>0.6:
        delta_rho_max = 1 - R
        delta_rho_min = 0 - R/4
    elif R==0.6:
        delta_rho_max = 1 - R
        delta_rho_min = 0 - R        
    else:
        delta_rho_max = 1 - R
        delta_rho_min = 0 - R        
        
    delta_rho = np.linspace(delta_rho_min, delta_rho_max, 1000)
    
    
    DRHO, SR = np.meshgrid(delta_rho, sigma_ratio)
    
    delta_curr =  delta_func(DRHO, SR, R)   
    beta2_curr = beta2_func(DRHO, SR, R)
    alpha_curr = alpha_func(DRHO, SR, R)

    #######################################################
    ## spread-error (delta)
    #######################################################
    if R==R_all[0]:
        min_ = -6.0
        max_ = 1.0
        tick_values = np.around(np.arange(min_, max_+0.5, 1.0))
    elif R==R_all[1]:
        min_ = -6
        max_ = 6
        tick_values = np.around(np.arange(min_, max_+0.5, 2.0))
    elif R==R_all[2]:
        min_ = -6
        max_ = 6
        tick_values = np.around(np.arange(min_, max_+0.5, 2.0))      
        
        
    def plot_zero(ax, data):
        cs_zero = ax.contour(
            DRHO, SR, data,
            levels=np.array([0.0]), colors='0.5', linewidths=1.5, linestyles='-',
        )

    def plot_one(ax, data, ls=None):
        cs_zero = ax.contour(
            DRHO, SR, data,
            levels=np.array([1.0]), colors='k', linewidths=1.5, linestyles=ls,
        )

        
    clevs = np.linspace(min_, max_, 100) 
    cmap_orig = cm.bwr
    cmap = ff.truncate_colormap(cmap_orig, 0.1, 0.9)
    cmap = ff.truncate_div_cmap(cmap, frac_subtract_low=0.05, frac_subtract_high=0.05)
    cmap.set_over(cmap_orig(cmap_orig.N))
    cmap.set_under(cmap_orig(0))
    norm = mpl.colors.TwoSlopeNorm(vmin=min_, vcenter=0.0, vmax=max_)
    
    ax = fig.add_subplot(gs[count_R,0])
    im = ax.contourf(delta_rho, sigma_ratio, delta_curr, cmap=cmap, levels=clevs,
                     extend='both', norm=norm)

    # plot where spread change after calibration is opposite that 
    # indicated by spread-error  
    mask_pos = np.zeros(delta_curr.shape)
    mask_neg = np.zeros(delta_curr.shape)
    
    # mask_pos[(delta_curr > 0)&(beta2_curr > 1)] = 1.0
    mask_neg[(delta_curr < 0)&(beta2_curr < 1)] = 1.0

    h1 = ax.contourf(delta_rho, sigma_ratio, mask_neg, levels=[0.5, 1.5], 
                     hatches=['////'], colors='none')    
    
    # plot_one(ax, alpha_curr, ls='--')
    plot_one(ax, beta2_curr, ls='-')
    plot_zero(ax, delta_curr)
    
    ax.hlines(1.0, delta_rho.min(), delta_rho.max(), colors='k', lw=0.5)
    ax.vlines(0.0, sigma_ratio.min(), sigma_ratio.max(), colors='k', lw=0.5)

    ax.set_yscale("log")
    ax.set_yticks([0.15, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0])
    ax.get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    
    ax.text(0.99,0.01, f"$R={R}$", fontsize=9, ha='right', va='bottom', 
            transform=ax.transAxes)
    
    ax.set_xlim((delta_rho_min, delta_rho_max))
    ax.tick_params(labelsize=7)
    
    ax.set_ylabel(r'$\sigma_x/\sigma_y$', fontsize=10)
    
    if count_R==len(R_all)-1:
        ax.set_xlabel(r'$\Delta\rho=r-R$', fontsize=10)
 

    cbar = fig.colorbar(im, ax=ax, orientation='vertical', ticks=tick_values,
                        extend='both', pad=0.02)
    
    cbar.ax.tick_params(labelsize=7)
    cbar.ax.set_ylabel('spread-error', fontsize=9, labelpad=0.02)
