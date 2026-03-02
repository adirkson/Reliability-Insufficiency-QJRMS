#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 10 18:13:51 2023

@author: ard000
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec as GS
from scipy.stats import norm

###############################################################################
np.random.seed(13)

# Number of experiments 
Nexp = int(1e4)
nplot = int(31)

# ensemble size
Nens = 50

# mariginal mean 
mu_x_clim = 0.0 # forecasts
mu_y_clim = 0.0 # y represents truth

# marginal stddev
s_x_clim = 1.0
s_y_clim = 1.0

# normal distributions for marginals
rv_x = norm(mu_x_clim, s_x_clim)
rv_y = norm(mu_y_clim, s_y_clim)

# correlation between members  
r_all = np.array([0.95, 0.7, 0.0]) 

# function for drawing random samples from MVN distribution
def mvnorm_rvs(mu, cov, A=None, size=1):
    '''
    A more efficient way of generating random samples from a multivariate
    normal distribution than what's done in scipy.stats.
    see:
    https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Computational_methods    
    
    '''
    M = size
    N = len(mu)
    if A is None:
        # cholesky decomposition of the covariance matrix
        A = np.linalg.cholesky(cov)
    
    # draw random sample from standard normal
    samp_uncorr = np.random.randn(N,M)
    
    # transform to MVN sample
    samp_corr = mu[:,np.newaxis] + np.dot(A,samp_uncorr)
    
    return samp_corr.T
  
# mean vector
mu_joint = np.concatenate((np.array([mu_y_clim]), np.array([mu_x_clim]*Nens))) 
# covariance matrix will be defined in loop below

############
# Plotting
############

# set up figure
fig = plt.figure(figsize=(11, 6))

# gridspec for 3 timeseries
gs2_xmin = 0.08
gs2_xmax = 0.56
gs2_ymin = 0.08
gs2_ymax = 0.93
gs2 = GS(3,1, figure=fig, left=gs2_xmin, right=gs2_xmax, top=gs2_ymax, bottom=gs2_ymin, hspace=0.05)

spacing = 0.04
left = gs2_xmax+spacing
right = 0.98
space_available = right - left - spacing

width = space_available/2.

gs3 = GS(3,1, figure=fig, left=gs2_xmax+spacing, right=gs2_xmax+spacing+width, top=gs2_ymax, bottom=gs2_ymin, 
         hspace=0.15, wspace=0.05)

gs4 = GS(3,1, figure=fig, left=gs2_xmax+2*spacing+width, right=right, top=gs2_ymax, bottom=gs2_ymin, 
         hspace=0.15, wspace=0.05)

# add labels to gridspecs
fig.text(gs2_xmin-0.03, gs2_ymax+0.02, '(b)', fontsize=14, ha='center', va='center')
fig.text(gs2_xmax+spacing-0.01, gs2_ymax+0.02, '(c)', fontsize=14, ha='center', va='center')
fig.text(gs2_xmax+2*spacing+width-0.02, gs2_ymax+0.02, '(d)', fontsize=14, ha='center', va='center')

# loop over the three levels of predictability and plot
for count_r, r in enumerate(r_all):
    np.random.seed(35)
    ax1 = fig.add_subplot(gs2[count_r,0])
    
    gs3_sp = gs3[count_r,0].subgridspec(2, 2, width_ratios=(5, 1), height_ratios=(1, 5),
                      wspace=0.01, hspace=0.001)
    
    ax2_scatt = fig.add_subplot(gs3_sp[1, 0])
    
    ax2_histx = fig.add_subplot(gs3_sp[0, 0], sharex=ax2_scatt)
    ax2_histy = fig.add_subplot(gs3_sp[1, 1], sharey=ax2_scatt)

    gs4_sp = gs4[count_r,0].subgridspec(2, 2, width_ratios=(5, 1), height_ratios=(1, 5),
                      wspace=0.01, hspace=0.001)

    ax3_scatt = fig.add_subplot(gs4_sp[1, 0])
    ax3_histx = fig.add_subplot(gs4_sp[0, 0], sharex=ax2_scatt)
    ax3_histy = fig.add_subplot(gs4_sp[1, 1], sharey=ax2_scatt)

    R = r
    ###########################
    #### covariance matrix ####
    ###########################      
    # diagnoal variance matrix
    var_joint = s_x_clim**2. * np.eye(Nens+1)
    var_joint[0,0] = s_y_clim**2.
    
    # correlation matrix
    corr_joint = r*np.ones((Nens+1, Nens+1))
    corr_joint[1:,0] = R
    corr_joint[0,1:] = R
    np.fill_diagonal(corr_joint, 1.0)
    
    # covariance
    cov_joint = var_joint**0.5 @ corr_joint @ var_joint**0.5
    
    ####### Draw Nexp random samples from joint model
    joint_samp = mvnorm_rvs(mu_joint, cov_joint, size=Nexp)

    Y = joint_samp[:,0] # truth (Nexp)
    X = joint_samp[:,1:] # model ensemble (Nexp by Nens)
    
    # statistics for seperate grid points
    ensvar = X.var(axis=-1, ddof=1).mean(axis=0)
    mse = np.mean((X.mean(axis=-1) - Y)**2., axis=0)
    mse_f = mse - ensvar/Nens
    
    # mean of marginal mean of X
    X_mean = np.mean(X)
    # mean of marginal variance of X
    X_var = np.var(X,axis=0,ddof=1).mean()
    # print("marginal variance X: ", X_var)
    # marginal mean of Y
    Y_mean = np.mean(Y)
    # marginal variance of Y
    Y_var = np.var(Y,ddof=1)
    
    # anomalies relative to mean taken along dimension Nexp
    X_anom = X - X.mean(axis=0)
    Y_anom = Y - Y.mean(axis=0)
    
    # covariance between each ensemble member in X and Y
    cov_xy = np.mean(np.dot(X_anom.T, Y_anom)/Nexp)
    
    # covariance between each member with another member
    cov_xx_full = np.dot(X_anom.T, X_anom)/Nexp
    cov_xx_triu = np.triu(cov_xx_full, k=1)
    cov_xx = np.sum(cov_xx_triu)/((Nens**2.-Nens)/2)
    
    r_samp = cov_xx/X_var
    R_samp = cov_xy/(X_var**0.5*Y_var**0.5)


    xrange = np.arange(nplot)

    # special xticklabels
    nlab = 6
    nT = 3
    nblank = int((nplot-1)/2 - nlab - nT)
    #######################
    # Time series 
    #######################    
    # inds_plot = np.random.choice(np.arange(Nexp), size=nplot, replace=False)
    inds_plot = np.arange(nplot)
    xrange_long = np.arange(-1, nplot+1, 1)
    X_plot = X[inds_plot]
    Y_plot = Y[inds_plot]

    ax1.plot(xrange_long, mu_y_clim*np.ones(len(xrange_long)),  linestyle='--', color='k', lw=1)
    ax1.plot(xrange, X_plot, linestyle='none', marker='+', color='c', ms=6, mew=1, alpha=0.5)
    ax1.plot(xrange, X_plot.mean(axis=1), linestyle='none', marker='o', color='b', ms=2, mew=2)
    ax1.plot(xrange, Y_plot,  linestyle='none', marker='*', color='r', ms=6)

    ax1.grid(linestyle='--', color='0.5', alpha=0.5)
    
    ax1.set_ylim((-4,4))
    ax1.set_xticks(xrange[::2])
    ax1.set_xticks(xrange, minor=True)
    ax1.set_xlim((xrange_long.min(),xrange_long.max()))
    
    l1 = Line2D([], [], marker='+', ls='none', ms=8, mew=2, color='c')
    l2 = Line2D([], [], marker='o', ls='none', ms=5, mew=2, color='b')
    l3 = Line2D([], [], marker='*', ls='none', ms=10, color='r')
    ax_pos = ax1.get_position()
    mp = (ax_pos.ymin + ax_pos.ymax)/2
    if count_r==0:
        fig.text(0.02, mp, 'high \npredictability', fontsize=12, ha='center', va='center', 
                 rotation=90, fontweight='semibold')
        ax1.set_xticklabels([])
        ax1.legend(handles=[l1,l2,l3],labels=['ens. members','ens. mean','truth'], 
                       ncol=3, loc='lower center', 
                       bbox_to_anchor=(0.5,1.0), fontsize=9)
        
    elif count_r==1:
        fig.text(0.02, mp, 'moderate \npredictability', fontsize=12, ha='center', va='center', 
                 rotation=90, fontweight='semibold')       
        ax1.set_xticklabels([])
        
    elif count_r==2:        
        fig.text(0.02, mp, 'no \npredictability', fontsize=12, ha='center', va='center', 
                 rotation=90, fontweight='semibold')
        
        ax1.set_xlabel('valid time', fontsize=11)
        xticklabels = np.arange(0, nplot-1+0.1,2).astype(int)
        ax1.set_xticklabels(xticklabels, fontsize=10)

    ax1.set_ylabel('state', fontsize=11, labelpad=0.01)
          
    ########################
    # scatter plots between x1 and x2
    ########################
    fs = 11
    # Draw horizontal arrows above the subplot
    nbins=20

    # the scatter plot:
    ticks = [-4, 0, 4]
    ax2_scatt.scatter(X[:,0], X[:,1], color='0.5', s=5, marker='.', alpha=0.1)
    ax2_scatt.set_xticks(ticks)
    ax2_scatt.set_yticks(ticks)
    ax2_scatt.set_xticklabels(ticks)
    ax2_scatt.set_yticklabels(ticks)
    ax2_scatt.set_xlim((-5, 5))
    ax2_scatt.set_ylim((-5, 5))

    if count_r==len(r_all)-1:
        ax2_scatt.set_xlabel('$X_1$', labelpad=1)
    ax2_scatt.set_ylabel('$X_2$', labelpad=1)
    
    ax2_scatt.grid(linestyle='--', color='0.5', alpha=0.5)

    # marginal distributions on spines
    ax2_histx.hist(X[:,0], bins=nbins, density=True, color='c', alpha=0.5)
    ax2_histy.hist(X[:,1], bins=nbins, density=True, color='c', 
                   orientation='horizontal', alpha=0.5)
    
    ax2_histx.axes.set_axis_off()
    ax2_histy.axes.set_axis_off()
    
    ax2_histx.text(0.5, 0.02, '$\mu_x, ~\sigma_x^2$', fontsize=8, ha='center', va='bottom',
                   transform=ax2_histx.transAxes)
    

    ax2_histy.text(0.01, 0.5, '$\mu_x, ~\sigma_x^2$', fontsize=8, ha='left', va='center',
                   transform=ax2_histy.transAxes, rotation=-90)
    
    if r==0.95:
        ax2_scatt.text(0.01, 0.98, f"$r=${r:0.2f}", ha='left',va='top',
                       transform=ax2_scatt.transAxes, fontsize=9)
    else:
        ax2_scatt.text(0.01, 0.98, f"$r=${r:0.1f}", ha='left',va='top',
                       transform=ax2_scatt.transAxes, fontsize=9)            

    ########################
    # scatter plots between x1 and y
    ########################
    # the scatter plot:
    ax3_scatt.scatter(X[:,0], Y[:], color='0.5', s=5, marker='.', alpha=0.1)
    ax3_scatt.set_xticks(ticks)
    ax3_scatt.set_yticks(ticks)
    ax3_scatt.set_xticklabels(ticks)
    ax3_scatt.set_yticklabels(ticks)
    ax3_scatt.set_xlim((-5, 5))
    ax3_scatt.set_ylim((-5, 5))
    if r==0:
        ax3_scatt.set_xlabel('$X_1$', labelpad=1)
    ax3_scatt.set_ylabel('$Y$', labelpad=1)

    ax3_scatt.grid(linestyle='--', color='0.5', alpha=0.5)

    if r==0.95:
        ax3_scatt.text(0.01, 0.98, f"$R=${R:0.2f}", ha='left',va='top',
                       transform=ax3_scatt.transAxes, fontsize=9)
    else:
        ax3_scatt.text(0.01, 0.98, f"$R=${R:0.1f}", ha='left',va='top',
                       transform=ax3_scatt.transAxes, fontsize=9)        

    # marginal distributions on spines    
    ax3_histx.hist(X[:,0], bins=nbins, color='c', alpha=0.5)
    ax3_histy.hist(Y, bins=nbins, color='r', orientation='horizontal', alpha=0.5)

    ax3_histx.axes.set_axis_off()
    ax3_histy.axes.set_axis_off()

    ax3_histx.text(0.5, 0.02, '$\mu_x, ~\sigma_x^2$', fontsize=8, ha='center', va='bottom',
                   transform=ax3_histx.transAxes)
    

    ax3_histy.text(0.01, 0.5, '$\mu_y, ~\sigma_y^2$', fontsize=8, ha='left', va='center',
                   transform=ax3_histy.transAxes, rotation=-90)
