#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 10 18:13:51 2023

@author: ard000
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scores import scores_basic
from stat_funcs import round_nearest, mvnorm_rvs
from scipy.stats import norm
from matplotlib.gridspec import GridSpec as GS

# Number of experiments 
Nexp = int(1e5)

ntime_plot=50

# Ensemble size
Nens=50
Nbins = Nens+1  # for rank histogram

###############################################################################
########## Parameters for multivariate normal model
###############################################################################

np.random.seed(3)

bias = 0.0
stddev_rat = 2.0

# Climatological mean 
mu_y_clim = 0.0 # y represents truth
mu_x_clim = mu_y_clim + bias # forecasts

# climatological standard deviation
s_y_clim = 1.0 
s_x_clim = s_y_clim * stddev_rat

# covariance and correlation
R = 0.9
r = 0.9
cov_xy_mvn = R * s_x_clim * s_y_clim
cov_xx_mvn = r * s_x_clim**2.0

print("sigma_x^2", s_x_clim**2.0)
print("sigma_y^2", s_y_clim**2.0)
print("cov_xx: ", cov_xx_mvn)
print("cov_xy: ", cov_xy_mvn)
print(f"r={r}")
print(f"R={R}")

# ensemble spread
s_e_sq = s_x_clim**2.0 * (1 - r)

##############################################
############ MVN distribution ################   
# mean vector
mu_joint = np.concatenate((np.array([mu_y_clim]), np.array([mu_x_clim]*Nens))) 
  
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

scores = scores_basic(X, Y)
   
mse_f = scores.mse_fair
spread = scores.spread

# print("reliability component of CRPS:", scores.crps_rel)

#################################
# calibrate with QM
#################################
# Quantile map the forecasts
rv_x = norm(mu_x_clim, s_x_clim)
rv_y = norm(mu_y_clim, s_y_clim)
X_qm = np.zeros(X.shape)
for ii in range(Nexp):
    X_qm[ii] = rv_y.ppf(rv_x.cdf(X[ii]))
    
scores_qm = scores_basic(X_qm, Y)

#################################
# Additive inflation ensemble
#################################

noise = np.abs(scores.delta_tau_v)**0.5 * np.random.randn(Nexp,Nens)
noise -= noise.mean(axis=1)[:,np.newaxis]
X_add = X + noise
scores_add = scores_basic(X_add, Y)

##################################################################
plt.close()
fig = plt.figure(num=1, figsize=(10,5.5))
plt.clf()
gs1 = GS(3, 1, left=0.07, right=0.5, bottom=0.09, top=0.99, hspace=0.1)
gs2 = GS(3, 2, left=0.53, right=0.99, bottom=0.09, top=0.99, hspace=0.1)

def plot_timeseries(x, y, count):
    ### Plot timeseries
    
    ax = fig.add_subplot(gs1[count,0])
    
    xrange = np.arange(ntime_plot) + 1
    xrange_ticks = np.arange(0, ntime_plot+5, 5)
    ax.hlines(0.0, -1, ntime_plot+3, colors='k', ls='-', lw=0.5)
    
    ax.plot(xrange,x[:ntime_plot,:], ls='none', marker='+', color='k', alpha=0.2, markersize=5)
    ax.plot(xrange,x[:ntime_plot,:].mean(axis=1), ls='none', marker='o', color='b', markersize=4)
    ax.plot(xrange,y[:ntime_plot], ls='none', marker='*', color='r', markersize=6)
    ax.set_xticks(xrange_ticks, minor=True)
    ax.set_xticks(xrange_ticks[::2])
    if count<2:
        ax.set_xticklabels([])
    else:
        ax.set_xticklabels(xrange_ticks[::2])
    
    ax.set_xlim((-1, ntime_plot+3))
    ax.set_ylim((-6,6))
    ax.set_xlabel('valid time', fontsize=10)
    ax.set_ylabel('state', fontsize=10)
    
    ax.grid(ls=':', color='0.5')
    
    lines = [Line2D([], [], ls='none', marker='+', color='k', alpha=0.8, label='ens. members'),
             Line2D([], [], ls='none', marker='o', color='b', label='ens. mean'),
             Line2D([], [], ls='none', marker='*', color='r', label='truth')]
    
    if count==0:
        ax.legend(handles=lines, ncol=3, bbox_to_anchor=(0.5,1.02), loc='upper center',
                  fontsize=8)
    
    ax.text(0.98, 0.98, labels[count], fontsize=11, ha='right', va='top', transform=ax.transAxes)
    # ax.text(0.5, 0.01, label_spreaderror, fontsize=8, ha='center', va='bottom', transform=ax.transAxes)

def plot_diagnostics(sc, count, label_1, label_2):
    # spread-error and reliability budget
    ax = fig.add_subplot(gs2[count,0])
    labels = [r'$\delta_{\tau}$',
              r'$\delta_{\tau}^{\nu}$',
              r'$-(\Delta\mu)^2$',
              r'$\Delta \sigma^2$',
              r'$-2\Delta \Sigma$']  

    xrange = np.arange(len(labels))

    values = [sc.delta_tau, sc.delta_tau_v, 
              -1*(sc.margmean_x - sc.margmean_y)**2., 
              sc.margvar_x - sc.margvar_y, 
              -2*(sc.cov_xx - sc.cov_xy)]
        
    step = 2.0

    y_lim_min = -4
    y_lim_max = 4
        
    yticks = np.arange(y_lim_min, y_lim_max+step, step)
    yticks_minor = np.arange(y_lim_min, y_lim_max+step/5, step/5)

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    ax.bar(xrange, values, width=0.5, color=colors)
    x_lims = ax.get_xlim()
    ax.hlines(0.0, xmin=x_lims[0], xmax=x_lims[1], linestyle='--', color='k', lw=1)
    ax.set_xticks(xrange)
    ax.set_xlim(x_lims)
    if count<2:
        ax.set_xticklabels([])
    else:
        ax.set_xticklabels(labels, fontsize=10)
        
    ax.set_yticks(yticks_minor, minor=True)
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{tick:.0f}" for tick in yticks])
    ax.set_ylim((y_lim_min,y_lim_max))
    ax.grid(ls=':', color='0.5')

    ax.set_xlabel('spread-error & components', fontsize=10)

    ax.text(0.98, 0.98, label_1, fontsize=11, ha='right', va='top', transform=ax.transAxes)

    # Rank histogram
    ax = fig.add_subplot(gs2[count,1])
    ax.bar(np.arange(Nbins), sc.a_k/sc.E_k, width=1, edgecolor='purple',
            facecolor='thistle', lw=1.5)
    ax.hlines(1.0, -0.5, Nens+0.05, linestyles='--', colors='k')
    ax.set_xlabel('rank of $y_t$', fontsize=10)
    ax.set_ylabel('frequency', fontsize=10)
    ax.set_xticks(np.arange(0, Nens+10, 10))
    ax.set_xticks(np.arange(0, Nens+5, 5), minor=True)
    if count<2:
        ax.set_xticklabels([])
    else:
        ax.set_xticklabels(np.arange(0, Nens+10, 10), fontsize=10)    

    ax.set_xlim((-0.5,Nens+0.05))
    if count==0:
        ax.set_ylim((0.0,round_nearest(sc.a_k.max()/sc.E_k,0.5, up=True)))
    else:
        yticks = np.arange(0, 2.1, 1)
        ax.set_ylim((0.0, 2.0))
        ax.set_yticks(np.arange(0, 2.1, 1))
        ax.set_yticklabels([f"{tick:.0f}" for tick in yticks])
        
    ax.grid(ls=':', color='0.5')

    # ax.text(0.04, 0.99, r"$\mathrm{crps}="+f"${sc.crps:.2f} \n"+r"$\mathrm{crps_{rel}}="+f"${(sc.crps_rel*100):.2f}"+r"$\times10^{-2}$",
    #         va='top', ha='left', transform=ax.transAxes, fontsize=8)

    ax.text(0.98, 0.98, label_2, fontsize=11, ha='right', va='top', transform=ax.transAxes)


data_list = [X, X_qm, X_add]
scores_list = [scores, scores_qm, scores_add]

labels = ['(a)','(d)','(g)']

for ii in range(3):
    label_spreaderror = f"rms error={np.sqrt(scores_list[ii].mse):.2f}, rms spread={np.sqrt(scores_list[ii].spread):.2f}"
    plot_timeseries(data_list[ii], Y, ii)

### Plot diagnostics
labels1 = ['(b)', '(e)', '(h)']
labels2 = ['(c)', '(f)', '(i)']

for ii in range(3):
    plot_diagnostics(scores_list[ii],ii, labels1[ii], labels2[ii])





 


