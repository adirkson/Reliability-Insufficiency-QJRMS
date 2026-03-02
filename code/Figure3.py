#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 10 18:13:51 2023

@author: ard000
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scores import scores_basic, scores_secondary
from stat_funcs import round_nearest, mvnorm_rvs
from matplotlib.gridspec import GridSpec as GS

##### User-set parameters for this toy problem are set in section 1 
###############################################################################

# Number of experiments 
Nexp = int(1e5)

ntime_plot=75

# Ensemble size
Nens=50
Nbins = Nens+1  # for rank histogram

###############################################################################
########## Parameters for multivariate normal model
##############################################################################
np.random.seed(3)

bias = 0.0
vardiff = 3.0
covdiff = -1 * vardiff

# Climatological mean 
mu_y_clim = 0.0 # y represents truth
mu_x_clim = mu_y_clim + bias # forecasts

# climatological standard deviation
s_y_clim = 1.0 
s_x_clim = np.sqrt(s_y_clim**2.0 + vardiff)

# correlations and covariances
R = 0.9
cov_xy_mvn = R*s_x_clim*s_y_clim
cov_xx_mvn = (2*cov_xy_mvn - covdiff)/2
r = cov_xx_mvn/s_x_clim**2.0

print("sigma_x^2", s_x_clim**2.0)
print("sigma_y^2", s_y_clim**2.0)
print("cov_xx: ", cov_xx_mvn)
print("cov_xy: ", cov_xy_mvn)
print(f"r={r}")
print(f"R={R}")

# ensemble spread
s_e_sq = s_x_clim**2.0 * (1 - r)

##############################################
############ Build MVN distribution ################   
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

# compute basic scores
scores = scores_basic(X, Y)

# instantiate secondary scores class
scores_sec = scores_secondary()

##################################################################

fig = plt.figure(num=1, figsize=(10,5.5))
plt.clf()
gs1 = GS(1, 1, left=0.06, right=0.99, bottom=0.60, top=0.98)
gs2 = GS(1, 3, left=0.06, right=0.99, bottom=0.09, top=0.5, wspace=0.25)


### Plot timeseries

ax = fig.add_subplot(gs1[0,0])

xrange = np.arange(ntime_plot) + 1
xrange_ticks = np.arange(0, ntime_plot+5, 5)
ax.hlines(0.0, -1, ntime_plot+3, colors='k', ls='-', lw=0.5)

ax.plot(xrange,X[:ntime_plot,:], ls='none', marker='+', color='k', alpha=0.2, markersize=5)
ax.plot(xrange,X[:ntime_plot,:].mean(axis=1), ls='none', marker='o', color='b', markersize=4)
ax.plot(xrange,Y[:ntime_plot], ls='none', marker='*', color='r', markersize=6)
ax.set_xticks(xrange_ticks, minor=True)
ax.set_xticks(xrange_ticks[::2])
ax.set_xticklabels(xrange_ticks[::2])

ax.set_xlim((-1, ntime_plot+3))
ax.set_ylim((-6,6))
ax.set_xlabel('valid time', fontsize=11)
ax.set_ylabel('state', fontsize=11)

ax.grid(ls=':', color='0.5')

lines = [Line2D([], [], ls='none', marker='+', color='k', alpha=0.8, label='ens. members'),
         Line2D([], [], ls='none', marker='o', color='b', label='ens. mean'),
         Line2D([], [], ls='none', marker='*', color='r', label='truth')]

ax.legend(handles=lines, ncol=3, bbox_to_anchor=(0.5,1.02), loc='upper center',
          fontsize=8)

ax_pos = ax.get_position()
fig.text(ax_pos.xmin-0.03, ax_pos.ymax-0.005, '(a)', fontsize=12, ha='right', va='center')


###################################################################
### Plot diagnostics
###################################################################

# spread-error and reliability budget
ax = fig.add_subplot(gs2[0,0])
labels = [r'$\delta_{\tau}$',
          r'$\delta_{\tau}^{\nu}$',
          r'$-(\Delta\mu)^2$',
          r'$\Delta \sigma^2$',
          r'$-2\Delta \Sigma$']  

xrange = np.arange(len(labels))

values = [scores.delta_tau, scores.delta_tau_v, 
          -1*(mu_x_clim - mu_y_clim)**2., 
          s_x_clim**2.0 - s_y_clim**2.0, 
          -2*(cov_xx_mvn - cov_xy_mvn)]
    
step = 1.0

y_lim_max = max(np.max(values), step)    
y_lim_max_try = round_nearest(y_lim_max, step)
if y_lim_max_try<y_lim_max:
    y_lim_max = round_nearest(y_lim_max, step) + step
else:
    y_lim_max = y_lim_max_try

y_lim_min = min(np.min(values), -1*step)
if y_lim_min<0:
    y_lim_min = round_nearest(y_lim_min, step, down=True) 
else:
    y_lim_min = round_nearest(y_lim_min, step)   
    
yticks = np.arange(y_lim_min, y_lim_max+step, step)
yticks_minor = np.arange(y_lim_min, y_lim_max+step/5, step/5)

colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
#colors = ['orange','c', 'b', 'r', 'g','m']

ax.bar(xrange, values, width=0.5, color=colors)
x_lims = ax.get_xlim()
ax.hlines(0.0, xmin=x_lims[0], xmax=x_lims[1], linestyle='--', color='k', lw=1)
ax.set_xticks(xrange)
ax.set_xlim(x_lims)
ax.set_xticklabels(labels, fontsize=10)
ax.set_yticks(yticks_minor, minor=True)
ax.set_yticks(yticks)
ax.set_yticklabels([f"{tick:.2f}" for tick in yticks])
ax.set_ylim((y_lim_min-step/5,y_lim_max+step/5))
ax.grid(ls=':', color='0.5')

ax.set_xlabel('spread-error & components')

ax_pos = ax.get_position()
fig.text(ax_pos.xmin, ax_pos.ymax+0.005, '(b)', fontsize=12, ha='center', va='bottom')

# Rank histogram
ax = fig.add_subplot(gs2[0,1])
ax.bar(np.arange(Nbins), scores.a_k/scores.E_k, width=1, edgecolor='purple',
        facecolor='thistle', lw=1.5)
ax.hlines(1.0, -0.5, Nens+0.05, linestyles='--', colors='k')
ax.set_xlabel('rank of $y_t$', fontsize=10)
ax.set_ylabel('frequency', fontsize=10)
ax.set_xticks(np.arange(0, Nens+10, 10))
ax.set_xticks(np.arange(0, Nens+5, 5), minor=True)

ax.set_xlim((-0.5,Nens+0.05))
ax.set_ylim((0.0,round_nearest(scores.a_k.max()/scores.E_k,0.5, up=True)))
ax.grid(ls=':', color='0.5')

ax_pos = ax.get_position()
fig.text(ax_pos.xmin, ax_pos.ymax+0.005, '(c)', fontsize=12, ha='center', va='bottom')

###################################################################
# Reliability diagram

events = [[50, 50], [1/3*100, 2/3*100], [5, 95]]
events_str = ['$q_{0.5}$', '$q_{0.6\overline{6}}$', '$q_{0.95}$']

ls_all = ['-', '--', ':']
markers_all = ['o', 's', 'd']

ax = fig.add_subplot(gs2[0,2])
for count, event in enumerate(events):
    clim_percentile = np.percentile(Y, event)
    
    p_k, p_fcst_bn, p_fcst_nn, p_fcst_an, p_obs_bn, p_obs_nn, p_obs_an = scores_sec.get_fcst_probs(X, Y, clim_percentile)

    bs_o_k_an, bs_n_k_an, bs_rel_an, bs_res_an, bs_unc_an, bs_p_hat_an, bs_o_hat_an = scores_sec.brier_decomp(p_fcst_an, p_obs_an, p_k)

    prop=300
    s_k_an = prop*bs_n_k_an/np.nansum(bs_n_k_an)
    
           
    ax.plot(np.linspace(-0.05,1.05,100), np.linspace(-0.05,1.05,100),
              'k-', lw=1)
  
    ax.plot(p_k, bs_o_k_an, color='r', ls=ls_all[count])
    ax.scatter(p_k, bs_o_k_an, s=s_k_an, edgecolor='r', c='lightpink', marker=markers_all[count])
    
    ax.set_xticks(np.arange(0, 1.0 + 0.2, 0.2))
    ax.set_yticks(np.arange(0, 1.0 + 0.2, 0.2))
    ax.set_xlim((-0.05,1.05))
    ax.set_ylim((-0.05,1.05))
    ax.set_xlabel('$p_k$', fontsize=10)
    ax.set_ylabel('$\overline{o}_k$', fontsize=10)
    ax.grid(ls=':', color='0.5')
    
# make legend

legend_handles = [
    Line2D([0], [0], color='r', marker=markers_all[0], linestyle=ls_all[0], label=events_str[0], mfc='lightpink'),
    Line2D([0], [0], color='r', marker=markers_all[1], linestyle=ls_all[1], label=events_str[1], mfc='lightpink'),
    Line2D([0], [0], color='r', marker=markers_all[2], linestyle=ls_all[2], label=events_str[2], mfc='lightpink'),
]

ax.legend(handles=legend_handles, fontsize=9)

ax_pos = ax.get_position()
fig.text(ax_pos.xmin, ax_pos.ymax+0.005, '(d)', fontsize=12, ha='center', va='bottom')
