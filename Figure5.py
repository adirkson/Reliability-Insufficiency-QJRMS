#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 10 18:13:51 2023

@author: ard000
"""
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator
from matplotlib.colors import PowerNorm

from scores import scores_basic, scores_secondary
from stat_funcs import mvnorm_rvs
from matplotlib.gridspec import GridSpec as GS
import matplotlib.cm as cm
from figure_functions import FigureFunctions
import concurrent.futures

scores_sec = scores_secondary()

# Number of experiments 
Nexp = int(1e5)

# Ensemble size
Nens=50

Nbins = Nens+1  # for rank histogram

# for the reliability diagram
events = [[50, 50], [1/3*100, 2/3*100], [5, 95]]

R = 0.8

###############################################################################
########## Parameters for multivariate normal model
###############################################################################
np.random.seed(1)

# setup standard deviation ratios and correlation differences 
## CAUTION: parallelization will be done over stddevrat_all, so ensure there are
## at least as many cpu's available as the length of this array
stddevrat_all = np.concatenate((np.arange(0.25, 1.0+0.01, 0.05),np.arange(1.1,4.0+0.05,0.1)))
corrdiff_all = np.arange(-0.15, 0.15+0.01, 0.01)
nstddev = len(stddevrat_all)
ncorr = len(corrdiff_all)

ncpus = nstddev
while True:
    response = input(f"Ensure you have enough total CPU's (at least ncpus = {ncpus}) available; continue? [y/n]: ").strip().lower()
    if response in ("y", "yes"):
        approved = True
        break
    elif response in ("n", "no"):
        approved = False
        break
    else:
        print("Please enter 'y' or 'n'.")

# diagnostics for raw forecasts
sumvar2cov = np.full((nstddev, ncorr), np.nan)
crps_rel = np.full((nstddev, ncorr), np.nan)
bs_rel_q1 = np.full((nstddev, ncorr), np.nan)
bs_rel_q2 = np.full((nstddev, ncorr), np.nan)
bs_rel_q3 = np.full((nstddev, ncorr), np.nan)
chi_sq_stat = np.full((nstddev, ncorr), np.nan)
chi_sq_pval = np.full((nstddev, ncorr), np.nan)
spread = np.full((nstddev, ncorr), np.nan)
mse_f = np.full((nstddev, ncorr), np.nan)
spread_error = np.full((nstddev, ncorr), np.nan)

# diagnostics for mmc forecasts
crps_rel_mmc = np.full((nstddev, ncorr), np.nan)
bs_rel_q1_mmc = np.full((nstddev, ncorr), np.nan)
bs_rel_q2_mmc = np.full((nstddev, ncorr), np.nan)
bs_rel_q3_mmc = np.full((nstddev, ncorr), np.nan)
chi_sq_stat_mmc = np.full((nstddev, ncorr), np.nan)
chi_sq_pval_mmc = np.full((nstddev, ncorr), np.nan)
spread_mmc = np.full((nstddev, ncorr), np.nan)
mse_f_mmc = np.full((nstddev, ncorr), np.nan)
spread_error_mmc = np.full((nstddev, ncorr), np.nan)


def loop_stddev(stddevrat):
    sumvar2cov_curr = np.full(ncorr, np.nan)
    crps_rel_curr = np.full(ncorr, np.nan)
    bs_rel_q1_curr = np.full(ncorr, np.nan)
    bs_rel_q2_curr = np.full(ncorr, np.nan)
    bs_rel_q3_curr = np.full(ncorr, np.nan)
    chi_sq_stat_curr = np.full(ncorr, np.nan)
    chi_sq_pval_curr = np.full(ncorr, np.nan)
    spread_curr = np.full(ncorr, np.nan)
    mse_f_curr = np.full(ncorr, np.nan)
    spread_error_curr = np.full(ncorr, np.nan)

    # diagnostics for mmc forecasts
    crps_rel_mmc_curr = np.full(ncorr, np.nan)
    bs_rel_q1_mmc_curr = np.full(ncorr, np.nan)
    bs_rel_q2_mmc_curr = np.full(ncorr, np.nan)
    bs_rel_q3_mmc_curr = np.full(ncorr, np.nan)
    chi_sq_stat_mmc_curr = np.full(ncorr, np.nan)
    chi_sq_pval_mmc_curr = np.full(ncorr, np.nan)
    spread_mmc_curr = np.full(ncorr, np.nan)
    mse_f_mmc_curr = np.full(ncorr, np.nan)
    spread_error_mmc_curr = np.full(ncorr, np.nan)
        
    for count_corr, corrdiff in enumerate(corrdiff_all):
        # Climatological mean 
        mu_y_clim = 0.0 # y represents truth
        mu_x_clim = mu_y_clim # forecasts
        
        # climatological standard deviation
        s_y_clim = 1.0 
        s_x_clim = stddevrat*s_y_clim
        
        r = corrdiff + R
        
        cov_xx_mvn = r*s_x_clim**2.0
        cov_xy_mvn = R*s_x_clim*s_y_clim
        
        sumvar2cov_curr[count_corr] = (s_x_clim**2.0 - s_y_clim**2.0) + 2*(cov_xy_mvn - cov_xx_mvn)
        
        # ensemble spread
        s_e_sq = s_x_clim**2.0 * (1 - r)
        
        # print(f"s_x^2={s_x_clim**2}")
        # print(f"s_y^2={s_y_clim**2}")

        # print(f"r={r}")
        # print(f"R={R}")
        
        # print(f"spread={s_e_sq}")
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
            
        try:
            ####### Draw Nexp random samples from joint model
            joint_samp = mvnorm_rvs(mu_joint, cov_joint, size=Nexp)            
            Y = joint_samp[:,0] # truth (Nexp)
            X_orig = joint_samp[:,1:] # model ensemble (Nexp by Nens)
            
            scores_orig = scores_basic(X_orig, Y)
            
            # store scores for original/raw forecasts
            crps_rel_curr[count_corr] = scores_orig.crps_rel
    
            bs_rel_an_list = []
            for count, event in enumerate(events):
                clim_percentile = np.percentile(Y, event)              
                p_k, p_fcst_bn, p_fcst_nn, p_fcst_an, p_obs_bn, p_obs_nn, p_obs_an = scores_sec.get_fcst_probs(X_orig, Y, clim_percentile)           
                bs_o_k_an, bs_n_k_an, bs_rel_an, bs_res_an, bs_unc_an, bs_p_hat_an, bs_o_hat_an = scores_sec.brier_decomp(p_fcst_an, p_obs_an, p_k, n_thresh=50)
                bs_rel_an_list.append(bs_rel_an)
            
            bs_rel_q1_curr[count_corr] = bs_rel_an_list[0]
            bs_rel_q2_curr[count_corr] = bs_rel_an_list[1]
            bs_rel_q3_curr[count_corr] = bs_rel_an_list[2]    
    
            chi_sq_stat_curr[count_corr] = scores_orig.chi_sq_stat
            chi_sq_pval_curr[count_corr] = scores_orig.chi_sq_pval
            
            spread_error_curr[count_corr] = scores_orig.D_se
            mse_f_curr[count_corr]= scores_orig.mse_fair
            spread_curr[count_corr] = scores_orig.spread

        except:
            print("matrix non semi definite")
                
    return (sumvar2cov_curr, crps_rel_curr, bs_rel_q1_curr, bs_rel_q2_curr, bs_rel_q3_curr,
            chi_sq_stat_curr, chi_sq_pval_curr, spread_error_curr, mse_f_curr, spread_curr,
            crps_rel_mmc_curr, bs_rel_q1_mmc_curr, bs_rel_q2_mmc_curr, bs_rel_q3_mmc_curr,
            chi_sq_stat_mmc_curr, chi_sq_pval_mmc_curr, spread_error_mmc_curr, mse_f_mmc_curr, spread_mmc_curr)

with concurrent.futures.ProcessPoolExecutor(ncpus) as executor:
    results = executor.map(loop_stddev, stddevrat_all)

    

for count, result in enumerate(results):
    (sumvar2cov[count], crps_rel[count], bs_rel_q1[count], bs_rel_q2[count], bs_rel_q3[count],
            chi_sq_stat[count], chi_sq_pval[count], spread_error[count], mse_f[count], spread[count],
            crps_rel_mmc[count], bs_rel_q1_mmc[count], bs_rel_q2_mmc[count], bs_rel_q3_mmc[count],
            chi_sq_stat_mmc[count], chi_sq_pval_mmc[count], spread_error_mmc[count], mse_f_mmc[count], spread_mmc[count]) = result


def plot_contours(ax):
    # Define contour levels
    levels = np.array([-6, -4, -2, -1, 0, 0.25, 0.5, 0.75, 1])
    neg_levels = levels[levels < 0]
    pos_levels = levels[levels > 0]
    zero_level = levels[np.isclose(levels, 0)]
    
    # Plot negative contours with dashed lines
    if len(neg_levels) > 0:
        cs_neg = ax.contour(
            corrdiff_all, stddevrat_all, sumvar2cov,
            levels=neg_levels, colors='black', linestyles='dashed'
        )
        ax.clabel(cs_neg, fmt="%.2f", fontsize=6)
    
    # Plot positive contours with solid lines
    if len(pos_levels) > 0:
        cs_pos = ax.contour(
            corrdiff_all, stddevrat_all, sumvar2cov,
            levels=pos_levels, colors='black', linestyles='solid'
        )
        ax.clabel(cs_pos, fmt="%.2f", fontsize=6)
    
    # Optional: Plot zero contour with a thick gray line
    if len(zero_level) > 0:
        cs_zero = ax.contour(
            corrdiff_all, stddevrat_all, sumvar2cov,
            levels=zero_level, colors='gray', linewidths=1.5
        )
        ax.clabel(cs_zero, fmt="%.2f", fontsize=6)

def plot_zero(ax, data):
    cs_zero = ax.contour(
        corrdiff_all, stddevrat_all, data,
        levels=np.array([0.0]), colors='r', linewidths=1.5, ls=':',
    )
    ax.clabel(cs_zero, fmt="%.2f", fontsize=8)


ff = FigureFunctions()

fig = plt.figure(num=2, figsize=(10, 5))
plt.clf()
gs = GS(2, 3, left=0.05, right=0.98, bottom=0.08, top=0.95, wspace=0.15, hspace=0.25)

y_ticks = [0.3, 0.5, 0.6, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0]
#######################################################
## spread-error
#######################################################
min_ = -3.
max_ = 1.
clevs = np.linspace(min_, max_, 101)
cmap_orig = cm.bwr
cmap = ff.truncate_colormap(cmap_orig, 0.1, 0.9)
cmap = ff.truncate_div_cmap(cmap, frac_subtract_low=0.04, frac_subtract_high=0.0)
cmap.set_over(cmap_orig(cmap_orig.N))
cmap.set_under(cmap_orig(0))
norm = mpl.colors.TwoSlopeNorm(vmin=min_, vcenter=0.0, vmax=max_)
    
ax = fig.add_subplot(gs[0,0])
im = ax.contourf(corrdiff_all, stddevrat_all, spread_error, cmap=cmap, levels=clevs,
                 extend='both', norm=norm)

ax.hlines(1.0, corrdiff_all.min(), corrdiff_all.max(), colors='k', lw=0.5)
ax.vlines(0.0, stddevrat_all.min(), stddevrat_all.max(), colors='k', lw=0.5)
plot_contours(ax)

ax.set_yscale("log")
ax.set_yticks(y_ticks)

ax.get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter())

ax.set_title('spread-error', fontsize=10)
ax.set_xlim((corrdiff_all.min(), corrdiff_all.max()))
ax.tick_params(labelsize=8)

ax.set_ylabel(r'$\sigma_x/\sigma_y$', fontsize=10)

cbar = fig.colorbar(im, ax=ax, orientation='vertical', ticks=[-3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1],
                    extend='both')

cbar.ax.tick_params(labelsize=8)

ax.text(0.0, 1.01, '(a)', ha='center', va='bottom', transform=ax.transAxes,
        fontsize=10)


#######################################################
## chi sq p-value colormap (emphasize small p-values)
#######################################################

alpha = 0.05
chi_sq_critical = chi_sq_stat[chi_sq_pval>alpha].max()

min_ = chi_sq_critical
max_ = 1e7
clevs = np.logspace(np.log10(min_), np.log10(max_), 100)

cmap_orig = cm.BuPu
cmap = ff.truncate_colormap(cmap_orig, 0.15, 0.9)
cmap.set_under('w')
cmap.set_over(cmap_orig(cmap_orig.N))
norm = mpl.colors.LogNorm(vmin=min_, vmax=max_)

ax = fig.add_subplot(gs[0,1])
im = ax.contourf(
    corrdiff_all, stddevrat_all, chi_sq_stat,
    cmap=cmap, levels=clevs, norm=norm, extend='both'
)

ax.hlines(1.0, corrdiff_all.min(), corrdiff_all.max(), colors='k', lw=0.5)
ax.vlines(0.0, stddevrat_all.min(), stddevrat_all.max(), colors='k', lw=0.5)
plot_contours(ax)

ax.set_yscale("log")
ax.set_yticks(y_ticks)
ax.get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
ax.set_title(r'$\chi^2$ (rank histogram)', fontsize=10)
ax.set_xlim((corrdiff_all.min(), corrdiff_all.max()))
ax.tick_params(labelsize=8)

# Custom ticks
tick_values = [1, 100, 1e3, 1e4, 1e5, 1e6, 1e7]
cbar = fig.colorbar(im, ax=ax, orientation='vertical', ticks=tick_values)
cbar.ax.tick_params(labelsize=8)
cbar.ax.minorticks_off()

ax.text(0.0, 1.01, '(b)', ha='center', va='bottom', transform=ax.transAxes,
        fontsize=10)

#######################################################
## crps rel
#######################################################
gamma = 0.5

min_ = 0.0
max_ = 20
clevs = np.linspace(min_, max_, 100)

tick_values = [0, 5, 10, 15, 20]

cmap_orig = cm.BuPu
cmap = ff.truncate_colormap(cmap_orig, 0.05, 0.9)
cmap.set_over(cmap_orig(cmap_orig.N))
cmap.set_under('w')
norm = PowerNorm(gamma=gamma, vmin=clevs[1], vmax=max_) 

ax = fig.add_subplot(gs[0,2])
im = ax.contourf(corrdiff_all, stddevrat_all, crps_rel*100, cmap=cmap, norm=norm,
                   levels=clevs, extend='max')

ax.hlines(1.0, corrdiff_all.min(), corrdiff_all.max(), colors='k', lw=0.5)
ax.vlines(0.0, stddevrat_all.min(), stddevrat_all.max(), colors='k', lw=0.5)
plot_contours(ax)

ax.set_yscale("log")
ax.set_yticks(y_ticks)
ax.get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
ax.set_title(r'$\mathrm{CRPS_{rel}}\times 10^{2}$', fontsize=10)
ax.set_xlim((corrdiff_all.min(), corrdiff_all.max()))
ax.tick_params(labelsize=8)

cbar = fig.colorbar(im, ax=ax, orientation='vertical', ticks=tick_values)
cbar.ax.tick_params(labelsize=8)
cbar.ax.minorticks_off()

ax.text(0.0, 1.01, '(c)', ha='center', va='bottom', transform=ax.transAxes,
        fontsize=10)


#######################################################
## brier rel
#######################################################
####################

min_ = 0.0
max_ = 3
clevs = np.linspace(min_, max_, 100)
tick_values = [0, 1, 2, 3]

cmap_orig = cm.BuPu
cmap = ff.truncate_colormap(cmap_orig, 0.05, 0.9)
cmap.set_over(cmap_orig(cmap_orig.N))
cmap.set_under('w')
norm = PowerNorm(gamma=gamma, vmin=clevs[1], vmax=max_) 

ax = fig.add_subplot(gs[1,0])
im = ax.contourf(corrdiff_all, stddevrat_all, bs_rel_q1*100, cmap=cmap, norm=norm,
                 levels=clevs, extend='max')

ax.hlines(1.0, corrdiff_all.min(), corrdiff_all.max(), colors='k', lw=0.5)
ax.vlines(0.0, stddevrat_all.min(), stddevrat_all.max(), colors='k', lw=0.5)
plot_contours(ax)

ax.set_yscale("log")
ax.set_yticks(y_ticks)
ax.get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
ax.set_xlabel(r'$\Delta\rho=r-R$')
ax.set_title(r'$\mathrm{BS_{rel}}\times 10^{2}$: $q_{0.5}$', fontsize=10)
ax.set_xlim((corrdiff_all.min(), corrdiff_all.max()))
ax.tick_params(labelsize=8)
ax.set_ylabel(r'$\sigma_x/\sigma_y$', fontsize=10)

cbar1 = fig.colorbar(im, ax=ax, orientation='vertical', ticks=tick_values)
cbar1.set_ticks(tick_values)
cbar1.ax.yaxis.set_major_locator(FixedLocator(tick_values))
cbar1.ax.tick_params(labelsize=8)
cbar1.ax.minorticks_off()

ax.text(0.0, 1.01, '(d)', ha='center', va='bottom', transform=ax.transAxes,
        fontsize=10)


####################
min_ = 0.0
max_ = 15
clevs = np.linspace(min_, max_, 100)
tick_values = [0, 3, 6, 9, 12, 15]

cmap_orig = cm.BuPu
cmap = ff.truncate_colormap(cmap_orig, 0.05, 0.9)
cmap.set_over(cmap_orig(cmap_orig.N))
cmap.set_under('w')
norm = PowerNorm(gamma=gamma, vmin=clevs[1], vmax=max_) 

ax = fig.add_subplot(gs[1,1])
im = ax.contourf(corrdiff_all, stddevrat_all, bs_rel_q2*100, cmap=cmap, norm=norm,
                 levels=clevs, extend='max')

ax.hlines(1.0, corrdiff_all.min(), corrdiff_all.max(), colors='k', lw=0.5)
ax.vlines(0.0, stddevrat_all.min(), stddevrat_all.max(), colors='k', lw=0.5)
plot_contours(ax)

ax.set_yscale("log")
ax.set_yticks(y_ticks)
ax.get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
ax.set_xlabel(r'$\Delta\rho=r-R$')
ax.set_title(r'$\mathrm{BS_{rel}}\times 10^{2}$: $q_{0.6\overline{6}}$', fontsize=10)
ax.set_xlim((corrdiff_all.min(), corrdiff_all.max()))
ax.tick_params(labelsize=8)
ax.set_ylabel(r'$\sigma_x/\sigma_y$', fontsize=10)

cbar2 = fig.colorbar(im, ax=ax, orientation='vertical', ticks=tick_values)
cbar2.set_ticks(tick_values)
cbar2.ax.yaxis.set_major_locator(FixedLocator(tick_values))
cbar2.ax.tick_params(labelsize=8)
cbar2.ax.minorticks_off()

ax.text(0.0, 1.01, '(e)', ha='center', va='bottom', transform=ax.transAxes,
        fontsize=10)


####################
min_ = 0.0
max_ = 20
clevs = np.linspace(min_, max_, 400)
tick_values = [0, 5, 10, 15, 20]

cmap_orig = cm.BuPu
cmap = ff.truncate_colormap(cmap_orig, 0.05, 0.9)
cmap.set_over(cmap_orig(cmap_orig.N))
cmap.set_under('w')
norm = PowerNorm(gamma=gamma, vmin=clevs[1], vmax=max_) 

ax = fig.add_subplot(gs[1,2])
im = ax.contourf(corrdiff_all, stddevrat_all, bs_rel_q3*100, cmap=cmap, norm=norm,
                 levels=clevs, extend='max')

ax.hlines(1.0, corrdiff_all.min(), corrdiff_all.max(), colors='k', lw=0.5)
ax.vlines(0.0, stddevrat_all.min(), stddevrat_all.max(), colors='k', lw=0.5)
plot_contours(ax)

ax.set_yscale("log")
ax.set_yticks(y_ticks)
ax.get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
ax.set_xlabel(r'$\Delta\rho=r-R$')
ax.set_title(r'$\mathrm{BS_{rel}}\times 10^{2}$: $q_{0.95}$', fontsize=10)
ax.set_xlim((corrdiff_all.min(), corrdiff_all.max()))
ax.tick_params(labelsize=8)
ax.set_ylabel(r'$\sigma_x/\sigma_y$', fontsize=10)

cbar3 = fig.colorbar(im, ax=ax, orientation='vertical', ticks=tick_values)
cbar3.set_ticks(tick_values)
cbar3.ax.yaxis.set_major_locator(FixedLocator(tick_values))
cbar3.ax.tick_params(labelsize=8)
cbar3.ax.minorticks_off()

ax.text(0.0, 1.01, '(f)', ha='center', va='bottom', transform=ax.transAxes,
        fontsize=10)
