#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 20 14:27:59 2021

@author: ard000
"""

# import stdfiles
import domcmc.fst_tools as fst_tools
from figure_functions import FigureFunctions
from misc_funcs import convert_longitude
from scores import scores_basic, scores_secondary
from stat_funcs import round_nearest

import datetime
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
from scipy.spatial import cKDTree
import cartopy.crs as ccrs
import concurrent.futures
from matplotlib.patches import Rectangle
from matplotlib.gridspec import GridSpec as GS
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap
from statsmodels.stats.multitest import multipletests


##########################################################################################
bootstrap = True

# number of parallel processes for loading data
nprocesses = 20

exp_geps = 'E2FC8021E22'
exp_gdps = 'G2FC902V1E22'

exp_path_geps = '/path/to/geps/data/'
exp_path_gdps = '/path/to/gdps/data/'

# number of members
nens = 20

# initialization times
time_valid_s = datetime.datetime(2022, 7, 12, 0, tzinfo=datetime.timezone.utc)
time_valid_f = datetime.datetime(2022, 9, 12, 0, tzinfo=datetime.timezone.utc)
time_valid = [time_valid_s + datetime.timedelta(days=i) 
             for i in range((time_valid_f - time_valid_s).days + 1)] 
ntime = len(time_valid)

# variable (one of TT, HU, UUWE, VVSN)
var = 'TT'

# pressure levels
p_lev = 925

# domain used for n. america
# ll_lon, ur_lon, ll_lat, ur_lat
bbox = [-120, -50, 25, 75]
 
# lead time in hours (valid time is timestamp_exp + lead_num)
lead_num = 120

# grid for interpolation
lat_gauss = np.arange(-90, 90.01, 0.25)
lon_gauss = np.arange(0, 360, 0.25)
# lat_gauss = np.arange(-90, 90.01, 1.0)
# lon_gauss = np.arange(0, 360, 1.0)
lon_grid, lat_grid = np.meshgrid(lon_gauss, lat_gauss)
nrow_g, ncol_g = lat_grid.shape

#########################################################################################
# special treatment of var name if looking at winds
if var=='UUWE' or var=='VVSN':
    var_infile = 'wind_vectors'
    values = var.lower()
else:
    var_infile = var
    values = 'values'

# load initial file for geps to get metadata
fname_e = time_valid_s.strftime('%Y%m%d%H')+'_000_000'
p0_e = fst_tools.get_data(exp_path_geps+fname_e, datev=time_valid[0], var_name=var_infile, ip1=[p_lev], latlon=True) 

# grid for geps
lat_yin_e = p0_e['yin']['lat']
lon_yin_e = p0_e['yin']['lon']
nrow_yin_e, ncol_yin_e = lat_yin_e.shape

lat_yang_e = p0_e['yang']['lat']
lon_yang_e = p0_e['yang']['lon']
nrow_yang_e, ncol_yang_e = lat_yang_e.shape


# load initial file for gdps analysis
fname_d = time_valid[0].strftime('%Y%m%d%H')+'_000'
p0_d = fst_tools.get_data(exp_path_gdps+fname_d, datev=time_valid[0], var_name=var_infile, ip1=[p_lev], latlon=True) 

# grid for gdps
lat_yin_d = p0_d['yin']['lat']
lon_yin_d = p0_d['yin']['lon']
nrow_yin_d, ncol_yin_d = lat_yin_d.shape

lat_yang_d = p0_d['yang']['lat']
lon_yang_d = p0_d['yang']['lon']
nrow_yang_d, ncol_yang_d = lat_yang_d.shape

# convert lead time to string with lead 0 filled
lead_str = str(lead_num).zfill(3)

print("--------------------------------------")
print(f"WORKING ON LEAD TIME {lead_str}")
print("--------------------------------------")

# forecast files are output every 24 hours (in filename), and contain 
# two lead times EXCEPT when lead is 000hr. For example, lead 024hr file contains both lead 012hr and 024hr
# figure out the lead time in the file name accordingly
if lead_num==0:
    lead_num_fname = 0
elif lead_num % 24 == 0: 
    lead_num_fname = lead_num
else:
    lead_num_fname = lead_num + 12 

lead_num_fname_str = str(lead_num_fname).zfill(3)

# ensemble info
members_num = np.arange(1, nens+1)
members = [str(mm).zfill(3) for mm in members_num]

# Compute valid time of the forecast; this will be the timestamp used to load the analysis
time_init = [tt - datetime.timedelta(hours=lead_num) for tt in time_valid]
ndates = len(time_init)

# max number of dates to loop over in each subprocess
nperprocess = round_nearest(ndates/nprocesses, 1) 
# if number of nprocesses is larger than 1, do parallelization
if nprocesses>1:
    # organize dates into a list of lists for parallelization. The outer
    # list has length 'nprocesses'. The inner lists have length 'niter',except 
    # for possibly the last one which may be shorter.
    time_init_div = [time_init[int(ind_s):int(ind_f)] for ind_s, ind_f in 
                      zip(np.arange(0, ndates, nperprocess),
                          np.arange(nperprocess, ndates + nperprocess, nperprocess))]
else:
    time_init_div = [time_init]

# prep data arrays to be filled
X_orig_yin = np.full((ntime, nens, nrow_yin_e, ncol_yin_e), np.nan)
X_orig_yang = np.full((ntime, nens, nrow_yang_e, ncol_yang_e), np.nan)
Y_orig_yin = np.full((ntime, nrow_yin_d, ncol_yin_d), np.nan)
Y_orig_yang = np.full((ntime, nrow_yang_d, ncol_yang_d), np.nan)
P0_orig_yin = np.full((ntime, nrow_yang_d, ncol_yang_d), np.nan)
P0_orig_yang = np.full((ntime, nrow_yang_d, ncol_yang_d), np.nan)


X_interp = np.full((ntime, nens, lat_gauss.size, lon_gauss.size), np.nan)
Y_interp = np.full((ntime, lat_gauss.size, lon_gauss.size), np.nan)
P0_interp = np.full((ntime, lat_gauss.size, lon_gauss.size), np.nan)

# Load data in parallel
def load_data(time_init_list_curr):
    ntime_curr = len(time_init_list_curr)
    
    X_orig_yin_curr = np.full((ntime_curr, nens, nrow_yin_e, ncol_yin_e), np.nan)
    X_orig_yang_curr = np.full((ntime_curr, nens, nrow_yang_e, ncol_yang_e), np.nan)
    Y_orig_yin_curr = np.full((ntime_curr, nrow_yin_d, ncol_yin_d), np.nan)
    Y_orig_yang_curr = np.full((ntime_curr, nrow_yang_d, ncol_yang_d), np.nan)
    P0_orig_yin_curr = np.full((ntime_curr, nrow_yang_d, ncol_yang_d), np.nan)
    P0_orig_yang_curr = np.full((ntime_curr, nrow_yang_d, ncol_yang_d), np.nan)    
    
    for count_t, time_init_curr in enumerate(time_init_list_curr):
        count_t_real = np.where(np.asarray(time_init)==time_init_curr)[0][0]
        time_valid_curr = time_valid[count_t_real]
        print(f"Loading data for init time: {time_init_curr.strftime('%Y%m%d%H')}")
        # print(f"Loading data for valid time: {time_valid_curr.strftime('%Y%m%d%H')}")
        
        # load analysis
        fname_anal = time_valid_curr.strftime('%Y%m%d%H')+'_000'
        f0_anal = fst_tools.get_data(exp_path_gdps+fname_anal, datev=time_valid_curr, var_name=var_infile, ip1=[p_lev], latlon=True)  
        p0_anal = fst_tools.get_data(exp_path_gdps+fname_anal, datev=time_valid_curr, var_name='P0', latlon=True) 
        
        # print(f0_anal['yin']['values'].shape)
        Y_orig_yin_curr[count_t] = f0_anal['yin'][values]
        Y_orig_yang_curr[count_t] = f0_anal['yang'][values]   

        P0_orig_yin_curr[count_t] = p0_anal['yin'][values]
        P0_orig_yang_curr[count_t] = p0_anal['yang'][values] 
        
        for count_n, ens_id in enumerate(members):
            # load ensemble
            fname_ens = time_init_curr.strftime('%Y%m%d%H')+'_'+lead_num_fname_str+'_'+ens_id
            # print(f"Loading data for ensemble member: {ens_id}")
            f0_ens = fst_tools.get_data(exp_path_geps+fname_ens, datev=time_valid_curr, var_name=var_infile, ip1=[p_lev], latlon=True)  
    
            X_orig_yin_curr[count_t, count_n] = f0_ens['yin'][values]
            X_orig_yang_curr[count_t, count_n] = f0_ens['yang'][values]
        
    return X_orig_yin_curr, X_orig_yang_curr, Y_orig_yin_curr, Y_orig_yang_curr, P0_orig_yin_curr, P0_orig_yang_curr

# parallelization for reading data
with concurrent.futures.ProcessPoolExecutor(nprocesses) as executor:
    results = executor.map(load_data, time_init_div)
    
ind_s = 0   
ind_f = 0
for count, result in enumerate(results):
    ind_s = ind_f
    ind_f = ind_f+len(time_init_div[count])
    inds = np.arange(ind_s, ind_f)
    
    # stores data from 'result' produced by rsc.read_sqlite in dspd object
    X_orig_yin[inds], X_orig_yang[inds], Y_orig_yin[inds], Y_orig_yang[inds], P0_orig_yin[inds], P0_orig_yang[inds] = result  

## Interpolation for all levels
def latlon_to_cartesian(lat, lon):
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    x = np.cos(lat_rad) * np.cos(lon_rad)
    y = np.cos(lat_rad) * np.sin(lon_rad)
    z = np.sin(lat_rad)
    return np.column_stack((x, y, z))

def prepare_idw_weights(lat_yin, lon_yin,
                        lat_yang, lon_yang,
                        lat_out, lon_out,
                        k=4, eps=1e-12):
    """Precompute KDTree indices and weights for spherical inverse distance weighting interpolation."""

    # Flatten input grids and convert to Cartesian
    xyz_yin = latlon_to_cartesian(lat_yin.ravel(), lon_yin.ravel())
    xyz_yang = latlon_to_cartesian(lat_yang.ravel(), lon_yang.ravel())
    xyz_all = np.vstack([xyz_yin, xyz_yang])

    # Build tree from source points
    tree = cKDTree(xyz_all)

    # Output grid in Cartesian
    xyz_out = latlon_to_cartesian(lat_out.ravel(), lon_out.ravel())
    dists, idxs = tree.query(xyz_out, k=k)

    if k == 1:
        dists = dists[:, np.newaxis]
        idxs = idxs[:, np.newaxis]

    weights = 1.0 / np.clip(dists, eps, None)
    weights /= weights.sum(axis=1, keepdims=True)

    return idxs, weights

def apply_idw_values(data_yin, data_yang, idxs, weights):
    """Perform weighted interpolation using precomputed indices and weights."""
    vals_all = np.concatenate([data_yin.ravel(), data_yang.ravel()])
    return np.sum(vals_all[idxs] * weights, axis=1)


# Precompute weights
idxs_e, weights_e = prepare_idw_weights(
    lat_yin_e, lon_yin_e, lat_yang_e, lon_yang_e, lat_grid, lon_grid
)
idxs_d, weights_d = prepare_idw_weights(
    lat_yin_d, lon_yin_d, lat_yang_d, lon_yang_d, lat_grid, lon_grid
)

# Interpolate
for t in range(ntime):
    print(f"Interpolating (valid time): {time_valid[t]}")
    Y_interp[t] = apply_idw_values(Y_orig_yin[t], Y_orig_yang[t], idxs_d, weights_d).reshape(lat_grid.shape)
    P0_interp[t] = apply_idw_values(P0_orig_yin[t], P0_orig_yang[t], idxs_d, weights_d).reshape(lat_grid.shape)

    for n in range(nens):
        # print(f"  Ensemble member {n+1}")
        X_interp[t, n] = apply_idw_values(X_orig_yin[t, n], X_orig_yang[t, n], idxs_e, weights_e).reshape(lat_grid.shape)

mask_P0 = np.zeros(Y_interp.mean(axis=0).shape)
mask_P0[np.any(P0_interp<p_lev,axis=0)] = 1.0    
# mask_P0[np.mean(P0_interp, axis=0)<p_lev] = 1.0    
    
########################################################
# Extract north american domain
########################################################
pad_lon = 60
pad_lat = 15 
inds_lat_nm = np.where((lat_gauss>=bbox[2]-pad_lat)&(lat_gauss<=bbox[3]+pad_lat))[0]
inds_lon_nm = np.where((lon_gauss>=convert_longitude(bbox[0])-pad_lon)&(lon_gauss<=convert_longitude(bbox[1])+pad_lon))[0]

ind_lat_s = inds_lat_nm[0]
ind_lat_f = inds_lat_nm[-1] + 1
ind_lon_s = inds_lon_nm[0]
ind_lon_f = inds_lon_nm[-1] + 1

X_interp = X_interp[:,:,ind_lat_s:ind_lat_f, ind_lon_s:ind_lon_f]
Y_interp = Y_interp[:,ind_lat_s:ind_lat_f, ind_lon_s:ind_lon_f]
mask_P0 = mask_P0[ind_lat_s:ind_lat_f, ind_lon_s:ind_lon_f]

lat_gauss = lat_gauss[inds_lat_nm]
lon_gauss = lon_gauss[inds_lon_nm]

lon_grid, lat_grid = np.meshgrid(lon_gauss, lat_gauss)

nrow_g, ncol_g = lat_grid.shape

# Prep statistics arrays
cov_xx = np.full((nrow_g, ncol_g), 0.0)
cov_xy = np.full((nrow_g, ncol_g), 0.0)    
    
r = np.full((nrow_g, ncol_g), np.nan)
R = np.full((nrow_g, ncol_g), np.nan)        

delta_tau_v = np.full((nrow_g, ncol_g), np.nan)    

var_diff_sc = np.full((nrow_g, ncol_g), 0.0)
delta_tau_v_sc = np.full((nrow_g, ncol_g), 0.0)
corr_diff_sc = np.full((nrow_g, ncol_g), 0.0)
cov_diff_sc = np.full((nrow_g, ncol_g), 0.0)
    
########################################################
# compute marginal variance, covariance, and correlation
########################################################
print("Computing statistics: marginal variance, covariance, and correlation")
# marginal variance
margvar_x = np.var(X_interp, axis=0).mean(axis=0)
margvar_y = np.var(Y_interp, axis=0)

var_diff = (margvar_x - margvar_y)/margvar_y

# covariance between members
X_anom = X_interp - X_interp.mean(axis=0)[np.newaxis]
Y_anom = Y_interp - Y_interp.mean(axis=0)[np.newaxis]       

for ii in range(nens):
    cov_xy += np.mean(X_anom[:,ii] * Y_anom, axis=0)/nens
    for jj in range(nens):
        if ii < jj:
            cov_xx += np.mean(X_anom[:,ii]*X_anom[:,jj], axis=0)/((nens**2.-nens)/2)     

cov_diff = 2*(cov_xx - cov_xy)/margvar_y
cov_xymean_all = np.mean(X_anom.mean(axis=1) * Y_anom, axis=0)
            
r = cov_xx/margvar_x
R = cov_xy/(margvar_x**0.5 * margvar_y**0.5)
 
# reliability budget
spread = np.var(X_interp, axis=1, ddof=1).mean(axis=0)
depart = np.mean((X_interp.mean(axis=1) - Y_interp)**2.0, axis=0)
bias_sq = np.mean(X_interp.mean(axis=1) - Y_interp, axis=0)**2.0

delta_tau_v =  (nens + 1)/nens * spread - ( ntime/(ntime - 1) * (depart - bias_sq))
delta_tau = (nens + 1)/nens * spread  - depart

# determine statistical significance using bootstrapping
nboot = 1000
block_size = 1

var_diff_bs = np.zeros((nboot, nrow_g, ncol_g))
delta_tau_v_bs = np.zeros((nboot, nrow_g, ncol_g))
delta_tau_bs = np.zeros((nboot, nrow_g, ncol_g))
corr_diff_bs = np.zeros((nboot, nrow_g, ncol_g))
cov_diff_bs = np.zeros((nboot, nrow_g, ncol_g))


def block_bootstrap_indices(ntime, block_size):
    """
    Generate block bootstrap indices for time series resampling.
    ntime: total length of series
    block_size: length of each block
    """
    nblocks = int(np.ceil(ntime / block_size))
    
    # choose blocks with replacement
    block_starts = np.random.randint(0, ntime - block_size + 1, size=nblocks)
    
    # build indices from chosen blocks
    inds = np.concatenate([np.arange(start, start + block_size) for start in block_starts])
    
    # truncate to exactly ntime (in case it overshoots)
    return inds[:ntime]

def loop_boot(nboot_curr, block_size):
    # print(nboot_curr)
    var_diff_bs_curr = np.zeros((nboot_curr, nrow_g, ncol_g))
    delta_tau_v_bs_curr = np.zeros((nboot_curr, nrow_g, ncol_g))
    cov_diff_bs_curr = np.zeros((nboot_curr, nrow_g, ncol_g))         
    corr_diff_bs_curr = np.zeros((nboot_curr, nrow_g, ncol_g)) 
    
    for bb in range(nboot_curr):
        inds_bs = block_bootstrap_indices(ntime, block_size)
        X_bs = X_interp[inds_bs]
        Y_bs = Y_interp[inds_bs]
        
        # marginal variance
        margvar_x_bs = np.var(X_bs, axis=0).mean(axis=0)
        margvar_y_bs = np.var(Y_bs, axis=0)    
        
        var_diff_bs_curr[bb] = margvar_x_bs - margvar_y_bs
        
        # reliability budget
        spread_bs = np.var(X_bs, axis=1, ddof=1).mean(axis=0)
        depart_bs = np.mean((X_bs.mean(axis=1) - Y_bs)**2.0, axis=0)
        bias_sq_bs = np.mean(X_bs.mean(axis=1) - Y_bs, axis=0)**2.0
    
        delta_tau_v_bs_curr[bb] =  (nens + 1)/nens * spread_bs - ( ntime/(ntime - 1) * (depart_bs - bias_sq_bs))
        # delta_tau_v_bs_curr[bb] /= margvar_y_bs
        
        # correlation difference
        X_bs_anom = X_bs - X_bs.mean(axis=0)[np.newaxis]
        Y_bs_anom = Y_bs - Y_bs.mean(axis=0)[np.newaxis]       

        cov_xy_bs = 0.0
        cov_xx_bs = 0.0
        for ii in range(nens):
            cov_xy_bs += np.mean(X_bs_anom[:,ii] * Y_bs_anom, axis=0)/nens
            for jj in range(nens):
                if ii < jj:
                    cov_xx_bs += np.mean(X_bs_anom[:,ii]*X_bs_anom[:,jj], axis=0)/((nens**2.-nens)/2)     

        cov_diff_bs_curr[bb] = 2*(cov_xy_bs - cov_xx_bs)
                    
        r_bs = cov_xx_bs/margvar_x_bs
        R_bs = cov_xy_bs/(margvar_x_bs**0.5 * margvar_y_bs**0.5)
        corr_diff_bs_curr[bb] = r_bs - R_bs
        
    
    return var_diff_bs_curr, delta_tau_v_bs_curr, cov_diff_bs_curr, corr_diff_bs_curr

def bootstrap_pvals_percentile(bs, theta0=0.0):
    """
    Two-sided percentile bootstrap p-values for H0: theta = theta0.
    
    Parameters
    ----------
    bs : ndarray, shape (Nboot, ..., ...)
        Bootstrap replicates.
    theta0 : float
        Null value.
    
    Returns
    -------
    pvals : ndarray, shape (..., ...)
        Two-sided bootstrap p-values.
    """
    p_lo = np.mean(bs <= theta0, axis=0)
    p_hi = np.mean(bs >= theta0, axis=0)
    pvals = 2.0 * np.minimum(p_lo, p_hi)

    pvals = np.clip(pvals, 1.0 / bs.shape[0], 1.0)
    return pvals

def fdr_mask(pvals, alpha=0.05):
    """
    Benjamini–Hochberg FDR mask for spatial p-values.
    
    Parameters
    ----------
    pvals : ndarray, shape (Nlat, Nlon)
    alpha : float
    
    Returns
    -------
    mask : ndarray, shape (Nlat, Nlon), bool
        True where FDR-significant.
    """
    p_flat = pvals.ravel()
    valid = np.isfinite(p_flat)

    reject = np.zeros_like(p_flat, dtype=bool)
    reject_valid, _, _, _ = multipletests(
        p_flat[valid], alpha=alpha, method="fdr_bh"
    )
    reject[valid] = reject_valid

    return reject.reshape(pvals.shape)

if bootstrap:
    print("Bootstrapping starts")
    nprocesses_curr = 20
    nboot_curr = int(nboot / nprocesses_curr)

    with concurrent.futures.ProcessPoolExecutor(nprocesses_curr) as executor:
        results = executor.map(
            loop_boot,
            [nboot_curr] * nprocesses_curr,
            [block_size] * nprocesses_curr,
        )

        ind_s = 0
        ind_f = 0
        for result in results:
            ind_s = ind_f
            ind_f += nboot_curr
            inds = np.arange(ind_s, ind_f)

            (
                var_diff_bs[inds],
                delta_tau_v_bs[inds],
                cov_diff_bs[inds],
                corr_diff_bs[inds],
            ) = result

    # percentile bootstrap p-values
    p_var_diff = bootstrap_pvals_percentile(var_diff_bs)
    p_delta_tau_v = bootstrap_pvals_percentile(delta_tau_v_bs)
    p_cov_diff = bootstrap_pvals_percentile(cov_diff_bs)
    p_corr_diff = bootstrap_pvals_percentile(corr_diff_bs)

    # FDR correction (done per field): values of 1 in _sc arrays indicate significance
    alpha_fdr = 0.05
    var_diff_sc = fdr_mask(p_var_diff, alpha_fdr).astype(float)
    delta_tau_v_sc = fdr_mask(p_delta_tau_v, alpha_fdr).astype(float)
    cov_diff_sc = fdr_mask(p_cov_diff, alpha_fdr).astype(float)
    corr_diff_sc = fdr_mask(p_corr_diff, alpha_fdr).astype(float)


# clip values of correlations very close to zero and one
eps = 0.001
r = np.clip(r, a_min=r+eps, a_max=r-eps)
R = np.clip(R, a_min=R+eps, a_max=R-eps)

corr_diff = r - R


# Regions that will have a black box in the first figure
bbox_1 = [30.0,  43,  -97.5, -75.0] # southeastern US

mask_1 = np.zeros((nrow_g, ncol_g))
mask_1[(lat_grid>bbox_1[0])&(lat_grid<bbox_1[1])&
       (lon_grid>convert_longitude(bbox_1[2]))&
       (lon_grid<convert_longitude(bbox_1[3]))] = 1.0

# select region to compute scores for in second figure
##########################################
mask_curr = mask_1.copy()

# extract a particular location in the region of interest for time series
dist_v = (np.abs(var_diff[mask_curr==1.0]) - np.abs(var_diff[mask_curr==1.0]).mean())**2.0
dist_d = (np.abs(delta_tau_v[mask_curr==1.0]) - np.abs(delta_tau_v[mask_curr==1.0]).mean())**2.0

sum_vd = dist_d + dist_v

lat_curr, lon_curr = 33.75, convert_longitude(-79)

dist_lat = lat_curr - lat_grid[mask_curr==1.0]
dist_lon = lon_curr - lon_grid[mask_curr==1.0]
dist = np.sqrt(dist_lat**2.0 + dist_lon**2.0)

ind_curr = np.where(dist==dist.min())[0][0]

X_interp_bc = Y_interp.mean(axis=0)[np.newaxis,np.newaxis] + (X_interp - X_interp.mean(axis=0)[np.newaxis])

X_interp3 = X_interp[:,:,mask_curr==1.0]
X_interp3_bc = X_interp_bc[:,:,mask_curr==1.0]
Y_interp3 = Y_interp[:,mask_curr==1.0]
Np3 = len(mask_curr[mask_curr==1.0])

#
Nbins = nens + 1

# reshape forecasts and analyses to concatenate space and time
# for computing the rank histogram and reliability diagram

# Transpose so that p is first, then t, then n
X_interp3_c = X_interp3.transpose(2, 0, 1)   # shape (p, t, n)
# Reshape so (p, t) becomes one axis
X_interp3_c = X_interp3_c.reshape(-1, nens)  # shape (p*t, n)

# Transpose so that p is first, then t, then n
X_interp3_bc_c = X_interp3_bc.transpose(2, 0, 1)   # shape (p, t, n)
# Reshape so (p, t) becomes one axis
X_interp3_bc_c = X_interp3_bc_c.reshape(-1, nens)  # shape (p*t, n)

# Transpose so that p is first, then t, then n
Y_interp3_c = Y_interp3.T
Y_interp3_c = np.reshape(Y_interp3_c, (ntime*Np3))

scores = scores_basic(X_interp3_c, Y_interp3_c)
scores_bc = scores_basic(X_interp3_bc_c, Y_interp3_c)
scores_sec = scores_secondary()
########################################################
# Plot result (a plot for each reliability budget, vardiff, and covdiff)
########################################################
ff = FigureFunctions()

def add_rectangle(ax, bbox):
    rect = Rectangle(
        (convert_longitude(bbox[2]), bbox[0]),   # (lon_min, lat_min)
        convert_longitude(bbox[3]) - convert_longitude(bbox[2]),    # width (lon_max - lon_min)
        bbox[1] - bbox[0],    # height (lat_max - lat_min)
        linewidth=1.5,
        edgecolor='k',
        facecolor='none',
        transform=ccrs.PlateCarree(),  # IMPORTANT: tell Cartopy coords are lon/lat
        zorder=3
    )
    ax.add_patch(rect)

# hatching for significance
hatch = ['xxxx']

plt.close()
fig = plt.figure(num=1, figsize=(10,2.8))
fig.clf()

nrow = 1
# show reliability budget, vardiff, covdiff, corrdiff
ncol = 4

# set up subplot structure
gs = gridspec.GridSpec(nrow, ncol, wspace=0.02, top=0.99, bottom=0.08, left=0.01, right=0.99)


# colorbar for first three subplots
var_min = -0.5
var_max = 0.5
var_range_ticks = np.arange(var_min, var_max+0.05, 0.05)
var_range_labels = np.arange(var_min, var_max+0.2, 0.2)

cmap_orig = cm.bwr
cmap = ff.truncate_colormap(cmap_orig, 0.1, 0.9)
cmap.set_over(cmap_orig(cmap_orig.N))
cmap.set_under(cmap_orig(0))
norm = mpl.colors.BoundaryNorm(var_range_ticks, cmap.N)

color = 'k'
marker = 'o'
size = 40
####################
# reliability budget
####################
ax, transform = ff.add_map_subplot(fig, gs[0, 0], bbox)     
im = ax.pcolormesh(lon_grid, lat_grid, delta_tau_v/margvar_y, 
                    cmap=cmap, norm=norm, rasterized=True, transform=ccrs.PlateCarree(), zorder=2)

# # ax.gridlines(color='0.5', linestyle='--', linewidth=1)

if bootstrap:
    cs = ax.contourf(lon_grid, lat_grid, delta_tau_v_sc, levels=[0.5, 1.5], colors='none', hatches=hatch, 
                     transform=ccrs.PlateCarree(),
                     zorder=3)
    
    cs.set_hatch_linewidth(0.5)

add_rectangle(ax, bbox_1)

ax.scatter(lon_grid[mask_curr==1.0][ind_curr], lat_grid[mask_curr==1.0][ind_curr],
           marker=marker, s=size, edgecolors=color, c='none', transform=ccrs.PlateCarree(), zorder=3)

cmap_black = ListedColormap(["none", "black"])

ax.pcolormesh(lon_grid, lat_grid, mask_P0, cmap=cmap_black, shading='nearest',
              rasterized=True, transform=ccrs.PlateCarree(), zorder=3)


ax.set_title(r'$\delta_{\tau}^\nu~/~\sigma_y^2$', fontsize=9)
ax.text(0.0, 1.01, '(a)', transform=ax.transAxes, ha='left', va='bottom',
        fontsize=11)

######################
# marginal variance
######################
ax, transform = ff.add_map_subplot(fig, gs[0, 1], bbox)     
im = ax.pcolormesh(lon_grid, lat_grid, var_diff, 
                    cmap=cmap, norm=norm, rasterized=True, transform=ccrs.PlateCarree(), zorder=2)

if bootstrap:
    cs = ax.contourf(lon_grid, lat_grid, var_diff_sc, levels=[0.5, 1.5], colors='none', hatches=hatch, 
                transform=ccrs.PlateCarree(),
                zorder=3)
    
    cs.set_hatch_linewidth(0.5)
       
add_rectangle(ax, bbox_1)

ax.scatter(lon_grid[mask_curr==1.0][ind_curr], lat_grid[mask_curr==1.0][ind_curr],
           marker=marker, s=size, edgecolors=color, c='none', transform=ccrs.PlateCarree(), zorder=3)

ax.pcolormesh(lon_grid, lat_grid, mask_P0, cmap=cmap_black, shading='nearest',
              rasterized=True, transform=ccrs.PlateCarree(), zorder=3)

ax.set_title(r'$(\sigma_x^2 - \sigma_y^2)~/~\sigma_y^2$', fontsize=9)

ax.text(0.0, 1.01, '(b)', transform=ax.transAxes, ha='left', va='bottom',
        fontsize=11)

ax_pos = ax.get_position() # (to be used for colorbar)

###############
# covariance
###############
ax, transform = ff.add_map_subplot(fig, gs[0, 2], bbox)     
im = ax.pcolormesh(lon_grid, lat_grid, cov_diff, 
                    cmap=cmap, norm=norm, rasterized=True, transform=ccrs.PlateCarree(), zorder=2)

if bootstrap:
    cs = ax.contourf(lon_grid, lat_grid, cov_diff_sc, levels=[0.5, 1.5], colors='none', hatches=hatch,
                transform=ccrs.PlateCarree(),
                zorder=3)
    cs.set_hatch_linewidth(0.5)

add_rectangle(ax, bbox_1)

ax.scatter(lon_grid[mask_curr==1.0][ind_curr], lat_grid[mask_curr==1.0][ind_curr],
           marker=marker, s=size, edgecolors=color, c='none', transform=ccrs.PlateCarree(), zorder=3)

ax.pcolormesh(lon_grid, lat_grid, mask_P0, cmap=cmap_black, shading='nearest',
              rasterized=True, transform=ccrs.PlateCarree(), zorder=3)

ax.set_title(r'$2(\mathrm{Cov}(X_i,X_j)-\mathrm{Cov}(X_i,Y))~/~\sigma_y^2$', fontsize=9)

ax.text(0.0, 1.01, '(c)', transform=ax.transAxes, ha='left', va='bottom',
        fontsize=11)

# colorbar for first three
xmin = ax_pos.xmin
xmax = ax_pos.xmax
ymin = ax_pos.ymin
pad = 0.1 # on each side

axes_cb = fig.add_axes((xmin - pad, ymin - 0.05, xmax - xmin + pad*2.0, 0.03))
cb = fig.colorbar(im, cax=axes_cb, orientation="horizontal", 
              extend='both', spacing='uniform', ticks=var_range_labels)

cb.minorticks_off()       
axes_cb.set_xticklabels([f"{x:.1f}" for x in var_range_labels], fontsize=8)


###############
# correlation
###############
rho_min = -0.15
rho_max = 0.15
rho_range_ticks = np.arange(rho_min, rho_max+0.01, 0.01)
rho_range_labels = np.arange(rho_min, rho_max+0.05, 0.05)

cmap_orig = cm.bwr
cmap = ff.truncate_colormap(cmap_orig, 0.1, 0.9)
cmap.set_over(cmap_orig(cmap_orig.N))
cmap.set_under(cmap_orig(0))
norm = mpl.colors.BoundaryNorm(rho_range_ticks, cmap.N)

ax, transform = ff.add_map_subplot(fig, gs[0, 3], bbox)     
im = ax.pcolormesh(lon_grid, lat_grid, corr_diff, 
                    cmap=cmap, norm=norm, rasterized=True, transform=ccrs.PlateCarree(), zorder=2)

if bootstrap:
    cs = ax.contourf(lon_grid, lat_grid, corr_diff_sc, levels=[0.5, 1.5], colors='none', hatches=hatch,
                transform=ccrs.PlateCarree(),
                zorder=3)
    cs.set_hatch_linewidth(0.5)

add_rectangle(ax, bbox_1)

ax.scatter(lon_grid[mask_curr==1.0][ind_curr], lat_grid[mask_curr==1.0][ind_curr],
           marker=marker, s=size, edgecolors=color, c='none', transform=ccrs.PlateCarree(), zorder=3)

ax.pcolormesh(lon_grid, lat_grid, mask_P0, cmap=cmap_black, shading='nearest',
              rasterized=True, transform=ccrs.PlateCarree(), zorder=3)

ax.set_title(r'$r-R$', fontsize=9)
ax.text(0.0, 1.01, '(d)', transform=ax.transAxes, ha='left', va='bottom',
        fontsize=11)

# colorbar for first three
ax_pos = ax.get_position() # 

xmin = ax_pos.xmin
xmax = ax_pos.xmax
ymin = ax_pos.ymin

axes_cb = fig.add_axes((xmin, ymin - 0.05, xmax-xmin, 0.03))
cb = fig.colorbar(im, cax=axes_cb, orientation="horizontal", 
              extend='both', spacing='uniform', ticks=rho_range_labels)

cb.minorticks_off()       
axes_cb.set_xticklabels([f"{x:.2f}" for x in rho_range_labels], fontsize=8)

########################################################
########################################################

fig = plt.figure(num=2, figsize=(10,5.5))
plt.clf()
gs1 = GS(1, 1, left=0.07, right=0.98, bottom=0.60, top=0.98)
gs2 = GS(1, 3, left=0.07, right=0.98, bottom=0.09, top=0.5, wspace=0.25)

### Plot timeseries

ax = fig.add_subplot(gs1[0,0])

xrange = np.arange(ntime)
ax.hlines(Y_interp3[:,ind_curr].mean(), -1, ntime+3, colors='k', ls='-', lw=0.5)

ax.plot(xrange,X_interp3_bc[:,:,ind_curr], ls='none', marker='+', color='k', alpha=0.2, markersize=5)
ax.plot(xrange,X_interp3_bc[:,:,ind_curr].mean(axis=1), ls='none', marker='o', color='b', markersize=4)
ax.plot(xrange,Y_interp3[:,ind_curr], ls='none', marker='*', color='r', markersize=6)
ax.set_xticks(xrange, minor=True)
ax.set_xticks(xrange[::5])
ax.set_xticklabels([tt.strftime('%m-%d-%y') for tt in time_valid[::5]])

ax.set_xlim((-1, ntime))

ax.set_ylabel('Temperature ($^\circ$C)', fontsize=11)

ax.grid(ls=':', color='0.5')

lines = [Line2D([], [], ls='none', marker='+', color='k', alpha=0.8, label='ens. members'),
         Line2D([], [], ls='none', marker='o', color='b', label='ens. mean'),
         Line2D([], [], ls='none', marker='*', color='r', label='analysis')]

ax.legend(handles=lines, ncol=3, bbox_to_anchor=(0.5,1.02), loc='upper center',
          fontsize=8)

ax_pos = ax.get_position()
fig.text(ax_pos.xmin-0.03, ax_pos.ymax-0.005, '(a)', fontsize=12, ha='right', va='center')

# spread-error and reliability budget
ax = fig.add_subplot(gs2[0,0])
labels = [r'$\delta_{\tau}$',
          r'$\delta_{\tau}^{\nu}$',
          r'$-(\Delta\mu)^2$',
          r'$\Delta \sigma^2$',
          r'$-2\Delta \Sigma$']  

xrange = np.arange(len(labels))

values = [delta_tau[mask_curr==1.0].mean()/margvar_y[mask_curr==1.0].mean(), delta_tau_v[mask_curr==1.0].mean()/margvar_y[mask_curr==1.0].mean(), 
          -1*np.mean(bias_sq[mask_curr==1.0])/margvar_y[mask_curr==1.0].mean(), 
          var_diff[mask_curr==1.0].mean(), 
          -1*cov_diff[mask_curr==1.0].mean()]
    
step = 0.2

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

ax.set_xlabel('normalized spread-error & components')

ax_pos = ax.get_position()
fig.text(ax_pos.xmin, ax_pos.ymax+0.02, '(b)', fontsize=12, ha='center', va='center')

###################################################################
# Rank histogram
ax = fig.add_subplot(gs2[0,1])
ax.bar(np.arange(Nbins), scores.a_k/scores.E_k, width=1, edgecolor='0.5',
        facecolor='0.9', lw=1.5, label='raw')

ax.bar(np.arange(Nbins), scores_bc.a_k/scores_bc.E_k, width=1, edgecolor='purple',
        facecolor='thistle', lw=1.5, ls='--', alpha=0.5, label='bias subt.')
ax.legend(fontsize=8)
ax.hlines(1.0, -0.5, nens+0.05, linestyles='--', colors='k')
ax.set_xlabel('rank of $y_t$', fontsize=10)
ax.set_ylabel('frequency', fontsize=10)
ax.set_xticks(np.arange(0, nens+10, 10))
ax.set_xticks(np.arange(0, nens+5, 5), minor=True)

ax.set_xlim((-0.5,nens+0.05))
ax.set_ylim((0.0, round_nearest(scores.a_k.max()/scores.E_k,0.5, up=True)))
ax.grid(ls=':', color='0.5')

ax_pos = ax.get_position()
fig.text(ax_pos.xmin, ax_pos.ymax+0.02, '(c)', fontsize=12, ha='center', va='center')

###################################################################
# Reliability diagram
events = [[50, 50], [1/3*100, 2/3*100], [5, 95]]
events_str = ['$q_{0.5}$', '$q_{0.6\overline{6}}$', '$q_{0.95}$']

ls_all = ['-', '--', ':']
markers_all = ['o', 's', 'd']
labels = ['raw', 'bias subt.']

ax = fig.add_subplot(gs2[0,2])
for label in labels:
    for count, event in enumerate(events):
        clim_percentile = np.percentile(Y_interp3_c, event)
        
        if label=='raw':
            p_k, p_fcst_bn, p_fcst_nn, p_fcst_an, p_obs_bn, p_obs_nn, p_obs_an = scores_sec.get_fcst_probs(X_interp3_c, Y_interp3_c, clim_percentile)
        else:
            p_k, p_fcst_bn, p_fcst_nn, p_fcst_an, p_obs_bn, p_obs_nn, p_obs_an = scores_sec.get_fcst_probs(X_interp3_bc_c, Y_interp3_c, clim_percentile)
            
        bs_o_k_an, bs_n_k_an, bs_rel_an, bs_res_an, bs_unc_an, bs_p_hat_an, bs_o_hat_an = scores_sec.brier_decomp(p_fcst_an, p_obs_an, p_k, n_thresh=50)
    
        prop=300
        s_k_an = prop*bs_n_k_an/np.nansum(bs_n_k_an)
        
               
        ax.plot(np.linspace(-0.05,1.05,100), np.linspace(-0.05,1.05,100),
                  'k-', lw=1)
      
        if label=='raw':        
            ax.plot(p_k, bs_o_k_an, color='0.5', ls=ls_all[count])
            ax.scatter(p_k, bs_o_k_an, s=s_k_an, edgecolor='0.5', c='0.9', marker=markers_all[count])    
        else:
            ax.plot(p_k, bs_o_k_an, color='r', ls=ls_all[count])
            ax.scatter(p_k, bs_o_k_an, s=s_k_an, edgecolor='r', c='lightpink', marker=markers_all[count])        
        
        ax.set_xticks(np.arange(0, 1.0 + 0.2, 0.2))
        ax.set_yticks(np.arange(0, 1.0 + 0.2, 0.2))
        ax.set_xlim((-0.05,1.05))
        ax.set_ylim((-0.05,1.05))
        ax.set_xlabel('$p_k$', fontsize=10)
        ax.set_ylabel('$\overline{o}_k$', fontsize=10)
        ax.grid(ls=':', color='0.5')
    
# make legend for quantiles
legend_handles = [
    Line2D([0], [0], color='0.5', marker=markers_all[0], linestyle=ls_all[0], label=events_str[0], mfc='0.9'),
    Line2D([0], [0], color='0.5', marker=markers_all[1], linestyle=ls_all[1], label=events_str[1], mfc='0.9'),
    Line2D([0], [0], color='0.5', marker=markers_all[2], linestyle=ls_all[2], label=events_str[2], mfc='0.9'),
]

first_legend=ax.legend(handles=legend_handles, fontsize=8, title_fontsize=8, title='raw', loc='upper left')
first_legend.get_title().set_color("0.25")
    
# make legend for quantiles
legend_handles = [
    Line2D([0], [0], color='r', marker=markers_all[0], linestyle=ls_all[0], label=events_str[0], mfc='lightpink'),
    Line2D([0], [0], color='r', marker=markers_all[1], linestyle=ls_all[1], label=events_str[1], mfc='lightpink'),
    Line2D([0], [0], color='r', marker=markers_all[2], linestyle=ls_all[2], label=events_str[2], mfc='lightpink'),
]

second_legend=ax.legend(handles=legend_handles, fontsize=8, title_fontsize=8, title='bias subt.', loc='upper center')
second_legend.get_title().set_color("r")
 

ax.add_artist(first_legend)


ax_pos = ax.get_position()
fig.text(ax_pos.xmin, ax_pos.ymax+0.02, '(d)', fontsize=12, ha='center', va='center')
