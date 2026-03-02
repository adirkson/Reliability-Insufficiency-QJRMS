#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 18:49:03 2022

@author: ard000
"""
import numpy as np
from collections import namedtuple
from scipy.stats import chisquare
from stat_funcs import round_nearest

class scores_basic:
    def __init__(self, x, y):
        T, n = x.shape
        sf = scores_secondary()
        
        # mean ensemble variance
        self.spread = np.var(x, axis=1, ddof=1).mean()
        # mean square error
        self.mse = np.mean((x.mean(axis=1) - y)**2.0)
        # unbiased mean square error (w.r.t. ensmeble size)
        self.mse_fair = self.mse  - self.spread/n
        # clim. means and bias bias
        self.margmean_x = x.mean()
        self.margmean_y = y.mean()
        self.bias = self.margmean_x - self.margmean_y
        # spread-error difference
        self.delta_tau = self.spread - self.mse_fair  
        # reliability budget
        self.delta_tau_v = self.spread - T/(T-1)*(self.mse_fair - self.bias**2.0)
        
        r = []
        R = []
        cov_xx = []
        cov_xy = []
        for ii in range(n):
            R.append(np.corrcoef(x[:,ii], y)[0,1])
            cov_xy.append(np.cov(x[:,ii], y)[0,1])
            for jj in range(ii+1,n):
                r.append(np.corrcoef(x[:,ii],x[:,jj])[0,1])
                cov_xx.append(np.cov(x[:,ii],x[:,jj])[0,1])
        
        self.R = np.mean(np.asarray(R))
        self.r = np.mean(np.asarray(r))
        self.rho_diff = self.r - self.R

        self.cov_xx = np.mean(np.asarray(cov_xx))
        self.cov_xy = np.mean(np.asarray(cov_xy))
        
        self.margvar_x = np.var(x, axis=0).mean()
        self.margvar_y = np.var(y)

        self.cov_diff = 2*(self.cov_xx - self.cov_xy)
        self.var_diff = self.margvar_x - self.margvar_y

        self.cov_diff_norm = 2*(self.cov_xx - self.cov_xy)/self.margvar_y
        self.var_diff_norm = (self.margvar_x-self.margvar_y)/self.margvar_y
        self.stddev_rat = self.margvar_x**0.5 / self.margvar_y**0.5
        
        # rank histogram
        self.a_k, self.E_k = sf.rank_histogram(x, y)
        self.chi_sq_stat, self.chi_sq_pval = chisquare(self.a_k, self.E_k)

        # crps
        result_crps = sf.crps_decomp(x , y)
        self.crps_rel, self.crps_res, self.crps = result_crps.crps_rel, result_crps.crps_res, result_crps.crps



class scores_secondary:
    
    def ecdf(self, x, data):
        r'''
        For computing the empirical cumulative distribution function (ecdf) for a
        data sample at values x.
        
        Args:
            x (float or ndarray):
                The value(s) at which the ecdf is evaluated
               
            data (float or ndarray):
                A sample for which to compute the ecdf. 
                
        Returns: ecdf_vals (ndarray):            
            The ecdf for data, evaluated at x.            
        '''
        
        if isinstance(x,float):
            #if x comes in as float, turn it into a numpy array
            x = np.array([x])
    
        if isinstance(data,float):
            #if data comes in as float, turn it into a numpy array
            data = np.array([data])  
    
       
        # sort the values of X_samp from smallest to largest
        xs = np.sort(data)
        
        # get the sample size of xs satisfying xs<=x for each x
        def func(vals):
            return len(xs[xs<=vals])
        
        ys = [len(xs[xs<=vals]) for vals in x]
        
    
        return np.array(ys)/float(len(xs))
    
    def mvnorm_rvs(self, mu, cov, A=None, size=1, return_uncorrelated=False):
        '''
        A much more efficient way of generating random samples from a multivariate
        normal distribution than what's done in scipy.stats
        '''
        M = size
        N = len(mu)
        if A is None:
            A = np.linalg.cholesky(cov)
            
        samp_uncorr = np.random.randn(N,M)
        
        samp_corr = mu[:,np.newaxis] + np.dot(A,samp_uncorr)
        
        if return_uncorrelated:
            return samp_corr.T, samp_uncorr.T
        else:
            return samp_corr.T
        
    def rank_histogram(self, X, Y):
        # rank histogram
        Nexp = X.shape[0]
        Nbins = X.shape[1] + 1
        joint_samp = np.concatenate((np.array([Y]),X.T)).T
        order_y = np.argsort(joint_samp,axis=1)
        rank_y = np.argsort(order_y,axis=1)[:,0]
        a_k = np.asarray([len(rank_y[rank_y==bb]) for bb in range(Nbins)])
        E_k = Nexp/float(Nbins)  
        
        return a_k, E_k

    def crps_decomp(self, X, Y, W=None):
        '''
        Function for computing the decomposition of the continuous rank
        probability score (CRPS), following the method outlined in 
        Hersbach 2000.
        
        Args:
            X (ndarray), shape=(Nsamp, Nens):
                Array containing the forecasts, where Nsamp is the number of 
                samples (e.g. over space/time), and Nens is the ensemble size.
            Y (ndarray), shape=(Nsamp,):
                Array containing the observations
            W (ndarray), shape=(Nsamp,) optional:
                Array containing the weight assigned to each sample. If not provided
                each weight is assigned a value 1/Nsamp
                
        Returns:            
            crps (float):
                The CRPS averaged over the sample. This is a weighted average if 
                W is provided as an input argument.
                
            crps_rel (float):
                The reliability component of the CRPS.
                
            crps_pot (float):
                The potential component of the CRPS.
                
            crps_unc (float):
                The uncertainty component of the CRPS.
                
            crps_res (float):
                The resolution component of the CRPS.
                
            cprs_k (ndarray), shape=(Nsamp,):
                The CRPS for each sample.
             
        '''
        
        # get number of samples and ensemble size from X
        Nsamp, Nens = X.shape
        
        # make sure Y is shape Nsamp
        if len(Y) != Nsamp:
            print("Number of samples in Y doesn't match number of samples in X")
            raise(ValueError)
    
        # make sure W (if provided) is shape Nsamp,
        # and that values sum to 1; if not        
        if W:
            if len(W) != Nsamp:
                print("Number of weights in W doesn't match number of samples Nsamp")
                raise(ValueError)            
            W = W/np.sum(W)
        else:
            W = np.ones(Nsamp)/float(Nsamp)  
    
        # prep arrays to be filled
        beta_ki = np.zeros((Nsamp, Nens+1))
        alpha_ki = np.zeros((Nsamp, Nens+1))
        c_ki = np.zeros((Nsamp, Nens+1))
        H_0 = np.zeros(Nsamp)
        H_N = np.ones(Nsamp)
        
        # for computing crps_unc
        # need weights sorted by the sorting of y
        inds_Y_sort = np.argsort(Y)
        Y_sort = Y[inds_Y_sort]
        W_sort = W[inds_Y_sort]
        p_k = 0.0
        # crps_unc = 0.0
        crps_unc_k = np.zeros(Nsamp)   
        
        # cumulative distribution function for the forecast (size Nens+1)
        # p = 0 for index 0, p=1 for index Nens
        p_i = np.asarray([i/Nens for i in range(Nens+1)])
        for kk in range(Nsamp):
            # extract forecast and observation for current sample
            # and sort forecast ensemble from smallest to largest
            x = np.sort(X[kk])
            y = Y[kk]
    
            if y<=x[0]:
                # y is less than (or equal to) smallest member
                # note all alpha's are 0
                beta_ki[kk][0] = x[0] - y
                beta_ki[kk][1:-1] = x[1:] - x[:-1]                
                H_0[kk] = 1.0
                
            elif y>=x[-1]:
                # y is greater than (or equal to) largest member
                # note all beta's are 0
                alpha_ki[kk][-1] = y - x[-1]
                alpha_ki[kk][1:-1] = x[1:] - x[:-1]
                H_N[kk] = 0.0          
            else:
                # y is within the range of the ensemble (exclusive of end values)
                
                # identify bin index where observation lies
                # this corresponds to the index of alpha and beta
                # such that alpha and beta are computed with y
                bin_ind_y = np.where(x<y)[0][-1]          
                
                # first and last elements of alpha are zero
                alpha_ki[kk][0] = 0.0
                alpha_ki[kk][-1] = 0.0
                x_list_ip1 = np.append(x[1:(bin_ind_y+1)], y)
                x_list_i = x[0:(bin_ind_y+1)]
                alpha_ki[kk][1:-1][0:(bin_ind_y+1)] = x_list_ip1 - x_list_i 
                
                # first and last elements of beta are zero
                beta_ki[kk][0] = 0.0
                beta_ki[kk][-1] = 0.0
                x_list_ip1 = x[(bin_ind_y+1):]
                x_list_i = np.append(y, x[(bin_ind_y+1):-1])
                beta_ki[kk][1:-1][bin_ind_y:] = x_list_ip1 - x_list_i
             
            
            # update the uncertainty component
            if kk<Nsamp-1:
                p_k += W_sort[kk]
                # crps_unc += p_k*(1.0 - p_k)*(Y_sort[kk+1] - Y_sort[kk])
                crps_unc_k[kk] = p_k * (1.0 - p_k) * (Y_sort[kk + 1] - Y_sort[kk])
                        
            
        # crps for each case (sum c over ensemble)
        c_ki = alpha_ki*p_i[np.newaxis]**2. + beta_ki[np.newaxis]*(1-p_i)**2.
        crps_k = np.sum(c_ki, axis=-1)
    
        # averages
        alpha_i = np.sum(W[:,np.newaxis]*alpha_ki, axis=0)
        beta_i = np.sum(W[:,np.newaxis]*beta_ki, axis=0)
        # average crps
        crps = np.sum(alpha_i * p_i**2. + beta_i*(1.0 - p_i)**2.)    
        # average bin width
        g_i = alpha_i + beta_i
        # related to average frequency that the 
        # observation is below midpoint of each bin
        o_i = np.zeros(Nens+1)
        good_inds = np.where(alpha_i + beta_i != 0.0)
        o_i[good_inds] = beta_i[good_inds]/(alpha_i[good_inds] + beta_i[good_inds])
        o_i[0] = np.sum(H_0)/Nsamp
        o_i[-1] = np.sum(H_N)/Nsamp
    
        if o_i[0] != 0.0:
            g_i[0] = beta_i[0]/o_i[0]
        else:
            g_i[0] = 0.0
        
        if o_i[-1] != 1.0:
            g_i[-1] = alpha_i[-1]/(1.0 - o_i[-1])
        else:
            g_i[-1] = 1.0        
        
        
        crps_unc = np.sum(crps_unc_k)
        
        # reliability component of crps
        crps_rel = np.sum(g_i * (o_i - p_i)**2.0)
        crps_pot = np.sum(g_i * o_i * (1.0 - o_i))
        
        # resolution component of the crps
        crps_res = crps_rel - crps + crps_unc 

        crps_result = namedtuple('crps_result', ['crps','crps_rel','crps_res','crps_unc', 'crps_pot'])
        return crps_result(crps, crps_rel, crps_res, crps_unc, crps_pot)

    def crps_decomp_test(self, X, Y, W=None):
        '''
        Function for computing the decomposition of the continuous rank
        probability score (CRPS), following the method outlined in 
        Hersbach 2000.
        
        Args:
            X (ndarray), shape=(Nsamp, Nens):
                Array containing the forecasts, where Nsamp is the number of 
                samples (e.g. over space/time), and Nens is the ensemble size.
            Y (ndarray), shape=(Nsamp,):
                Array containing the observations
            W (ndarray), shape=(Nsamp,) optional:
                Array containing the weight assigned to each sample. If not provided
                each weight is assigned a value 1/Nsamp
                
        Returns:            
            crps (float):
                The CRPS averaged over the sample. This is a weighted average if 
                W is provided as an input argument.
                
            crps_rel (float):
                The reliability component of the CRPS.
                
            crps_pot (float):
                The potential component of the CRPS.
                
            crps_unc (float):
                The uncertainty component of the CRPS.
                
            crps_res (float):
                The resolution component of the CRPS.
                
            cprs_k (ndarray), shape=(Nsamp,):
                The CRPS for each sample.
             
        '''
        
        # get number of samples and ensemble size from X
        Nsamp, Nens = X.shape
        
        # make sure Y is shape Nsamp
        if len(Y) != Nsamp:
            print("Number of samples in Y doesn't match number of samples in X")
            raise(ValueError)
    
        # make sure W (if provided) is shape Nsamp,
        # and that values sum to 1; if not        
        if W:
            if len(W) != Nsamp:
                print("Number of weights in W doesn't match number of samples Nsamp")
                raise(ValueError)            
            W = W/np.sum(W)
        else:
            W = np.ones(Nsamp)/float(Nsamp)  
    
        # prep arrays to be filled
        beta_ki = np.zeros((Nsamp, Nens+1))
        alpha_ki = np.zeros((Nsamp, Nens+1))
        c_ki = np.zeros((Nsamp, Nens+1))
        
        # for computing crps_unc
        # need weights sorted by the sorting of y
        inds_Y_sort = np.argsort(Y)
        Y_sort = Y[inds_Y_sort]
        W_sort = W[inds_Y_sort]
        p_k = 0.0
        crps_unc = 0.0
              
        # cumulative distribution function for the forecast (size Nens+1)
        # p = 0 for index 0, p=1 for index Nens
        p_i = np.asarray([i/Nens for i in range(Nens+1)])
        o_0 = 0.0
        o_N = 0.0
        
        X_sorted = np.sort(X, axis=1)
        
        inds_low = np.where(Y <= X_sorted[:,0])
        inds_high = np.where(Y >= X_sorted[:,-1])
        inds_mid = np.where((Y > X_sorted[:,0])&(Y < X_sorted[:,-1]))
        
        # y <= x[0]
        beta_ki[inds_low,0] = X_sorted[inds_low,0] - Y[inds_low]
        beta_ki[inds_low,1:-1] = X_sorted[inds_low,1:] - X_sorted[inds_low,:-1]
        o_0 = np.sum(W[inds_low])
        o_N = np.sum(W[inds_low])
        
        # y >= x[-1]
        alpha_ki[inds_high,-1] = Y[inds_high] - X_sorted[inds_high,-1]
        alpha_ki[inds_high,1:-1] = X_sorted[inds_high,1:] - X_sorted[inds_high,:-1]
        
        # y > x[0] and y < x[-1]
                    
        o_N = np.sum(W[inds_mid])     
        
        ######
        c_ki = alpha_ki*p_i**2. + beta_ki*(1-p_i)**2.
        p_k[:-1] = sum(W_sort[:-1])

        crps_unc = np.sum(p_k*(1.0 - p_k)*(Y_sort[1:] - Y_sort[:-1]))    
            
        # crps for each case (sum c over ensemble)
        crps_k = np.sum(c_ki, axis=-1)
        # crps_k_fair = crps_k - crps_samperr
        
        # averages
        alpha_i = np.sum(W[:,np.newaxis]*alpha_ki, axis=0)
        beta_i = np.sum(W[:,np.newaxis]*beta_ki, axis=0)
        # average crps
        crps = np.sum(alpha_i * p_i**2. + beta_i*(1.0 - p_i)**2.)    
        
        # related to average frequency that the 
        # observation is below midpoint of each bin
        o_i = np.zeros(Nens+1)
        good_inds = np.where(alpha_i + beta_i != 0.0)
        o_i[good_inds] = beta_i[good_inds]/(alpha_i[good_inds] + beta_i[good_inds])
        o_i[0] = o_0
        o_i[-1] = o_N 
 
        # average bin width
        g_i = alpha_i + beta_i
        g_i[0] = beta_i[0]/o_i[0]
        g_i[-1] =  alpha_i[-1]/(1 - o_i[-1])
 
        # reliability component of crps
        crps_rel = np.sum(g_i * (o_i - p_i)**2.0)
        crps_pot = np.sum(g_i * o_i * (1.0 - o_i))
        
        # resolution component of the crps
        crps_res = crps_rel - crps + crps_unc 
        
        # return crps, crps_rel, crps_pot, crps_unc, crps_res, crps_k, crps_k_fair
        result = namedtuple('result', ('crps', 'crps_rel', 'crps_pot', 'crps_unc', 
                                       'crps_res'))
        return result(crps, crps_rel, crps_pot, crps_unc, crps_res)


    def crps_fair(self, X, Y, W=None):
        '''
        Function for computing the fair CRPS.
        
        Args:
            X (ndarray), shape=(Nsamp, Nens):
                Array containing the forecasts, where Nsamp is the number of 
                samples (e.g. over space/time), and Nens is the ensemble size.
            Y (ndarray), shape=(Nsamp,):
                Array containing the observations
            W (ndarray), shape=(Nsamp,) optional:
                Array containing the weight assigned to each sample. If not provided
                each weight is assigned a value 1/Nsamp
                
        Returns:            
            crpsf (float):
                The CRPS averaged over the sample. This is a weighted average if 
                W is provided as an input argument.
                
        '''
        
        # get number of samples and ensemble size from X
        Nsamp, Nens = X.shape
        
        # make sure Y is shape Nsamp
        if len(Y) != Nsamp:
            print("Number of samples in Y doesn't match number of samples in X")
            raise(ValueError)
    
        # make sure W (if provided) is shape Nsamp,
        # and that values sum to 1; if not        
        if W:
            if len(W) != Nsamp:
                print("Number of weights in W doesn't match number of samples Nsamp")
                raise(ValueError)            
            W = W/np.sum(W)
        else:
            W = np.ones(Nsamp)/float(Nsamp)  
    
        # prep arrays to be filled
        beta_ki = np.zeros((Nsamp, Nens+1))
        alpha_ki = np.zeros((Nsamp, Nens+1))
        c_ki = np.zeros((Nsamp, Nens+1))
        
        # for computing crps_unc
        # need weights sorted by the sorting of y
        inds_Y_sort = np.argsort(Y)
        Y_sort = Y[inds_Y_sort]
        W_sort = W[inds_Y_sort]
        p_k = 0.0
        crps_unc = 0.0
           
        # cumulative distribution function for the forecast (size Nens+1)
        # p = 0 for index 0, p=1 for index Nens
        p_i = np.asarray([i/Nens for i in range(Nens+1)])
        for kk in range(Nsamp):
            # extract forecast and observation for current sample
            # and sort forecast ensemble from smallest to largest
            x = np.sort(X[kk])
            y = Y[kk]
    
            if y<=x[0]:
                # y is less than (or equal to) smallest member
                # note all alpha's are 0
                beta_ki[kk][0] = x[0] - y
                beta_ki[kk][1:-1] = x[1:] - x[:-1]
                
            elif y>=x[-1]:
                # y is greater than (or equal to) largest member
                # note all beta's are 0
                alpha_ki[kk][-1] = y - x[-1]
                alpha_ki[kk][1:-1] = x[1:] - x[:-1]
           
            else:
                # y is within the range of the ensemble (exclusive of end values)
                
                # identify bin index where observation lies
                # this corresponds to the index of alpha and beta
                # such that alpha and beta are computed with y
                bin_ind_y = np.where(x<y)[0][-1]          
                
                # first and last elements of alpha are zero
                alpha_ki[kk][0] = 0.0
                alpha_ki[kk][-1] = 0.0
                x_list_ip1 = np.append(x[1:(bin_ind_y+1)], y)
                x_list_i = x[0:(bin_ind_y+1)]
                alpha_ki[kk][1:-1][0:(bin_ind_y+1)] = x_list_ip1 - x_list_i 
                
                # first and last elements of beta are zero
                beta_ki[kk][0] = 0.0
                beta_ki[kk][-1] = 0.0
                x_list_ip1 = x[(bin_ind_y+1):]
                x_list_i = np.append(y, x[(bin_ind_y+1):-1])
                beta_ki[kk][1:-1][bin_ind_y:] = x_list_ip1 - x_list_i
             
            c_ki[kk] = alpha_ki[kk]*p_i**2. + beta_ki[kk]*(1-p_i)**2.
            
            # update the uncertainty component
            if kk<Nsamp-1:
                p_k += W_sort[kk]
                crps_unc += p_k*(1.0 - p_k)*(Y_sort[kk+1] - Y_sort[kk])
                        
            
        # crps for each case (sum c over ensemble)
        crps_k = np.sum(c_ki, axis=-1)
        
        # averages
        alpha_i = np.sum(W[:,np.newaxis]*alpha_ki, axis=0)
        beta_i = np.sum(W[:,np.newaxis]*beta_ki, axis=0)
        # average crps
        crps = np.sum(alpha_i * p_i**2. + beta_i*(1.0 - p_i)**2.)    
        # average bin width
        g_i = alpha_i + beta_i
        # related to average frequency that the 
        irange = np.arange(1,Nens)
        crps_fair = crps - 1/(Nens**2.*(Nens-1))*np.sum(irange*(Nens-irange)*g_i[1:-1])

        return crps_fair


    def get_fcst_probs(self, x, y, clim_percentile, d_rel=0.1):
        Nexp, Nens = x.shape
        # brier score / reliability diagram
        p_k = round_nearest(np.arange(0.0, 1.0 + d_rel, d_rel),d_rel) 
        p_fcst_bn = np.asarray([len(x[xx][x[xx]<=clim_percentile[0]])/float(Nens) for xx in range(Nexp)])
        p_fcst_nn = np.asarray([len(x[xx][(x[xx]>clim_percentile[0])&(x[xx]<=clim_percentile[1])])/float(Nens) for xx in range(Nexp)])
        p_fcst_an = np.asarray([len(x[xx][x[xx]>clim_percentile[1]])/float(Nens) for xx in range(Nexp)])
        
        p_fcst_bn = round_nearest(p_fcst_bn, 0.1)
        p_fcst_nn = round_nearest(p_fcst_nn, 0.1)
        p_fcst_an = round_nearest(p_fcst_an, 0.1)

        p_obs_bn = np.zeros(Nexp)
        p_obs_bn[y<=clim_percentile[0]] = 1.0
        p_obs_nn = np.zeros(Nexp)
        p_obs_nn[(y>clim_percentile[0])&(y<=clim_percentile[1])] = 1.0
        p_obs_an = np.zeros(Nexp)
        p_obs_an[y>clim_percentile[1]] = 1.0
        
        return p_k, p_fcst_bn, p_fcst_nn, p_fcst_an, p_obs_bn, p_obs_nn, p_obs_an

    
    def brier_decomp(self, p_fcst, p_obs, p_k, area=None, n_thresh=0.0):
        '''
        Function for computing the relevant statistics for assessing forecast reliability 
        and computing various terms in the Brier Score. Note that the `p_fcst' and `p_obs' arrays 
        can contain ``np.nan'' values, such as when masking is used. This is useful for evaluating sea ice concentration forecasts
        since you may want to mask out places where sea ice is never observed or when sea ice concnentration
        varies very little (e.g. sometimes I will use a mask based on standard devation exceeding 10%
        so that the `p_fcst' array doesn't become dominated by locations where sea ice variability is negligable)
        
        Args:
            p_fcst (ndarray):
                Array containing forecast probabilities (values between zero and one inclusive)
                
            p_obs (ndarray):
                Array containing observed probabilities (zero when event is not observed, one when event is observed)
                of same shape as p_fcst
                
            p_k (ndarray):
                Array containing the values of the bins that the forecast probabilities will be grouped into
                
            area (ndarray):
                Array containing the area of each grid cell corresponding to the spatial locations of p_fcst and p_obs
                of same shape as p_fcst
                
        Returns:
            o_k (ndarray):
                Array containing the observed relative frequency for the event for each bin k, of size p_k 
                
            n_k (ndarray):
                Array containing the number of forecasts in bin k, of size p_k
                
            rel (float):
                The "reliability" term in the Brier Score
                
            res (float):
                The "resolution" term in the Brier Score
                
            unc (float):
                The "uncertainty" term in the Brier Score
                
            p_hat (float):
                The climatological forecast probability for the event
                
            o_hat (float):
                The climatological observed probability for the event
        '''
        if area is None:
            area = np.ones(len(p_obs))
        
        nbins = len(p_k)
        o_hat = np.nansum(p_obs*area)/np.nansum(area) # compute the observed climatological frequency of the event
        p_hat = np.nansum(p_fcst*area)/np.nansum(area)
        
        rel_k = np.zeros(nbins)
        res_k = np.zeros(nbins)
        o_k = np.zeros(nbins)
        w_k = np.zeros(nbins)
        n_k = np.zeros(nbins)
        
        for ii in np.arange(nbins): # loop through the different probability bins
            o_k_curr = p_obs[p_fcst==p_k[ii]]      # the subset of observation corresponding to when a probability p_k[ii] is issued        
            area_k = area[p_fcst==p_k[ii]]  # the subset of gridcell areas for the locations where a probability p_k[ii] is issued
    
            n_k[ii] = len(p_fcst[p_fcst==p_k[ii]]) # number of forecasts where a probability p_k[ii] is issued
            w_k[ii] = np.nansum(area_k) # the total area of all grid cells where a probability p_k[ii] is issued (for weighting)
            o_k[ii] = np.nansum(o_k_curr*area_k)/w_k[ii] # the observed relative frequency of the event when a probability p_k[ii] is issued
            rel_k[ii] = (o_k[ii]-p_k[ii])**2. # the reliability term of the brier score for bin k
            res_k[ii] = (o_k[ii]-o_hat)**2. # the resolution term of the brier score for bin k
              
        
        n_k[n_k<=n_thresh] = 0.0
        w_k[n_k==0.0] = np.nan
        o_k[n_k==0.0] = np.nan
        rel_k[n_k==0.0] = np.nan
        res_k[n_k==0.0] = np.nan
    
        rel = np.nansum(w_k*rel_k)/np.nansum(w_k) # reliability term computed as a weighted average 
        res = np.nansum(w_k*res_k)/np.nansum(w_k) # resolution term computed as a weighted average
        
        unc = o_hat*(1-o_hat) # uncertainty term computed as a weighted average (note o_hat definition above)
    
        # return o_k, n_k, rel
    
        return o_k, n_k, rel, res, unc, p_hat, o_hat
    
            