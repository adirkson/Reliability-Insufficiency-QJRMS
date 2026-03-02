#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 27 15:27:18 2026

@author: ard000
"""
import numpy as np


def round_nearest(x, a, up=False, down=False): # function for rounding "prob_fcst" array to nearest "a=rn" for binning
    if up:            
        return np.ceil(x / a) * a 
    elif down:
        return np.floor(x / a) * a 
    else:
        return np.round(x / a) * a 
    

def mvnorm_rvs(mu, cov, A=None, size=1):
    '''
    Generates random samples from a multivariate
    normal distribution (faster than scipy's implementation)
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
