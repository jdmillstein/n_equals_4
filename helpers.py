#useful functions for rheology project
from __future__ import division

import sys, os, re, scipy
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.ndimage
from osgeo import gdal, gdalconst 
from osgeo.gdalconst import *
from scipy.signal import fftconvolve
from sklearn.linear_model import LinearRegression
from random import randint,gauss,seed



'''
Load_data and get_hdr_info functions found in this useful blog post
http://chris35wills.github.io/binary_geotiff_processing/
'''


### copied from SciPy Cookbook: http://scipy-cookbook.readthedocs.io/items/SavitzkyGolay.html
def sgolay2d ( z, window_size, order, derivative=None):
    """
    """
    # number of terms in the polynomial expression
    n_terms = ( order + 1 ) * ( order + 2)  / 2.0

    if  window_size % 2 == 0:
        raise ValueError('window_size must be odd')

    if window_size**2 < n_terms:
        raise ValueError('order is too high for the window size')

    half_size = window_size // 2

    # exponents of the polynomial.
    # p(x,y) = a0 + a1*x + a2*y + a3*x^2 + a4*y^2 + a5*x*y + ...
    # this line gives a list of two item tuple. Each tuple contains
    # the exponents of the k-th term. First element of tuple is for x
    # second element for y.
    # Ex. exps = [(0,0), (1,0), (0,1), (2,0), (1,1), (0,2), ...]
    exps = [ (k-n, n) for k in range(order+1) for n in range(k+1) ]

    # coordinates of points
    ind = np.arange(-half_size, half_size+1, dtype=np.float64)
    dx = np.repeat( ind, window_size )
    dy = np.tile( ind, [window_size, 1]).reshape(window_size**2, )

    # build matrix of system of equation
    A = np.empty( (window_size**2, len(exps)) )
    for i, exp in enumerate( exps ):
        A[:,i] = (dx**exp[0]) * (dy**exp[1])

    # pad input array with appropriate values at the four borders
    new_shape = z.shape[0] + 2*half_size, z.shape[1] + 2*half_size
    Z = np.zeros( (new_shape) )
    # top band
    band = z[0, :]
    Z[:half_size, half_size:-half_size] =  band -  np.abs( np.flipud( z[1:half_size+1, :] ) - band)
    # bottom band
    band = z[-1, :]
    Z[-half_size:, half_size:-half_size] = band  + np.abs( np.flipud( z[-half_size-1:-1, :] ) - band)
    # left band
    band = np.tile( z[:,0].reshape(-1,1), [1,half_size])
    Z[half_size:-half_size, :half_size] = band - np.abs( np.fliplr( z[:, 1:half_size+1] ) - band)
    # right band
    band = np.tile( z[:,-1].reshape(-1,1), [1,half_size] )
    Z[half_size:-half_size, -half_size:] =  band + np.abs( np.fliplr( z[:, -half_size-1:-1] ) - band)
    # central band
    Z[half_size:-half_size, half_size:-half_size] = z

    # top left corner
    band = z[0,0]
    Z[:half_size,:half_size] = band - np.abs( np.flipud(np.fliplr(z[1:half_size+1,1:half_size+1]) ) - band )
    # bottom right corner
    band = z[-1,-1]
    Z[-half_size:,-half_size:] = band + np.abs( np.flipud(np.fliplr(z[-half_size-1:-1,-half_size-1:-1]) ) - band )

    # top right corner
    band = Z[half_size,-half_size:]
    Z[:half_size,-half_size:] = band - np.abs( np.flipud(Z[half_size+1:2*half_size+1,-half_size:]) - band )
    # bottom left corner
    band = Z[-half_size:,half_size].reshape(-1,1)
    Z[-half_size:,:half_size] = band - np.abs( np.fliplr(Z[-half_size:, half_size+1:2*half_size+1]) - band )

    # solve system and convolve
    if derivative == None:
        m = np.linalg.pinv(A)[0].reshape((window_size, -1))
        return fftconvolve(Z, m, mode='valid')
    elif derivative == 'col':
        c = np.linalg.pinv(A)[1].reshape((window_size, -1))
        return fftconvolve(Z, -c, mode='valid')
    elif derivative == 'row':
        r = np.linalg.pinv(A)[2].reshape((window_size, -1))
        return fftconvolve(Z, -r, mode='valid')
    elif derivative == 'both':
        c = np.linalg.pinv(A)[1].reshape((window_size, -1))
        r = np.linalg.pinv(A)[2].reshape((window_size, -1))
        return fftconvolve(Z, -r, mode='valid'), fftconvolve(Z, -c, mode='valid')


def plot_regions(strain,thick,mask):
    '''
    simplify the process of plotting after loading all the files 
    '''
    # general constants
    rho_ice = 910.0 # kg / m-3
    rho_water = 1026.0 # kg / m-3
    gravity_acc = 9.81 # m / s-2
    gdash = gravity_acc*(1-rho_ice/rho_water)
    shear_Pa = 0.25*thick*gdash*rho_ice # in Pa

    # prep to make into dataframe
    shear = shear_Pa.flatten(); strain = strain.flatten(); mask = mask.flatten();
    
    if np.size(shear) != np.size(strain):
        print('sizes do not match')
    if np.size(shear) != np.size(mask):
        print('sizes do not match')
     
    # create dataframe and remove zero regions
    df = pd.DataFrame({'strain':strain,'shear':shear,'mask':mask})
    df_crop_shear = df[df.shear > 200000]
    df_crop_strain = df_crop_shear[df_crop_shear.strain > 0.0000005]

    # plotting for n without masking for regions in purely extensional regime
    x = np.log(df_crop_strain.shear)
    y = np.log(df_crop_strain.strain)
    model = LinearRegression().fit(x[:, None],y)
    plt.plot(x,model.coef_*x+model.intercept_,'b')
    plt.plot(x,y,linestyle='', marker='o', markersize=0.7,alpha=0.08)
    plt.xlabel('log(effective stress) Pa'); plt.ylabel('log(effective strain) $a^{-1}$')
    
    # remove regions in compression / not-extension
    df_masked = df_crop_strain[df_crop_strain['mask'] > np.float32(input())]
    x_log = np.log(df_masked.shear)
    y_log = np.log(df_masked.strain)
    model2 = LinearRegression().fit(x_log[:, None],y_log)
    plt.plot(x,model2.coef_*x+model2.intercept_,'g')
    plt.plot(x_log,y_log,linestyle='', marker='o', markersize=0.7,alpha=0.08,)
    plt.xlabel('log(effective stress Pa)'); plt.ylabel('log(effective strain $a^{-1}$)')
    
    print('all',model.coef_,model.intercept_,'masked',model2.coef_,model2.intercept_)
    return

def plot_figure(strain,thick,mask):
    '''
    simplify the process of plotting after loading all the files 
    '''
    # general constants
    rho_ice = 910.0 # kg / m-3
    rho_water = 1026.0 # kg / m-3
    gravity_acc = 9.81 # m / s-2
    gdash = gravity_acc*(1-rho_ice/rho_water)
    shear_Pa = 0.25*thick*gdash*rho_ice # in Pa

    # prep to make into dataframe
    shear = shear_Pa.flatten(); strain = strain.flatten(); mask = mask.flatten();
    
    if np.size(shear) != np.size(strain):
        print('sizes do not match')
    if np.size(shear) != np.size(mask):
        print('sizes do not match')
     
    # create dataframe and remove zero regions
    df = pd.DataFrame({'strain':strain,'shear':shear,'mask':mask})
    df_crop_shear = df[df.shear > 200000]
    df_crop_strain = df_crop_shear[df_crop_shear.strain > 0.0000005]

    # remove regions in compression / not-extension
    df_masked = df_crop_strain[df_crop_strain['mask'] > np.float32(input())]
    x_log = np.log10(df_masked.shear)
    y_log = np.log10(df_masked.strain)
    maskval = df_masked['mask']
    model2 = LinearRegression().fit(x_log[:, None],y_log)
    plt.plot(x_log,model2.coef_*x_log+model2.intercept_,'g')
    plt.plot(x_log,y_log,linestyle='', marker='o', markersize=0.7,alpha=0.08,)
    plt.xlabel('log(effective stress Pa)'); plt.ylabel('log(effective strain $a^{-1}$)')
    rsquared = model2.score(x_log[:, None],y_log)
    print('masked',model2.coef_,model2.intercept_,rsquared)

    return x_log, y_log, maskval, model2.coef_, model2.intercept_, rsquared

