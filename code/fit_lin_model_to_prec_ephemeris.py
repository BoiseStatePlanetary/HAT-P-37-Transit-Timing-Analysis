"""
A collection of functions to estimate the corrections to T0 and Ps when fitting a linear model to a quadratic ephemeris
"""

import numpy as np
from utils import *

def calc_Delta(x, sigma):
    S = calc_S(x, sigma)
    Sx = calc_Sx(x, sigma)
    Sxx = calc_Sxx(x, sigma)

    term1 = Sxx*S
    term2 = -Sx*Sx

    return term1 + term2

def calc_T0_lin_factor(x, cosw, sigma):
    Delta = calc_Delta(x, sigma)

    Sx = calc_Sx(x, sigma)
    Sxx = calc_Sxx(x, sigma)
    Sxxx = calc_Sxxx(x, sigma)

    Scosw = calc_Scosw(x, cosw, sigma)
    Sxcosw = calc_Sxcosw(x, cosw, sigma)

    term1 = Sx*Sxcosw - Sxx*Scosw

    return term1/Delta

def calc_Ps_lin_factor(x, cosw, sigma):
    Delta = calc_Delta(x, sigma)

    S = calc_S(x, sigma)
    Sx = calc_Sx(x, sigma)

    Scosw = calc_Scosw(x, cosw, sigma)
    Sxcosw = calc_Sxcosw(x, cosw, sigma)

    term1 = Sx*Scosw - S*Sxcosw 

    return term1/Delta

def calc_Delta_BIC(x, cosw, sigma, scaling_factor):
    """
    Delta BIC for linear vs. precession model fit to precession ephemeris

    Args:
        x (float): orbit epochs
        cosw (float): cos(omega)
        sigma (float): timing uncertainties
        scaling_factor (float): e*Pa/pi
    Returns:
        Equation Delta_BIC_lin_fit_vs_prec_fit_to_prec_ephem from 
        Jackson+ (2025)
    """
    Delta_T0 = calc_T0_lin_factor(x, cosw, sigma)
    Delta_Ps = calc_Ps_lin_factor(x, cosw, sigma)
    return scaling_factor**2*\
        np.sum((cosw + x*Delta_Ps + Delta_T0)**2/sigma**2) -\
        3.*np.log(len(x)) + 3.
