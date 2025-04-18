import numpy as np
from scipy.integrate import quad
import pandas as pd

BoiseState_blue = "#0033A0"
BoiseState_orange = "#D64309"
aspect_ratio = 16./9.

# This is the REAL DATA for WASP-12 b pulled from Yee et al.
url = "https://raw.githubusercontent.com/BoiseStatePlanetary/susie/refs/heads/main/example_data/wasp12b_tra_occ.csv"
data = pd.read_csv(url)
tra_or_occs = np.array(data["tra_or_occ"])
epochs = np.array(data["epoch"].astype('int'))
mid_times = np.array(data["mid_time"])
mid_time_errs = np.array(data["mid_time_err"])

sigma = np.median(mid_time_errs)

# Taken from Yee+ (2020)
T0_precession = 2456305.45488
Ps_precession = 1.091419633
e = 0.00310
omega_0 = 2.62 # radians
domega_dE = 0.000984 # rad per orbit
Pa = Ps_precession*(1. - domega_dE/2./np.pi)

# WASP12b_scaling_factor = e*Pa/np.pi/sigma

T0_decay = 2456305.455809
Ps_decay = 1.091420107
dPdE_decay = -10.04e-10

# 2025 Feb 20 - Just as a check, I want to see how different the eccentricity metric would be if I only include part of the dataset.

#print("WARNING! You're only using part of the WASP-12 b dataset!!")
#ind = epochs < 250
#E = epochs[ind]
#E_tra = epochs[ind][tra_or_occs[ind] == "tra"]
#E_occ = epochs[ind][tra_or_occs[ind] == "occ"]
#t = mid_times[ind]
#ttra = mid_times[ind][tra_or_occs[ind] == "tra"]
#tocc = mid_times[ind][tra_or_occs[ind] == "occ"]
#sigma = mid_time_errs[ind]
#sigma_tra = mid_time_errs[ind][tra_or_occs[ind] == "tra"]
#sigma_occ = mid_time_errs[ind][tra_or_occs[ind] == "occ"]
#tra_or_occs = tra_or_occs[ind]

E = epochs
E_tra = E[tra_or_occs == "tra"]
E_occ = E[tra_or_occs == "occ"]
t = mid_times
ttra = mid_times[tra_or_occs == "tra"]
tocc = mid_times[tra_or_occs == "occ"]
sigma = mid_time_errs
sigma_tra = mid_time_errs[tra_or_occs == "tra"]
sigma_occ = mid_time_errs[tra_or_occs == "occ"]

def calc_omega(E, omega_0, domega_dE):
    return omega_0 + domega_dE*E

def calc_Pa(Ps, domega_dE):
    return Ps/(1. - 1./2/np.pi*domega_dE)

def calc_ttra_precession(E, T0, Ps, e, omega_0, domega_dE):
    omega = calc_omega(E, omega_0, domega_dE)
    Pa = calc_Pa(Ps, domega_dE)

    return T0 + E*Ps - e*Pa/np.pi*np.cos(omega)

def calc_tocc_precession(E, T0, Ps, e, omega_0, domega_dE):
    omega = calc_omega(E, omega_0, domega_dE)
    Pa = calc_Pa(Ps, domega_dE)

    return T0 + Pa/2. + E*Ps + e*Pa/np.pi*np.cos(omega)

def calc_ttra_decay(E, T0, Ps, dP_dE):
    return T0 + Ps*E + 0.5*dP_dE*E*E

def calc_tocc_decay(E, T0, Ps, dP_dE):
    return T0 + Ps*(E + 1/2) + 0.5*dP_dE*E**2

def calc_anal_sigma_T0(E, sigma, T0, Ps, e, omega_0, domega_dE):
    return np.sqrt(np.sum(sigma**2))    

def calc_anal_sigma_Ps(E, sigma, T0, Ps, e, omega_0, domega_dE):
    # Only good for transit data; need to expand to occultations!
    return np.sqrt(np.sum(sigma**2/E**2))

def calc_anal_sigma_e(E, sigma, T0, Ps, e, omega_0, domega_dE):
    omega = calc_omega(E, omega_0, domega_dE)
    
    return (np.pi/Ps)*np.sqrt(np.sum(sigma**2/np.cos(omega)**2))

def calc_anal_sigma_omega_0(E, sigma, T0, Ps, e, omega_0, domega_dE):
    omega = calc_omega(E, omega_0, domega_dE)

    return (np.pi/e/Ps)*np.sqrt(np.sum(sigma**2/np.sin(omega)**2))

def calc_anal_sigma_omega_domega_dE(E, sigma, T0, Ps, e, omega_0, domega_dE):
    omega = calc_omega(E, omega_0, domega_dE)

    return (np.pi/e/Ps)*np.sqrt(np.sum(sigma**2/E**2/np.sin(omega)**2))

def calc_analytic_ecc(Delta_t_prime, sigma_Delta_t_prime, cos_omega):
    return -np.sum(Delta_t_prime*cos_omega/sigma_Delta_t_prime**2)/np.sum(cos_omega**2/sigma_Delta_t_prime**2)

def calc_analytic_sigma_ecc(sigma_Delta_t_prime, cos_omega):
    return np.sqrt(1./np.sum(cos_omega**2/sigma_Delta_t_prime**2))

def calc_S(x, sigma):
    return np.sum(1./sigma/sigma)
def calc_Sxx(x, sigma):
    return np.sum(x*x/sigma/sigma)
def calc_Sx(x, sigma):
    return np.sum(x/sigma/sigma)
def calc_Sy(x, y, sigma):
    return np.sum(y/sigma/sigma)
def calc_Sxy(x, y, sigma):
    return np.sum(x*y/sigma/sigma)
def calc_Sxxx(x, sigma):
    return np.sum(x*x*x/sigma/sigma)
def calc_Sxxxx(x, sigma):
    return np.sum(x*x*x*x/sigma/sigma)
def calc_Scosw(x, cosw, sigma):
    return np.sum(cosw/sigma/sigma)
def calc_Sxcosw(x, cosw, sigma):
    return np.sum(x*cosw/sigma/sigma)
def calc_Sxxcosw(x, cosw, sigma):
    return np.sum(x*x*cosw/sigma/sigma)
def calc_Scos2w(x, cosw, sigma):
    return np.sum(cosw*cosw/sigma/sigma)

def chisqg(ydata,ymod,sd=None):
    """
    Returns the chi-square error statistic as the sum of squared errors between
    Ydata(i) and Ymodel(i). If individual standard deviations (array sd) are supplied,
    then the chi-square error statistic is computed as the sum of squared errors
    divided by the standard deviations.     Inspired on the IDL procedure linfit.pro.
    See http://en.wikipedia.org/wiki/Goodness_of_fit for reference.

    x,y,sd assumed to be Numpy arrays. a,b scalars.
    Returns the float chisq with the chi-square statistic.

    Rodrigo Nemmen
    http://goo.gl/8S1Oo
    """
    # Chi-square statistic (Bevington, eq. 6.9)
    if np.all(sd==None):
        chisq=np.sum((ydata-ymod)**2)
    else:
        chisq=np.sum( ((ydata-ymod)/sd)**2 )

    return chisq

def redchisqg(ydata,ymod,deg=2,sd=None):
    """
    Returns the reduced chi-square error statistic for an arbitrary model,
    chisq/nu, where nu is the number of degrees of freedom. If individual
    standard deviations (array sd) are supplied, then the chi-square error
    statistic is computed as the sum of squared errors divided by the standard
    deviations. See http://en.wikipedia.org/wiki/Goodness_of_fit for reference.

    ydata,ymod,sd assumed to be Numpy arrays. deg integer.

      Usage:
          >>> chisq=redchisqg(ydata,ymod,n,sd)
          where
          ydata : data
          ymod : model evaluated at the same x points as ydata
          n : number of free parameters in the model
          sd : uncertainties in ydata

          Rodrigo Nemmen
          http://goo.gl/8S1Oo
    """
    # Chi-square statistic
    if np.all(sd==None):
        chisq=np.sum((ydata-ymod)**2)
    else:
        chisq=np.sum( ((ydata-ymod)/sd)**2 )

    # Number of degrees of freedom
    nu=ydata.size-1-deg

    return chisq/nu

def calc_BIC(ydata,ymod,deg=2,sd=None):
    """
    Calculate BIC
    """

    chisq = chisqg(ydata,ymod,sd=sd)
    penalty = deg*np.log(len(ydata))

    return chisq + penalty

def calc_T0(x, y, sigma):
    S = calc_S(x, sigma)
    Sxx = calc_Sxx(x, sigma)
    Sx = calc_Sx(x, sigma)
    Delta = S*Sxx - Sx*Sx

    Sy = calc_Sy(x, y, sigma)
    Sxy = calc_Sxy(x, y, sigma)

    return (Sxx*Sy - Sx*Sxy)/Delta

def calc_T0_correction(x, sigma):
    S = calc_S(x, sigma)
    Sx = calc_Sx(x, sigma)
    Sxx = calc_Sxx(x, sigma)
    Sxxx = calc_Sxxx(x, sigma)

    return (Sxx*Sxx - Sx*Sxxx)/(Sxx*S - Sx*Sx)

def calc_P_correction(x, sigma):
    S = calc_S(x, sigma)
    Sx = calc_Sx(x, sigma)
    Sxx = calc_Sxx(x, sigma)
    Sxxx = calc_Sxxx(x, sigma)

    return (S*Sxxx - Sx*Sxx)/(S*Sxx - Sx*Sx)

def calc_P(x, y, sigma):
    S = calc_S(x, sigma)
    Sxx = calc_Sxx(x, sigma)
    Sx = calc_Sx(x, sigma)
    Delta = S*Sxx - Sx*Sx

    Sy = calc_Sy(x, y, sigma)
    Sxy = calc_Sxy(x, y, sigma)

    return (S*Sxy - Sx*Sy)/Delta

def calc_analytic_chisq(E, sigma, dPdE):
    Delta_P_prime = calc_P_correction(E, sigma)
    Delta_T0_prime = calc_T0_correction(E, sigma)
    Sxxxx = calc_Sxxxx(E, sigma)
    Sxx = calc_Sxx(E, sigma)
    S = calc_S(E, sigma)
    Sxxx = calc_Sxxx(E, sigma)
    Sx = calc_Sx(E, sigma)
        
    chi_sq = 0.25*dPdE**2*np.sum(((E**2 -\
            Delta_P_prime*E - Delta_T0_prime)/sigma)**2) + (len(E) - 2.)
    return chi_sq


