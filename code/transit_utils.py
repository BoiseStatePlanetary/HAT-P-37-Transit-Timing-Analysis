# Collection of utility functions for working with transit analysis
import numpy as np

def num_elapsed_transits(input_mid, ref_per, ref_mid ):
    """ How many transits is the input_mid since since the reference transit at ref_mid (using orbital ref_per). """
    n_elapsed = round((input_mid - ref_mid) / ref_per)
    return n_elapsed

def predict_linear_eph(ntr, period, ref_mid):
    """ A linear transit ephemeris. """
    return ref_mid + period * ntr
    
def predict_decaying_eph(ntr, period, ref_time, delta_p):
    """ Transit ephemeris has a linear rate of orbital decay. From Adams et al 2010a (ApJ 721 1829).\\
    delta_p = amount by which orbital period changes per orbit, assuming P = P0 + delta_P * N. \\
    Note that Pdot = delta_P x 365.25 / P0 """
    return ref_time + period * ntr + delta_p * ntr * (ntr-1)/2.
    
def get_nearest_transit(tt, ref_per, ref_mid):
    nearest_n = num_elapsed_transits(tt, ref_per, ref_mid )
    pred_tmid = predict_linear_eph(nearest_n, ref_per, ref_mid)
    return pred_tmid 


# Carter Model: https://iopscience.iop.org/article/10.1086/592321/pdf
def Carter_model(time, tc, f0, delta, T, tau):
    # tc - central transit time
    # f0 - out-of-transit baseline
    # delta - difference in flux from the background to the deepest part of the transit f0*r**2
    # T - transit duration (full planet in front of the star) 2*tau0*sqrt(1-b**2) 
    # tau =  duration of ingress / egress 2*tau0*r**2/sqrt(1 - b**2)

    flux = np.zeros_like(time)

    ind = np.abs(time - tc) <= (T/2. - tau/2.)
    if(len(time[ind]) > 0):
        flux[ind] = f0 - delta

    ind = ((T/2. - tau/2.) < np.abs(time - tc)) & ((T/2. + tau/2.) > np.abs(time - tc))
    if(len(time[ind]) > 0):
        flux[ind] = f0 - delta + (delta/tau)*(np.abs(time[ind] - tc) - T/2. + tau/2.)

    ind = np.abs(time - tc) >= (T/2. + tau/2.)
    if(len(time[ind]) > 0):
        flux[ind] = f0

    return flux

def calc_Carter_initial_guesses(lightcurve):
    # tc - central transit time
    # f0 - out-of-transit baseline
    # delta - difference in flux from the background to the deepest part of the transit f0*r**2
    # T - transit duration (full planet in front of the star) 2*tau0*sqrt(1-b**2) 
    # tau =  duration of ingress / egress 

    tc = lightcurve.time.value[np.argmin(lightcurve.flux.value)] # estimate the central transit time to be the time w/ the minimum flux value
    background = np.median(lightcurve.flux.value)  # The background value is very nearly the median for all the data (f0)
    
    # for delta, we approx. as f0 - min flux
    min_flux = np.min(lightcurve.flux.value)
    delta = background - min_flux
    
    # T = T2 to T3 all values from Bakos et. al 2012
    T14 = 0.0971  # days +/- 0.0015
    T12 = 0.0153  # days +/- 0.0013
    duration = T14 - (2*T12)

    # (T r) / (1-b^2)
    ratio_of_planet_to_star_radius = 0.1378  # Bakos et al. 2012 +/- 0.0030
    b = 0.505 # +0.041/-0.062
    tau = (duration * ratio_of_planet_to_star_radius) / (1-b**2)

    # Let's set our initial guesses - tc, f0, delta, T, tau
    initial_guesses = np.array([tc, background, delta, duration, tau])
    return initial_guesses