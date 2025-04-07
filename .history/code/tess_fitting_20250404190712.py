import numpy as np
import os
from scipy.optimize import leastsq
import matplotlib.pyplot as plt

## HAT-P-37 b system parameters
# known_ref_mid["HAT-P-37b"] = 2457938.84392#±0.00016
# known_period["HAT-P-37b"] = 2.79744256#±0.00000041 Kokori

def read_data_file(file_name):
    phot_data = np.loadtxt(file_name, delimiter=",", skiprows=1)
    # correct time from BTJD to BJD
    tess_zero_time = 2457000.
    bjd_tdb = tess_zero_time + phot_data[:,0]
    # return np.column_stack((bjd_tdb, phot_data[:, 1], phot_data[:, 2]))
    return {"BJD_TDB" : bjd_tdb, 
            "flux" : phot_data[:, 1], 
            "error" : phot_data[:, 2]}

def plot_data(file_name, color_dict):
    photdata = read_data_file(file_name)
    file_split = file_name.split("_")
    planet_name = file_split[-2]
    sector = file_split[-1].split(".")[0]
    plt.errorbar(photdata["BJD_TDB"], photdata["flux"], yerr=photdata["error"], color='k', marker='none', ls='none', alpha=0.3, zorder=0)
    plt.plot(photdata["BJD_TDB"], photdata["flux"], color=color_dict[sector], marker=".", ls='none', label=f"{planet_name} {sector}") 

def combine_data_and_fit(data_files):
    data_dict = {}
    for i, file in enumerate(data_files):
        data_dict[i] = read_data_file(file)
    all_data = np.vstack((data_dict[0], data_dict[1], data_dict[2]))  

    all_times = all_data[:, 0] - np.min(all_data[:, 0])      

    # test out a sine fit
    guess_mean = np.mean(all_data[:, 1])
    guess_std = 3*np.std(all_data[:, 1])/(2**0.5)/(2**0.5)
    guess_phase = 0
    guess_freq = 0.6 # period of the eclipsing binary
    guess_amp = 0.6

    # we'll use this to plot our first estimate. This might already be good enough for you
    data_first_guess = guess_std*np.sin(all_times+guess_phase) + guess_mean

    # Define the function to optimize, in this case, we want to minimize the difference
    # between the actual data and our "guessed" parameters
    optimize_func = lambda x: x[0]*np.sin(x[1]*all_times+x[2]) + x[3] - all_data[:, 1]
    est_amp, est_freq, est_phase, est_mean = leastsq(optimize_func, [guess_amp, guess_freq, guess_phase, guess_mean])[0]

    # recreate the fitted curve using the optimized parameters
    data_fit = est_amp*np.sin(est_freq*all_data[:, 1]+est_phase) + est_mean

    # recreate the fitted curve using the optimized parameters

    fine_t = np.arange(0,max(all_times),)
    data_fit=est_amp*np.sin(est_freq*fine_t+est_phase)+est_mean

    plt.plot(all_data[:, 0], all_data[:, 1], '.')
    plt.plot(all_data[:, 0], data_first_guess, label='first guess')
    plt.plot(fine_t, data_fit, label='after fitting')
    plt.legend()
    plt.show()                              

    

if __name__ == "__main__":
    color_dict = {"TESS-Sector-53" : "C0", 
                  "TESS-Sector-54" : "C1", 
                  "TESS-Sector-55" : "C2"}

    name_53 = "code/HAT-P-37b/TESS/folded_before_smoothing_HAT-P-37b_TESS-Sector-53.tsv"
    name_54 = "code/HAT-P-37b/TESS/folded_before_smoothing_HAT-P-37b_TESS-Sector-54.tsv"
    name_55 = "code/HAT-P-37b/TESS/folded_before_smoothing_HAT-P-37b_TESS-Sector-55.tsv"
    file_list = [name_53, name_54, name_55]
    # combine_data_and_fit(file_list)

    plot_data(name_53, color_dict)
    plot_data(name_54, color_dict)
    plot_data(name_55, color_dict)
    plt.legend()
    plt.show()