import numpy as np
import os
import scipy
import matplotlib.pyplot as plt

## HAT-P-37 b system parameters
# known_ref_mid["HAT-P-37b"] = 2457938.84392#±0.00016
# known_period["HAT-P-37b"] = 2.79744256#±0.00000041 Kokori

def read_data_file(file_name):
    phot_data = np.loadtxt(file_name, delimiter=",", skiprows=1)
    # correct time from BTJD to BJD
    tess_zero_time = 2457000.
    bjd_tdb = tess_zero_time + phot_data[:,0]
    return {"BJD_TDB" : bjd_tdb, 
            "flux" : phot_data[:, 1], 
            "error" : phot_data[:, 2]}

def plot_data(file_name, color_dict):
    photdata = read_data_file(file_name)
    file_split = file_name.split("_")
    planet_name = file_split[-2]
    sector = file_split[-1]
    plt.errorbar(photdata["BJD_TDB"], photdata["flux"], yerr=photdata["error"], color='k', marker='.', ls='none', alpha=0.3)
    plt.plot(photdata["BJD_TDB"], photdata["flux"], color=color_dict[sector], label=f"{planet_name} {sector}") 
    

if __name__ == "__main__":
    color_dict = {"TESS-Sector-53" : }

    name_53 = "code/HAT-P-37b/TESS/folded_before_smoothing_HAT-P-37b_TESS-Sector-53.tsv"
    name_54 = "code/HAT-P-37b/TESS/folded_before_smoothing_HAT-P-37b_TESS-Sector-54.tsv"
    name_55 = "code/HAT-P-37b/TESS/folded_before_smoothing_HAT-P-37b_TESS-Sector-55.tsv"
    plot_data(name_53)
    plot_data(name_54)
    plot_data(name_55)
    plt.legend()
    plt.show()