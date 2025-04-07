import numpy as np
import os
import scipy
import matplotlib.pyplot as plt

## HAT-P-37 b system parameters
# known_ref_mid["HAT-P-37b"] = 2457938.84392#±0.00016
# known_period["HAT-P-37b"] = 2.79744256#±0.00000041 Kokori

def read_data_file(file_name):
    phot_data = np.loadtxt(file_name, delimiter=",")
    # correct time from BTJD to BJD
    tess_zero_time = 2457000.
    bjd_tdb = tess_zero_time + phot_data[:,0]
    return {"BJD_TDB" : bjd_tdb, 
            "flux" : phot_data[:, 1], 
            "error" : phot_data[:, 2]}

def plot_data(file_name):
    data_dict = read_data_file(file_name)
    plt.errorbar(time, photdata["flux"]+shift, yerr=photdata["error"], color=color, marker='.', ls='none', alpha=0.7, label=data_set)    

