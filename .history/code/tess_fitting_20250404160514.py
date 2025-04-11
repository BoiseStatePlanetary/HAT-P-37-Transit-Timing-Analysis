import numpy as np
import os
import scipy

## HAT-P-37 b system 

def read_data_file(file_name):
    phot_data = np.loadtxt(file_name, delimiter=",")
    # correct time from BTJD to BJD
    tess_zero_time = 2457000.
    bjd_tdb = tess_zero_time + phot_data[:,0]
    return {"BJD_TDB" : bjd_tdb, 
            "flux" : phot_data[:, 1], 
            "error" : phot_data[:, 2]}

