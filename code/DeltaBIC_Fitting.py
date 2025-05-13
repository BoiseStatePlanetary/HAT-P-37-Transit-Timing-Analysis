# write a script to use susie to calculate all precession fits

import os
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import matplotlib.cm as mplcm
import matplotlib.colors as colors
from susie.timing_data import TimingData
from susie.ephemeris import Ephemeris
# Update font sizes in plots
plt.rcParams.update({'font.size': 22}) # Sets global plot font size to 22
plt.rcParams.update({'legend.fontsize':10}) # sets global legend font size to 20

#define global variables for directories
code_dir = os.path.dirname(__file__)
main_dir = os.path.dirname(os.path.dirname(__file__))
fig_dir = os.path.join(main_dir, "figures")

# define global variables for literature values
tess_zero_time = 2457000.
Athano_T0 = 2455642.14768
Athano_per = 2.797440 #±0.000001 

def get_epochs(T, T0=Athano_T0, P=Athano_per, E0=0):
    """
    T: float
        Mid-time for your new point
    T0: float
        The mid-time of the very first epoch (corresponding to epoch=0)
    P: float
        Orbital period
    E0: int
        The first epoch number, if not zero. Will be added to make sure initial epoch number is accounted for
    """
    N = (T-T0)/P
    return round(N + E0)  # must use round not INT because midtime can be earlier or later than expected and int will not round.

def read_exoWatch_json_data(json_file=os.path.join(code_dir, "ExoWatch_HAT-P-37b.json")):
    with open(json_file) as f: 
        json_data = json.load(f)
        # NOTE: we can ignore nea data in json because it is a duplicaate of Elisbeth's lit data
        # ephem_dict = json_data['ephemeris']
        # mid_times = [float(x) for x in ephem_dict["nea_tmids"]]
        # mid_times_err = [float(x) for x in ephem_dict["nea_tmids_err"]]
        # src_flg = ["NASA Exoplanet Archive"] * len(mid_times)
        mid_times = []
        mid_times_err = []
        src_flg = []
        obs_dict = json_data["observations"]
        for obs in obs_dict:
            if obs["data_flag_ephemeris"] == True: 
                # print(obs.keys())
                dat = obs["parameters"]
                mid_times.append(float(dat['Tc']))
                mid_times_err.append(float(obs["errors"]["Tc"]))
                # observers.append(obs["obscode"]["id"])
                if len(obs["secondary_obscodes"]) == 0:
                    src_flg.append("AAVSO")
                else:
                    for observer in obs["secondary_obscodes"]:
                        if observer["id"] == "UNIS":
                            src_flg.append("Unistellar")
                        elif observer["id"] == "MOBS":
                            src_flg.append("MOBS/EXOTIC")
    
        return np.array(mid_times), np.array(mid_times_err), src_flg
    
def read_Sec7475_data(file_name=os.path.join(code_dir, "best_fit_values_HAT-P-37b.csv")):
    data = pd.read_csv(file_name, comment='#', header=0)
    # print(data.columns)
    # epochs = np.array(data["Epoch"].astype('int'))
    mid_times = np.array(data["Midtime"])
    mid_times_errs = np.array(data["Midtime_err_minus_days"])
    long_src = np.array(data["Name"])
    src = [src.split("_")[-1] for src in long_src]
    src_flg = np.array([flg.replace("-", " ") for flg in src])
    return mid_times, mid_times_errs, src_flg

def read_ETD_data(file_name=os.path.join(code_dir, "all_lit_times_HAT-P-37b.csv")):
    data = pd.read_csv(file_name, comment='#', header=0)
    # print(data.columns)
    # epochs = np.array(data["Epoch"].astype('int'))
    mid_times = np.array(data["Midtime"])
    mid_times_errs = np.array(data["Midtime_err_minus_days"])
    src_flg = np.array(data["Source"])
    # simplify sources
    # tresca_inds = [i for i, s in enumerate(src_flg) if 'TRESCA' in s]
    # src_flg[tresca_inds] = "TRESCA"
    lit_inds = np.array([i for i, s in enumerate(src_flg) if "2022ApJS" in s])
    # remove lit values already in Athano table:
    mid_times = mid_times[~lit_inds]
    mid_times_errs = mid_times_errs[~lit_inds]
    src_flg = ["TRESCA"] * len(mid_times)
    return mid_times, mid_times_errs, src_flg

def read_Athano_data(file_name=os.path.join(code_dir, "Athano2022_Table6.csv")):
    data = pd.read_csv(file_name, comment='#', header=0)
    # print(data.columns)
    epochs = np.array(data["Epoch"].astype('int'))
    # For A-thano+ 2022 the mid-ties must be corrected. they are in -2450000 BJD_TDB
    mid_times = np.array(data["Tm"])
    adjusted_midtimes = mid_times + 2450000
    mid_times_errs = np.array(data["sigma_Tm"])
    src_flg = np.array(["A-thano+ 2022"] * len(epochs))
    return epochs, adjusted_midtimes, mid_times_errs, src_flg

def read_Tess_data(file_name=os.path.join(code_dir, "tess_hatp37b_allsector_fits_constantEBper.csv")):
        data = pd.read_csv(file_name, comment='#', header=0)
        epochs = np.array(data["Epoch"].astype('int'))
        mid_times = np.array(data["Tm"])
        mid_times_errs = np.array(data["sigma_Tm"])
        src_flg = np.array(data["src_flg"])
        return epochs, mid_times, mid_times_errs, src_flg

def create_susie_obj():
    # first compile all data and calculate the epochs for full data set 
    ETD_midtimes, ETD_midtime_errs, ETD_srcs = read_ETD_data() 
    A_epochs, A_midtimes, A_midtime_errs, A_srcs = read_Athano_data()
    tess_epochs, tess_midtimes, tess_midtime_errs, tess_srcs = read_Tess_data()  
    EW_midtimes, EW_midtime_errs, EW_srcs = read_exoWatch_json_data()
    s745_midtimes, s745_midtime_errs, s745_srcs = read_Sec7475_data()

    # print("ETD: ", [get_epochs(tm) for tm in ETD_midtimes])
    # print("Athano: ", [get_epochs(tm) for tm in A_midtimes])
    # print("Diff: ", A_epochs)
    # print("TESS: ", [get_epochs(tm) for tm in tess_midtimes])
    # print("Diff: ", tess_epochs)
    # print("EW: ", [get_epochs(tm) for tm in EW_midtimes])
    # print("s745: ", [get_epochs(tm) for tm in s745_midtimes])
    
    rand_all_midtimes = np.concatenate((ETD_midtimes, A_midtimes, tess_midtimes, s745_midtimes))
    rand_all_midtime_errs = np.concatenate((ETD_midtime_errs, A_midtime_errs, tess_midtime_errs, s745_midtime_errs))
    rand_all_src_flgs = np.concatenate((ETD_srcs, A_srcs, tess_srcs, s745_srcs))

    # return ALL SORTED values
    sort_idx = np.argsort(rand_all_midtimes)
    all_midtimes = rand_all_midtimes[sort_idx]
    all_epochs = np.array([get_epochs(tm) for tm in all_midtimes])
    all_midtime_errs = rand_all_midtime_errs[sort_idx]
    all_src_flgs = rand_all_src_flgs[sort_idx]

    # delte duplicates, first attempt - keep first value
    e, u_inds = np.unique(all_epochs, return_index=True)
    unique_epochs = all_epochs[u_inds]
    unique_midtimes = all_midtimes[u_inds]
    unique_errors = all_midtime_errs[u_inds]
    unique_src_flgs = all_src_flgs[u_inds]


    all_data_timing_obj = TimingData(time_format="jd", epochs=unique_epochs, mid_times=unique_midtimes, 
                                     mid_time_uncertainties=unique_errors, time_scale="tdb")  
                                    #  object_ra=284.2960393, object_dec=51.2691212)
    all_data_ephemeris_obj = Ephemeris(all_data_timing_obj)

    return unique_src_flgs, all_data_ephemeris_obj

def create_tess_susie_obj():
    tess_epochs, tess_midtimes, tess_midtime_errs, tess_srcs = read_Tess_data()  
    timing_obj = TimingData(time_format="jd", epochs=tess_epochs, mid_times=tess_midtimes, 
                                     mid_time_uncertainties=tess_midtime_errs, time_scale="tdb")  
                                    #  object_ra=284.2960393, object_dec=51.2691212)
    ephemeris_obj = Ephemeris(timing_obj)

    return tess_srcs, ephemeris_obj

def create_colormap(src_flgs):
    cm = plt.get_cmap('managua')
    num_colors = len(src_flgs)
    inds = range(num_colors)
    cNorm  = colors.Normalize(vmin=0, vmax=num_colors-1)
    scalarMap = mplcm.ScalarMappable(norm=cNorm, cmap=cm)
 
    color_dict = {}
    for i in inds:
        color_dict[src_flgs[i]] = scalarMap.to_rgba(i)

    return color_dict  

def plot_lin_BIC(src_flgs, ephemeris_obj):
    ax = ephemeris_obj.plot_oc_plot("precession")
    oc_vals = ephemeris_obj.oc_vals
    epochs = ephemeris_obj.timing_data.epochs
    # bad_inds = np.argwhere(oc_vals >= 2000)
    # print(epochs[bad_inds], ephemeris_obj.timing_data.mid_times[bad_inds])
    # print(src_flgs[bad_inds])

    color_dict = create_colormap(np.unique(src_flgs))

    for data_point, time, src in zip(epochs, oc_vals, src_flgs):
        ax.scatter(data_point, time, label=src, color=color_dict[src], zorder=110)
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = set(labels)
    handles = [handles[labels.index(label)] for label in unique_labels]
    labels = list(unique_labels)
    ax.legend(handles, labels, loc="upper center", 
              bbox_to_anchor=(0.5, 1.0), ncols=5, fancybox=True, shadow=True)
    plt.show()

    ephemeris_obj.plot_running_delta_bic("linear", "precession")
    plt.show()
    
    lp_delta_bic_value = ephemeris_obj.calc_delta_bic("linear", "precession")
    print(f"Linear vs Precession \u0394 BIC: {lp_delta_bic_value}")

    # ephemeris_obj.plot_oc_plot("linear")
    # plt.show()



if __name__ == "__main__":
    flags, obj = create_susie_obj()
    plot_lin_BIC(flags, obj)

