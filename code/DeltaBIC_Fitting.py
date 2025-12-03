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
from scipy.optimize import curve_fit
from utils import *
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

def read_Wang_data(file_name=os.path.join(code_dir, "Wang2024_Table1_hatp37b.csv")):
    data = pd.read_csv(file_name, comment='#', header=0)
    # print(data.columns)
    epochs = np.array(data["Epoch"].astype('int'))
    mid_times = np.array(data["Tm"])
    mid_times_errs = np.array(data["sigma_Tm"])
    src_flg = np.array(["Wang+ 2024"] * len(epochs))
    return epochs, mid_times, mid_times_errs, src_flg

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


    # all_data_timing_obj = TimingData(time_format="jd", epochs=unique_epochs, mid_times=unique_midtimes, 
                                    #  mid_time_uncertainties=unique_errors, time_scale="tdb")  
                                    #  object_ra=284.2960393, object_dec=51.2691212)
    # all_data_ephemeris_obj = Ephemeris(all_data_timing_obj)

    return unique_epochs, unique_midtimes, unique_errors, unique_src_flgs


def create_athano_tess_bsu_asu_susie_obj():
    A_epochs, A_midtimes, A_midtime_errs, A_srcs = read_Athano_data()
    tess_epochs, tess_midtimes, tess_midtime_errs, tess_srcs = read_Tess_data()
    s745_midtimes, s745_midtime_errs, s745_srcs = read_Sec7475_data()

    data = pd.read_csv(os.path.join(code_dir, "BSU_midtimes_2025.csv"), comment='#', header=0)
    BSU_midtimes = np.array(data["Tm"])
    BSU_midtime_errs = np.array(data["sigma_Tm"])
    BSU_src_flg = np.array(data["src_flg"])

    asu_data = pd.read_csv(os.path.join(code_dir, "ASU_midtimes_2025.csv"), comment='#', header=0)
    ASU_midtimes = np.array(asu_data["Tm"])
    ASU_midtime_errs = np.array(asu_data["sigma_Tm"])
    ASU_src_flg = np.array(asu_data["src_flg"])

    rand_all_midtimes = np.concatenate((A_midtimes, tess_midtimes, s745_midtimes, BSU_midtimes, ASU_midtimes))
    rand_all_midtime_errs = np.concatenate((A_midtime_errs, tess_midtime_errs, s745_midtime_errs,
                                            BSU_midtime_errs, ASU_midtime_errs))
    rand_all_src_flgs = np.concatenate((A_srcs, tess_srcs, s745_srcs, BSU_src_flg, ASU_src_flg))

    # return ALL SORTED values
    sort_idx = np.argsort(rand_all_midtimes)
    all_midtimes = rand_all_midtimes[sort_idx]
    all_epochs = np.array([get_epochs(tm) for tm in all_midtimes])
    all_midtime_errs = rand_all_midtime_errs[sort_idx]
    all_src_flgs = rand_all_src_flgs[sort_idx]

    timing_obj = TimingData(time_format="jd", epochs=all_epochs, mid_times=all_midtimes,
                            mid_time_uncertainties=all_midtime_errs, time_scale="tdb")

    ephemeris_obj = Ephemeris(timing_obj)

    return all_src_flgs, ephemeris_obj, timing_obj

def create_colormap(src_flgs):
    cm = plt.get_cmap('jet')
    num_colors = len(src_flgs)
    inds = range(num_colors)
    cNorm  = colors.Normalize(vmin=0, vmax=num_colors-1)
    scalarMap = mplcm.ScalarMappable(norm=cNorm, cmap=cm)
 
    color_dict = {}
    for i in inds:
        color_dict[src_flgs[i]] = scalarMap.to_rgba(i)
    color_dict["A-thano+ 2022"] = 'k'
    color_dict["Wang+ 2022"] = 'grey'
    color_dict["SuPerPiG"] = 'fuchsia'
    color_dict["Unistellar"] = 'orange'
    color_dict["MicroObservatory(ASU)"] = 'maroon'
    color_dict["Bruneau"] = 'g'

    return color_dict  

def plot_lin_BIC(src_flgs, ephemeris_obj):
    ax = ephemeris_obj.plot_oc_plot("quadratic")
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
    # ax.legend(handles, labels, loc="upper center", 
    #           bbox_to_anchor=(0.5, 1.0), ncols=5, fancybox=True, shadow=True)
    plt.show()

    # ephemeris_obj.plot_running_delta_bic("linear", "precession")
    # plt.show()
    
    lp_delta_bic_value = ephemeris_obj.calc_delta_bic("linear", "precession")
    print(f"Linear vs Precession \u0394 BIC: {lp_delta_bic_value}")

    # ephemeris_obj.plot_oc_plot("linear")
    # plt.show()

def calc_athano_precession_params():
    # dictionary of values from Table 7 of A-thano (2022)
    precession_params = {}

    precession_params["Ps"] = 2.797440 # days - sidereal period
    precession_params["T0"] = 2455642.14436 # - 2450000. # BJD
    precession_params["omega_0"] = -0.30 # rad
    precession_params["domega_dE"] = 0.0143 # rad/orbit
    precession_params["e"] = 0.0013
    precession_params["Pa"] = calc_Pa(precession_params["Ps"], precession_params["domega_dE"])  # anomalistic period

    T0_precession_unc_plus = 0.00042
    T0_precession_unc_minus = -0.0004

    Ps_precession_unc_plus = 0.000001
    Ps_precession_unc_minus = 0.000001

    omega_0_unc_plus = 0.51
    omega_0_unc_minus = -0.65

    domega_dE_unc_plus = 0.0009
    domega_dE_unc_minus = -0.0007

    e_unc_plus = 0.0005
    e_unc_minus = -0.0004

    precession_params["unc"] = np.sqrt([T0_precession_unc_plus**2 + T0_precession_unc_minus**2,
                               Ps_precession_unc_plus**2 + Ps_precession_unc_minus**2,
                               e_unc_plus**2 + e_unc_minus**2,
                               omega_0_unc_plus**2 + omega_0_unc_minus**2,
                               domega_dE_unc_plus**2 + domega_dE_unc_minus**2])
    return precession_params

def calc_precession_best_fit(p0_params, Epochs, Tms, sigma_Tms):
    popt, pcov = curve_fit(calc_ttra_precession, Epochs, Tms, sigma=sigma_Tms, 
                        p0=[p0_params["T0"], p0_params["Ps"], p0_params["e"], p0_params["omega_0"], p0_params["domega_dE"]],
                        method='trf')
    
    best_fit_params = {}
    best_fit_params["unc"] = np.sqrt(np.diag(pcov))
    best_fit_params["T0"] = popt[0]
    best_fit_params["Ps"] = popt[1]
    best_fit_params["e"] = popt[2]
    best_fit_params["omega_0"] = popt[3]
    best_fit_params["domega_dE"] = popt[4]
    best_fit_params["Pa"] = popt[1]/(1. - 1./2./np.pi*popt[4])
    return best_fit_params

def linear_midtime_model(E, Tm, sigma_Tm):
    # Fit a linear model - returns coefficent of model (P)
    # since degree=1 this is fitting Tm = p[1] + p[0] * E**1
    # where p[0] = orbital period and p[1] = T0   
    lin_coeffs = np.polyfit(E, Tm, 1, w=1./sigma_Tm)
    # lin_coeffs is the array [orbital period, T0]
    return lin_coeffs

def plot_omni_precession(Epochs, Tms, sigma_Tms, src_flgs):
    '''_summary_

    Parameters
    ----------
    Epochs : np.array
        all the Epochs to be used for the precession model calculation
    Tms : np.array
        corresponding OBSERVED midtimes for the model
    sigma_Tms : np.array
        corresponding midtime errors
    '''
    num_prec_params = 5.
    num_lin_params = 2.

    fig = plt.figure(figsize=(6*aspect_ratio, 8))
    fig.suptitle("HAT-P-37 b - Precession", fontsize=24)
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212, sharex=ax1)
    # ax3 = fig.add_subplot(313, sharex=ax1)

    #fit linear model
    lin_P_T0 = linear_midtime_model(Epochs, Tms, sigma_Tms)
    print("Linear Model Params: (P, T0)", lin_P_T0)
    linear_Tms = np.polyval(lin_P_T0, Epochs) # evaluates a polynomial with the given coeffs (period & T0) at the values of Epochs
    lin_BIC = calc_BIC(Tms, linear_Tms, deg=2., sd=sigma_Tms) # calculate the linear BIC
    print("BIC for linear with all points: ", lin_BIC)

    # plot all data colored by source
    # O - C using best-fit linear portion of precession model
    oc_vals = (Tms - linear_Tms)*24*60  # convert to minutes
    color_dict = create_colormap(np.unique(src_flgs))
    for data_point, time, sigma, src in zip(Epochs, oc_vals, sigma_Tms, src_flgs):
        ax1.errorbar(data_point, time, yerr=sigma*24*60, color='k')
        ax1.scatter(data_point, time, label=src, color=color_dict[src], zorder=110)
    
    #fit apsidal precesssion model
    athano_dict = calc_athano_precession_params()
    precession_popt = calc_precession_best_fit(athano_dict, Epochs, Tms, sigma_Tms)
    precession_Tms = calc_ttra_precession(Epochs, precession_popt["T0"], precession_popt["Ps"], precession_popt["e"], precession_popt["omega_0"], precession_popt["domega_dE"])
    prec_BIC = calc_BIC(Tms, precession_Tms, deg=5., sd=sigma_Tms)
    print("BIC for precession with all points", prec_BIC)
    
    final_Delta_BIC = lin_BIC - prec_BIC # delta bic linear model - precession model
    error_bars = np.sqrt(2.*(len(Epochs) - (5. + 2.)))

    print("final Delta BIC with all points +- unc: ", final_Delta_BIC, error_bars)

    # plot precession model (as a line)
    model_Epochs = np.arange(Epochs[0], Epochs[-1], 1)
    model_prec = calc_ttra_precession(model_Epochs, precession_popt["T0"], precession_popt["Ps"], precession_popt["e"], precession_popt["omega_0"], precession_popt["domega_dE"])
    model_lin = np.polyval(lin_P_T0, model_Epochs)
    prec_O_minus_C = (model_prec - model_lin)*24*60
    ax1.plot(model_Epochs, prec_O_minus_C, ls='--', color=BoiseState_orange, label="Apsidal precession model")
    ax1.axhline(0., color='gray', lw=2, ls='-')

    # create legend for first panel
    handles, labels = ax1.get_legend_handles_labels()
    unique_labels = set(labels)
    handles = [handles[labels.index(label)] for label in unique_labels]
    labels = list(unique_labels)
    ax1.legend(handles, labels, loc="upper center", 
              bbox_to_anchor=(0.5, 1.4), ncols=5, fancybox=True, shadow=True)

    # Running Delta BIC - panel B
    running_Delta_BIC = []
    sigma_running_Delta_BIC = []
    running_anal_Delta_BIC = []

    # ind = np.full(len(Epochs), False, dtype=bool)
    # np.cos(saved_popt[4]*data['Epoch'] + saved_popt[3])
    cosw = np.cos(precession_popt["domega_dE"]*Epochs + precession_popt["omega_0"])

    for i in range(7, len(Epochs)+1):
        # NOTE: we start at ind 7, which means we throw out the first SIX points, therefore the number of running BIC calculations should be 6 less than the total number of epochs   
        # since indexing :i is non-inclusive, we MUST use len(Epochs)+1 to ensure we get all of the points
        # calculate precession model and linear model up to the epoch looping over. 
        running_popt = calc_precession_best_fit(athano_dict, Epochs[:i], Tms[:i], sigma_Tms[:i]) 
        prec_model = calc_ttra_precession(Epochs[:i], running_popt["T0"], running_popt["Ps"], running_popt["e"], running_popt["omega_0"], running_popt["domega_dE"])
        prec_BIC = calc_BIC(Tms[:i], prec_model, deg=5., sd=sigma_Tms[:i])
        lin_coeffs = linear_midtime_model(Epochs[:i], Tms[:i], sigma_Tms[:i])
        lin_model = np.polyval(lin_coeffs, Epochs[:i])
        lin_BIC = calc_BIC(Tms[:i], lin_model, deg=2., sd=sigma_Tms[:i])

        Delta_T0_prime = calc_T0_lin_factor(Epochs[:i], cosw[:i], sigma_Tms[:i])
        Delta_Ps_prime = calc_Ps_lin_factor(Epochs[:i], cosw[:i], sigma_Tms[:i])
        anal_lin_chi_sq = (precession_popt["e"]*precession_popt["Pa"]/np.pi)**2*np.sum((cosw[:i] +\
                            Delta_Ps_prime*Epochs[:i] + Delta_T0_prime)**2/sigma_Tms[:i]**2) + (len(Epochs[:i]) - 2.)

        anal_lin_fit_BIC = anal_lin_chi_sq + num_lin_params*np.log(len(Epochs[:i]))
        # the way this is written the analytic prec delta BIC does not include a chi sq term which does not seem right.. also probably why the deviation...
        anal_prec_BIC = (len(Epochs[:i]) - num_prec_params) + num_prec_params*np.log(len(Epochs[:i]))

        running_Delta_BIC.append(lin_BIC - prec_BIC)
        running_anal_Delta_BIC = np.append(running_anal_Delta_BIC, anal_lin_fit_BIC - anal_prec_BIC)
        sigma_running_Delta_BIC.append(np.sqrt(2.*(len(Epochs[:i]) - (5. + 2.))))

    running_Delta_BIC = np.array(running_Delta_BIC)
    sigma_running_Delta_BIC = np.array(sigma_running_Delta_BIC)    
    # EDITED(6/17) first point: plots @ epoch index 6, calculated running delta BIC includes midtimes from index 0-6
    ax2.plot(Epochs[6:], running_Delta_BIC, lw=3, color=BoiseState_blue, label="Numeric")
    ax2.fill_between(Epochs[6:], running_Delta_BIC - sigma_running_Delta_BIC, running_Delta_BIC + sigma_running_Delta_BIC, 
                    color=BoiseState_blue, alpha=0.25)
    ax2.plot(Epochs[6:], running_anal_Delta_BIC, lw=3, color=BoiseState_orange, label="Analytic")
    ax2.fill_between(Epochs[6:], running_anal_Delta_BIC - sigma_running_Delta_BIC, running_anal_Delta_BIC + sigma_running_Delta_BIC, 
                    color=BoiseState_orange, alpha=0.25)
    
    print("final Numeric BIC:", running_Delta_BIC[-1])
    print(running_Delta_BIC)
    print("Final Analytic BIC:", running_anal_Delta_BIC[-1])
    print(running_anal_Delta_BIC)

    ### end Panel b ###

    ax1.grid(True)
    ax2.grid(True)
    # ax3.grid(True)

    # ax1.legend(loc='best', fontsize=16)
    # ax2.legend(loc='best', fontsize=16)
    # ax3.legend(loc='best', fontsize=16)

    ax1.tick_params(labelbottom=True, labelsize=16)
    ax2.tick_params(labelbottom=False, labelsize=16)
    # ax3.tick_params(labelsize=16)

    ax1.set_ylabel(r'$t_i - T_0 - P_{\rm s} E_i\,\left( {\rm min}\right)$', fontsize=18)
    ax2.set_ylabel(r'$\Delta {\rm BIC}$', fontsize=24)
    # ax3.set_ylabel(r'$^{\Delta \left( \Delta {\rm BIC} \right)}/{\left( \Delta {\rm BIC} \right)_{\rm final}}$', fontsize=16)
    # ax3.set_xlabel("Epoch", fontsize=24)

    ax1.text(0.01, 0.8, "(a)", fontsize=36, transform=ax1.transAxes)
    ax2.text(0.01, 0.8, "(b)", fontsize=36, transform=ax2.transAxes)
    # ax3.text(0.01, 0.05, "(c)", fontsize=36, transform=ax3.transAxes)

    # ax1.set_xlim(1730, 1760)
    # ax3.set_ylim([-0.5, 0.25])

    plt.tight_layout()
    plt.show()
    # fig.savefig("../figures/OmniPlot_HAT-P-37b_Precession.jpg", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    flags, obj, timing_obj = create_athano_tess_bsu_asu_susie_obj()
    # flags, obj, timing_obj = create_susie_obj()
    # plot_lin_BIC(flags, obj)
    plot_omni_precession(timing_obj.epochs, timing_obj.mid_times,
                         timing_obj.mid_time_uncertainties, flags)

