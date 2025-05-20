# Converting "Conditioning" notebook to a script for ease of use. building functions to aid and streamline process
import os
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as mplcm
import matplotlib.colors as colors
import lightkurve as lk
from astropy.units import Quantity
from astropy.time import Time
from scipy.stats import median_abs_deviation as mad
from scipy.optimize import curve_fit
from transit_utils import *
from utils import *


plt.rcParams.update({'font.size': 14}) # Sets global plot font size to 22
plt.rcParams.update({'legend.fontsize': 12}) # sets global legend font size to 20
plt.rcParams['figure.constrained_layout.use'] = True
tess_zero_time = 2457000.
main_dir = os.path.dirname(os.path.dirname(__file__))
fig_dir = os.path.join(main_dir, "figures")

# pull all SPOC data for a given planet
def download_tess_sectors(planet="TIC 267572272"):
    search_result = lk.search_lightcurve(planet, author='SPOC', exptime='short')
    tess_lc_collection = search_result.download_all()  # changed name of variable to make it easier to identify
    print(tess_lc_collection)

    return tess_lc_collection

# clean sectors - use sigma clipped PDCSAP flux
def clean_sectors(lc_collection, savefig=False):
    if savefig:
        fig = plt.figure(figsize=(17, 22), layout="constrained")
    
    cleaned_lcs = []
    for i, lc in enumerate(lc_collection):
        no_nans_lc = lc.remove_nans()
        clipped_lc = no_nans_lc.remove_outliers(sigma=5.0)

        cleaned_lcs.append(clipped_lc)

        if savefig:
            ax = fig.add_subplot(4, 3, i+1)
            clipped_lc.scatter(column='pdcsap_flux', label='PDCSAP Flux', normalize=True, ax=ax)
            ax.set_title(f"TESS Sector {lc.sector}")
            ax.get_legend().remove()
        
    if savefig:
        fig_name = os.path.join(fig_dir, "allSectors_sigma5_PDCSAPflux_lc.png")
        fig.savefig(fig_name, dpi=300, bbox_inches="tight")
        print(f"Figure saved to {fig_name}")

    return cleaned_lcs

def calc_BLS_period(light_curve, savefig=False):
    # Use the PDCSAP to calculate the Box-Least Squares periodogram (fit to box ie Transit shapes) for each sector of TESS data and plot
    pg = light_curve.to_periodogram(method="bls")
    
    if savefig: 
        fig = plt.figure()
        ax = plt.add_subplot(111)

        pg.plot(ax=ax)
        ax.set_title(f"Peak = {pg.period_at_max_power}")
        ax.get_legend().remove() 
        fig_name = os.path.join(fig_dir, f"S{light_curve.sector}_BLS_periodogram.png")
        fig.savefig(fig_name, dpi=300, bbox_inches='tight')
    
    return pg.period_at_max_power.value

# create a transit mask, use the argument 'plot_data=True' for a sanity check of the mask
def mask_hatp37_transits(light_curve, plot_data=False):
    # note the literature values for hat-p-37
    # From A-thano et. al. (https://ui.adsabs.harvard.edu/abs/2023ApJS..265....4K/abstract) - Table 7 - constant-period model
    orbital_period = 2.797440 #±0.000001 
    epoch_0 = 2455642.14409 # ±0.00046
    # convert t0 to TESS BJD (BTJD)
    tess_zero_time = 2457000.
    planet_epoch_0_BTJD = epoch_0 - tess_zero_time

    '''returns lightkurve object of a full sector without the transits, sector_lk MUST be a lightkurve LightCurve obj'''
    known_transit_mask = light_curve.create_transit_mask(transit_time=planet_epoch_0_BTJD, 
                                                         period=orbital_period,
                                                         duration=0.194) # duration (in days) == 2*transit duration (2.33 h ~ 0.097d)
    # NOTE: create transit mask returns an array with True for every data point IN TRANSIT, so we need to flip it to remove the transits
    no_transit_mask = known_transit_mask == False  # this inverts the mask to mask out ONLY the transits
    NO_transits_lk = light_curve[no_transit_mask]

    if plot_data:
        fig = plt.figure()
        ax=fig.add_subplot(111)
        light_curve.scatter(ax=ax, normalize=True, color='k', label="All Data")
        NO_transits_lk.scatter(ax=ax, normalize=True, label="Transits Masked")
        ax.get_legend() 
        fig_name = os.path.join(fig_dir, f"FullSector_{light_curve.sector}_NO_transits.png")
        fig.savefig(fig_name, dpi=300, bbox_inches="tight")

    return NO_transits_lk 

def bin_lightcurve(light_curve, binsize=120./86400.):  # 120 seconds to days
    time = light_curve.time.value
    flux = light_curve.flux.value 
    
    binned_time = np.array([])
    binned_flux = np.array([])
    binned_err = np.array([])

    cur_point = np.min(time) + 0.5*binsize
    while(cur_point <= np.max(time) - 0.5*binsize):
        ind = np.abs(cur_point - time) < 0.5*binsize

        if(len(time[ind]) > 0):
            binned_time = np.append(binned_time, np.mean(time[ind]))
            binned_flux = np.append(binned_flux, np.mean(flux[ind]))
            binned_err = np.append(binned_err, np.std(flux[ind]))
        
        cur_point += binsize
    
    astropy_time = Time(binned_time, format=light_curve.time.format)
    binned_lightcurve = lk.LightCurve(time=astropy_time, flux=binned_flux, flux_err=binned_err, meta=light_curve.meta)
    return binned_lightcurve

def fold_lightcurve(light_curve, period, ref_mid, output_rel_times=False):
    # return a lightkurve object with the time folded
    # print(f"Folding to period {period} with ref mid {ref_mid}")
    times = light_curve.time.value
    rel_times = times - ref_mid - period/2. # to plot transit in center
    mod_rel_times = rel_times % period
    
    if output_rel_times:
        folded_lightcurve = lk.LightCurve(time=mod_rel_times, flux=light_curve.flux, flux_err=light_curve.flux_err, meta=light_curve.meta)
    else:
        mid_elapsed = num_elapsed_transits(min(times), period, ref_mid) 
        midsector_time = predict_linear_eph(mid_elapsed, period, ref_mid)
        folded_times_assigned_mid_sector = mod_rel_times+midsector_time-period/2.
        folded_astropy_time = Time(folded_times_assigned_mid_sector, format="btjd")
        folded_lightcurve = lk.LightCurve(time=folded_astropy_time, flux=light_curve.flux, flux_err=light_curve.flux_err, meta=light_curve.meta)
    return folded_lightcurve

def detrend_by_EB(sector_lc, use_lit_time=True):
    # calculate the t0 for the eclipsing binary (from https://iopscience.iop.org/article/10.3847/1538-4365/ab9cae#apjsab9caes5)
    EB_t0_BJD = 2458641.929750  # calculated using the Ohio State Applet (https://astroutils.astronomy.osu.edu/time/hjd2bjd.html)
    # finally convert t0 to TESS BJD (BTJD)
    EB_t0_BTJD = EB_t0_BJD - tess_zero_time

    if use_lit_time:
        binary_period = 0.4354712  # d - lit period from Chen et al. 2020
    else:
        # calc binary period using BLS
        binary_period = calc_BLS_period(sector_lc)

    # create the mask, fold on the binary, and bin the points to create the data to interpolate
    masked_lc = mask_hatp37_transits(sector_lc) # Mask out transit
    folded_lc = fold_lightcurve(masked_lc, period=binary_period, ref_mid=EB_t0_BTJD)
    binned_lc = bin_lightcurve(folded_lc) # default bins to 120s 

    # the time series that needs interpolating should included ALL time folded (ie no transit mask)
    fold_sector = fold_lightcurve(sector_lc, period=binary_period, ref_mid=EB_t0_BTJD)
    interp_time = fold_sector.time.value

    # Now interpolate across whole time-series unfolded
    interp_trend = np.interp(interp_time, binned_lc.time.value, binned_lc.flux.value)

    # detrend by dividing the WHOLE sector flux by the interpolated
    detrended_flux = sector_lc.flux.value / interp_trend
    detrended_err = sector_lc.flux_err.value / np.median(interp_trend)
    
    detrended_lc = lk.LightCurve(time=sector_lc.time, flux=detrended_flux, flux_err=detrended_err, meta=sector_lc.meta)
    return binary_period, detrended_lc, interp_trend

def detrend_by_flatten(sector_lc):
    # mask out known transits 
    masked_lc = mask_hatp37_transits(sector_lc)
    flatten_lc_no_transits, trend_no_transits = masked_lc.flatten(window_length=251, return_trend=True)

    # ues Elisabeth's code to detrend w/ transit mask
    xx = trend_no_transits.time.value
    yy = trend_no_transits.flux.value
    xnew = sector_lc.time.value
    ynew = np.interp(xnew, xx, yy)
    # plt.plot(xx, yy, 'o', xnew, ynew, '-')
    # plt.show()

    detrended_flux = sector_lc.flux.value/ynew
    # detrended_err = sector_lc.flux_err.value/np.median(ynew)

    detrended_lc = lk.LightCurve(time=sector_lc.time, flux=detrended_flux, flux_err=sector_lc.flux_err, meta=sector_lc.meta)
    return detrended_lc

def calc_sector_chunks(sector_lc, orbital_period, ref_mid):
    ## Note, we want the UNFOLDED time here
    detrended_time = sector_lc.time.value
    min_t = min(detrended_time)
    max_t = max(detrended_time)  

    min_elapsed = num_elapsed_transits(min_t, orbital_period, ref_mid ) 
    max_elapsed = num_elapsed_transits(max_t, orbital_period, ref_mid )
    n_range = np.arange(min_elapsed, max_elapsed + 1)  # arange is NOT inclusive, must add 1 to max

    pred_midtimes = predict_linear_eph(n_range, orbital_period, ref_mid)
    pred_chunk_start = pred_midtimes -  orbital_period/2.
    pred_chunk_end = pred_midtimes +  orbital_period/2.

    chunk_times= []
    for nn in range(len(pred_midtimes)):
        chunk_times.append( [pred_chunk_start[nn],pred_chunk_end[nn]] )
    return n_range, chunk_times

def chunk_tess_sector(sector_lc, orbital_period, ref_mid, savefig=False, sector_dir=None):
    epochs, chunk_times = calc_sector_chunks(sector_lc, orbital_period, ref_mid)
    detrended_time = sector_lc.time.value
    print(f"Epochs in Sector {sector_lc.sector}:", epochs)
    # chunk_masks = []
    chunk_epochs = []
    chunked_lc_list = []

    for ii in range(len(chunk_times)):
        chunk_mask = (detrended_time >= chunk_times[ii][0]) & (detrended_time < chunk_times[ii][1])
        # chunk_masks.append( (detrended_time >= chunk_times[ii][0]) & (detrended_time < chunk_times[ii][1]) )
        if(list(chunk_mask).count(True)>0): 
            chunk_lc = sector_lc[chunk_mask]
            if len(chunk_lc.time.value) >= 1500 and np.min(chunk_lc.flux.value) <= 0.975:  
            # throw out chunks with less than x points & check for transit, should throw out half-transits
                print(f"Epoch {epochs[ii]} collected,  {len(chunk_lc.time.value)} points.")
                chunk_epochs.append(epochs[ii])
                chunked_lc_list.append(chunk_lc)

    # fig = plt.figure(figsize=(16,9))
    # ax = fig.add_subplot(111)
    # sector_lc.scatter(ax=ax, label=sector_lc.sector)
    # for val in chunk_times:
    #     ax.axvline(val[0], color='orange')
    #     ax.axvline(val[1], color='g')
    # plt.show()
    # fig_name = sector_dir + "flattened_sector_w_chunks_drawn.png"
    # fig.savefig(fig_name, dpi=300, bbox_inches="tight")

    return chunk_epochs, chunked_lc_list

def create_sector_dir_for_figures(sector):  # Sector must be a string
    # make a sector folder to hold plots
    use_dir = "../figures/"
    sector_dir = use_dir + "S"+sector+"/"
    if os.path.isdir(sector_dir) == False:
        print("making",sector_dir)
        os.makedirs(sector_dir)
    
    return sector_dir

def fit_transit_params_for_sector(sector_lc, orbital_period, ref_mid):
    # Use the folded light curve for the whole sector to establish all fit parameters except the midtime
    transit_folded_lc = fold_lightcurve(sector_lc, orbital_period, ref_mid)
    # transit_folded_lc.scatter()
    # plt.show()
    initial_params = calc_Carter_initial_guesses(transit_folded_lc)

    print(np.min(transit_folded_lc.flux_err), np.max(transit_folded_lc.flux_err))

    model_params, pcov = curve_fit(Carter_model, transit_folded_lc.time.value, transit_folded_lc.flux.value, p0=initial_params, sigma=transit_folded_lc.flux_err)
    model_params_unc = np.sqrt(np.diag(pcov))

    # note: we will drop the estimate of tc because it is not meaningful for the folded light curve
    return model_params[1:], model_params_unc[1:]

def fit_transit_midtime(lc, transit_shape_params):
    # use Carter model to fit ONLY midtimes of chunked transits
    initial_tc = lc.time.value[np.argmin(lc.flux.value)] # estimate the central transit time to be the time w/ the minimum flux value
    carter_fit_tc = lambda x, tc: Carter_model(x, tc, *transit_shape_params)

    fit_midtime, fit_err = curve_fit(carter_fit_tc, lc.time.value, lc.flux.value, p0=initial_tc, sigma=lc.flux_err)

    return fit_midtime, fit_err[0]

def plot_LS_by_sector(sector_list):
    # REMEMBER: figsize takes width by height!!
    fig = plt.figure(figsize=(17, 11), layout="constrained")  # (w:h) 3:2 aspect ratio to create full page figure, 1.5455:1 for half page
    behaved_sector_list = []
    for lc in sector_list:
        if lc.sector in [59, 74, 75]:
            pass
        else:
            behaved_sector_list.append(lc)
    print(behaved_sector_list)

    for i, lc in enumerate(behaved_sector_list):
        pg = lc.to_periodogram(oversample_factor=10)
        ax = fig.add_subplot(2, 4, i+1)
        pg.plot(ax=ax, view="period", scale="log", label=f"Peak = {pg.period_at_max_power:.4f}")
        ax.set_title(f"Sector {lc.sector}") 

    figname = os.path.join(fig_dir, "wellbehaved_sectors_LS_periodogram.png")
    plt.savefig(figname, dpi=300) #, bbox_inches="tight")
    print(f"Saved figure to {figname}")
        


def plot_sectors_folded_by_EB(sector_list):
    fig = plt.figure(figsize=(8.5, 5.5))
    ax = fig.add_subplot(111)

    #establish colors for the plot
    NUM_COLORS = len(sector_list) + 2

    cm = plt.get_cmap('managua')
    cNorm  = colors.Normalize(vmin=0, vmax=NUM_COLORS-1)
    scalarMap = mplcm.ScalarMappable(norm=cNorm, cmap=cm)
    ax.set_prop_cycle(color=[scalarMap.to_rgba(i) for i in range(NUM_COLORS)])                                    


    EB_period = 0.4354712  # d - lit period from Chen et al. 2020
    # calculate the t0 for the eclipsing binary (from https://iopscience.iop.org/article/10.3847/1538-4365/ab9cae#apjsab9caes5)
    EB_t0_BJD = 2458641.929750  # calculated using the Ohio State Applet (https://astroutils.astronomy.osu.edu/time/hjd2bjd.html)
    # finally convert t0 to TESS BJD (BTJD)
    EB_t0_BTJD = EB_t0_BJD - tess_zero_time

    bin_fold_list = []
    sectors = []
    for lc in sector_list:
        folded_lc = fold_lightcurve(lc, period=EB_period, ref_mid=EB_t0_BTJD, output_rel_times=True)
        bin_fold_lc = bin_lightcurve(folded_lc)
        bin_fold_list.append(bin_fold_lc)
        sectors.append(lc.sector)


    # sort the light curve sectors by their minimum flux to make the plot easier to see
    sector_mins = [np.min(x.flux.value) for x in bin_fold_list]
    sort_idx = np.argsort(sector_mins)
    num_idx = np.argsort(sectors)


    for i in num_idx:
        lc = bin_fold_list[i]
        lc.plot(ax=ax, label=f"Sector {lc.sector}", normalize=True, linewidth=3.0, zorder=sort_idx[i])
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])  # shrink axis by 20%
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    
    figname = os.path.join(fig_dir, "all_sectors_folded_EB_period.png")
    plt.savefig(figname, dpi=300, bbox_inches="tight")
    print(f"Saved figure to {figname}")

def detrend_and_fit_sector(sector_lc):
    # From A-thano et. al. (https://ui.adsabs.harvard.edu/abs/2023ApJS..265....4K/abstract)
    # Table 7 - constant-period model
    orbital_period = 2.797440 #±0.000001 
    epoch_0 = 2455642.14409 # ±0.00046
    # convert t0 to TESS BJD (BTJD)
    ref_mid = epoch_0 - tess_zero_time

    EB_per, detrended_sector_EB, interp_trend = detrend_by_EB(sector_lc, use_lit_time=True)
    detrended_sector = detrend_by_flatten(detrended_sector_EB)
    print(f"Sector {sector_lc.sector} detrended by EB period = {EB_per}")

    # calc shape params - [f0, delta, T, tau]
    shape_params, shape_params_unc = fit_transit_params_for_sector(detrended_sector, orbital_period, ref_mid)
    print("Transit params (f0, delta, T, tau): ", shape_params)
    print("Transit params (f0, delta, T, tau) Unc: ", shape_params_unc)

    # get chunked lc
    # save_dir = create_sector_dir_for_figures(str(sector_lc.sector))
    sector_epochs, chunked_lcs = chunk_tess_sector(detrended_sector, orbital_period, ref_mid)
    
    sector_midtimes = []
    sector_midtime_errs = []
    for i, lc in enumerate(chunked_lcs):
        fit_tc, fit_tc_err = fit_transit_midtime(lc, shape_params)
        print(f"Fitted Epoch {sector_epochs[i]}: Midtime = {fit_tc[0] + tess_zero_time}, {fit_tc_err[0]}")
        sector_midtimes.append(fit_tc[0] + tess_zero_time)
        sector_midtime_errs.append((fit_tc_err[0]*1000))
    
    return sector_epochs, sector_midtimes, sector_midtime_errs

def detrend_and_fit_multiple_sectors(lc_collection, sector_list):
    all_sector_epochs = []
    all_sector_midtimes = []
    all_sector_midtime_errs = []
    all_sector_src_flgs = []
    for lc in lc_collection:
        if lc.sector in sector_list:
            epochs, mid_times, errors = detrend_and_fit_sector(lc)
            all_sector_epochs.append(epochs)
            all_sector_midtimes.append(mid_times)
            all_sector_midtime_errs.append(errors)
            all_sector_src_flgs.extend([f"TESS Sector {lc.sector}"] * len(epochs))
    
    csv_name = "code/tess_hatp37b_allsector_fits_constantEBper.csv"
    results = np.vstack((np.concatenate(all_sector_epochs), np.concatenate(all_sector_midtimes), 
                         np.concatenate(all_sector_midtime_errs), np.array(all_sector_src_flgs))).T
    with open(csv_name, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Tm', 'sigma_Tm', "src_flg"])
        writer.writerows(results)
    print(f"CSV file {csv_name} has been created successfully!") 
    
    return csv_name      

def plot_wang_vs_huchmala_sec_59():
    code_dir = os.path.dirname(__file__)
    wang_data_file = os.path.join(code_dir, "Wang2024_Table1_hatp37b.csv")
    huchmala_data_file = os.path.join(code_dir, "Huchmala_Sector59_fits.csv")

    w_data = pd.read_csv(wang_data_file, comment='#', header=0)
    w_epochs = np.array(w_data["Epoch"].astype('int'))
    w_mid_times = np.array(w_data["Tm"])
    w_mid_times_errs = np.array(w_data["sigma_Tm"])

    r_data = pd.read_csv(huchmala_data_file, comment='#', header=0)
    r_epochs = np.array(r_data["Epoch"].astype('int'))
    r_mid_times = np.array(r_data["Tm"])
    r_mid_times_errs = np.array(r_data["sigma_Tm"])

    #create linear o-c plot
    #C = T0 +E(orbital_period)

    fig = plt.figure(figsize=(8.5, 5.5))
    ax = fig.add_subplot(111)
    ax.errorbar(w_epochs, w_mid_times, w_mid_times_errs, ls='', marker='o', elinewidth=1.0, label="Wang et. al. 2024", color=BoiseState_blue)
    ax.errorbar(r_epochs, r_mid_times, r_mid_times_errs, ls='', marker='o', elinewidth=1.0, label="This work", color=BoiseState_orange)

    figname = os.path.join(fig_dir, "Wang_vs_Huchmala_Sec59.png")
    plt.savefig(figname, dpi=300, bbox_inches="tight")
    print(f"Saved figure to {figname}")
         

if __name__ == '__main__':
    tess_lc_collection = download_tess_sectors()
    cleaned_collection = clean_sectors(tess_lc_collection, savefig=True)
    # 0: <TessLightCurve LABEL="TIC 267572272" SECTOR=26 AUTHOR=SPOC FLUX_ORIGIN=pdcsap_flux>
    # 1: <TessLightCurve LABEL="TIC 267572272" SECTOR=40 AUTHOR=SPOC FLUX_ORIGIN=pdcsap_flux>
    # 2: <TessLightCurve LABEL="TIC 267572272" SECTOR=41 AUTHOR=SPOC FLUX_ORIGIN=pdcsap_flux>
    # 3: <TessLightCurve LABEL="TIC 267572272" SECTOR=53 AUTHOR=SPOC FLUX_ORIGIN=pdcsap_flux>
    # 4: <TessLightCurve LABEL="TIC 267572272" SECTOR=54 AUTHOR=SPOC FLUX_ORIGIN=pdcsap_flux>
    # 5: <TessLightCurve LABEL="TIC 267572272" SECTOR=55 AUTHOR=SPOC FLUX_ORIGIN=pdcsap_flux>
    # 6: <TessLightCurve LABEL="TIC 267572272" SECTOR=59 AUTHOR=SPOC FLUX_ORIGIN=pdcsap_flux>
    # 7: <TessLightCurve LABEL="TIC 267572272" SECTOR=74 AUTHOR=SPOC FLUX_ORIGIN=pdcsap_flux>
    # 8: <TessLightCurve LABEL="TIC 267572272" SECTOR=75 AUTHOR=SPOC FLUX_ORIGIN=pdcsap_flux>
    # 9: <TessLightCurve LABEL="TIC 267572272" SECTOR=80 AUTHOR=SPOC FLUX_ORIGIN=pdcsap_flux>
    # 10: <TessLightCurve LABEL="TIC 267572272" SECTOR=82 AUTHOR=SPOC FLUX_ORIGIN=pdcsap_flux>

    # plot_LS_by_sector(cleaned_collection)
    # plot_sectors_folded_by_EB(cleaned_collection)

    # con_secs = [26., 40., 41., 53., 54., 55., 59., 80., 82.] # Elisabeth has 74 & 75!!
    # detrend_and_fit_multiple_sectors(cleaned_collection, con_secs)
    # plot_wang_vs_huchmala_sec_59()
