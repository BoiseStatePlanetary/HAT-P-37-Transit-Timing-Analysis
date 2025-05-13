
# calculate the t0 for the eclipsing binary (from https://iopscience.iop.org/article/10.3847/1538-4365/ab9cae#apjsab9caes5)
HJD_shift = 2400000.5
EB_t0_HJD = 58641.4289616 + HJD_shift
# print(EB_t0_HJD)
EB_t0_BJD = 2458641.929750  # calculated using the Ohio State Applet (https://astroutils.astronomy.osu.edu/time/hjd2bjd.html)
# finally convert t0 to TESS BJD (BTJD)
EB_t0_BTJD = EB_t0_BJD - 2457000.
print(EB_t0_BTJD)

# From A-thano et. al. (https://ui.adsabs.harvard.edu/abs/2023ApJS..265....4K/abstract)
# Table 7 - constant-period model
orbital_period = 2.797440 #±0.000001 
epoch_0 = 2455642.14409 # ±0.00046
# convert t0 to TESS BJD (BTJD)
tess_zero_time = 2457000.
planet_epoch_0_BTJD = epoch_0 - tess_zero_time