import matplotlib
# comment out design values in this list 
# you don't want to run through the pipeline
lmap = [
'#B544A6',
'#884DB2',
'#5856AF',
'#6089AC',
'#6AA8A2',
'#64AE90',
'#62B07A',
'#75B85B',
'#B1BF53',
'#C78E4B',
][::-1]
custom_cmap = matplotlib.colors.ListedColormap(lmap)


station_dvs = [
    "RL50 (kPa)",
    "SL50 (kPa)",
    "moisture_index",
    "mean RH (%)",
    "HDD (degC-day)",
    "TJan2.5 (degC)",
    "TJan1.0 (degC)",
    "TJul2.5 (degC)",
    "TwJul2.5 (degC)",
    "Tmin (degC)",
    "Tmax (degC)",
    "WP10",
    "WP50",
    "DRWP-RL5 (Pa)",
    "annual_pr (mm)",
    "annual_rain (mm)",
    "RL50 (mm)",
    "Gum-LM RL10 (mm)"
]


filenames = {
    "RL50 (kPa)": "RL50",
    "mean RH (%)": "RHann",
    "HDD (degC-day)": "HDD",
    "SL50 (kPa)": "SL50",
    "WP10": "WP10",
    "WP50": "WP50",
    "TJan2.5 (degC)": "TJan2.5",
    "TJan1.0 (degC)": "TJan1.0",
    "Tmin (degC)": "Tmin",
    "Tmax (degC)": "Tmax",
    "TJul2.5 (degC)": "TJul97.5",
    "TwJul2.5 (degC)": "TwJul97.5",
    "DRWP-RL5 (Pa)": "DRWP5",
    "annual_pr (mm)": "PAnn",
    "annual_rain (mm)": "RAnn",
    "RL50 (mm)": "R1d50",
    "moisture_index": "MI",
    "Gum-LM RL10 (mm)": "R15m10"
}

model_paths = {
    'RL50 (kPa)': 'data/model_inputs/snw_rain_CanRCM4-LE_ens35_1951-2016_max_rl50_load_ensmean.nc',
    'mean RH (%)': 'data/model_inputs/hurs_CanRCM4-LE_ens15_1951-2016_ensmean.nc',
    'HDD (degC-day)': 'data/model_inputs/hdd_CanRCM4-LE_ens35_1951-2016_ann_ensmean.nc',
    'SL50 (kPa)': 'data/model_inputs/snw_CanRCM4-LE_ens35_1951-2016_max_rl50_load_ensmean.nc',
    'WP10': 'data/model_inputs/wpress_CanRCM4-LE_ens35_1951-2016_max_rl10_kpa_ensmean.nc',
    'WP50': 'data/model_inputs/wpress_CanRCM4-LE_ens35_1951-2016_max_rl50_kpa_ensmean.nc',
    'TJan2.5 (degC)': 'data/model_inputs/tas_CanRCM4-LE_ens35_1951-2016_1hr_jan2.5p_ensmean.nc',
    'TJan1.0 (degC)': 'data/model_inputs/tas_CanRCM4-LE_ens35_1951-2016_1hr_jan1.0p_ensmean.nc',
    'Tmax (degC)': 'data/model_inputs/tas_CanRCM4-LE_ens35_1951-2016_ann_max_ensmean.nc',
    'Tmin (degC)': 'data/model_inputs/tas_CanRCM4-LE_ens35_1951-2016_ann_min_ensmean.nc',
    'TwJul2.5 (degC)': 'data/model_inputs/twb_CanRCM4-LE_ens35_1951-2016_1hr_jul97.5p_ensmean.nc',
    'TJul2.5 (degC)': 'data/model_inputs/tas_CanRCM4-LE_ens35_1951-2016_1hr_jul97.5p_ensmean.nc',
    'DRWP-RL5 (Pa)': 'data/model_inputs/DRWP_CanRCM4-LE_ens15_1951-2016_RL5_ensmean.nc',
    'annual_pr (mm)': 'data/model_inputs/pr_CanRCM4-LE_ens35_1951-2016_ann_sum_ensmean.nc',
    'annual_rain (mm)': 'data/model_inputs/rain_CanRCM4-LE_ens35_1951-2016_ann_sum_ensmean.nc',
    'RL50 (mm)': 'data/model_inputs/rain_CanRCM4-LE_ens35_1951-2016_annmax_rl50_ensmean.nc',
    "moisture_index": "data/model_inputs/moisture_index_CanRCM4-LE_ens15_1951-2016_ensmean.nc"
}

station_paths = {
    'RL50 (kPa)': 'data/station_inputs/Interim_snow_rain_load_LR_composite_stations_tbd_v4.csv',
    'mean RH (%)': 'data/station_inputs/rh_annual_mean_10yr_for_maps.csv',
    'HDD (degC-day)': 'data/station_inputs/hdd_Tmax_Tmin_allstations_v3_for_maps.csv',
    'SL50 (kPa)': 'data/station_inputs/Interim_snow_rain_load_LR_composite_stations_tbd_v4.csv',
    'WP10': 'data/station_inputs/wpress_stations_rl10_rl50_for_maps.csv',
    'WP50': 'data/station_inputs/wpress_stations_rl10_rl50_for_maps.csv',
    'TJan2.5 (degC)': 'data/station_inputs/janT2.5p_T1.0p_allstations_v3_min8yr_for_maps.csv',
    'TJan1.0 (degC)': 'data/station_inputs/janT2.5p_T1.0p_allstations_v3_min8yr_for_maps.csv',
    'Tmin (degC)': 'data/station_inputs/Interim_hdd_Tmax_Tmin_delivered.csv',
    'Tmax (degC)': 'data/station_inputs/Interim_hdd_Tmax_Tmin_delivered.csv',
    'TJul2.5 (degC)': 'data/station_inputs/julT97.5p_allstations_v3_min8yr_for_maps.csv',
    'TwJul2.5 (degC)': 'data/station_inputs/julTwb97.5p_allstations_v3_for_maps.csv',
    'DRWP-RL5 (Pa)': 'data/station_inputs/drwp_rl5_for_maps.csv',
    'annual_pr (mm)': 'data/station_inputs/pr_annual_mean_doy_MSC_25yr_for_maps.csv',
    'annual_rain (mm)': 'data/station_inputs/rain_annual_mean_doy_MSC_25yr_for_maps.csv',
    'RL50 (mm)':'data/station_inputs/Interim_snow_rain_load_LR_composite_stations_tbd_v4.csv',
    'moisture_index':'data/station_inputs/moisture_index_for_maps.csv'
}

plot_dict = {
    'RL50 (kPa)': (custom_cmap, True, 2),
    'mean RH (%)': (custom_cmap, False, 0),
    'HDD (degC-day)': ('RdBu_r', False, 0),
    'SL50 (kPa)': (custom_cmap, False, 0),
    'WP10': (custom_cmap, False, 2),
    'WP50': (custom_cmap, False, 2),
    'TJan2.5 (degC)': ('RdBu_r', False, 0),
    'TJan1.0 (degC)': ('RdBu_r', False, 0),
    'Tmin (degC)': ('RdBu_r', False, 0),
    'Tmax (degC)': ('RdBu_r', False, 0),
    'TJul2.5 (degC)': ('RdBu_r', False, 0),
    'TwJul2.5 (degC)': ('RdBu_r', False, 0),
    'annual_rain (mm)': (custom_cmap, True, 0),
    'annual_pr (mm)': (custom_cmap, True, 0),
    'DRWP-RL5 (Pa)': (custom_cmap, False, 0),
    'RL50 (mm)': (custom_cmap, True, 0),
    'moisture_index': (custom_cmap, False, 2),
    "Gum-LM RL10 (mm)": (custom_cmap, True, 1)
}
