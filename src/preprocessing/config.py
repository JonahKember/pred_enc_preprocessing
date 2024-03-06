params = {
    'overwrite':False,
    'h_freq_filter':90,
    'l_freq_filter':.1,
    'notch_filter':60,
    'epoch_tmin':-.5,
    'epoch_tmax':3,
    'baseline':(-.5,0),
    'downsample':250
}

# 'raw': 15 minutes, 16G.
# 'epochs': 10 minutes, 2G.
# 'dataframe': 10 minutes, 1G.

job_params = {
    'stage':'raw',
    'hours':00,
    'minutes':15,
    'mem_per_cpu':'10G',
    'n_jobs':750
}
