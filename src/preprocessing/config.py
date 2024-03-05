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

job_params = {
    'stage':'epochs',
    'hours':00,
    'minutes':15,
    'mem_per_cpu':'2G',
    'n_jobs':1000
}

# stage = {'stage':'raw','hours':00,'minutes':15,'mem_per_cpu':'16G'},
# stage = {'stage':'epochs','hours':00,'minutes':15,'mem_per_cpu':'2G'},
# stage = {'stage':'dataframe','hours':00,'minutes':10,'mem_per_cpu':'4G'}
