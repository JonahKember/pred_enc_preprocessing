# Specify preprocessing parameters.
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

# Specify parameters for creating/running SBATCH jobs.
job_params = {
    'stage':'dataframe',
    'hours':00,
    'minutes':2,
    'mem_per_cpu':'1G',
    'n_jobs':-1
}

# Note 'raw': (15 min, 16G), 'epochs': (10 min, 2G), 'dataframe': (2 min, 1G).
