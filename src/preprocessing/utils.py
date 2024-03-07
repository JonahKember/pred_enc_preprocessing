import os
import mne
import json
import inspect
import scipy.io
import numpy as np
import pandas as pd

from dotenv import load_dotenv

from preprocessing.config import params

load_dotenv()
project_dir = os.getenv('project_dir')
account = os.getenv('account')
bids_root = f'{project_dir}/data/raw/ds004395'

def get_subject_id(subject_idx):
    '''Return subject ID from index, or all subject IDs if subject_idx='all'.'''

    subject_ids = pd.read_csv(f'{bids_root}/participants.tsv',sep='\t')['participant_id']

    if subject_idx == 'all': subject = [s.replace('sub-','') for s in subject_ids]
    else: subject = subject_ids[int(subject_idx) - 1].replace('sub-','')

    return(subject)


def change_stimulus_labels(subject, session):
    '''Change stimulus labels for the specified session-task according to subsequent memory.'''

    event_file = f'{bids_root}/sub-{subject}/ses-{session}/eeg/sub-{subject}_ses-{session}_task-ltpFR_events.tsv'
    df = pd.read_csv(event_file, sep='\t')
    
    # Loop through trial blocks and change the stimulus labels based on whether they were subsequently remembered.
    data = df[(df['trial_type'] == 'WORD') | (df['trial_type'] == 'REC_WORD')]
    for trial_id in range(1,17):

        trial = data[data['trial'] == trial_id]
        words_presented = trial.loc[trial['trial_type'] == 'WORD','item_name']
        words_spoken = trial.loc[trial['trial_type'] == 'REC_WORD','item_name'].unique()
        words_recalled = set(words_presented).intersection(words_spoken)
        words_recalled_indices = trial.index[(trial['trial_type'] == 'WORD') & (trial['item_name'].isin(words_recalled))]
        df.loc[words_recalled_indices,'trial_type'] = 'WORD_R'

    df.loc[df['trial_type'] == 'WORD','trial_type'] = 'WORD_F'
    df = df.fillna('n/a')
    df.to_csv(event_file, index=False, sep='\t')


def has_been_processed(subject, session):

    processed = {'raw':False, 'epochs':False, 'dataframe':False}

    if params['overwrite']:
        return processed

    if os.path.exists(f'{project_dir}/data/processed/sub-{subject}/sub-{subject}_ses-{session}_task-ltpFR-raw.fif'):
        processed['raw'] = True

    if os.path.exists(f'{project_dir}/data/processed/sub-{subject}/sub-{subject}_ses-{session}_task-ltpFR-epo.fif'):
        processed['epochs'] = True

    if os.path.exists(f'{project_dir}/data/dataframes/{subject}.h5'):
        processed['dataframe'] = True

    return processed


def inspect_data(subject, session):
    '''Ensure data is ready to be pre-processed.'''

    is_clean = True

    # Check that raw EEG (.edf/.bdf) data is included and not corrupt.
    eeg_path = f'{bids_root}/sub-{subject}/ses-{session}/eeg/sub-{subject}_ses-{session}_task-ltpFR_eeg'

    if not any([os.path.exists(f'{eeg_path}.edf'), os.path.exists(f'{eeg_path}.bdf')]):
        is_clean = False
        return is_clean

    if os.path.exists(f'{eeg_path}.edf'):
        try: raw = mne.io.read_raw_edf(f'{eeg_path}.edf')
        except: 
            is_clean = False
            return is_clean

    if os.path.exists(f'{eeg_path}.bdf'):
        try: raw = mne.io.read_raw_bdf(f'{eeg_path}.bdf')
        except: 
            is_clean = False
            return is_clean

    # Ensure events file exists.
    events = f'{bids_root}/sub-{subject}/ses-{session}/eeg/sub-{subject}_ses-{session}_task-ltpFR_events.tsv'
    try: events_df = pd.read_csv(events, sep='\t')
    except:
        is_clean = False
        return is_clean

    # Ensure onset times are not constant
    if events_df['onset'][0] == events_df['onset'][1]:
        is_clean = False
        return is_clean

    # Ensure at least 10 trials are included (minimum for AutoReject).
    change_stimulus_labels(subject, session)
    events_df = events_df[events_df.trial_type.isin(['WORD_R', 'WORD_F'])].reset_index(drop=True)
    if len(events_df) <= 10:
        is_clean = False
        return is_clean

    # Ensure experiment is not PEERS2 (note: PEERS2 has 12 blocks).
    if int(events_df.trial.max()) == 12: 
        is_clean = False
        return is_clean

    return is_clean


def get_subject_sessions(subject, overwrite=False):

    save_path = f'{project_dir}/data/interim/clean_sessions/{subject}.csv'
    if not overwrite and os.path.exists(save_path):
        df = pd.read_csv(save_path)
    else:
        session_dirs = [entry for entry in os.listdir(f'{bids_root}/sub-{subject}') if 'ses' in entry]
        sessions = [session.replace('ses-','') for session in session_dirs]

        clean_sessions = sessions.copy()
        for session in sessions:
            if not inspect_data(subject, session):
                clean_sessions.remove(session)
                continue

        df = pd.DataFrame(columns=['sessions'])
        df['sessions'] = clean_sessions 
        df.to_csv(save_path, index=False)

    return [str(ses) for ses in df.sessions.tolist()]


def get_cap_manufacturer(subject, session):
    '''Return electrode cap ('EGI' or 'BioSemi') used for session-task as str.'''

    with open(f'{bids_root}/sub-{subject}/ses-{session}/eeg/sub-{subject}_ses-{session}_task-ltpFR_eeg.json', 'r') as json_file:
        eeg_info = json.load(json_file)

    return(eeg_info['CapManufacturer'])


def get_bad_channels(manufacturer):
    '''Return the to-be-excluded channels around the neck/chin for the specified manufacturer.'''

    # EGI.
    if manufacturer == 'EGI':
        bad_channels = ['E17','E48','E49','E56','E63','E68','E73','E81','E88','E94','E99','E107','E113','E119','E125']

    # BioSemi.
    if manufacturer == 'BioSemi':
        bad_channels = ['A12','A13','A25','A26','B9','C15','C16','C17','C18','C28','C29','C30','C31','C6','C7','C8','C9','D32','EXG5','EXG6','EXG7','EXG8',
                            'Status','GSR1','GSR2','Erg1','Erg2','Resp','Plet','Temp','E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7', 'E8', 'E9', 'E10', 'E11', 'E12',
                            'E13', 'E14', 'E15', 'E16', 'E17', 'E18', 'E19', 'E20', 'E21', 'E22', 'E23', 'E24', 'E25', 'E26', 'E27', 'E28', 'E29', 'E30', 'E31',
                            'E32', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10', 'F11', 'F12', 'F13', 'F14', 'F15', 'F16', 'F17', 'F18', 'F19', 
                            'F20', 'F21', 'F22', 'F23', 'F24', 'F25', 'F26', 'F27', 'F28', 'F29', 'F30', 'F31', 'F32', 'G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7',
                            'G8', 'G9', 'G10', 'G11', 'G12', 'G13', 'G14', 'G15', 'G16', 'G17', 'G18', 'G19', 'G20', 'G21', 'G22', 'G23', 'G24', 'G25', 'G26',
                            'G27', 'G28', 'G29', 'G30', 'G31', 'G32', 'H1', 'H2', 'H3', 'H4','H5', 'H6', 'H7', 'H8', 'H9', 'H10', 'H11', 'H12', 'H13', 'H14',
                            'H15', 'H16', 'H17', 'H18', 'H19', 'H20', 'H21', 'H22', 'H23', 'H24', 'H25', 'H26', 'H27', 'H28', 'H29', 'H30', 'H31', 'H32']

    return(bad_channels)


def set_eog_channels(raw, manufacturer):
    '''Ensure EOG channels are marked (recurrent issue with some sessions).'''

    if manufacturer == 'BioSemi':
        raw.set_channel_types(dict(zip(['EXG1','EXG2','EXG3','EXG4'], ['eog','eog','eog','eog'])))

    if manufacturer == 'EGI':
        raw.set_channel_types(dict(zip(['E8','E25','E126','E127'], ['eog','eog','eog','eog'])))

    return raw


def get_report(subject, session):

    report = {}

    epochs = mne.read_epochs(f'{project_dir}/data/processed/sub-{subject}/sub-{subject}_ses-{session}_task-ltpFR-epo.fif')
    raw = mne.io.read_raw_fif(f'{project_dir}/data/processed/sub-{subject}/sub-{subject}_ses-{session}_task-ltpFR-raw.fif')
    events, event_id = mne.events_from_annotations(raw, regexp='WORD_[RF]')

    report['N_trials_remembered_raw'] = sum(events[:,2] == event_id['WORD_R'])
    report['N_trials_forgotten_raw'] = sum(events[:,2] == event_id['WORD_F'])

    report['N_trials_remembered_clean'] = sum(epochs.events[:,2] == epochs.event_id['WORD_R'])
    report['N_trials_forgotten_clean'] = sum(epochs.events[:,2] == epochs.event_id['WORD_F'])

    return report


def get_all_reports():
    '''Write /results/report.csv with information about N trials for each session.'''

    subjects = get_subject_id('all')
    df = pd.DataFrame(
        columns=[
        'subject',
        'session',
        'N_trials_remembered_raw',
        'N_trials_forgotten_raw',
        'N_trials_remembered_clean',
        'N_trials_forgotten_clean'
        ]
    )

    for subject in subjects:
        for session in get_subject_sessions(subject):
            try:
                report = get_report(subject, session)
                df_session = pd.DataFrame.from_dict(report, orient='index').T
                df_session['subject'] = subject
                df_session['session'] = session
                df = pd.concat([df, df_session])
            except:
                continue

    df.to_csv(f'{project_dir}/results/report.csv', index=False)


def get_channels(manufacturer):
    '''Get a list of the 110 good channels in either EGI or BioSemi electrode caps.'''

    if manufacturer == 'EGI':
        # Get subject with EGI channels for use as template, remove neck/chin/eog channels.
        ch_names = pd.read_csv(f'{bids_root}/sub-LTP063/ses-0/eeg/sub-LTP063_ses-0_task-ltpFR_channels.tsv', sep='\t').name.tolist()
        bad_channels = get_bad_channels('EGI')
        bad_channels.extend(['E8','E25','E126','E127'])

    if manufacturer == 'BioSemi':
        # Get subject with BioSemi channels for use as template, remove neck/chin/eog channels.
        ch_names = pd.read_csv(f'{bids_root}/sub-LTP331/ses-5/eeg/sub-LTP331_ses-5_task-ltpFR_channels.tsv', sep='\t').name.tolist()
        bad_channels = get_bad_channels('BioSemi')
        bad_channels.extend(['EXG1','EXG2','EXG3','EXG4'])

    ch_names = [channel for channel in ch_names if channel not in bad_channels]
    return ch_names


def get_area_2_cortex():
    '''Return dictionary mapping HCP-MMP1 area labels to HCP-MMP1_combined cortex labels.'''

    hcpex_labels = scipy.io.loadmat(f'{project_dir}/data/external/HCPex/HCPex_v1.1/HCPex_LabelID.mat')

    n_labels = len(hcpex_labels['LabelID'])
    hcpmmp1 = [hcpex_labels['LabelID'][idx,2][0] for idx in range(n_labels)]
    hcpmmp1_combined = [hcpex_labels['LabelID'][idx,5][0] for idx in range(n_labels)]
    area__2_cortex = dict(zip(hcpmmp1, hcpmmp1_combined))
    return area__2_cortex


def get_rois(cortices):
    '''Return a list of the ROIs in the specified cortices of the HCP-MMP1 atlas.'''

    hcp = get_area_2_cortex()

    if cortices == 'all':
        cortices=[
            'Inferior_Frontal',
            'Medial_Temporal',
            'Lateral_Temporal',
            'Ventral_Stream_Visual',
            'Superior_Parietal',
            'Premotor'
            ]

    if type(cortices) == str:
        cortices = [cortices]

    rois = list()
    for cortex in cortices:
        rois.extend([key for key, value in hcp.items() if cortex in value and 'R_' not in key])

    return rois


def get_current_density(time_series):
    '''Convert minimum norm estimate of source time-series (A-m) to surface current density (nA/mm).'''

    # Specify surface area (mm^2) per vertex.
    mm2_per_vertex = 6.246565416014217

    # Convert time-series from A-m to A-mm.
    time_series = time_series * 1e3

    # Normalize by surface area to get surface current density (from A-mm to A/mm).
    amps_per_mm = time_series/mm2_per_vertex

    # Change units from A/mm to nA/mm.
    nanoamps_per_mm = amps_per_mm * 1e9

    return(nanoamps_per_mm)


def get_cleaned_events_df(subject, session, epochs, initial=False):
    '''Return a pandas dataframe with all the information accompanying 'WORD_R' 
    and 'WORD_F events, ready to be used in the creation of dataframes.'''

    df_columns = [
            'subject',
            'experiment',
            'session',
            'block',
            'task',
            'response',
            'condition',
            'manufacturer',
            'date',
            'word',
            'case',
            'font',
            'color'
        ]

    if initial:
        return pd.DataFrame(columns=df_columns)

    events_path = f'{bids_root}/sub-{subject}/ses-{session}/eeg/sub-{subject}_ses-{session}_task-ltpFR_events.tsv'
    events_df = pd.read_csv(events_path, sep='\t')

    # Contain events dataframe to 'WORD_R' and 'WORD_F' trials.
    events_df = events_df[events_df.trial_type.isin(['WORD_R', 'WORD_F'])].reset_index(drop=True)

    # Remove trials rejected through autoreject.
    dropped_trials = [trial == ('AUTOREJECT',) for trial in epochs.drop_log]
    events_df = events_df.drop(index=np.where(dropped_trials)[0]).reset_index(drop=True)

    # Create single RGB column.
    events_df['color'] = list(zip(events_df['color_r'], events_df['color_g'], events_df['color_b']))

    # Rename columns.
    events_df = events_df.rename(columns={
        'item_name':'word',
        'trial_type':'condition',
        'resp':'response',
        'trial':'block'
    })

    # Add date, manufacturer, roi, area, and time_series columns.
    events_df['date'] = epochs.info['meas_date'].strftime('%Y-%m-%d %H:%M:%S')
    events_df['manufacturer'] = get_cap_manufacturer(subject, session)

    # Drop irrelevant columns.
    events_df = events_df.drop(columns=[
        'sample',
        'onset',
        'duration',
        'stim_file',
        'recog_resp',
        'recog_conf',
        'color_r',
        'color_g',
        'color_b',
        'item_num'
    ])

    # Sort dataframe.
    events_df = events_df.reindex(columns=df_columns)

    # Change coded values to explicit strings.
    events_df['response'] = events_df['response'].replace({0:'small/nonliving',1:'big/living',-1:'control'})
    events_df['task'] = events_df['task'].replace({0:'size',1:'animacy',-1:'control'})

    # Change datatypes to be easilty written/read.
    events_df['block'] = events_df['block'].astype(int)
    events_df.color = [list(rgb) for rgb in events_df.color]
    events_df.response = [str(resp) for resp in events_df.response]

    return events_df


def create_inverse_model():
    '''Create MNE-python inverse models for each electrode cap (EGI and BioSemi).'''

    cap = [
        ('EGI','GSN-HydroCel-129'),
        ('BioSemi','biosemi128')
    ]

    src = f'{project_dir}/data/external/fsaverage/bem/fsaverage-ico-5-src.fif'
    bem = f'{project_dir}/data/external/fsaverage/bem/fsaverage-5120-5120-5120-bem-sol.fif'

    for manufacturer, montage in cap:

        save_file = f'{project_dir}/data/processed/{manufacturer}-inv.fif'
        if os.path.exists(save_file):
            continue

        # Create info object.
        info = mne.create_info(
            ch_names=get_channels(manufacturer),
            sfreq=params['downsample'],
            ch_types='eeg'
        )

        info.set_montage(mne.channels.make_standard_montage(montage))

        # Create and save inverse model.
        cov = mne.make_ad_hoc_cov(info)
        fwd = mne.make_forward_solution(info, trans='fsaverage', src=src, bem=bem, eeg=True, mindist=5.0, verbose=0)
        inverse = mne.minimum_norm.make_inverse_operator(info, forward=fwd, noise_cov=cov, loose=0.2, depth=3.5, verbose=0)
        mne.minimum_norm.write_inverse_operator(save_file, inverse, overwrite=True)


def create_sbatch(subject, session, stage, hours, minutes, mem_per_cpu):
    '''Write a shell script for a specific job (a subject, session, and preprocessing stage).'''

    fname = f'sub-{subject}_ses-{session}_{stage}'

    sbatch = inspect.cleandoc(
        f'''#!/bin/bash
        #SBATCH --time={hours}:{minutes}:00
        #SBATCH --nodes=1
        #SBATCH --cpus-per-task=1
        #SBATCH --account={account}
        #SBATCH --mem-per-cpu={mem_per_cpu}
        #SBATCH --output={project_dir}/slurm/output/{fname}.out

        # Set-up the environment.
        source .env
        source ENV/bin/activate

        # Preprocess data.
        python {project_dir}/src/pipeline.py --subject {subject} --session {session} --stage {stage}

        # If pipeline fails, copy output to error directory.
        if [ $? -ne 0 ]; then
            cp {project_dir}/slurm/output/{fname}.out {project_dir}/slurm/error/{fname}.out
            rm {project_dir}/slurm/output/{fname}.out
        fi
        rm {project_dir}/jobs/{fname}.sh
        '''
    )

    # Write sbatch script.
    script = f'{project_dir}/jobs/{fname}.sh'
    with open(script, 'w') as f:
        f.write(sbatch)

    # Make script executable.
    os.chmod(script, 0o755)

    return script