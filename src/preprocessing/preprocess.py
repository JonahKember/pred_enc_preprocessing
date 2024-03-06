import os
import mne
import numpy as np
import pandas as pd

from dotenv import load_dotenv
from pyprep import NoisyChannels
from autoreject import AutoReject
from mne_bids import BIDSPath, read_raw_bids

from preprocessing import utils
from preprocessing.config import params

load_dotenv()
project_dir = os.getenv('project_dir')
bids_root = f'{project_dir}/data/raw/ds004395'


def preprocess_raw(subject, session):
    '''Run MNE-python preprocessing pipeline on raw EEG data.'''

    # Set-up.
    manufacturer = utils.get_cap_manufacturer(subject, session)
    bids_path = BIDSPath(root=bids_root, session=session, datatype='eeg', subject=subject, task='ltpFR', suffix='eeg')
    save_path = f'{project_dir}/data/processed/sub-{subject}'
    preload_path = f'{save_path}/sub-{subject}_ses-{session}_task-ltpFR_eeg-preloaded.edf'

    if not os.path.exists(save_path): 
        os.makedirs(save_path)

    if manufacturer == 'EGI': montage = mne.channels.make_standard_montage('GSN-HydroCel-129')
    if manufacturer == 'BioSemi': montage = mne.channels.make_standard_montage('biosemi128')

    # Load raw EEG.
    raw = read_raw_bids(bids_path, extra_params={'preload':preload_path}, verbose=0)
    raw = utils.set_eog_channels(raw, manufacturer)

    # Drop neck/chin channels.
    bad_channels = utils.get_bad_channels(manufacturer=manufacturer)
    raw.drop_channels(bad_channels, on_missing='warn')
    raw.set_montage(montage, on_missing='warn')

    # Donwsample.
    raw.resample(params['downsample'])

    # Identify and interpolate noisy/flat channels.
    nd = NoisyChannels(raw.copy().pick('eeg'))
    nd.find_all_bads(ransac=False)
    raw.info['bads'].extend(nd.get_bads())
    raw.interpolate_bads()

    # Filter.
    raw.filter(
        l_freq=params['l_freq_filter'],
        h_freq=params['h_freq_filter']
    )
    raw.notch_filter(params['notch_filter'])

    # Re-reference.
    raw.set_eeg_reference(ref_channels='average', projection=True)

    # Remove ocular artifacts (ICA).
    ica = mne.preprocessing.ICA(n_components=.99, max_iter='auto', random_state=0).fit(raw)
    eog_idx = ica.find_bads_eog(raw)

    if eog_idx[0]: 
        ica.exclude = eog_idx[0]
        ica.apply(raw)

    # Save cleaned raw data.
    raw.save(f'{save_path}/sub-{subject}_ses-{session}_task-ltpFR-raw.fif', overwrite=True)
    os.remove(preload_path)


def preprocess_epochs(subject, session):
    '''Epoch cleaned EEG data.'''

    save_path = f'{project_dir}/data/processed/sub-{subject}'
    raw = mne.io.read_raw_fif(f'{save_path}/sub-{subject}_ses-{session}_task-ltpFR-raw.fif')
    events, event_id = mne.events_from_annotations(raw, regexp='WORD_[RF]', verbose=0)

    # Epoch data.
    epochs = mne.Epochs(
        raw,
        events=events,
        event_id=event_id,
        preload=True,
        baseline=None,
        tmin=params['epoch_tmin'],
        tmax=params['epoch_tmax'],
        verbose=False,
        event_repeated='merge'
    )

    # Interpolate/remove bad trials (AutoReject).
    epochs = AutoReject().fit_transform(epochs)

    # Baseline-correct and save.
    epochs.apply_baseline(params['baseline'])
    epochs.save(f'{save_path}/sub-{subject}_ses-{session}_task-ltpFR-epo.fif', overwrite=True)


def preprocess_dataframe(subject, sessions):
    '''Extract single-epoch ROI time-series for each session, store in a dataframe, and write to Hierarchical Data Format (Version 5).'''

    # Set-up.
    utils.create_inverse_model()
    area_2_cortex = utils.get_area_2_cortex()
    rois = utils.get_rois(['Inferior_Frontal','Medial_Temporal'])

    # Get MNE labels for HCP-MMP1 ROIs.
    labels = mne.read_labels_from_annot('fsaverage', 'HCPMMP1', 'lh', subjects_dir=f'{project_dir}/data/external/')[1::]
    label_names = [label.name.replace('_ROI-lh','') for label in labels]

    # Contain labels to ROIS.
    indices = np.where([label in rois for label in label_names])[0]
    labels = [labels[i] for i in indices]

    for label in labels:

        # Initialize dataframe.
        df_label = utils.get_cleaned_events_df(subject=None, session=None, epochs=None, initial=True)

        for session in sessions:

            epochs = mne.read_epochs(f'{project_dir}/data/processed/sub-{subject}/sub-{subject}_ses-{session}_task-ltpFR-epo.fif', verbose=0)
            manufacturer = utils.get_cap_manufacturer(subject, session)
            inverse = mne.minimum_norm.read_inverse_operator(f'{project_dir}/data/processed/{manufacturer}-inv.fif', verbose=0)

            # Extract 1st PC within label (time_series).
            stcs = mne.minimum_norm.apply_inverse_epochs(epochs, inverse, label=label, method='MNE', pick_ori='normal', lambda2=.1, verbose=0)
            time_series = [stc.extract_label_time_course(label, src=inverse['src'], mode='pca_flip', verbose=0) for stc in stcs]
            time_series = np.concatenate(time_series)

            # Convert units of time_series.
            time_series = utils.get_current_density(time_series)

            # Add to dataframe.
            df_session = utils.get_cleaned_events_df(subject, session, epochs)
            df_session['roi'] = label.name.replace('_ROI-lh','')
            df_session['area'] = area_2_cortex[label.name.replace('_ROI-lh','')]
            df_session['time_series'] = [xs for xs in time_series]

            df_label = pd.concat([df_label, df_session], ignore_index=True)

        df_label.to_hdf(f'{project_dir}/data/datafranes/{subject}.h5', key=label.name.replace('_ROI-lh',''), complevel=9, index=False)
