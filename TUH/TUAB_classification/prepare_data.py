"""
Multiple discrete targets with the TUH EEG Corpus
=================================================

In this example, we showcase usage of multiple discrete targets per recording
with the TUH EEG Corpus.
"""

# Author: Lukas Gemein <l.gemein@gmail.com>
#
# License: BSD (3-clause)

import os
import mne
from torch.utils.data import DataLoader

mne.set_log_level('ERROR')  # avoid messages everytime a window is extracted

import tempfile

import numpy as np
import matplotlib.pyplot as plt

from braindecode.preprocessing import (
    preprocess, Preprocessor, create_fixed_length_windows, scale as multiply)

mne.set_log_level('ERROR')  # avoid messages everytime a window is extracted
###############################################################################
# If you want to try this code with the actual data, please delete the next
# section. We are required to mock some dataset functionality, since the data
# is not available at creation time of this example.
from braindecode.datasets.tuh import TUH, TUHAbnormal #_TUHMock as TUH  # noqa F811


###############################################################################
# We start by creating a TUH dataset. Instead of just a str, we give it
# multiple strings as target names. Each of the strings has to exist as a
# column in the description DataFrame.
#
# import hydra
# from omegaconf import DictConfig, OmegaConf

# @hydra.main(config_path="configs", config_name="prepare-args")
def prepare_tuab(TUH_PATH):
    #-> None:
    # print(OmegaConf.to_yaml(cfg))
    print(TUH_PATH)
    TUH_PATH = TUH_PATH #cfg.args.TUH_PATH
    tuh = TUHAbnormal(
        path=TUH_PATH,
        recording_ids=None, #range(3), #None, #range(10), #  [1,2,3,4,5],
        target_name=( 'pathological'),  # 'age', 'gender',
        preload=False,
        add_physician_reports=False,
    )
    tuh.description


    ###############################################################################
    # Iterating through the dataset gives x as ndarray(n_channels x 1) as well as
    # the target as [age of the subject, gender of the subject]. Let's look at the last example
    # as it has more interesting age/gender labels (compare to the last row of the dataframe above).
    print('dataset info:\n', tuh.description)
    x, y = tuh[-1]
    print('x:', x.shape)
    print('y:', y)

    x, y = tuh[-2]
    print('x:', x.shape)
    print('y:', y)

    ###############################################################################
    # preprocessing steps
    ###############################################################################
    # Next, we will perform some preprocessing steps. First, we will do some
    # selection of available recordings based on the duration. We will select those
    # recordings, that have at least five minutes duration. Data is not loaded here.

    def select_by_duration(ds, tmin=0, tmax=None):
        if tmax is None:
            tmax = np.inf
        # determine length of the recordings and select based on tmin and tmax
        split_ids = []
        for d_i, d in enumerate(ds.datasets):
            duration = d.raw.n_times / d.raw.info['sfreq']
            if tmin <= duration <= tmax:
                split_ids.append(d_i)
        splits = ds.split(split_ids)
        split = splits['0']
        return split


    tmin = 5 * 60
    tmax = None
    tuh = select_by_duration(tuh, tmin, tmax)


    ###############################################################################
    # Next, we will discard all recordings that have an incomplete channel
    # configuration (wrt the channels that we are interested in, i.e. the 21
    # channels of the international 10-20-placement). The dataset is subdivided into
    # recordings with 'le' and 'ar' reference which we will have to consider. Data
    # is not loaded here.

    short_ch_names = sorted([
        'A1', 'A2',
        'FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
        'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'FZ', 'CZ', 'PZ'])
    ar_ch_names = sorted([
        'EEG A1-REF', 'EEG A2-REF',
        'EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF',
        'EEG C4-REF', 'EEG P3-REF', 'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF',
        'EEG F7-REF', 'EEG F8-REF', 'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF',
        'EEG T6-REF', 'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF'])
    le_ch_names = sorted([
        'EEG A1-LE', 'EEG A2-LE',
        'EEG FP1-LE', 'EEG FP2-LE', 'EEG F3-LE', 'EEG F4-LE', 'EEG C3-LE',
        'EEG C4-LE', 'EEG P3-LE', 'EEG P4-LE', 'EEG O1-LE', 'EEG O2-LE',
        'EEG F7-LE', 'EEG F8-LE', 'EEG T3-LE', 'EEG T4-LE', 'EEG T5-LE',
        'EEG T6-LE', 'EEG FZ-LE', 'EEG CZ-LE', 'EEG PZ-LE'])
    assert len(short_ch_names) == len(ar_ch_names) == len(le_ch_names)
    ar_ch_mapping = {ch_name: short_ch_name for ch_name, short_ch_name in zip(
        ar_ch_names, short_ch_names)}
    le_ch_mapping = {ch_name: short_ch_name for ch_name, short_ch_name in zip(
        le_ch_names, short_ch_names)}
    ch_mapping = {'ar': ar_ch_mapping, 'le': le_ch_mapping}


    def select_by_channels(ds, ch_mapping):
        split_ids = []
        for i, d in enumerate(ds.datasets):
            ref = 'ar' if d.raw.ch_names[0].endswith('-REF') else 'le'
            # these are the channels we are looking for
            seta = set(ch_mapping[ref].keys())
            # these are the channels of the recoding
            setb = set(d.raw.ch_names)
            # if recording contains all channels we are looking for, include it
            if seta.issubset(setb):
                split_ids.append(i)
        return ds.split(split_ids)['0']


    tuh = select_by_channels(tuh, ch_mapping)


    ###############################################################################
    # Next, we will chain several preprocessing steps that are realized through
    # `mne`. Data will be loaded by the first preprocessor that has a mention of it
    # in brackets:
    #
    # #. crop the recordings to a region of interest
    # #. re-reference all recordings to 'ar' (requires load)
    # #. rename channels to short channel names
    # #. pick channels of interest
    # #. scale signals to micro volts (requires load)
    # #. clip outlier values to +/- 800 micro volts (requires load)
    # #. resample recordings to a common frequency (requires load)

    def custom_rename_channels(raw, mapping):
        # rename channels which are dependent on referencing:
        # le: EEG 01-LE, ar: EEG 01-REF
        # mne fails if the mapping contains channels as keys that are not present
        # in the raw
        reference = raw.ch_names[0].split('-')[-1].lower()
        assert reference in ['le', 'ref'], 'unexpected referencing'
        reference = 'le' if reference == 'le' else 'ar'
        raw.rename_channels(mapping[reference])


    def custom_crop(raw, tmin=0.0, tmax=None, include_tmax=True):
        # crop recordings to tmin – tmax. can be incomplete if recording
        # has lower duration than tmax
        # by default mne fails if tmax is bigger than duration
        tmax = min((raw.n_times - 1) / raw.info['sfreq'], tmax)
        raw.crop(tmin=tmin, tmax=tmax, include_tmax=include_tmax)


    tmin = 1 * 60
    tmax = 6 * 60
    sfreq = 100
    # Parameters for exponential moving standardization
    factor_new = 1e-3
    init_block_size = 1000
    preprocessors = [
    Preprocessor(custom_crop, tmin=tmin, tmax=tmax, include_tmax=False,
                 apply_on_array=False),
    Preprocessor('set_eeg_reference', ref_channels='average', ch_type='eeg'),
    Preprocessor(custom_rename_channels, mapping=ch_mapping,
                 apply_on_array=False),
    Preprocessor('pick_channels', ch_names=short_ch_names, ordered=True),
    Preprocessor(multiply, factor=1e6, apply_on_array=True),
    Preprocessor(np.clip, a_min=-800, a_max=800, apply_on_array=True),
    Preprocessor('resample', sfreq=sfreq),
    ]       


    ###############################################################################
    # Next, we apply the preprocessors on the selected recordings in parallel.
    # We additionally use the serialization functionality of
    # :func:`braindecode.preprocessing.preprocess` to limit memory usage during
    # preprocessing (as each file must be loaded into memory for some of the
    # preprocessing steps to work). This also makes it possible to use the lazy
    # loading capabilities of :class:`braindecode.datasets.BaseConcatDataset`, as
    # the preprocessed data is automatically reloaded with ``preload=False``.
    #
    # .. note::
    #    Here we use ``n_jobs=2`` as the machines the documentation is build on
    #    only have two cores. This number should be modified based on the machine
    #    that is available for preprocessing.
    N_JOBS = 1
    OUT_PATH = TUH_PATH + "/preprocessed/" #tempfile.mkdtemp()  # please insert actual output directory here
    from pathlib import Path
    Path(OUT_PATH).mkdir(parents=True, exist_ok=True)
    print(OUT_PATH)

    tuh_preproc = preprocess(
        concat_ds=tuh,
        preprocessors=preprocessors,
        n_jobs=N_JOBS,
        # save_dir=OUT_PATH
    )

    return tuh_preproc 
    # #Save dataset
    # tuh.save(
    #     path=OUT_PATH,
    #     overwrite=True,
    # )

    # dataset_loaded = load_concat_dataset(
    # path=OUT_PATH,
    # preload=True,
    # ids_to_load=[1, 3],
    # target_name=None,
    # )


    # we will create compute windows. We specify a
    # mapping from genders 'M' and 'F' to integers, since this is required for
    # decoding.

    tuh_windows = create_fixed_length_windows(
        tuh_preproc,
        start_offset_samples=0,
        stop_offset_samples=None,
        window_size_samples=1000,
        window_stride_samples=1000,
        drop_last_window=False,
        mapping={'M': 0, 'F': 1},  # map non-digit targets
    )
    # store the number of windows required for loading later on
    tuh_windows.set_description({
        "n_windows": [len(d) for d in tuh_windows.datasets]})


    ###############################################################################
    # Iterating through the dataset gives x as ndarray(n_channels x 1000), y as
    # [age, gender], and ind. Let's look at the last example again.
    x, y, ind = tuh_windows[-2]
    print('x:', x)
    print('y:', y)
    print('ind:', ind)


    ###############################################################################
    # We give the dataset to a pytorch DataLoader, such that it can be used for
    # model training.
    dl = DataLoader(
        dataset=tuh_windows,
        batch_size=4,
    )


    ###############################################################################
    # Iterating through the DataLoader gives batch_X as tensor(4 x n_channels x
    # 1000), batch_y as [tensor([4 x age of subject]), tensor([4 x gender of
    # subject])], and batch_ind. We will iterate to the end to look at the last example
    # again.
    for batch_X, batch_y, batch_ind in dl:
        pass
    print('batch_X:', batch_X.shape)
    print('batch_y:', batch_y)
    print('batch_ind:', batch_ind)

    return tuh_preproc

if __name__ == '__main__':
    main()
