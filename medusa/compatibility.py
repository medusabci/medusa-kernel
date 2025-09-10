# External imports
import numpy as np
import traceback
import mne
from dateutil import parser
from datetime import timezone

# MEDUSA imports
from medusa import components
from medusa.meeg.meeg_montages import get_camel_case_labels
from medusa import meeg, ecg, emg, eog
from medusa.bci import (mi_paradigms, cvep_spellers, ssvep_spellers,
                        erp_spellers, nft_paradigms)
from medusa.epoching import get_nearest_idx


class MNEData:

    def __init__(self):
        pass

    def mne_to_medusa(self):
        # TODO: MNE to MEDUSA format
        raise NotImplemented

    def medusa_to_mne(self, medusa_rec, save_path=None,
                      custom_onsets=None, custom_durations=None,
                      custom_descriptions=None):
        """ This function converts a MEDUSA recording into a MNE recording.
        However, it should be noted that the MNE format is restricted in
        terms of the data that could be stored. For that reason,
        it is recommended to maintain the original MEDUSA recording to access
        to additional paradigm information. Moreover, MNE only supports
        recording several biosignals if all of them share the same timestamps
        and sampling frequency. As this is not a requirement in MEDUSA,
        the signals will be stored separately.

        Note: for CustomExperimentData paradigms, it is required to specify
        the MNE annotations as custom_onsets, custom_durations and
        custom_descriptions. All of them must be lists with the same
        dimensions. Onsets are stored in seconds, relative to the start of
        the recording (where time is 0).

        Parameters
        -----------------------
        medusa_rec : medusa.components.Recording
            MEDUSA recording to convert
        save_path : str or None
            Filename path for the output signal (e.g., "C:\signal"). If
            several signals must be stored, the filename will append the
            signal name at the end (e.g., "C:\signal_eeg.fif",
            "C:\signal_ecg.fif"). If None, the conversion will not be saved.
        custom_onsets : list
            Custom onsets
        custom_durations : list
            Custom durations
        custom_descriptions : list
            Custom descriptions

        Returns
        -------------------------
        list(mne.io.RawArray)
            List of MNE files
        """

        # Anonymous function to convert each signal
        def convert_signal(signal_attr):
            # Create the info
            signal = getattr(rec, signal_attr)
            if hasattr(signal.channel_set, "l_cha"):
                ch_names = get_camel_case_labels(signal.channel_set.l_cha)
            else:
                ch_names = get_camel_case_labels(signal.channel_set["l_cha"])
            ch_types = [signal_attr] * len(ch_names)
            sampling_freq = signal.fs
            meas_date = parser.parse(rec.date)
            info = mne.create_info(
                ch_names=ch_names,
                ch_types=ch_types,
                sfreq=sampling_freq
            )
            info["subject_info"] = {
                "his_id": str(rec.subject_id)
            }
            info["description"] = str(rec.recording_id)
            info.set_meas_date(meas_date.replace(tzinfo=timezone.utc))

            # Set montage if any
            if hasattr(signal.channel_set, "montage"):
                montage = signal.channel_set.montage
            else:
                montage = signal.channel_set["montage"]
            if montage in ('10-20', '10-10', '10-05'):
                if signal.channel_set.montage == '10-20':
                    montage = 'standard_1020'
                else:
                    montage = 'standard_1005'
                info.set_montage(montage, match_case=False, on_missing='warn')

            # Set data
            raw_data = mne.io.RawArray(np.array(signal.signal).T, info)

            # Set events if any
            exp_annotations = {
                "onset": list(),
                "duration": list(),
                "description": list()
            }
            if len(rec.experiments) == 0 and hasattr(rec, "experiment_data"):
                rec.experiments = rec.experiment_data
            for key in rec.experiments:
                # Get the type of experiment (e.g., cvepspellerdata)
                exp_type = rec.get_experiments_with_class_name(
                    rec.experiments[key]['class_name'])
                # For each experiment of this type
                for exp in exp_type.values():
                    ann = {}
                    if rec.experiments[key]['class_name'] == "CVEPSpellerData":
                        ann = self.__get_cvepdata_annotations(signal.times, exp)
                    elif rec.experiments[key]['class_name'] == "MIData":
                        ann = self.__get_midata_annotations(signal.times, exp)
                    elif rec.experiments[key][
                        'class_name'] == "SSVEPSpellerData":
                        ann = self.__get_ssvepdata_annotations(signal.times, exp)
                    elif rec.experiments[key][
                        'class_name'] == "NeurofeedbackData":
                        ann = self.__get_nftdata_annotations(signal.times, exp)
                    elif rec.experiments[key]['class_name'] == "ERPSpellerData":
                        ann = self.__get_erpdata_annotations(signal.times, exp)
                    else:
                        if (custom_onsets is not None) and \
                            (custom_durations is not None) and \
                            (custom_descriptions is not None):
                            ann = {
                                "onset": custom_onsets,
                                "duration": custom_durations,
                                "description": custom_descriptions
                            }
                        else:
                            print("[WARNING] No annotations are included in "
                                  "this experiment")
                    if len(ann) > 0:
                        exp_annotations["onset"] += ann["onset"]
                        exp_annotations["duration"] += ann["duration"]
                        exp_annotations["description"] += ann["description"]

            annotations = mne.Annotations(
                onset=exp_annotations["onset"],
                duration=exp_annotations["duration"],
                description=exp_annotations["description"]
            )
            raw_data.set_annotations(annotations)
            return raw_data

        # Check errors
        if isinstance(medusa_rec, str):
            rec =  components.Recording.load(medusa_rec)
        elif isinstance(medusa_rec, components.Recording):
            rec = medusa_rec
        else:
            raise TypeError("The parameter 'medusa_rec' should be a path or a "
                            "Recording instance")

        # Detect the number of signals
        # Note: MNE only supports signals with the same timestamps and
        # sampling rate. As this is not guaranteed in MEDUSA, signals will be
        # converted separately
        mne_output = list()
        for signal_attr in rec.biosignals.keys():
            print(f"> Converting signal {signal_attr} to MNE format...")
            try:
                mne_data = convert_signal(signal_attr)
                mne_output.append(mne_data)

                # Save
                if save_path is not None:
                    mne_data.save(f"{save_path}/_{signal_attr}.fif")
            except Exception as e:
                print(f"[EXCEPTION] Cannot convert signal {signal_attr}. "
                      f"More information: {traceback.format_exc()}")

        return mne_output

    @staticmethod
    def __get_midata_annotations(times, midata):
        exp_annotations = {}
        start_offset = midata.w_trial_t[0] / 1000
        trial_duration = (midata.w_trial_t[1] - midata.w_trial_t[0])/1000
        sample_onsets = midata.onsets - times[0] + start_offset
        exp_annotations["onset"] = sample_onsets.tolist()
        exp_annotations["description"] = midata.mi_labels.tolist()
        exp_annotations["duration"] = [trial_duration] * len(sample_onsets)
        return exp_annotations

    @staticmethod
    def __get_erpdata_annotations(times, erpdata):
        # TODO: train y test
        raise NotImplemented
        # exp_annotations = {}
        #
        # sample_onsets = np.array(erpdata.onsets) - times[0]
        # exp_annotations["onset"] = sample_onsets.tolist()
        # exp_annotations["description"] = ["trial_onset"] * len(sample_onsets)
        # exp_annotations["duration"] = [np.min(np.diff(sample_onsets))] * len(
        #     sample_onsets)

    @staticmethod
    def __get_nftdata_annotations(times, nftdata):
        exp_annotations = {}
        sample_onsets = np.array(nftdata["run_onsets"]) - times[0]
        exp_annotations["onset"] = sample_onsets.tolist()
        exp_annotations["description"] = nftdata["run_success"]
        exp_annotations["duration"] = nftdata["run_durations"]
        return exp_annotations

    @staticmethod
    def __get_ssvepdata_annotations(times, ssvepspellerdata):
        exp_annotations = {}
        sample_onsets = ssvepspellerdata.onsets - times[0]
        exp_annotations["onset"] = sample_onsets.tolist()
        exp_annotations["description"] = ["trial_onset"] * len(sample_onsets)
        exp_annotations["duration"] = [ssvepspellerdata.stim_time] * len(sample_onsets)
        return exp_annotations

    @staticmethod
    def __get_custom_annotations(times, expdata, custom_onsets,
                                 custom_durations, custom_descriptions):
        exp_annotations = {}
        if custom_onsets is not None and hasattr(expdata, custom_onsets):
            sample_onsets = np.array(getattr(expdata, custom_onsets)) - times[0]
            exp_annotations["onset"] = sample_onsets.tolist()
        if custom_durations is not None and hasattr(expdata, custom_durations):
            exp_annotations["duration"] = getattr(expdata, custom_durations)
        if (custom_descriptions is not None) and \
            (hasattr(expdata, custom_descriptions)) and \
            (custom_onsets is not None):
            desc = getattr(expdata, custom_descriptions)
            if not isinstance(desc, list):
                desc = [desc] * len(sample_onsets)
            exp_annotations["description"] = desc
        return exp_annotations

    @staticmethod
    def __get_cvepdata_annotations(times, cvepspellerdata):
        exp_annotations = {}

        # Cycle onsets
        sample_onsets = cvepspellerdata.onsets - times[0]
        exp_annotations["onset"] = sample_onsets.tolist()
        exp_annotations["description"] = ["cycle_onset"] * len(sample_onsets)

        # If training
        fps = cvepspellerdata.fps_resolution
        if cvepspellerdata.mode == "train":
            # Get sequences of each calibrated command
            seqs_by_cycle = list()
            for idx in range(len(cvepspellerdata.command_idx)):
                m_ = int(cvepspellerdata.matrix_idx[idx])
                c_ = str(int(cvepspellerdata.command_idx[idx]))
                seqs_by_cycle.append(
                    cvepspellerdata.commands_info[m_][c_]['sequence']
                )
            # Set the duration for cycle onsets
            exp_annotations["duration"] = [len(s)/fps for s in seqs_by_cycle]
            # Annotate bit-wise
            for o_idx in range(len(sample_onsets)):
                bw_onsets = [sample_onsets[o_idx] + i/fps for i in range(
                        len(seqs_by_cycle[o_idx]))]
                bw_duration = [1/fps] * len(bw_onsets)
                bw_desc = seqs_by_cycle[o_idx]
                exp_annotations["onset"] += bw_onsets
                exp_annotations["duration"] += bw_duration
                exp_annotations["description"] += bw_desc

        # If test
        else:
            # Duration: we assume the same length for each command
            seq = cvepspellerdata.commands_info[0]['0']['sequence']
            exp_annotations["duration"] = [len(seq) / fps] * len(sample_onsets)

        return exp_annotations

if __name__ == '__main__':
    EXAMPLE_MEDUSA_CVEP = (r"D:\Users\Victor\OneDrive - "
                           r"UVa\Datasets\cvep-pary\data\btfc\btfc_2_train1.cvep.bson")
    EXAMPLE_M3ROB = (r"D:\Users\Victor\OneDrive - "
                      r"UVa\Datasets\m3rob-benito-menni\eegs\s001_KG099_QJ189\sesion04\KG099_TEST_04_10TRIAL.m3rob.bson")
    EXAMPLE_SSVEP = r"Z:\BBDD\BCI\studies\2025_neurobot\S01\ssvep\ej2.cvep.bson"
    EXAMPLE_BRAINGYM = (r"Z:\BBDD\BCI\studies\2024_braingym\Recordings"
                        r"\Braingym\BRAINGYM\Subject_5\Session_9\R4.mi.bson")
    EXAMPLE_VIDEOGAME = r"Z:\BBDD\BCI\studies\2025_videogame\S2\R8.rec.bson"
    EXAMPLE_NFT = (r"Z:\BBDD\BCI\databases\2022_nft_itaca_diego\Sujetos\S02"
                   r"\Sesiones\s_3\registros\run_4.nft.bson")
    EXAMPLE_ERP = r"Z:\BBDD\BCI\databases\asynchrony\medusa20\U01-control-r10.rcp.bson"
    convert = MNEData()
    convert.medusa_to_mne(EXAMPLE_ERP, None)