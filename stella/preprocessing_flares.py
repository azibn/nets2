import os
import numpy as np
from tqdm import tqdm
from astropy.table import Table
from scipy.interpolate import interp1d
import re
import inspect
import random
import pickle
import matplotlib.pyplot as plt

from .utils import break_rest, do_the_shuffle, split_data

__all__ = ["FlareDataSet"]


class FlareDataSet(object):
    """
    Given a directory of files, reformat data to
    create a training set for the convolutional
    neural network.
    Files must be in '.npy' file format and contain
    at minimum the following indices:
         - 0th index = array of time
         - 1st index = array of flux
         - 2nd index = array of flux errors
    All other indices in the files are ignored.
    This class additionally requires a catalog of flare
    start times for labeling. The flare catalog can be
    in either '.txt' or '.csv' file format. This class will
    be passed into the stella.neural_network() class to
    create and train the neural network.
    """

    def __init__(
        self,
        fn_dir=None,
        catalog=None,
        downloadSet=None,
        additional_dirs=None,
        cadences=168,
        frac_balance=0.73,
        training=0.80,
        validation=0.10,
        time_offset=0,  # 2457000.0,
        merge_datasets=False,
        other_datasets=None,
        other_datasets_labels=None,
        num_subset=None,
        augment_portion=None,
        save_global_context=False,      # NEW: Enable global context preservation
        global_window_size=2000,        # NEW: RNN input size after downsampling
        global_window_days=3.0,         # NEW: Days around event for global context
        orbit_gap_threshold=0.5,        # NEW: TESS orbital gap detection (days)
        global_downsampling=4           # NEW: Downsampling factor for global context
    ):
        """
        Loads in time, flux, flux error data. Reshapes
        arrays into `cadences`-sized bins and labels
        flares vs. non-flares using the input catalog.

        Parameters
        ----------
        fn_dir : str, optional
             The path to where the files for the training
             set are stored.
        catalog : str, optional
             The path and filename of the catalog with
             marked flare start times
        downloadSet : stella.DownloadSets, optional
             The stella.DownloadSets class, which contains the
             flare catalog name and directory where light curves
             and the catalog are saved.
        additional_dirs: If you have an extra set of negative class
             objects that are not created in the standard `stella` way,
             specify the path here (e.g: a directory of exoplanet models).
        cadences : int, optional
             The size of each training set. Default is 200.
        frac_balance : float, optional
             The amount of the negative class to remove.
             Default is 0.75. If merged datasets are used, this only removes a fraction from the original dataset, not the merged dataset.
        training : float, optional
             Assigns the percentage of training set data for the
             model. Default is 80%
        validation : float, optionl
             Assigns the percentage of validation and testing set
             data for the model. Default is 10%.
        time_offset: optional
             Time correction from flare catalog to light curve and is
             necessary when using Max Guenther's catalog.
             Default is 0 as I fixed it from Feinstein et al's work. But need to use this
             for Kepler data.
        merge_datasets: optional
             If you have more than one dataset, you can combine them. Default is False.
        other_datasets: FlareDataSet, optional
             The datasets to merge into the current FlareDataSet.
        num_subset: int, optional
             Select a subset of positive class data if you do not want to use the entire catalog.
        augment_portion: float, optional
             Augments a portion of the positive class and assigns them as part of the negative class.
             Important if the shape is a characteristic (such as exocomets).
        save_global_context: bool, optional
             Whether to preserve full lightcurves for local+global RNN training.
             Default is False (no additional storage).
        global_window_size: int, optional
             Size of global context after downsampling for RNN input. Default is 2000.
        global_window_days: float, optional
             Time span in days around each event for global context. Default is 3.0 days.
        orbit_gap_threshold: float, optional
             Gap size in days to split TESS orbits. Default is 0.5 days.
        global_downsampling: int, optional
             Downsampling factor for global context. Default is 4.

        """
        if fn_dir is not None:
            self.fn_dir = fn_dir

        if additional_dirs is not None:
            self.additional_dirs = additional_dirs

        if catalog is not None:
            self.catalog = Table.read(catalog, format="ascii")

        if downloadSet is not None:
            self.fn_dir = downloadSet.fn_dir
            self.catalog = downloadSet.flare_table

        self.cadences = cadences

        self.frac_balance = frac_balance
        self.num_subset = num_subset
        
        # NEW: Global context attributes (only initialized if enabled)
        self.save_global_context = save_global_context
        if save_global_context:
            self.global_window_size = global_window_size
            self.global_window_days = global_window_days
            self.orbit_gap_threshold = orbit_gap_threshold
            self.global_downsampling = global_downsampling
            self.global_lightcurves = []
            self.orbit_segments = []
            self.window_to_global_map = None
        
        self.load_files(time_offset=time_offset)

        # NEW: Preserve global lightcurves before windowing (only if enabled)
        if self.save_global_context:
            self.preserve_global_lightcurves()

        self.reformat_data()
        self.original_labels = np.copy(self.labels)

        if merge_datasets == True:
            self.merge(other_datasets, labels=other_datasets_labels)
        


        misc = split_data(
            self.labels,
            self.full_matrix,
            self.full_ids,
            self.full_peaks,
            training,
            validation,
            self.original_labels,
        )



        if self.num_subset:
            subset_data = self.subsets(num_subset=self.num_subset)
            for attr, value in subset_data.items():
                setattr(self, attr, value)
            misc = split_data(
                self.labels,
                self.full_matrix,
                self.full_ids,
                self.full_peaks,
                training,
                validation,
                self.original_labels,
            )



        
        self.train_data = misc[0]
        self.train_labels = misc[1]
        self.val_data = misc[2]
        self.val_labels = misc[3]
        self.val_ids = misc[4]  # This was already assigned but keeping for clarity
        self.val_tpeaks = misc[5]
        self.test_data = misc[6]
        self.test_labels = misc[7]
        self.test_ids = misc[8]  # This was already assigned but keeping for clarity
        self.test_tpeaks = misc[9]
        self.train_labels_ori = misc[10]
        self.val_labels_ori = misc[11]
        self.test_labels_ori = misc[12]
        self.train_ids = misc[13]  # New assignments
        self.val_ids = misc[14]    # If you want to reassign
        self.test_ids = misc[15]   # If you want to reassign

        if (augment_portion is not None): 
            self.flip_exocomets(portion=augment_portion)
        self.print_properly(portion=augment_portion)

    def load_files(
        self,
        id_keyword="TIC",
        ft_keyword="tpeak",
        time_offset=0,
    ):
        """
        Loads in light curves from the assigned training set
        directory. Files must be formatted such that the ID
        of each star is first and followed by '_'
        (e.g. 123456789_sector09.npy).

        Attributes
        ----------
        times : np.ndarray
             An n-dimensional array of times, where n is the
             number of training set files.
        fluxes : np.ndarray
             An n-dimensional array of fluxes, where n is the
             number of training set files.
        flux_errs : np.ndarray
             An n-dimensional array of flux errors, where n is
             the number of training set files.
        ids : np.array
             An array of light curve IDs for each time/flux/flux_err.
             This is essential for labeling flare events.
        id_keyword : str, optional
             The column header in catalog to identify target ID.
             Default is 'tic_id'.
        ft_keyword : str, optional
             The column header in catalog to identify flare peak time.
             Default is 'tpeak'.
        time_offset : float, optional
             Time correction from flare catalog to light curve and is
             necessary when using Max Guenther's catalog.
             Default is 0
        subset: bool, optional
             Returns a specific amount of data from the catalog (shuffled).
             Default is False, where the import consists of all the times in the catalog.
        num_subset: int, optional
             Specify a number of rows to call from the catalog.
        """


        files = os.listdir(self.fn_dir)

        files = np.sort([i for i in files if i.endswith(".npy") and any(x in i for x in ["sector", "_q", "_c"])])


        tics, time, flux, err, real, model, tpeaks = [], [], [], [], [], [], []

        for fn in files:
            data = np.load(os.path.join(self.fn_dir, fn), allow_pickle=True)
            
            split_fn = fn.split("_")
            tic = int(split_fn[0])

            tics.append(tic)
            
            # Single regex to match all three formats: sector_07, q11, c00
            sector_match = re.search(r'(?:sector[-_]?|[qc])(\d+)', split_fn[1], re.IGNORECASE)
            if sector_match:
                sector = int(sector_match.group(1))
            else:
                print(f"Could not extract sector/quarter/campaign number from {fn}")
                continue


            time.append(data[0])
            flux.append(data[1])
            err.append(data[2])

            try:
                real.append(data[3])
                model.append(data[4])
            except:
                pass

            peaks = self.catalog[(self.catalog[id_keyword] == tic)][ft_keyword].data

            peaks = peaks - time_offset
            tpeaks.append(peaks)
        self.ids = np.array(tics)
        self.time = np.array(time, dtype=np.ndarray)  # in TBJD
        self.flux = np.array(flux, dtype=np.ndarray)
        self.flux_err = np.array(err, dtype=np.ndarray)
        self.real = np.array(real, dtype=np.ndarray)
        self.model = np.array(model, dtype=np.ndarray)
        self.tpeaks = tpeaks  # in TBJD



    # def reformat_data(self, random_seed=321):
    #     """
    #     Reformats the data into `cadences`-sized array and assigns
    #     a label based on flare times defined in the catalog.
    #     """
    #     ss = 300000
    #     training_matrix = np.zeros((ss, self.cadences))
    #     training_labels = np.zeros(ss, dtype=int)
    #     training_peaks = np.zeros(ss)
    #     training_ids = np.zeros(ss)

    #     x = 0
    #     print(f"Starting to process {len(self.time)} files")
    #     dropped_files = []

    #     for i in tqdm(range(len(self.time))):
    #         flares = np.array([], dtype=int)
    #         file_added = False

    #         # Track file processing
    #         current_file = {
    #             'id': self.ids[i],
    #             'n_peaks': len(self.tpeaks[i]),
    #             'peaks_processed': 0,
    #             'peaks_added': 0,
    #             'reason_dropped': []
    #         }

    #         for peak in self.tpeaks[i]:
    #             current_file['peaks_processed'] += 1
                
    #             # Find points around peak
    #             arg = np.where(
    #                 (self.time[i] > (peak - 0.08)) & (self.time[i] < (peak + 0.08))
    #             )[0]

    #             if len(arg) == 0:
    #                 current_file['reason_dropped'].append(f"No points found around peak {peak}")
    #                 continue

    #             closest = arg[np.argmin(np.abs(peak - self.time[i][arg]))]
    #             start = int(closest - self.cadences / 2)
    #             end = int(closest + self.cadences / 2)

    #             flare_region = np.arange(start, end, 1, dtype=int)

    #             # Check window boundaries
    #             if not ((start >= 0) and (end < len(self.time[i]))):
    #                 current_file['reason_dropped'].append(f"Window out of bounds: start={start}, end={end}, len={len(self.time[i])}")
    #                 continue

    #             # Add example
    #             try:
    #                 training_peaks[x] = self.time[i][closest] + 0.0
    #                 training_ids[x] = self.ids[i] + 0.0
    #                 training_matrix[x] = self.flux[i][flare_region]
    #                 training_labels[x] = 1
    #                 x += 1
    #                 current_file['peaks_added'] += 1
    #                 file_added = True
    #                 flares = np.append(flares, flare_region)
    #             except Exception as e:
    #                 current_file['reason_dropped'].append(f"Error adding example: {str(e)}")

    #         if not file_added:
    #             dropped_files.append(current_file)

    #         # Process negative examples
    #         time_removed = np.delete(self.time[i], flares)
    #         flux_removed = np.delete(self.flux[i], flares)
    #         flux_err_removed = np.delete(self.flux_err[i], flares)

    #         nontime, nonflux, nonerr = break_rest(
    #             time_removed, flux_removed, flux_err_removed, self.cadences
    #         )
            
    #         for j in range(len(nonflux)):
    #             if x >= ss:
    #                 break
    #             else:
    #                 training_ids[x] = self.ids[i] + 0.0
    #                 training_peaks[x] = nontime[j][int(self.cadences / 2)]
    #                 training_matrix[x] = nonflux[j]
    #                 training_labels[x] = 0
    #                 x += 1

    #     print("\nSummary of dropped files:")
    #     for file in dropped_files:
    #         print(f"\nFile ID {file['id']}:")
    #         print(f"- Total peaks: {file['n_peaks']}")
    #         print(f"- Peaks processed: {file['peaks_processed']}")
    #         print(f"- Peaks added: {file['peaks_added']}")
    #         print("- Reasons dropped:")
    #         for reason in file['reason_dropped']:
    #             print(f"  * {reason}")

    #     print(f"\nBefore trimming arrays:")
    #     print(f"- Total examples created: {x}")
    #     print(f"- Positive examples: {np.sum(training_labels[:x] == 1)}")
    #     print(f"- Negative examples: {np.sum(training_labels[:x] == 0)}")

    #     # DELETE EXTRA END OF TRAINING MATRIX AND LABELS
    #     training_matrix = np.delete(
    #         training_matrix, np.arange(x, ss, 1, dtype=int), axis=0
    #     )
    #     labels = np.delete(training_labels, np.arange(x, ss, 1, dtype=int))
    #     training_peaks = np.delete(training_peaks, np.arange(x, ss, 1, dtype=int))
    #     training_ids = np.delete(training_ids, np.arange(x, ss, 1, dtype=int))

    #     print(f"\nAfter trimming arrays:")
    #     print(f"- Total examples: {len(labels)}")
    #     print(f"- Positive examples: {np.sum(labels == 1)}")
    #     print(f"- Negative examples: {np.sum(labels == 0)}")

    #     ids, matrix, label, peaks = do_the_shuffle(
    #         training_matrix, labels, training_peaks, training_ids, self.frac_balance
    #     )

    #     print(f"\nAfter shuffling and balancing:")
    #     print(f"- Total examples: {len(label)}")
    #     print(f"- Positive examples: {np.sum(label == 1)}")
    #     print(f"- Negative examples: {np.sum(label == 0)}")

    #     self.labels = label
    #     self.original_labels = np.copy(label)
    #     self.full_peaks = peaks
    #     self.full_ids = ids
    #     self.full_matrix = matrix


    def reformat_data(self, random_seed=321):
        """
        Reformats the data into `cadences`-sized array and assigns
        a label based on flare times defined in the catalog.

        Parameters
        ----------
        random_seed : int, optional
             A random seed to set for randomizing the order of the
             training_matrix after it is constructed. Default is 321.

        Attributes
        ----------
        training_matrix : np.ndarray
             An n x `cadences`-sized array used as the training data.
        labels : np.array
             An n-sized array of labels for each row in the training
             data.
        """
        ss = 300000

        training_matrix = np.zeros((ss, self.cadences))
        training_labels = np.zeros(ss, dtype=int)
        training_peaks = np.zeros(ss)
        training_ids = np.zeros(ss)
        
        # NEW: Track global mapping if enabled
        if hasattr(self, 'save_global_context') and self.save_global_context:
            training_global_idx = np.zeros(ss, dtype=int)

        x = 0



        for i in tqdm(range(len(self.time))):
            flares = np.array([], dtype=int)

            for peak in self.tpeaks[i]:

                arg = np.where(
                    (self.time[i] > (peak - 0.06)) & (self.time[i] < (peak + 0.06))
                )[
                    0
                ]  

                # expanded the peak to one hour (in days) rather than
                ## 30 minutes (in days)

                # DOESN'T LIKE FLARES AT THE VERY END OF THE LIGHT CURVE
                # (AND NEITHER DO I)
                if len(arg) > 0:
                    closest = arg[np.argmin(np.abs(peak - self.time[i][arg]))]
                    start = int(closest - self.cadences / 2)
                    end = int(closest + self.cadences / 2)



                    flare_region = np.arange(start, end, 1, dtype=int)
       

                    if (start > 0) and (end < len(self.time[i])):
                        flares = np.append(flares, flare_region)
                        ### ADD LABELS AND MATRIX PROPERLY
                        fails = []
                        try:
                            ### ADD ASSERTION HERE FOR INTERPOLATION CHECKING
                            training_peaks[x] = self.time[i][closest] + 0.0
                            training_ids[x] = self.ids[i] + 0.0
                            training_matrix[x] = self.flux[i][flare_region]
                            training_labels[x] = 1
                            
                            # NEW: Map this window to its orbital segment if enabled
                            if hasattr(self, 'save_global_context') and self.save_global_context:
                                window_time = self.time[i][closest]
                                global_idx = self._find_orbital_segment(i, window_time)
                                training_global_idx[x] = global_idx
                            
                            x += 1

                        except IndexError:
                            fails.append(self.ids)
                            continue
      

            time_removed = np.delete(self.time[i], flares)
            flux_removed = np.delete(self.flux[i], flares)
            flux_err_removed = np.delete(self.flux_err[i], flares)

            nontime, nonflux, nonerr = break_rest(
                time_removed, flux_removed, flux_err_removed, self.cadences
            )

            for j in range(len(nonflux)):
                if x >= ss:
                    break
                else:

                    training_ids[x] = self.ids[i] + 0.0
                    training_peaks[x] = nontime[j][int(self.cadences / 2)]
                    training_matrix[x] = nonflux[j]
                    training_labels[x] = 0
                    
                    # NEW: Map this window to its orbital segment if enabled
                    if hasattr(self, 'save_global_context') and self.save_global_context:
                        window_time = nontime[j][int(self.cadences / 2)]
                        global_idx = self._find_orbital_segment(i, window_time)
                        training_global_idx[x] = global_idx

                    x += 1


        # DELETE EXTRA END OF TRAINING MATRIX AND LABELS
        training_matrix = np.delete(
            training_matrix, np.arange(x, ss, 1, dtype=int), axis=0
        )
        labels = np.delete(training_labels, np.arange(x, ss, 1, dtype=int))
        training_peaks = np.delete(training_peaks, np.arange(x, ss, 1, dtype=int))
        training_ids = np.delete(training_ids, np.arange(x, ss, 1, dtype=int))
        
        # NEW: Also trim global mapping if enabled
        if hasattr(self, 'save_global_context') and self.save_global_context:
            training_global_idx = np.delete(training_global_idx, np.arange(x, ss, 1, dtype=int))


        # Use original shuffle function (global context is reconstructed on-demand)
        ids, matrix, label, peaks = do_the_shuffle(
            training_matrix, labels, training_peaks, training_ids, self.frac_balance
        )

        self.labels = label
        self.original_labels = np.copy(label)
        self.full_peaks = peaks
        self.full_ids = ids
        self.full_matrix = matrix



    def merge(self, other, labels=0):
        """Merge one FlareDataSet instance into this one.

        other: your other FlareDataSet instances.
        set_to_negative: Set all the positive class data from the other instances to 0. Default False.
        labels: Assign each FlareDataSet instance's positive class its own label. Default None.

        """
        ### READS IN DATASETS (IF MULTIPLE, THIS IS HANDLED TOO)
        for i, o in enumerate(other):
            if labels != 0:

                if isinstance(labels, (list, np.ndarray)):
                    current_label = labels[i]
                else:
                    current_label = labels
                
                o.original_labels[:] = current_label
                
                if current_label != 1:
                    o.labels[:] = 0


            self.full_matrix = np.concatenate(
                [self.full_matrix, o.full_matrix]
            )
            self.labels = np.concatenate([self.labels, o.labels])
            self.original_labels = np.concatenate(
                [self.original_labels, o.original_labels]
            )
            self.full_ids = np.concatenate([self.full_ids, o.full_ids])
            self.full_peaks = np.concatenate(
                [self.full_peaks, o.full_peaks]
            )

            self.ids = np.concatenate([self.ids, o.ids])
            self.time = np.concatenate([self.time, o.time], axis=0)
            self.flux = np.concatenate([self.flux, o.flux], axis=0)
            self.flux_err = np.concatenate([self.flux_err, o.flux_err], axis=0)
            self.real = np.concatenate([self.real, o.real], axis=0)

            ### NOTE: TPEAKS IS NOT CONCATENATED HERE.



    def subsets(self, num_subset):
        """Returns subset of positive class.
        FUNCTION STILL UNDER CONSTRUCTION.
        """

        indices = np.where(self.labels == 1)[0]
        print(len(indices))

        if isinstance(num_subset, int):
            if num_subset > len(indices):
                raise ValueError(
                    f"Requested {num_subset} positive class samples, but only {len(indices)} are available."
                )

            random_indices = random.sample(list(indices), num_subset)

            attributes = [
                "training_matrix",
                "labels",
                "original_labels",
                "train_ids",
                "tpeaks",
            ]
            for attr in attributes:
                attr_len = len(getattr(self, attr))
                print(f"Length of {attr}: {attr_len}")
        return {
            attribute: getattr(self, attribute)[random_indices]
            for attribute in attributes
        }
    

    def flip_exocomets(self, portion=None, horizontal_ratio=0.5, vertical_ratio=0.5):
        """Function to augment a portion of the positive class data by flipping exocomet transits.
        By default, splits the portion equally between horizontal and vertical flips.
        
        Parameters:
        ------------
        portion: float, optional
            Total portion of the positive class to flip. Default is None.
        horizontal_ratio: float, optional
            Ratio of flips that should be horizontal. Default is 0.5 (50% of flips).
        vertical_ratio: float, optional
            Ratio of flips that should be vertical. Default is 0.5 (50% of flips).
            The portion of the positive class data to flip. Default is None.
        """
        if portion is None:
            return
                
        if not 0 <= portion <= 1:
            raise ValueError("portion must be between 0 and 1")    
        if not np.isclose(horizontal_ratio + vertical_ratio, 1.0):
            raise ValueError("horizontal_ratio and vertical_ratio must sum to 1.0")

        ind_pc = np.where(self.train_labels == 1)[0]
        val_pc = np.where(self.val_labels == 1)[0]
        total_flips_train = int(len(ind_pc) * portion)
        total_flips_val = int(len(val_pc) * portion)
        
        horizontal_flips_train = int(total_flips_train * horizontal_ratio)
        horizontal_flips_val = int(total_flips_val * horizontal_ratio)
        
        vertical_flips_train = int(total_flips_train * vertical_ratio)
        vertical_flips_val = int(total_flips_val * vertical_ratio)

        # Handle horizontal flips
        if horizontal_flips_train > 0:
            flip_ind = np.random.choice(ind_pc, size=horizontal_flips_train, replace=False)
            flip_ind_val = np.random.choice(val_pc, size=horizontal_flips_val, replace=False)
            
            flipped_data = [self.train_data[i][::-1] for i in flip_ind]
            flipped_data_val = [self.val_data[i][::-1] for i in flip_ind_val]
        
            
            self._add_flipped_data(flip_ind, flip_ind_val, flipped_data, flipped_data_val)

        # Handle vertical flips
        if vertical_flips_train > 0:
            flip_ind = np.random.choice(ind_pc, size=vertical_flips_train, replace=False)
            flip_ind_val = np.random.choice(val_pc, size=vertical_flips_val, replace=False)
            
            flipped_data = np.array([-self.train_data[i] for i in flip_ind])
            flipped_data_val = np.array([-self.val_data[i] for i in flip_ind_val])
            
            self._add_flipped_data(flip_ind, flip_ind_val, flipped_data, flipped_data_val)
       

    def _add_flipped_data(self, flip_ind, flip_ind_val, flipped_data, flipped_data_val):
        flipped_labels = np.zeros(len(flipped_data))
        flipped_labels_val = np.zeros(len(flipped_data_val))
        
        flipped_labels_ori = np.full(shape=(len(flipped_data),), fill_value=99)
        flipped_labels_ori_val = np.full(shape=(len(flipped_data_val),), fill_value=99)

        self.train_data = np.concatenate((self.train_data, flipped_data), axis=0)
        self.train_labels = np.concatenate((self.train_labels, flipped_labels))
        self.train_labels_ori = np.concatenate((self.train_labels_ori, flipped_labels_ori))
        self.train_ids = np.concatenate((self.train_ids, self.train_ids[flip_ind]))

        self.val_data = np.concatenate((self.val_data, flipped_data_val), axis=0)
        self.val_labels = np.concatenate((self.val_labels, flipped_labels_val))
        self.val_labels_ori = np.concatenate((self.val_labels_ori, flipped_labels_ori_val))
        self.val_ids = np.concatenate((self.val_ids, self.val_ids[flip_ind_val]))
        self.val_tpeaks = np.concatenate((self.val_tpeaks, self.val_tpeaks[flip_ind_val]))



    def print_properly(self, portion=None):
        ind_pc = np.where(self.train_labels == 1)
        ind_nc = np.where(self.train_labels != 1)
        print(f"Number of positive class training data: {len(ind_pc[0])}")
        print(f"Number of negative class training data: {len(ind_nc[0])}")

        val_pc = np.where(self.val_labels == 1)
        val_nc = np.where(self.val_labels != 1)
        print(f"Number of positive class validation data: {len(val_pc[0])}")
        print(f"Number of negative class validation data: {len(val_nc[0])}")

        ### SORT THIS BIT OUT
        if portion is not None:
            print(
                f"Size of augmented data (training set only): {int(len(ind_pc[0]) * portion)}"
            )
        else:
            print(f"Size of augmented data (training set only): 0")

        # I need to change original labels to original labels, original train labels and original val labels
        unique_train, counts_train = np.unique(self.train_labels, return_counts=True)
        unique_val, counts_val = np.unique(self.val_labels_ori, return_counts=True)

        for value, count in zip(unique_train, counts_train):
            print(f"Class label (training): {value}, Count: {count}")

        for value, count in zip(unique_val, counts_val):
            print(f"Class label (validation): {value}, Count: {count}")

    
        print(f"Total size of training set: {len(self.train_data)}")

        
        print(f"Total size of validation set: {len(self.val_data)}")
        print(f"Total size of test set: {len(self.test_data)}")


    def print_properly(self, portion=None):
        ind_pc = np.where(self.train_labels == 1)
        ind_nc = np.where(self.train_labels != 1)
        print(f"Number of positive class training data: {len(ind_pc[0])}")
        print(f"Number of negative class training data: {len(ind_nc[0])}")

        val_pc = np.where(self.val_labels == 1)
        val_nc = np.where(self.val_labels != 1)
        print(f"Number of positive class validation data: {len(val_pc[0])}")
        print(f"Number of negative class validation data: {len(val_nc[0])}")

        ### SORT THIS BIT OUT
        if portion is not None:
            print(
                f"Size of augmented data (training set only): {int(len(ind_pc[0]) * portion)}"
            )
        else:
            print(f"Size of augmented data (training set only): 0")

        # I need to change original labels to original labels, original train labels and original val labels
        unique_train, counts_train = np.unique(self.train_labels, return_counts=True)
        unique_val, counts_val = np.unique(self.val_labels_ori, return_counts=True)

        for value, count in zip(unique_train, counts_train):
            print(f"Class label (training): {value}, Count: {count}")

        for value, count in zip(unique_val, counts_val):
            print(f"Class label (validation): {value}, Count: {count}")

    
        print(f"Total size of training set: {len(self.train_data)}")
        
        if self.val_data is not None:
            print(f"Total size of validation set: {len(self.val_data)}")
        
        if self.test_data is not None:
            print(f"Total size of test set: {len(self.test_data)}")

        try:
            print(
                f"Approximate class imbalance: {np.round(100 * (1 - len(ind_pc[0]) / len(ind_nc[0])))}"
            )
        except ZeroDivisionError:
            print("No second class to calculate imbalance.")

    def save(self,output='ds.pkl'):
        """Save the FlareDataSet instance to a file."""
        with open(output, 'wb') as f:
            pickle.dump(self, f)

    def verify_stratification(self, y_train, y_val, y_train_ori, y_val_ori):
        """
        Verifies stratification of both binary and original labels in train/val splits.
        
        Parameters:
        -----------
        y_train: array-like
            Binary labels for training set
        y_val: array-like
            Binary labels for validation set
        y_train_ori: array-like
            Original labels for training set
        y_val_ori: array-like
            Original labels for validation set
        """
        # Calculate proportions for binary labels
        train_total = len(y_train)
        val_total = len(y_val)
        
        print("Binary Label Distribution:")
        print("-" * 30)
        for label in np.unique(np.concatenate([y_train, y_val])):
            train_prop = np.sum(y_train == label) / train_total
            val_prop = np.sum(y_val == label) / val_total
            print(f"Label {label}:")
            print(f"  Training: {train_prop:.3f} ({np.sum(y_train == label)} samples)")
            print(f"  Validation: {val_prop:.3f} ({np.sum(y_val == label)} samples)")
            print(f"  Difference: {abs(train_prop - val_prop):.3f}")
            print()
        
        print("Original Label Distribution:")
        print("-" * 30)
        for label in np.unique(np.concatenate([y_train_ori, y_val_ori])):
            train_prop = np.sum(y_train_ori == label) / train_total
            val_prop = np.sum(y_val_ori == label) / val_total
            print(f"Label {label}:")
            print(f"  Training: {train_prop:.3f} ({np.sum(y_train_ori == label)} samples)")
            print(f"  Validation: {val_prop:.3f} ({np.sum(y_val_ori == label)} samples)")
            print(f"  Difference: {abs(train_prop - val_prop):.3f}")
            print()

    def preserve_global_lightcurves(self):
        """
        Save processed global lightcurves split by TESS orbits.
        Each orbit becomes a separate global context segment.
        """
        print("Preserving global lightcurves with orbit splitting...")
        
        self.global_lightcurves = []
        self.orbit_segments = []  # Track which orbit each global segment belongs to
        
        for i in range(len(self.time)):
            # Extract arrays from object array
            time = np.array(self.time[i], dtype=float)
            flux = np.array(self.flux[i], dtype=float) 
            flux_err = np.array(self.flux_err[i], dtype=float)
            
            # Clean data first
            mask = ~np.isnan(time) & ~np.isnan(flux) & ~np.isnan(flux_err)
            time_clean = time[mask]
            flux_clean = flux[mask]
            flux_err_clean = flux_err[mask]
            
            # Split into orbital segments
            orbit_segments = self._split_by_orbits(time_clean, flux_clean, flux_err_clean)
            
            # Process each orbit separately
            for orbit_idx, (orbit_time, orbit_flux, orbit_err) in enumerate(orbit_segments):
                # Skip very short segments (< 200 cadences or < 0.2 days)
                if len(orbit_time) < 200 or (orbit_time[-1] - orbit_time[0]) < 0.2:
                    continue
                    
                # Normalize flux for this orbit
                flux_normalized = orbit_flux / np.nanmedian(orbit_flux)
                
                # Find which transits fall in this orbit
                orbit_tpeaks = []
                for tpeak in self.tpeaks[i]:
                    if orbit_time[0] <= tpeak <= orbit_time[-1]:
                        orbit_tpeaks.append(tpeak)
                
                # Store global context for this orbit
                global_lc = {
                    'tic_id': self.ids[i],
                    'orbit_idx': orbit_idx,
                    'time': orbit_time,
                    'flux': flux_normalized,
                    'flux_err': orbit_err,
                    'tpeaks': orbit_tpeaks,
                    'time_span': (orbit_time[0], orbit_time[-1])
                }
                
                self.global_lightcurves.append(global_lc)
                self.orbit_segments.append((i, orbit_idx))  # (lightcurve_idx, orbit_idx)
        
        print(f"Preserved {len(self.global_lightcurves)} orbital segments from {len(self.time)} targets")
        self._print_orbit_stats()

    def _split_by_orbits(self, time, flux, flux_err, gap_threshold=None):
        """
        Split lightcurve by orbital gaps.
        
        Parameters
        ----------
        time, flux, flux_err : np.ndarray
            Cleaned time series data
        gap_threshold : float, optional
            Gap size in days to split orbits. Uses self.orbit_gap_threshold if None.
            
        Returns
        -------
        segments : list
            List of (time, flux, flux_err) tuples for each orbit
        """
        if gap_threshold is None:
            gap_threshold = self.orbit_gap_threshold
            
        if len(time) == 0:
            return []
        
        # Find gaps
        time_diffs = np.diff(time)
        gap_indices = np.where(time_diffs > gap_threshold)[0]
        
        # Split points (end of each orbit)
        split_points = np.concatenate([[0], gap_indices + 1, [len(time)]])
        
        segments = []
        for i in range(len(split_points) - 1):
            start = split_points[i] 
            end = split_points[i + 1]
            
            segment_time = time[start:end]
            segment_flux = flux[start:end]
            segment_err = flux_err[start:end]
            
            segments.append((segment_time, segment_flux, segment_err))
        
        return segments

    def _print_orbit_stats(self):
        """Print statistics about orbit segments."""
        if not self.global_lightcurves:
            return
            
        segment_lengths = [len(lc['time']) for lc in self.global_lightcurves]
        segment_durations = [lc['time'][-1] - lc['time'][0] for lc in self.global_lightcurves]
        
        print(f"Orbit segment statistics:")
        print(f"- Average length: {np.mean(segment_lengths):.0f} cadences")
        print(f"- Average duration: {np.mean(segment_durations):.2f} days") 
        print(f"- Min duration: {np.min(segment_durations):.2f} days")
        print(f"- Max duration: {np.max(segment_durations):.2f} days")
        
        # Count segments per target
        targets_with_segments = {}
        for lc in self.global_lightcurves:
            tic = lc['tic_id']
            targets_with_segments[tic] = targets_with_segments.get(tic, 0) + 1
        
        avg_segments_per_target = np.mean(list(targets_with_segments.values()))
        print(f"- Average orbits per target: {avg_segments_per_target:.1f}")

    def _find_orbital_segment(self, lightcurve_idx, window_time):
        """
        Find which orbital segment contains the given window time.
        
        Parameters
        ----------
        lightcurve_idx : int
            Index of the source lightcurve
        window_time : float
            Time at center of the window
            
        Returns
        -------
        global_idx : int
            Index into self.global_lightcurves, or -1 if no match
        """
        for global_idx, global_lc in enumerate(self.global_lightcurves):
            # Check if this global segment matches our lightcurve
            if global_lc['tic_id'] != self.ids[lightcurve_idx]:
                continue
                
            # Check if window time falls within this orbital segment
            start_time, end_time = global_lc['time_span']
            if start_time <= window_time <= end_time:
                return global_idx
        
        # No matching orbital segment found
        return -1
