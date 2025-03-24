import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from astropy.table import Table

__all__ = ["ExocometFinder"]


class ExocometFinder(object):
    """
    Uses the predictions from the neural network
    and identifies exocomet transit events based on consecutive
    points. Users define both high and low probability thresholds
    for accepting an exocomet event as real.
    """

    def __init__(self, id, time, flux, flux_err=None, predictions=None):
        """
        Uses the times, fluxes, and predictions defined
        in stella.ConvNN to identify and fit exocomet transits.
        
        Parameters
        ----------
        id : np.array
             Array of target IDs
        time : np.array
             Array of times to find transits on.
        flux : np.array
             Array of light curves.
        flux_err : np.array, optional
             Array of errors on light curves. If None, dummy errors will be created.
        predictions : np.array
             Array of predictions for each light curve
             passed in.
        
        Attributes
        ----------
        IDs : np.array
        time : np.ndarray
        flux : np.ndarray
        flux_err : np.ndarray
        predictions : np.ndarray
        """
        self.IDs = id
        self.time = time
        self.flux = flux
        self.predictions = predictions
        
        # Handle missing flux errors by creating dummy errors
        # if flux_err is None:
        #     self.create_dummy_errors()
        # else:
        #     self.flux_err = flux_err

    def create_dummy_errors(self, error_fraction=0.01):
        """
        Creates dummy flux errors if none were provided.
        
        Parameters
        ----------
        error_fraction : float, optional
            Fraction of flux to use as error. Default is 0.01 (1%).
        """
        print("Creating dummy flux errors (1% of flux values)")
        
        # Check if flux is a list of arrays or a single array
        if isinstance(self.flux, list) or (isinstance(self.flux, np.ndarray) and self.flux.ndim > 1):
            self.flux_err = []
            for f in self.flux:
                self.flux_err.append(error_fraction * np.abs(f))
        else:
            # Single flux array
            self.flux_err = error_fraction * np.abs(self.flux)


    def group_inds(self, values, max_gap=20):
        """
        Groups regions marked as potential exocomets.
        Indices within max_gap of each other are grouped
        as one exocomet transit.

        Parameters
        ----------
        values : np.array
            Array of indices to group
        max_gap : int, optional
            Maximum gap between indices to be considered part of the same event.
            Default is 20, which is suitable for exocomet transits.

        Returns
        -------
        results: np.ndarray
             An array of arrays, which are groups of indices
             supposedly attributed with a single exocomet transit.
        """
        results = []

        for i, v in enumerate(values):
            if i == 0:
                mini = maxi = v
                temp = [v]
            else:
                if (np.abs(v-maxi) <= max_gap):
                    temp.append(v)
                    if v > maxi:
                        maxi = v
                    if v < mini:
                        mini = v
                else:
                    results.append(temp)
                    mini = maxi = v
                    temp = [v]
                
                # GETS THE LAST GROUP
                if i == len(values)-1:
                    results.append(temp)

        return results


    def group_inds_dual_threshold(self, prob, high_threshold=0.9, low_threshold=0.5, max_gap=10):
        """
        Groups regions using a dual-threshold approach to capture full transit events.
        
        Parameters
        ----------
        prob : array
            Prediction probabilities from the CNN
        high_threshold : float
            Primary threshold for high-confidence detections
        low_threshold : float
            Secondary threshold for expanding detections
        max_gap : int
            Maximum gap between indices to be considered part of the same event
        
        Returns
        -------
        grouped_indices : list of arrays
            Groups of indices corresponding to detected events
        """
        # First identify high-confidence points
        high_conf_indices = np.where(prob >= high_threshold)[0]
        
        # Group high-confidence points using original method
        initial_groups = []
        if len(high_conf_indices) > 0:
            for i, v in enumerate(high_conf_indices):
                if i == 0:
                    mini = maxi = v
                    temp = [v]
                else:
                    if (np.abs(v-maxi) <= max_gap):
                        temp.append(v)
                        if v > maxi:
                            maxi = v
                        if v < mini:
                            mini = v
                    else:
                        initial_groups.append(temp)
                        mini = maxi = v
                        temp = [v]
                    
                    if i == len(high_conf_indices)-1:
                        initial_groups.append(temp)
        
        # If no high-confidence detections found, check if there are any low-confidence ones
        if len(initial_groups) == 0:
            low_conf_indices = np.where(prob >= low_threshold)[0]
            if len(low_conf_indices) > 0:
                # Use the standard grouping method for these
                return self.group_inds(low_conf_indices, max_gap)
            else:
                return []
        
        # For each high-confidence group, expand with nearby low-confidence points
        expanded_groups = []
        for group in initial_groups:
            # Define search range for this group (with padding)
            min_idx = max(0, min(group) - max_gap)
            max_idx = min(len(prob), max(group) + max_gap)
            
            # Find low-confidence points in this range
            low_conf_in_range = np.where((prob >= low_threshold) & 
                                        (prob < high_threshold) & 
                                        (np.arange(len(prob)) >= min_idx) & 
                                        (np.arange(len(prob)) <= max_idx))[0]
            
            # Combine high and low confidence points and sort
            expanded_group = np.sort(np.concatenate([group, low_conf_in_range]))
            expanded_groups.append(expanded_group)
        
        # Check if any expanded groups should be merged
        merged_groups = []
        current_group = expanded_groups[0]
        
        for i in range(1, len(expanded_groups)):
            # If this group is close to the current group, merge them
            if min(expanded_groups[i]) - max(current_group) <= max_gap:
                # Merge groups
                current_group = np.sort(np.concatenate([current_group, expanded_groups[i]]))
            else:
                # Complete current group and start a new one
                merged_groups.append(current_group)
                current_group = expanded_groups[i]
        
        # Add the last group
        merged_groups.append(current_group)
        
        return merged_groups


    def get_init_guesses(self, groupings, time, flux, prob, region=40):
        """
        Guesses at the initial transit parameters based on 
        probability groups.

        Parameters
        ----------
        groupings : np.ndarray
             Group of indices for a single transit event.
        time : np.array
            Time array
        flux : np.array
            Flux array
        err : np.array
            Error array
        prob : np.array
            Probability array
        region : int, optional
            Number of points to add on each side for analysis. Default is 40.

        Returns
        -------
        tcenters : np.ndarray
             Array of transit centers for each group.
        depths : np.ndarray
             Array of depths at each center.
        durations : np.ndarray
             Array of estimated durations for each group.
        """
        tcenters = np.array([])
        depths = np.array([])
        durations = np.array([])

        if len(groupings) > 0:
            for g in groupings:
                # Define a region around the group for analysis
                if g[0]-region < 0:
                    subreg = np.arange(0, g[-1]+region, 1, dtype=int)
                elif g[-1]+region > len(time):
                    subreg = np.arange(len(time)-region, len(time), 1, dtype=int)
                else:
                    subreg = np.arange(g[0]-region, g[-1]+region, 1, dtype=int)
                
                # Extract region data
                subt = time[subreg]
                subf = flux[subreg]
                #sube = err[subreg]
                subp = prob[subreg]
                
                # Find the deepest point in the transit
                min_idx = np.argmin(subf)
                t_center = subt[min_idx]
                depth = 1.0 - subf[min_idx]
                
                # Estimate duration from group span
                duration = (subt[-1] - subt[0]) * 0.2  # Simple estimate
                
                tcenters = np.append(tcenters, t_center)
                depths = np.append(depths, depth)
                durations = np.append(durations, duration)

        return tcenters, depths, durations


    def identify_exocomet_candidates(self, high_threshold=0.9, low_threshold=0.5, 
                               max_gap=10, min_points=5):
        """
        Finds regions with high prediction probabilities for manual inspection.
        
        Parameters
        ----------
        high_threshold : float, optional
            The high probability threshold for confident detection.
            Default is 0.9.
        low_threshold : float, optional
            The low probability threshold for expanding detections.
            Default is 0.5.
        max_gap : int, optional
            Maximum gap between indices to consider as same event.
            Default is 20.
        min_points : int, optional
            Minimum number of points required for a valid detection.
            Default is 5.

        Returns
        -------
        candidate_table : astropy.table.Table
            A table of candidate times and probabilities for manual inspection.
        """
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold

        table = Table(names=['Target_ID', 'tpeak', 'high_conf_points', 
                            'total_points', 'max_prob', 'event_id'])
        
        for i in range(len(self.IDs)):
            time = self.time[i]
            prob = self.predictions[i]
            
            # Use dual-threshold grouping
            groupings = self.group_inds_dual_threshold(
                prob, high_threshold, low_threshold, max_gap)
            
            # Filter out groups that are too small
            valid_groupings = [g for g in groupings if len(g) >= min_points]
            
            event_id = 1
            
            for g in valid_groupings:
                peak_idx = g[np.argmax(prob[g])]
                tp = time[peak_idx]
                
                # Count high-confidence points in this group
                high_conf_count = np.sum(prob[g] >= high_threshold)
                max_prob = np.max(prob[g])

                if max_prob > 0.9 and high_conf_count == 0:
                    print("\nDEBUG - Unusual case detected:")
                    high_conf_count = np.sum(prob[g] >= high_threshold)
                    print(f"Group indices: {g}")
                    print(f"Probabilities: {prob[g]}")
                    print(f"High threshold: {high_threshold}")
                    print(f"High conf points: {high_conf_count}")
                    print(f"Max prob: {np.max(prob[g])}")

                
                table.add_row([
                    self.IDs[i], tp, high_conf_count, 
                    len(g), max_prob, event_id
                ])
                
                event_id += 1

        if len(table) > 0:  
            table['Target_ID'] = table['Target_ID'].astype(np.int64)
            table['high_conf_points'] = table['high_conf_points'].astype(np.int32)
            table['total_points'] = table['total_points'].astype(np.int32)
            table['event_id'] = table['event_id'].astype(np.int32)

       
        self.candidate_table = table
        self.transit_table = table  
        return table
    
    
    def plot_exocomet_candidate(self, index, padding_factor=2):
        """
        Plots an identified exocomet candidate for manual inspection.
        
        Parameters
        ----------
        index : int
            Index of the candidate in the candidate_table
        padding_factor : float, optional
            Factor to multiply by a fixed window size for display. Default is 2.
                
        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object containing the plot
        """
        if not hasattr(self, 'candidate_table') or len(self.candidate_table) <= index:
            print(f"No candidate found at index {index}")
            return None
        
        # Get candidate info
        candidate = self.candidate_table[index]
        target_id = candidate['Target_ID']
        t0 = candidate['tpeak']
        
        # Find target in data arrays
        target_idx = np.where(self.IDs == target_id)[0][0]
        
        time = self.time[target_idx]
        flux = self.flux[target_idx]
        prob = self.predictions[target_idx]
        
        # Define window around candidate (fixed 0.5 day window by default)
        window_size = 0.5 * padding_factor
        #mask = (time >= t0 - window_size) & (time <= t0 + window_size)
        
        # Create plot
        fig, ax1 = plt.subplots(2, 1, figsize=(7,3), 
                                    gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot flux
        ax1.plot(time, flux, 'k.', alpha=0.7, label='Data')
        ax1.axvline(t0, color='blue', ls='--', alpha=0.5, label='Peak Probability')
        ax1.set_title(f"Exocomet Candidate - Target ID: {target_id}")
        ax1.set_ylabel("Normalised Flux")
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        # Display candidate parameters
        param_text = (f"High-conf points: {candidate['high_conf_points']}\n"
                    f"Total points: {candidate['total_points']}\n"
                    f"Max probability: {candidate['max_prob']:.3f}\n"
                    f"Event ID: {candidate['event_id']}")
        
        ax1.annotate(param_text, xy=(0.02, 0.02), xycoords='axes fraction',
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
        
        plt.tight_layout()
        return fig
    
    def plot_candidates(self, figsize=(7,3), target_id=None):
        """
        Plot the light curve with vertical lines at identified exocomet candidates.
        
        Parameters
        ----------
        figsize : tuple, optional
            Figure size. Default is (7,3).
        target_id : int, optional
            Target ID to plot. If None, uses the first target.
            
        Returns
        -------
        fig : matplotlib figure
            Figure containing the plot
        """
        if not hasattr(self, 'candidate_table') or len(self.candidate_table) == 0:
            print("No candidates identified. Run identify_exocomet_candidates first.")
            return None
        
        # If no target_id specified, use the first one in the candidate table
        if target_id is None:
            target_id = self.candidate_table['Target_ID'][0]
        
        # Find index of this target in the data arrays
        idx = np.where(self.IDs == target_id)[0]
        if len(idx) == 0:
            print(f"Target ID {target_id} not found in data.")
            return None
        idx = idx[0]
        
        # Create plot
        fig = plt.figure(figsize=figsize)
        plt.scatter(self.time[idx], self.flux[idx], c=self.predictions[idx], s=5)
        
        # Draw vertical lines at each candidate peak time
        for tpeak in self.candidate_table[self.candidate_table['Target_ID'] == target_id]['tpeak']:
            plt.vlines(tpeak, np.min(self.flux[idx]), np.max(self.flux[idx]), color='k', alpha=0.5, linewidth=1, zorder=0)
        
        plt.xlabel('Time [BJD]')
        plt.ylabel('Normalised Flux')
        plt.tight_layout()
        return fig