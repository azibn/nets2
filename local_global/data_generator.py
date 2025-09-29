"""
Data generator for local+global exocomet detection.
Works with enhanced FlareDataSet that preserves global lightcurves.
"""

import numpy as np
import tensorflow as tf
from typing import Tuple, List, Optional
import pickle
from scipy.signal import medfilt
from scipy.interpolate import interp1d

class LocalGlobalDataGenerator(tf.keras.utils.Sequence):
    """
    Generates local+global pairs for training the fusion model.
    Works with FlareDataSet that has save_global_context=True.
    """
    
    def __init__(self, 
                 flare_dataset,
                 split='train',
                 local_window: int = 168,
                 global_window_size: int = 500,  # After downsampling
                 global_window_days: float = 3.0,  # Days around event
                 batch_size: int = 32,
                 shuffle: bool = True,
                 augment: bool = True):
        """
        Parameters
        ----------
        flare_dataset : FlareDataSet
            Enhanced FlareDataSet with preserved global lightcurves
        split : str
            'train', 'val', or 'test'
        local_window : int
            Number of cadences for local view (should match dataset cadences)
        global_window_size : int
            Size of global context after downsampling
        global_window_days : float
            Time span in days around each event for global context
        batch_size : int
            Batch size for training
        shuffle : bool
            Whether to shuffle data between epochs
        augment : bool
            Whether to apply data augmentation
        """
        self.flare_ds = flare_dataset
        self.split = split
        self.local_window = local_window
        self.global_window_size = global_window_size
        self.global_window_days = global_window_days
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        
        # Verify dataset has global context enabled
        if not hasattr(flare_dataset, 'save_global_context') or not flare_dataset.save_global_context:
            raise ValueError("FlareDataSet must have save_global_context=True")
        
        # Get data splits
        if split == 'train':
            self.windows = flare_dataset.train_data
            self.labels = flare_dataset.train_labels
            self.window_ids = flare_dataset.train_ids
            self.window_times = getattr(flare_dataset, 'train_tpeaks', 
                                       [flare_dataset.full_peaks[i] for i in range(len(flare_dataset.train_data))])
        elif split == 'val':
            self.windows = flare_dataset.val_data
            self.labels = flare_dataset.val_labels  
            self.window_ids = flare_dataset.val_ids
            self.window_times = flare_dataset.val_tpeaks
        elif split == 'test':
            self.windows = flare_dataset.test_data
            self.labels = flare_dataset.test_labels
            self.window_ids = flare_dataset.test_ids
            self.window_times = flare_dataset.test_tpeaks
        else:
            raise ValueError(f"Unknown split: {split}")
        
        # Create indices for batching
        self.indices = np.arange(len(self.windows))
        self.on_epoch_end()
        
    def __len__(self):
        """Number of batches per epoch."""
        return int(np.ceil(len(self.indices) / self.batch_size))
    
    def __getitem__(self, index):
        """Generate one batch of data."""
        # Get batch indices
        start_idx = index * self.batch_size
        end_idx = min(start_idx + self.batch_size, len(self.indices))
        batch_indices = self.indices[start_idx:end_idx]
        
        # Generate data
        X, y = self._generate_batch(batch_indices)
        return X, y
    
    def on_epoch_end(self):
        """Shuffle indices after each epoch if specified."""
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def _generate_batch(self, batch_indices):
        """Generate batch of local+global pairs."""
        local_batch = []
        global_batch = []
        label_batch = []
        
        for idx in batch_indices:
            # Get local window (already processed by FlareDataSet)
            local_data = self.windows[idx]
            
            # Get global context
            global_data = self._get_global_context(idx)
            
            # Apply augmentation if specified
            if self.augment and self.labels[idx] == 1:
                local_data, global_data = self._augment_data(local_data, global_data)
            
            local_batch.append(local_data)
            global_batch.append(global_data)
            label_batch.append(self.labels[idx])
        
        # Convert to numpy arrays
        local_batch = np.array(local_batch)
        global_batch = np.array(global_batch)
        label_batch = np.array(label_batch)
        
        return [local_batch, global_batch], label_batch
    
    def _get_global_context(self, window_idx):
        """
        Get global context for a specific window using on-demand matching.
        
        Parameters
        ----------
        window_idx : int
            Index into the windows array
            
        Returns
        -------
        global_context : np.ndarray
            Global context array of shape (global_window_size, 1)
        """
        # Get window information
        window_tic = self.window_ids[window_idx]
        window_time = self.window_times[window_idx]
        
        # Find matching orbital segment
        matching_segment = None
        for global_lc in self.flare_ds.global_lightcurves:
            if (global_lc['tic_id'] == window_tic and 
                global_lc['time_span'][0] <= window_time <= global_lc['time_span'][1]):
                matching_segment = global_lc
                break
        
        if matching_segment is None:
            # No global context available - return zeros
            print(f"Warning: No global context found for TIC {window_tic} at time {window_time}")
            return np.zeros((self.global_window_size, 1), dtype=np.float32)
        
        # Extract global context around the window time
        return self._extract_global_window(matching_segment, window_time)
    
    def _extract_global_window(self, global_lc, center_time):
        """
        Extract global context window from orbital segment.
        
        Parameters
        ----------
        global_lc : dict
            Orbital segment data
        center_time : float
            Center time for the global window
            
        Returns
        -------
        global_context : np.ndarray
            Processed global context of shape (global_window_size, 1)
        """
        orbit_time = global_lc['time']
        orbit_flux = global_lc['flux']
        
        # Define global window boundaries
        half_window = self.global_window_days / 2
        start_time = max(orbit_time[0], center_time - half_window)
        end_time = min(orbit_time[-1], center_time + half_window)
        
        # Extract time range
        mask = (orbit_time >= start_time) & (orbit_time <= end_time)
        context_time = orbit_time[mask]
        context_flux = orbit_flux[mask]
        
        if len(context_flux) == 0:
            # Fallback: use entire orbital segment
            context_flux = orbit_flux.copy()
            context_time = orbit_time.copy()
        
        # Downsample to target size
        if len(context_flux) > self.global_window_size:
            # Simple downsampling
            step = len(context_flux) / self.global_window_size
            indices = np.round(np.arange(0, len(context_flux), step)).astype(int)
            indices = indices[indices < len(context_flux)][:self.global_window_size]
            context_flux = context_flux[indices]
        
        # Pad if too short
        elif len(context_flux) < self.global_window_size:
            # Pad with edge values to maintain temporal continuity
            pad_length = self.global_window_size - len(context_flux)
            if len(context_flux) > 0:
                # Symmetric padding
                pad_before = pad_length // 2
                pad_after = pad_length - pad_before
                context_flux = np.pad(context_flux, (pad_before, pad_after), mode='edge')
            else:
                # No data - return zeros
                context_flux = np.zeros(self.global_window_size)
        
        # Additional normalization for global context
        context_flux = self._normalize_global_flux(context_flux)
        
        return context_flux.reshape(-1, 1).astype(np.float32)
    
    def _normalize_global_flux(self, flux):
        """
        Normalize global flux with robust scaling.
        """
        # Remove extreme outliers (beyond 5 sigma)
        median_flux = np.nanmedian(flux)
        mad = np.nanmedian(np.abs(flux - median_flux))
        if mad > 0:
            outlier_mask = np.abs(flux - median_flux) > 5 * mad
            flux[outlier_mask] = median_flux
        
        # Robust min-max scaling
        p5, p95 = np.nanpercentile(flux, [5, 95])
        if p95 > p5:
            flux = (flux - p5) / (p95 - p5)
            flux = np.clip(flux, 0, 1)  # Ensure [0, 1] range
        else:
            flux = flux * 0  # All values are the same
        
        return flux
    
    def _augment_data(self, local_data, global_data):
        """
        Apply data augmentation to local and global data.
        
        Parameters
        ----------
        local_data : np.ndarray
            Local window data
        global_data : np.ndarray  
            Global context data
            
        Returns
        -------
        augmented_local, augmented_global : tuple of np.ndarray
            Augmented data
        """
        local_aug = local_data.copy()
        global_aug = global_data.copy()
        
        # Flip exocomet transit (reverse time) with 50% probability
        if np.random.random() > 0.5:
            local_aug = local_aug[::-1]
            # Don't flip global - preserve stellar context chronology
        
        # Add small amount of noise with 50% probability  
        if np.random.random() > 0.5:
            noise_level = 0.001
            local_aug += np.random.normal(0, noise_level, local_aug.shape).astype(np.float32)
            global_aug += np.random.normal(0, noise_level * 0.5, global_aug.shape).astype(np.float32)
        
        # Small time shifts for local data with 30% probability
        if np.random.random() > 0.7:
            shift = np.random.randint(-2, 3)  # Shift by up to 2 cadences
            if shift != 0:
                local_aug = np.roll(local_aug, shift, axis=0)
        
        return local_aug, global_aug


def create_local_global_generators(dataset_path: str,
                                  local_window: int = 168,
                                  global_window_size: int = 500,
                                  global_window_days: float = 3.0,
                                  batch_size: int = 32):
    """
    Create train/val/test generators from saved FlareDataSet.
    
    Parameters
    ----------
    dataset_path : str
        Path to pickle file with FlareDataSet that has global context
    local_window : int
        Size of local window (should match dataset cadences)
    global_window_size : int  
        Size of global context after processing
    global_window_days : float
        Days around event for global context
    batch_size : int
        Batch size for training
        
    Returns
    -------
    train_gen, val_gen, test_gen : LocalGlobalDataGenerator
        Data generators for each split
    """
    print(f"Loading dataset from {dataset_path}...")
    
    with open(dataset_path, 'rb') as f:
        ds_info = pickle.load(f)
        dataset = ds_info['dataset']
    
    # Verify global context is available
    if not hasattr(dataset, 'save_global_context') or not dataset.save_global_context:
        raise ValueError(
            f"Dataset at {dataset_path} does not have global context enabled. "
            "Please create dataset with save_global_context=True"
        )
    
    print(f"Found {len(dataset.global_lightcurves)} orbital segments for global context")
    
    # Create generators for each split
    train_gen = LocalGlobalDataGenerator(
        dataset,
        split='train',
        local_window=local_window,
        global_window_size=global_window_size,
        global_window_days=global_window_days,
        batch_size=batch_size,
        shuffle=True,
        augment=True
    )
    
    val_gen = LocalGlobalDataGenerator(
        dataset,
        split='val', 
        local_window=local_window,
        global_window_size=global_window_size,
        global_window_days=global_window_days,
        batch_size=batch_size,
        shuffle=False,
        augment=False
    )
    
    test_gen = LocalGlobalDataGenerator(
        dataset,
        split='test',
        local_window=local_window,
        global_window_size=global_window_size, 
        global_window_days=global_window_days,
        batch_size=batch_size,
        shuffle=False,
        augment=False
    )
    
    print(f"Created generators:")
    print(f"- Training: {len(train_gen)} batches ({len(train_gen.windows)} samples)")
    print(f"- Validation: {len(val_gen)} batches ({len(val_gen.windows)} samples)")
    print(f"- Test: {len(test_gen)} batches ({len(test_gen.windows)} samples)")
    
    return train_gen, val_gen, test_gen


# Example usage
if __name__ == "__main__":
    # Example of how to use the data generator
    
    # First, create a dataset with global context enabled
    # This would be done in a separate script
    """
    import sys
    sys.path.insert(1, '../stella')
    import stella
    
    dataset = stella.FlareDataSet(
        fn_dir="path/to/lightcurves",
        catalog="path/to/catalog.txt",
        cadences=168,
        training=0.8,
        validation=0.1,
        frac_balance=0.73,
        save_global_context=True,      # Enable global context
        global_window_size=2000,
        global_window_days=3.0,
        orbit_gap_threshold=0.5
    )
    
    # Save enhanced dataset
    with open('dataset_with_global.pkl', 'wb') as f:
        pickle.dump({'dataset': dataset}, f)
    """
    
    # Then load and create generators
    try:
        train_gen, val_gen, test_gen = create_local_global_generators(
            dataset_path="dataset_with_global.pkl",
            local_window=168,
            global_window_size=500,
            global_window_days=3.0,
            batch_size=32
        )
        
        # Test generator
        print("\nTesting generator...")
        X, y = train_gen[0]  # Get first batch
        local_batch, global_batch = X
        
        print(f"Local batch shape: {local_batch.shape}")
        print(f"Global batch shape: {global_batch.shape}")
        print(f"Label batch shape: {y.shape}")
        print(f"Labels in batch: {np.unique(y, return_counts=True)}")
        
    except FileNotFoundError:
        print("Example dataset not found. Please create dataset with global context first.")
    except Exception as e:
        print(f"Error: {e}")