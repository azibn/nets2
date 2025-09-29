#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exocomet Candidate Analysis Script
Convert Jupyter notebook to standalone script for exocomet candidate identification and visualization

This script:
1. Loads prediction data from a pickle file
2. Identifies exocomet candidates using defined thresholds
3. Creates and saves plots for each candidate to a folder
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from astropy.table import Table
import sys
import os
import matplotlib.cm as cm
import argparse
sys.path.insert(1, 'scripts/')
sys.path.insert(1, 'stella/')
from mpl_toolkits.axes_grid1 import make_axes_locatable
from stella.mark_exocomets import ExocometFinder

parser = argparse.ArgumentParser(
    description="Plot the candidates."
)

# This is a required positional argument
parser.add_argument(
    "f",
    help="The target file of predictions.",
    type=str
)

# These should be optional arguments with defaults
parser.add_argument('--t1', dest='t1', help='Threshold 1', type=float, default=0.9)
parser.add_argument('--t2', dest='t2', help='Threshold 2', type=float, default=0.5)
parser.add_argument('--max_gap', dest='max_gap',help='Max gap', type=int, default=20)
parser.add_argument('--min_points', dest='min_points',help='Min points', type=int, default=5)

# This is another required positional argument
parser.add_argument(
    '--p',
    dest="plot_path",
    help="The path to save the plots.",
    type=str,
    default = None
)

parser.add_argument(
    '-n',
    '--n-plots', 
    dest="n_plots",
    help="Number of plots to create (top N plots sorted by max probability)", 
    type=int, 
    default=None  
)

args = parser.parse_args()



def load_predictions(file_path):
    """Load the pickle file containing predictions"""
    data = []
    import pickle
    with open(file_path, "rb") as f:
        while True:
            try:
                data.append(pickle.load(f))
            except EOFError:
                break
    return data


def plot_candidate(target_id, target_group, plot_path, info_text, show_plot=False):
    """
    Create a plot for a single exocomet candidate target with all its events.
    Include both normalized and original flux in separate panels.
    
    Parameters:
    -----------
    target_id : int or str
        The ID of the target
    target_group : pandas.DataFrame
        DataFrame containing all events for this target
    plot_path : str
        Directory where to save the plot
    info_text : str
        Text with detection parameters to add to the plot
    show_plot : bool, optional
        Whether to display the plot (default: False)
        
    Returns:
    --------
    None
    """
    # Get the first row to extract the time and flux data
    first_row = target_group.iloc[0]
    time = first_row['time']
    flux = first_row['flux']
    predictions = first_row['predictions']
    original_time = first_row['original_time']
    original_flux = first_row['original_flux']
    max_prob = target_group['max_prob'].max()
    
    # Create figure with two subplots stacked vertically
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # First subplot - Normalized flux
    scatter = ax1.scatter(time, flux, c=predictions, cmap='viridis', 
                         s=10, alpha=0.8, zorder=5)
    
    # Hide x-axis labels on top plot
    ax1.set_xticklabels([])
    ax1.set_xlabel('')
    ax1.set_ylabel('Min-Max Scaled Flux', fontsize=14)
    
    # Second subplot - Original flux
    ax2.scatter(time, original_flux, s=5, alpha=0.8)
    
    # Add labels for second subplot
    ax2.set_xlabel('Time (BJD)')
    ax2.set_ylabel('Original Flux', fontsize=14)
    
    # Calculate the full time range across both datasets
    min_time = min(np.min(time), np.min(original_time))
    max_time = max(np.max(time), np.max(original_time))
    
    # Add a small padding (1% on each side)
    time_range = max_time - min_time
    padding = 0.01 * time_range
    
    # Set identical x-axis limits for both plots
    ax1.set_xlim(min_time - padding, max_time + padding)
    
    # Add the title at the top of the figure
    fig.suptitle(f'TIC ID: {target_id} - {len(target_group)} Events - Max Prob: {round(max_prob,2)}', 
                fontsize=18, y=0.98)
    
    # Mark each event with vertical lines on both subplots
    for i, (_, event) in enumerate(target_group.iterrows()):
        # Add line to first subplot with normalized flux
        ax1.axvline(x=event['tpeak'], color='red', linestyle='--', zorder=2,
                   label=f'Event {event["event_id"]} (t={event["tpeak"]:.2f})')
        
        # Add same line to second subplot with original flux
        ax2.axvline(x=event['tpeak'], color='red', linestyle='--', zorder=2)
        
        # Add event number above the first subplot
        ax1.annotate(f'{event["event_id"]}', 
                    xy=(event['tpeak'], 1.02), xycoords=('data', 'axes fraction'),
                    xytext=(0, 5), textcoords='offset points',
                    ha='center', va='bottom', fontsize=12,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', 
                              alpha=0.9, edgecolor='black'))
    
    # Add the info text box in the figure space
    fig.text(0.02, 0.93, info_text, 
            fontsize=10, ha='left', va='center',
            bbox=dict(facecolor='white', alpha=0.9, boxstyle='round,pad=0.5', 
                     edgecolor='gray'))
    
    # Apply tight layout first to get proper alignment of subplots
    plt.tight_layout(rect=[0, 0, 0.93, 0.95])  # Reserve space for colorbar (90% width)
    
    # Add colorbar AFTER tight_layout but BEFORE saving
    cbar_ax = fig.add_axes([0.94, 0.47, 0.02, 0.38])  # [left, bottom, width, height]
    cbar = fig.colorbar(scatter, cax=cbar_ax)
    cbar.set_label('Prediction Probability')
    
    # Save plot
    plt.savefig(f'{plot_path}/TIC {target_id}.png', dpi=300, bbox_inches='tight')
    
    # Show plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close()


def main():
    # Create output directory for plots
    plot_path = args.plot_path
    os.makedirs(plot_path, exist_ok=True)
    
    # Load data
    print("Loading prediction data...")
    try:
        df = pd.DataFrame(load_predictions(args.f))
        print(f"Loaded {len(df)} predictions")
    except FileNotFoundError:
        # Try alternative path
        try:
            df = pd.DataFrame(load_predictions(args.f))
            print(f"Loaded {len(df)} predictions")
        except FileNotFoundError:
            print("Error: Could not find prediction file. Please check the file path.")
            return
    
    # Create histogram of probabilities
    plt.figure(figsize=(8, 6))
    plt.hist(df.pred.values, bins=100, log=True)
    plt.title('Histogram of probabilities')
    plt.ylabel('Count')
    plt.xlabel('Probability')
    plt.savefig(os.path.join(plot_path, 'probability_histogram.png'), bbox_inches='tight')
    plt.close()
    
    print("Initialising ExocometFinder...")
    finder = ExocometFinder(
        id=np.array(df['ID']),             # Array of target IDs
        time=np.array(df['time']),         # Array of time arrays
        flux=np.array(df['flux']),         # Array of flux arrays
        predictions=np.array(df['predictions'])  # Array of CNN prediction arrays
    )
    
    # Set thresholds based on the notebook values
    high_threshold =args.t1
    low_threshold = args.t2
    max_gap = args.max_gap
    min_points =  args.min_points
    
    # Create info text for the plots
    info_text = (f"Detection parameters:\n"
                f"High threshold: {high_threshold}\n"
                f"Low threshold: {low_threshold}\n"
                f"Max gap: {max_gap}\n"
                f"Min points: {min_points}")
    
    print(f"Identifying exocomet candidates with high_threshold={high_threshold}, "
          f"low_threshold={low_threshold}, max_gap={max_gap}, min_points={min_points}...")
    
    # Identify candidates
    candidates = finder.identify_exocomet_candidates(
        high_threshold=high_threshold,
        low_threshold=low_threshold,
        max_gap=max_gap,
        min_points=min_points
    ).to_pandas()
    
    print(f"Found {len(candidates)} candidate events across {len(set(candidates['Target_ID']))} targets")
    
    # Merge candidates with original data
    merged = pd.merge(df, candidates, left_on='ID', right_on='Target_ID')
    
    # Group by Target_ID to create one plot per target
    print("Creating candidate plots...")
    grouped = merged.groupby('Target_ID')

    target_max_probs = merged.groupby('Target_ID')['max_prob'].max().reset_index()
    target_max_probs = target_max_probs.sort_values('max_prob', ascending=False)
    
    # Limit to top N if n_plots is specified
    
    if args.plot_path is not None:
        if args.n_plots is not None:
            top_targets = target_max_probs.head(args.n_plots)['Target_ID'].values
            print(f"Limiting to top {args.n_plots} targets by maximum probability")
            targets_to_plot = top_targets
        else:
            targets_to_plot = target_max_probs['Target_ID'].values
    
    # Plot only the selected targets
    total_plotted = 0
    for target_id in tqdm(targets_to_plot):
        if target_id in grouped.groups:
            target_group = grouped.get_group(target_id)
            plot_candidate(target_id, target_group, plot_path, info_text)
            total_plotted += 1
    
    # # For each unique Target_ID, create one plot with all its events
    # for target_id, target_group in tqdm(list(grouped)):
    #     plot_candidate(target_id, target_group, plot_path, info_text)
    
    print(f"Analysis complete. {len(set(merged['Target_ID']))} candidate plots saved to {plot_path}/")


if __name__ == "__main__":
    main()