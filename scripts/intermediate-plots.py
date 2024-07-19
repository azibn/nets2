"""This script takes the folders created from `make_grid.py` and plots the injection recovery plot. It is a 2D "histogram" that shows the recovery rate of 
    exocomets that have a probability > 0.5 per skew/duration bin.
    
    
    This is not exactly my finest work of a script, but it does the job.
    """

import os
import re
import sys
import cProfile
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from itertools import product
from multiprocessing import Pool, cpu_count
from functools import partial
import stella


def get_max_around_time(file, target_time=1496.5, window_size=4):
    time = file[0]
    predictions = file[-1]

    closest_time_index = np.argmin(np.abs(time - target_time))
    
    left_bound = max(0, closest_time_index - window_size)
    right_bound = min(len(predictions), closest_time_index + window_size + 1)
    
    # Get the maximum within the window

    window_max = np.max(predictions[left_bound:right_bound])
    
    # # Get the corresponding time
    window_max_index = np.argmax(predictions[left_bound:right_bound]) + left_bound
    window_max_time = time[window_max_index]
    
    return window_max 


def extract_values_from_dirs(base_dir):
    folders = [
        f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))
    ]

    pattern = re.compile(
        r"injected-skew_(?P<skew>-?\d+(\.\d+)?)-duration-(?P<duration>\d+\.\d+)-snr-5"
    )

    skew_values = sorted(
        set(int(pattern.search(f).group("skew")) for f in folders if pattern.search(f))
    )
    duration_values = sorted(
        set(
            float(pattern.search(f).group("duration"))
            for f in folders
            if pattern.search(f)
        )
    )

    return skew_values, duration_values

def gridplot(array, skew_vals, duration_vals,outputfile='recovery-plot.png'):
    
    fig, ax = plt.subplots(figsize=(20, 10))
    # Create mesh for pcolormesh
    X, Y = np.meshgrid(skew_vals, duration_values)

    # Create the color plot
    pcm = ax.pcolormesh(X, Y, recovery, shading='auto', vmin=0, vmax=1, cmap='viridis')

    # Add text annotations for each cell
    for i in range(recovery.shape[0]):
        for j in range(recovery.shape[1]):
            text_color = 'white' if recovery[i, j] < 0.5 else 'black'
            ax.text(skew_vals[j], duration_values[i], f'{recovery[i, j]:.2f}', 
                    ha='center', va='center', color=text_color, fontsize=6)

    # Set ticks and labels
    ax.set_xticks(skew_vals)
    ax.set_xticklabels(skew_vals)

    # Add colorbar
    cbar = plt.colorbar(pcm)
    cbar.set_label('Recovery Rate')


    # Set labels and title
    ax.set_xlabel('Skewness')
    ax.set_ylabel('Duration')
    ax.set_title('Recovery Rate by Skewness and Duration')

    # Rotate and align the tick labels
    plt.setp(ax.get_xticklabels(), rotation=90, ha='center')
    plt.setp(ax.get_yticklabels(), ha='right')

    # Use a tight layout, but adjust for the rotated x-axis labels
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    ax.set_yscale('log')

    ax.set_yticks(duration_values)
    ax.set_yticklabels(duration_values)
    plt.savefig(f'{}',dpi=300,bbox_inches='tight')
    plt.show()
    plt.close()

def main():
    
    base_dir = 'exo9/'
    skew_vals, duration_vals = extract_values_from_dirs(base_dir)

    recovery = []
    for duration in tqdm(duration_values):
        duration_recovery = []
        for j in skew_vals:
            pred = []
            folder_pattern = f"injected-skew_{j}-duration-{duration:.2f}-snr-5"
            files = glob(f'../exo9/{folder_pattern}/*.npy')
            np.random.shuffle(files)
            
            for file_path in files:
                file = np.load(file_path, allow_pickle=True)
                val_pred = get_max_around_time(file)
                accpred = 1 if val_pred > 0.5 else 0
                pred.append(accpred)

            rec = sum(pred)/len(pred)
            duration_recovery.append(rec)
        
        recovery.append(duration_recovery)
    
    return np.array(recovery)

    



if __name__ == "__main__":
    cProfile.run("main()", "output.prof")
    sys.exit()
