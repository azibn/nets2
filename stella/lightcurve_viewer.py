import matplotlib.pyplot as plt
import os
import argparse
import numpy as np
import sys
from pathlib import Path
sys.path.insert(1,'scripts')
sys.path.insert(1,'stella')
#import scripts
import stella


class LightcurveViewer:
    def __init__(self, directory, pattern="*.npy"):
        self.directory = Path(directory)
        self.files = list(self.directory.glob(pattern))
        self.current_idx = 0
        self.fig, self.ax = plt.subplots()
        print(f"Found {len(self.files)} files")
        
    def show_lightcurve(self):
        self.ax.clear()
        current_file = self.files[self.current_idx]
        print(f"Showing {current_file.name} ({self.current_idx + 1}/{len(self.files)})")
        
        lc = np.load(current_file)  # your import function
        self.ax.plot(lc[0], lc[1], '.')
        self.ax.set_title(f'File: {current_file.name}')
        plt.draw()
    
    def on_keyboard(self, event):
        if event.key == 'right':
            self.current_idx = min(self.current_idx + 1, len(self.files) - 1)
            self.show_lightcurve()
        elif event.key == 'left':
            self.current_idx = max(self.current_idx - 1, 0)
            self.show_lightcurve()
        elif event.key == 'q':
            plt.close()

def main():
    parser = argparse.ArgumentParser(description='Quick lightcurve viewer')
    parser.add_argument('directory', type=str, help='Directory containing lightcurve files')
    parser.add_argument('--pattern', type=str, default='*.npy', 
                        help='File pattern to match (default: *.npy)')
    args = parser.parse_args()

    viewer = LightcurveViewer(args.directory, args.pattern)
    viewer.show_lightcurve() 
    viewer.fig.canvas.mpl_connect('key_press_event', viewer.on_keyboard)
    plt.show()

if __name__ == "__main__":
    main()
