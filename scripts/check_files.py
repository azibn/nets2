import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

def process_file(file_path):
    try:
        # Load only the second array from the file
        data = np.load(file_path, allow_pickle=True)
        time = data[0]
        flux = data[1]
        
        # Check if the minimum of the array is less than 0
        if np.min(data) < 0:
            # Create the plot
            plt.figure(figsize=(10, 6))
            plt.plot(time,flux)
            plt.title(f"Plot for {os.path.basename(file_path)}")
            plt.xlabel("Time")
            plt.ylabel("Flux")
            plt.savefig(f"{os.path.splitext(file_path)[0]}_plot.png")
            plt.close()
            
            return f"Plotted {os.path.basename(file_path)} - minimum value: {np.min(data)}"

    except Exception as e:
        return f"Error processing {os.path.basename(file_path)}: {str(e)}"

def process_numpy_files(directory):
    # Get all .npy files in the directory
    npy_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.npy')]
    
    # Use ProcessPoolExecutor for parallel processing
    with ProcessPoolExecutor() as executor:
        # Submit all tasks
        future_to_file = {executor.submit(process_file, file): file for file in npy_files}
        
        # Process results as they complete with a progress bar
        for future in tqdm(as_completed(future_to_file), total=len(npy_files), desc="Processing files"):
            file = future_to_file[future]
            try:
                result = future.result()
                print(result)
            except Exception as e:
                print(f"Error processing {os.path.basename(file)}: {str(e)}")

# Usage
if __name__ == "__main__":
    directory_path = 'models/eleanor-lite'
    process_numpy_files(directory_path)