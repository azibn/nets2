import argparse
import glob
import json
import os
from utils import import_lightcurve
from tqdm import tqdm

# argparsing in the works
parser = argparse.ArgumentParser(description="Extract metadata from FITS files and save to a file")

parser.add_argument("directory", help="Path of directory containing FITS files. Can be rooted directory or subdirectory.",default='directory')
parser.add_argument("-o, --output", help="Output file to save metadata",default="metadata.txt",type=str,default='output')

args = parser.parse_args()


def serialise_value(value):
    """Converts a value to a serializable format (strings in this case). Useful when dealing with FITS files."""
    if isinstance(value, (int, float, str, bool, type(None))):
        return value
    else:
        return str(value)

def get_metadata(filepath):
    """Opens the file and returns the metadata. Used for making my EPIC ID target list.

	Inputs:
		- filepath: path to the file

	Returns:
		- metadata: the lightcurve information

	"""

    _, metadata = import_lightcurve(filepath,return_meta_as_dict=True)
    metadata = {**metadata, 'filepath': filepath} # add filepath to metadata
    metadata = {key: serialise_value(value) for key, value in metadata.items()}
    return metadata

def get_files(directory):
    """Get a list of all files ending with '.fits' in the given directory and its subdirectories."""

    files = glob.iglob(os.path.join(directory, '**/*.fits'), recursive=True)
    
    return files

def append_metadata(metadata, output_file):
    """Save metadata to a file."""
    with open(output_file, 'a') as f:
        metadata_json = json.dumps(metadata)
        f.write(metadata_json + "\n") 

if __name__ == '__main__':
    directory_path = args.directory # epic catalogue is based on /storage/astro2/phrdhx/k2/everest/
    output_file = args.output

    file_list = get_files(directory_path)

    with tqdm(desc="Extracting metadata", unit="file") as pbar:
        for file_path in file_list:
            metadata = get_metadata(file_path)
            append_metadata(metadata, output_file)
            pbar.update(1)  

    print("Metadata extracted. Saved to:", output_file)
