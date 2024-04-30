import argparse
import os
import build_synthetic_set as bss
import numpy as np
import glob
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Create the synthetic set.')

parser.add_argument(help="target directory(s)", default=".", nargs="+", dest="path")
parser.add_argument("-n", "--number", help="number of lightcurve files to inject exocomet models in", default=100, type=int, dest="number")
parser.add_argument('-o','--output', help="output directory", default=".", dest="output")


args = parser.parse_args()


paths = []
for path in args.path:
    paths.append(os.path.expanduser(path))


if __name__ == '__main__':
    files = glob.glob(os.path.join(path, "**/*lc.fits"),recursive=True)
    sample = np.random.choice(files, args.number, replace=False)

    for file in files:
        lc, lc_info = bss.import_lightcurve(file)
        lc_inj = bss.injected_lightcurve(lc)
        bss.save_lightcurve(lc_inj, format='npz')

