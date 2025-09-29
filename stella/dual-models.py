"""
Creates both exocomet and exoplanet models, injecting them into the same real lightcurve.
"""

import os
import sys
import random
from glob import glob
from tqdm import tqdm
import numpy as np
import pandas as pd
from astropy.table import Table
import batman
import astropy.constants as const
import argparse
from astroquery.mast import Catalogs

# Import your existing modules
sys.path.insert(1, 'scripts')
sys.path.insert(1, 'stella')
import modelmaker
from utils import *
import models

def normalize_dual_transit(flux):
    """
    Normalize a lightcurve with dual transits to preserve both depths.
    """
    # Find the baseline level (robust against outliers like transits)
    baseline = np.median(flux)
    
    # Calculate relative flux
    rel_flux = flux / baseline
    
    # Scale to [0,1] range
    norm_flux = (rel_flux - np.min(rel_flux)) / (np.max(rel_flux) - np.min(rel_flux))
    
    return norm_flux

def find_separate_injection_times(lc, window_size, min_separation_days=4.0, max_attempts=10000):
    """
    Find two separate valid times to inject transits, ensuring they're separated.
    """
    # First find a valid time for the first transit
    first_transit = modelmaker.find_valid_injection_time(lc, window_size)
    if first_transit is None:
        return None
    
    t0_first = first_transit["t0"]
    
    # Then find a valid time for the second transit that's sufficiently separated
    for _ in range(max_attempts):
        second_transit = modelmaker.find_valid_injection_time(lc, window_size)
        if second_transit is None:
            continue
        
        t0_second = second_transit["t0"]
        
        # Check if the separation is sufficient
        if abs(t0_first - t0_second) >= min_separation_days:
            return (t0_first, t0_second)
    
    # Could not find valid separated times
    return None

def create_dual_transit(
    file,
    folder,
    r_star,
    m_star,
    min_snr_comet=7,
    max_snr_comet=12,
    min_snr_planet=3,
    max_snr_planet=20,
    window_size=84,
    max_retries=100,
    comet_method=None,
    save_model=True,
    mission='TESS'
):
    """
    Creates both exocomet and exoplanet transits and injects them into the same lightcurve.
    """
    # Load and prepare the lightcurve using modelmaker functions
    lc = modelmaker.prepare_lightcurve(file, mission=mission)
    
    # Skip invalid lightcurves
    if np.isnan(lc["rms"]):
        return None
    
    # Generate SNR values for both transits
    comet_snr = modelmaker.SNR(lc["rms"], min_snr_comet, max_snr_comet)
    planet_snr = modelmaker.SNR(lc["rms"], min_snr_planet, max_snr_planet)
    
    valid_model_found = False
    retry_count = 0
    
    while not valid_model_found and retry_count < max_retries:
        # Find two valid and sufficiently separated injection times
        injection_times = find_separate_injection_times(lc, window_size)
        
        if injection_times is None:
            retry_count += 1
            continue
        
        t0_comet, t0_planet = injection_times
        
        # Create exocomet model (using your existing models)
        if comet_method == "comet_curve" or comet_method is None:
            sigma = np.round(np.random.uniform(0.25, 0.75), 3)
            tail = np.round(np.random.uniform(0.35, 0.7), 3)
            shape = np.round(np.random.uniform(1, 3.5), 3)
            comet_model = 1 - models.comet_curve2(
                lc["time"], 
                comet_snr["amplitude"], 
                t0_comet, 
                sigma=sigma, 
                tail=tail, 
                shape=shape
            )
        elif comet_method == "skewed_gaussian":
            skew = 3
            duration = 0.2
            comet_model = models.skewed_gaussian(
                lc["time"], 
                alpha=skew, 
                t0=t0_comet, 
                sigma=duration, 
                depth=comet_snr["amplitude"]
            )
        
  
        
        # Create planet transit model using batman
        params = batman.TransitParams()
        params.t0 = t0_planet
        params.u = [np.random.uniform(0.2, 0.8)]  # limb darkening
        #period = np.random.uniform(5, 20)  # random period
        period_min = 5
        period_max = 700
        alpha = 0.9
        random_value = np.random.uniform(0, 1)
        params.per = params.per = period_min * (period_max / period_min) ** (random_value ** (1 / alpha))
        depth = planet_snr["snr"] * lc["rms"]  # SNR = signal/noise
        params.rp = np.sqrt(depth)
        params.a = ((params.per * 86400.) ** 2 * const.G.value * m_star * const.M_sun.value / 
                    (4 * np.pi**2)) ** (1/3) / (r_star * const.R_sun.value)
        params.inc = 90  # orbital inclination (edge-on)
        params.ecc = 0  # eccentricity
        params.w = 90  # longitude of periastron
        params.limb_dark = "linear"  # limb darkening model
        
        m = batman.TransitModel(params, lc['time'], fac=0.02)
        planet_model = m.light_curve(params)
        
        # Combine both models (multiplicative because each represents fractional flux)
        combined_model = comet_model * planet_model
        
        # Apply to the lightcurve
        f = combined_model * (lc["flux"] / np.nanmedian(lc["flux"]))
        
        # Check if model is valid (no negative flux values)
        if np.all(f > 0):
            valid_model_found = True
        else:
            retry_count += 1
    
    if not valid_model_found:
        return None
    
    # Normalize the lightcurve using specialized normalization
    f_scaled = normalize_dual_transit(f)
    
    # Calculate flux errors
    fluxerror = lc["flux_error"] / lc["flux"]
    
    # Get metadata for saving (using the same approach as modelmaker)
    if mission == 'TESS':
        target_id = lc['lc_info']['TIC_ID']
        segment = f"sector{lc['lc_info']['sector']:02d}"
    elif mission == 'Kepler':
        target_id = lc['lc_info']['KEPLERID']
        segment = f"q{lc['lc_info']['quarter']:02d}"
    elif mission == 'K2':
        target_id = lc['lc_info']['EPIC_ID']
        segment = f"c{lc['lc_info']['campaign']:02d}"
    
    # Save the data

    real_mask = lc["real"] == 1

    if save_model:
        np.save(
        f"{folder}/{target_id}_{segment}_dual.npy",
        np.array([
            lc["time"][real_mask],
            f_scaled[real_mask], 
            fluxerror[real_mask],
            lc["real"][real_mask],
            comet_model[real_mask],
            planet_model[real_mask],
            f[real_mask],
        ]),
    )
    else:
        np.save(
            f"{folder}/{target_id}_{segment}_dual.npy",
            np.array([
                lc["time"],
                f_scaled,
                fluxerror,
                lc["real"],
                f,
            ]),
        )
    
    # Return metadata
    return [{
        "tic": target_id, 
        "comet_time": t0_comet, 
        "planet_time": t0_planet,
        "comet_snr": comet_snr['snr'], 
        "planet_snr": planet_snr['snr'],
        "rms": lc['rms']
    }]

def main(args):
    """
    Main function to run the dual transit creation process.
    """
    # Find lightcurve files
    files = glob(f"{args.dir}/**/*.fits", recursive=True)
    random.shuffle(files)
    
    # Create output directory
    os.makedirs(args.folder, exist_ok=True)
    
    # Process lightcurves
    successful_models = 0
    file_index = 0
    results = []

        # Get stellar parameters - borrowing from modelmaker.exoplanet function
    TIC_table = Catalogs.query_object(f'TIC 270577175', catalog="TIC")
    r_star = TIC_table['rad'][0]
    m_star = TIC_table['mass'][0]
    
    with tqdm(total=args.number, desc="Creating dual transit models") as pbar:
        while successful_models < args.number:
            if file_index >= len(files):
                random.shuffle(files)
                file_index = 0
            
            if len(files) == 0:
                print("Error: No files found.")
                break
            
            target_ID = files[file_index]
            
            try:
                result = create_dual_transit(
                    target_ID, 
                    args.folder,
                    r_star=r_star,
                    m_star=m_star,
                    comet_method=args.model,
                    mission=args.mission
                )
                
                if result is not None:
                    results.extend(result)
                    successful_models += 1
                    pbar.update(1)
            except Exception as e:
                print(f"Failed for {target_ID}: {e}")
            
            file_index += 1
    
    # Create catalog
    if results:
        # Process results
        tic = []
        comet_times = []
        planet_times = []
        comet_snr = []
        planet_snr = []
        rms_cat = []
        
        for result in results:
            tic.append(result["tic"])
            comet_times.append(result["comet_time"])
            planet_times.append(result["planet_time"])
            comet_snr.append(result["comet_snr"])
            planet_snr.append(result["planet_snr"])
            rms_cat.append(result["rms"])
        
        # Create DataFrame
        data = pd.DataFrame({
            "TIC": tic,
            "tpeak": comet_times,
            "planet_tpeak": planet_times,
            "comet_SNR": comet_snr,
            "planet_SNR": planet_snr,
            "RMS": rms_cat
        })
        
        data.TIC = data.TIC.astype(int)
        t = Table.from_pandas(data)
        t.write(f"{args.catalog}", format="ascii", overwrite=True)
        
        print(f"Successfully created {len(results)} dual transit models.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create dual transit models (exocomet + exoplanet)."
    )
    
    parser.add_argument(
        help="The target directory of lightcurves to use for model injection.",
        dest="dir",
    )
    parser.add_argument("-f", "--folder", help="Target output folder.", dest="folder")
    parser.add_argument(
        "-c",
        "--catalog-name",
        help="Target catalog file. Saved in a .txt format",
        dest="catalog",
    )
    parser.add_argument("-n", "--number", default=5000, dest="number", type=int)
    parser.add_argument(
        "-m",
        "--model",
        help="Select the model used to create exocomets. Options: 'comet_curve', 'skewed_gaussian'. Default 'comet_curve'.",
        dest="model",
    )
    parser.add_argument(
        "--mission",
        help="Specify mission (TESS, Kepler, K2). Default 'TESS'.",
        choices=['TESS', 'Kepler', 'K2'],
        default='TESS'
    )
    
    args = parser.parse_args()
    
    main(args)
    print("Dual injection complete.")