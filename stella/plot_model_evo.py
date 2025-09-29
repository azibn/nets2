#!/usr/bin/env python3
"""
Model Evolution Plotter

This script downloads lightcurves for specified targets and shows how 
different CNN model iterations perform on detecting exocomet transits.

Usage:
    python model_evolution_plotter.py "Beta Pic" --sector 6
    python model_evolution_plotter.py "Beta Pic" "TIC 270577175" "HD 172555" --sector 6
    python model_evolution_plotter.py "Beta Pic" "TIC 270577175" --sectors 6 9 12 --output-dir plots/
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import lightkurve as lk
from mpl_toolkits.axes_grid1 import make_axes_locatable
import re
from pathlib import Path

# Add your stella library to the path
sys.path.insert(1, 'scripts')
sys.path.insert(1, 'stella')
import stella

# Hardcoded model paths and descriptions
MODELS = [
    'cnn-models-es/ensemble_s0101_i0200_b0.63.h5',
    'cnn-models-es/ensemble_s0111_i0200_b0.7.h5',
    'cnn-models-es/ensemble_s0121_i0200_b0.85.h5',
    'cnn-models-es/ensemble_s0131_i0200_b0.56.h5',
    'cnn-models-es/ensemble_s0141_i0200_b0.63.h5',
    'cnn-models-es/ensemble_s1031_i0200_b0.56.h5',
    'cnn-models-es/ensemble_s1041_i0200_b0.63.h5',
    'cnn-models-es/ensemble_s2031_i0200_b0.56.h5',
    'cnn-models-es/ensemble_s2041_i0200_b0.63.h5'
]

TITLES = [
    'Stage 1: Comets vs Non-comets',
    'Stage 2: Stage 1 + Augmented Data'
    'Stage 3: Stage 2 + Planets & Binaries',
    'Stage 4: Stage 3 (but larger dataset for class balance)',
    'Stage 5: Stage 4 + Sines',
    'Stage 6a: Three-Layer CNN (with Stage 4 dataset)',
    'Stage 6b: Three-layer CNN (with Stage 5 dataset)',
    'Stage 7a: CNN-LSTM hybrid (with Stage 4 dataset)',
    'Stage 7b: CNN-LSTM hybrid (with Stage 5 dataset)'
]

def verify_models_exist():
    """Check if all model files exist."""
    missing_models = []
    for model_path in MODELS:
        if not os.path.exists(model_path):
            missing_models.append(model_path)
    
    if missing_models:
        print("ERROR: The following model files are missing:")
        for model in missing_models:
            print(f"  - {model}")
        print("\nPlease check the model paths in the script.")
        return False
    return True

def sanitize_filename(target_name):
    """Convert target name to a safe filename."""
    safe_name = re.sub(r'[^\w\s-]', '', target_name)
    safe_name = re.sub(r'[-\s]+', '_', safe_name)
    return safe_name.strip('_')

def plot_model_evolution(target, sector=None, mission='TESS', pipeline='TESS-SPOC', 
                        figsize=(10, 8), output_path=None, show_plot=True):
    """
    Plot the evolution of model predictions across different training iterations.
    """
    
    print(f"Processing {target}...")
    
    # Download and process lightcurve
    try:
        if sector is None:
            search_result = lk.search_lightcurve(target, mission=mission, author=pipeline)
            if len(search_result) == 0:
                raise ValueError(f"No lightcurves found for {target}")
            lc = search_result[0].download()
        elif mission != 'TESS':
            lc = lk.search_lightcurve(target, quarter=sector, mission=mission).download()
            lc = lc[lc.sap_quality == 0]
        else:
            lc = lk.search_lightcurve(target, sector=sector, mission=mission, author=pipeline).download()
            lc = lc[lc.quality.value == 0]
        
        print(f"  Downloaded lightcurve: {len(lc.time)} data points")
        
    except Exception as e:
        print(f"  ERROR: Failed to download lightcurve for {target}: {e}")
        return False, None
    
    # Process the flux
    try:
        f = np.array((lc.flux.value / np.nanmedian(lc.flux.value)) - 1)
        f = np.array(f / np.nanstd((lc.flux.value / np.nanmedian(lc.flux.value)) - 1))
        f = f + 1
        f = f[~np.isnan(f)]
        f = (f - np.min(f)) / (np.max(f) - np.min(f))
        
        # Align arrays
        min_len = min(len(lc.time.value), len(f), len(lc.flux_err.value))
        times = lc.time.value[:min_len]
        fluxes = f[:min_len]
        errs = lc.flux_err.value[:min_len]
        
    except Exception as e:
        print(f"  ERROR: Failed to process flux data for {target}: {e}")
        return False, None
    
    # Initialize CNN
    try:
        cnn = stella.ConvNN(output_dir='cnn-models-es', ds=None)
    except Exception as e:
        print(f"  ERROR: Failed to initialize CNN: {e}")
        return False, None
    
    # Create subplot grid
    fig, axes = plt.subplots(len(MODELS), 1, figsize=figsize)
    plt.style.use('seaborn-v0_8-paper')
    
    if len(MODELS) == 1:
        axes = [axes]
    
    sector_str = f" (Sector {sector})" if sector else ""
    fig.suptitle(f'Model Evolution for {target}{sector_str}', fontsize=16, y=1.02)
    
    # Plot each model's predictions
    successful_models = 0
    for i, (model_path, title, ax) in enumerate(zip(MODELS, TITLES, axes)):
        try:
            cnn.predict(modelname=model_path, times=times, fluxes=fluxes, errs=errs)
            
            scatter = ax.scatter(cnn.predict_time[0], cnn.predict_flux[0], 
                               c=cnn.predictions[0], vmin=0, vmax=1, s=5,
                               cmap='viridis')
            ax.grid(False)
            
            # Add colorbar
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='2%', pad=0.05)
            plt.colorbar(scatter, cax=cax, label='Probability')
            
            ax.set_title(title, fontsize=12, pad=10)
            ax.set_xlabel('Time (BJD - 2457000)' if i == len(MODELS)-1 else '')
            ax.set_ylabel('Normalized Flux')
            successful_models += 1
            
        except Exception as e:
            print(f"  WARNING: Failed to process model {i+1}: {e}")
            ax.text(0.5, 0.5, f'Model failed to load:\n{title}', 
                   transform=ax.transAxes, ha='center', va='center',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.3))
            ax.set_title(f"{title} (FAILED)", fontsize=12, pad=10, color='red')
    
    plt.tight_layout()
    
    # Save the plot
    if output_path:
        try:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"  Saved plot to {output_path}")
        except Exception as e:
            print(f"  ERROR: Failed to save plot: {e}")
    
    # Show the plot
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    
    return successful_models > 0, output_path

def main():
    parser = argparse.ArgumentParser(
        description='Plot CNN model evolution for exocomet detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "Beta Pic" --sector 6
  %(prog)s "Beta Pic" "TIC 270577175" "HD 172555" --sector 6
  %(prog)s "Beta Pic" "TIC 270577175" --sectors 6 9 12 --output-dir plots/
        """
    )
    
    parser.add_argument('targets', nargs='+',
                       help='One or more target names or TIC IDs')
    parser.add_argument('--sector', type=int,
                       help='Single sector/quarter/campaign to analyze')
    parser.add_argument('--sectors', nargs='+', type=int,
                       help='Multiple sectors/quarters/campaigns to analyze')
    parser.add_argument('--mission', default='TESS',
                       choices=['TESS', 'Kepler', 'K2'],
                       help='Mission name (default: TESS)')
    parser.add_argument('--pipeline', default='TESS-SPOC',
                       help='Pipeline name (default: TESS-SPOC)')
    parser.add_argument('--figsize', nargs=2, type=float, default=[10, 8],
                       help='Figure size as width height (default: 10 8)')
    parser.add_argument('--output-dir', '-d',
                       help='Directory to save plots')
    parser.add_argument('--show-plots', action='store_true',
                       help='Display plots (default: True for single target, False for multiple)')
    parser.add_argument('--list-models', action='store_true',
                       help='List all models and exit')
    
    args = parser.parse_args()
    
    # List models if requested
    if args.list_models:
        print("Configured models:")
        for i, (model, title) in enumerate(zip(MODELS, TITLES), 1):
            status = "✓" if os.path.exists(model) else "✗"
            print(f"{i:2d}. {status} {title}")
            print(f"     {model}")
        return 0
    
    # Verify models exist
    if not verify_models_exist():
        return 1
    
    # Handle sector specification
    sectors = []
    if args.sectors:
        sectors = args.sectors
    elif args.sector:
        sectors = [args.sector]
    else:
        sectors = [None]  # Process all available sectors
    
    # Create output directory if specified
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Determine whether to show plots
    show_plots = args.show_plots or (len(args.targets) == 1 and len(sectors) == 1)
    
    # Process each target-sector combination
    successful = 0
    total = len(args.targets) * len(sectors)
    
    print(f"Processing {len(args.targets)} targets × {len(sectors)} sectors = {total} combinations\n")
    
    for target in args.targets:
        for sector in sectors:
            # Generate output filename
            output_path = None
            if args.output_dir:
                safe_name = sanitize_filename(target)
                sector_suffix = f"_sector{sector}" if sector else ""
                output_path = os.path.join(args.output_dir, f"{safe_name}_model_evolution{sector_suffix}.png")
            
            # Process the target
            success, _ = plot_model_evolution(
                target=target,
                sector=sector,
                mission=args.mission,
                pipeline=args.pipeline,
                figsize=tuple(args.figsize),
                output_path=output_path,
                show_plot=show_plots
            )
            
            if success:
                successful += 1
            
            print()  # Add blank line between targets
    
    # Print summary
    print(f"Successfully processed {successful}/{total} combinations")
    
    return 0 if successful > 0 else 1

if __name__ == '__main__':
    sys.exit(main())