import numpy as np
import argparse

from src.util import defaults
from src.config.spectra_setup import spectra_setup
from src.synthetic.generate import generate_synthetic
from src.fitter.fit_spec import fit_spectra

def run_fitter(script_id: int) -> None:
    snr_range = np.arange(*spectra_setup['snr'])
    vcen_range = np.arange(*spectra_setup['vcen'])
    logN_range = np.arange(*spectra_setup['logn'])
    bD_range = np.arange(*spectra_setup['bd'])
    Cf_range = np.arange(*spectra_setup['cf'])

    modspec = generate_synthetic(snr_range, vcen_range, logN_range, bD_range, Cf_range)

    fit_spectra(script_id, modspec)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("script_id", type=int, help="A unique identifier for the realization of the fits")
    
    return parser.parse_args()

if __name__ == "__main__":

    count = defaults.count_nspec()
    print(f"Preparing to run fits for {count} spectra")

    args = get_args()
    run_fitter(script_id=args.script_id)