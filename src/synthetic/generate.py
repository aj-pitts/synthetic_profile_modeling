import os
import numpy as np
from astropy.io import fits
from src.util import defaults
from src.synthetic.synthspec import physical_profile

def generate_synthetic(
        snr_range: np.ndarray, 
        vcen_range: np.ndarray,
        logN_range: np.ndarray,
        bD_range: np.ndarray,
        Cf_range: np.ndarray
        ) -> dict[int, dict[str, object]]:

    print("generating spectra...")
    root = defaults.get_root_path()
    config_dir = os.path.join(root, 'src/config')
    fullpath = os.path.join(config_dir, 'wave.npy')
    with fits.open(fullpath) as hdul:
        wavelength = hdul['wave'].data
    w = (wavelength >= 5870) & (wavelength <= 5920)
    wave = wavelength[w]

    spectra = {}
    spec_num = 0
    for snr in snr_range:
        for vcen in vcen_range:
            for logn in logN_range:
                for bd in bD_range:
                    for cf in Cf_range:
                        spec = physical_profile(wave, snr, vcen, logn, bd, cf)
                        params = {'snr':snr, 'vsynth':vcen, 'logn':logn, 'bd':bd, 'cf':cf}
                        #params = (snr, vcen, logn, bd, cf)
                        spectra[spec_num] = {"params":params, "spec":spec}
                        spec_num+=1
    return spectra