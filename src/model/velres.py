from linetools.spectra.xspectrum1d import XSpectrum1D
import numpy as np
import os

def get_velres(redshift, modwave):
    c = 2.998e5
    fitlim = (np.min(modwave), np.max(modwave))
    src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    configdir = os.path.join(src_dir, 'config')
    LSF_file = os.path.join(configdir, 'LSF-Config_MUSE_WFM')
    if not os.path.isfile(LSF_file):
        raise ValueError(f'LSF-Config_MUSE_WFM does not exist within {configdir}')
    
    configLSF = np.genfromtxt(LSF_file, comments='#')
    configLSF_wv_air = configLSF[:, 0]
    configLSF_res = configLSF[:, 1]

    # convert to vacuum since LSF is in air
    xspec = XSpectrum1D.from_tuple((configLSF_wv_air, 0.0 * configLSF_wv_air))
    xspec.meta['airvac'] = 'air'
    xspec.airtovac()
    configLSF_wv_vac = xspec.wavelength.value
    # convert LSF wavelength to the restframe using galaxy's redshift
    configLSF_restwv = configLSF_wv_vac / (1.0 + redshift)
    whLSF = np.where((configLSF_restwv > fitlim[0]) & (configLSF_restwv < fitlim[1]))
    median_LSFAng = np.median(configLSF_res[whLSF[0]])
    median_LSFvel = c * median_LSFAng / np.median(configLSF_wv_vac[whLSF[0]])
    return median_LSFvel