import numpy as np
from linetools.spectra.xspectrum1d import XSpectrum1D
from src.model import model_nai
from src.fitter.velres import get_velres
from typing import Optional
import os
import astropy.units as u

def gaussian_profile(
        equiv_w: float,
        v_disp: float,
        ratio: float,
        wave: Optional[np.ndarray] = None, 
        snr: float = 50,
        equiv_width: float = 1.,
        vcen: float = 0.,
        ) -> dict:

    wave = np.arange(5870, 5920, 0.1) if wave is None else wave
    dmwv = np.mean(np.diff(wave))
    c = 2.998e5
    z = vcen / c

    transinfo = model_nai.transitions()

    lam_d2 = transinfo['lamblu0'] * (1 + z)
    lam_d1 = transinfo['lamred0'] * (1 + z)

    sigma_d2 = (v_disp / c) * lam_d2
    sigma_d1 = (v_disp / c) * lam_d1

    ew_d2 = equiv_w * ratio / (1 + ratio)
    ew_d1 = equiv_w / (1 + ratio)

    A_d2 = ew_d2 / (sigma_d2 * np.sqrt(2 * np.pi))
    A_d1 = ew_d1 / (sigma_d1 * np.sqrt(2 * np.pi))

    prof_d2 = A_d2 * np.exp(-(wave - lam_d2)**2 / (2 * sigma_d2**2))
    prof_d1 = A_d1 * np.exp(-(wave - lam_d1)**2 / (2 * sigma_d1**2))

    flux = 1 - (prof_d2 + prof_d1)
    modwave = u.Quantity(wave, u.angstrom)

    xspec = XSpectrum1D.from_tuple((modwave, flux))
    xspec_noise = xspec.add_noise(s2n=snr)

    velres = get_velres(z, modwave)
    
    midwv = (transinfo['lamblu0'] + transinfo['lamred0']) / 2.0
    wvres = midwv * velres / c
    pxres = wvres / dmwv

    smooth_xspec = xspec_noise.gauss_smooth(pxres)

    return


def physical_profile(
        obs_wave: np.ndarray,
        snr: float,
        vcen: float,
        logN: float,
        bD: float,
        Cf: float,
) -> dict:
    wave = np.arange(5870, 5920, 0.1)
    dmwv = np.mean(np.diff(wave))
    c = 2.998e5
    z = vcen / c

    transinfo = model_nai.transitions()
    velratio = 1.0 + (transinfo['lamblu0'] - transinfo['lamred0'])/transinfo['lamred0']

    lamred = transinfo['lamred0'] * (1+z)
    lamblu = lamred * velratio

    N = 10**logN

    taured0 = N * 1.497e-15 * transinfo['lamfred0'] / bD
    taublu0 = N * 1.497e-15 * transinfo['lamfblu0'] / bD

    exp_red = -1.0 * (wave - lamred)**2 / (lamred * bD / c)**2
    exp_blu = -1.0 * (wave - lamblu)**2 / (lamblu * bD / c)**2

    taured = taured0 * np.exp(exp_red)
    taublu = taublu0 * np.exp(exp_blu)

    flux = 1.0 - Cf + (Cf * np.exp(-1.0*(taublu + taured)))

    xspec = XSpectrum1D.from_tuple((wave, flux))

    midwv = (transinfo['lamblu0'] + transinfo['lamred0']) / 2.0
    velres = get_velres(z, wave)
    wvres = midwv * velres / c
    pxres = wvres / dmwv
    xspec = xspec.gauss_smooth(pxres)

    uwave = u.Quantity(obs_wave,unit=u.AA)
    xspec = xspec.rebin(uwave)
    
    xspec = xspec.add_noise(s2n=snr)

    mod_wave = xspec.wavelength.value
    mod_flux = xspec.flux.value
    mod_err = 1 / np.sqrt(xspec.ivar.value)

    return{'flux':mod_flux, 'wave':mod_wave, 'err':mod_err, 'velres':velres}