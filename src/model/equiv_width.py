import numpy as np

def measure_equiv_width(flux: np.ndarray, wave: np.ndarray) -> float:
    wavelims = (5885, 5905)
    w = (wave >= wavelims[0]) & (wave <= wavelims[1])

    f = flux[w]
    cont = np.ones_like(f)
    dl = np.gradient(wave[w])

    weq = np.sum( (cont - f) * dl )
    return weq