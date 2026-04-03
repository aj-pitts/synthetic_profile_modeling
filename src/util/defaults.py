from pathlib import Path
import numpy as np
from src.config.spectra_setup import spectra_setup

def get_root_path() -> str:
    return Path(__file__).resolve().parents[2]

def count_nspec() -> int:
    count = 1
    for key, tup in spectra_setup.items():
        count *= len(np.arange(*tup))
    return count