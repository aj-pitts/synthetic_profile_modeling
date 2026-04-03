from multiprocessing import Pool
import argparse
import os

import warnings

from run.run_model import run_fitter
from src.util.defaults import get_root_path


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("n_realizations", type=int, help="The number of separate fitting processes to run in parallel")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    n_realizations = args.n_realizations

    if n_realizations >= os.cpu_count():
        raise UserWarning(f"Number of realizations ({n_realizations}) is greater than the avaiable CPU cores.")

    root = get_root_path()
    datapath = os.path.join(root, 'output/data/data.h5')
    if os.path.exists(datapath):
        os.remove(datapath)

    warnings.filterwarnings("error", message="function 'max' is not known to astropy's Quantity")
    
    with Pool(processes=n_realizations) as pool:
        pool.map(run_fitter, range(n_realizations))