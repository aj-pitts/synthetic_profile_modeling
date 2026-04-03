from src.plotter.plotting import setup_figure, plot_spec, plot_results, plot_grids, plot_results_2
from src.util import defaults
import os
import pickle

def get_results(results: dict, modspec: dict) -> None:
    #plot_grids(modspec, results)
    plot_results_2(results=results)

if __name__ == "__main__":
    root = defaults.get_root_path()
    resul_path = os.path.join(root, 'results')

    resul_file = os.path.join(resul_path, 'results.pkl')
    spectra_file = os.path.join(resul_path, 'spectra.pkl')

    with open(resul_file, 'rb') as f:
        results = pickle.load(f)

    with open(spectra_file, 'rb') as f:
        modspec = pickle.load(f)

    get_results(results, modspec)