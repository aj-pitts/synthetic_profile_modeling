import h5py
import fcntl
import numpy as np
from src.util.defaults import get_root_path
import os

def write_output(
        spec_id: int, script_id: int, wave: np.ndarray, flux: np.ndarray, err: np.ndarray,
        fixed_parameters: dict[str, float], measurements: dict[str, float]
        ) -> None:
    root = get_root_path()
    data_dir = os.path.join(root, 'output/data')
    os.makedirs(data_dir, exist_ok=True)
    
    filename = 'data.h5'
    filepath = os.path.join(data_dir, filename)

    lockfile = filepath + ".lock"

    with open(lockfile, 'w') as lf:
        fcntl.fcntl(lf, fcntl.LOCK_EX)

        try:
            with h5py.File(filepath, 'a') as f:
                group = f.require_group(str(spec_id))
                subgroup = group.create_group(str(script_id))

                subgroup.create_dataset('wave', data=wave)
                subgroup.create_dataset('flux', data=flux)
                subgroup.create_dataset('err', data=err)

                params_group = subgroup.create_group("parameters")
                for key, param in fixed_parameters:
                    params_group.create_dataset(key, data=param)
                
                measurement_group = subgroup.create_group("measurements")
                for key, val in measurements:
                    measurement_group.create_dataset(key, data=val)

        finally:
            fcntl.fcntl(lf, fcntl.LOCK_UN)


def load_spec(spec_id: int) -> dict:
    root = get_root_path()
    data_dir = os.path.join(root, 'output/data')
    filename = 'data.h5'
    filepath = os.path.join(data_dir, filename)

    realizations = {}
    with h5py.File(filepath, 'r') as f:
        for script_id in f[str(spec_id)].keys():
            group = f[str(spec_id)][script_id]

            realizations[script_id] = {
                'wave':group['wave'][:],
                'flux':group['flux'][:],
                'err':group['err'][:],
                'params':{k : group['params'][k][()] for k in group['params']},
                'measurements':{k : group['measurements'][k][()] for k in group['measurements']}
            }
    return realizations


def unpack_output() -> dict:
    root = get_root_path()
    data_dir = os.path.join(root, 'output/data')
    filename = 'data.h5'
    filepath = os.path.join(data_dir, filename)

    results = {}
    with h5py.File(filepath, 'r') as f:
        for spec_id in f.keys():
            realizations = load_spec(spec_id)
            results[spec_id] = realizations
    
    return results