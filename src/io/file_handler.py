import h5py
import fcntl
import numpy as np
from src.util.defaults import get_root_path
import os
from astropy.table import Table, Column
from glob import glob

def write_output(
        spec_id: int, script_id: int, wave: np.ndarray, flux: np.ndarray, err: np.ndarray,
        fixed_parameters: dict[str, float], measurements: dict[str, float]
        ) -> None:
    root = get_root_path()
    data_dir = os.path.join(root, 'output/data')
    os.makedirs(data_dir, exist_ok=True)
    
    files = glob(os.path.join(data_dir, 'data*.h5'))
    filename = 'data.h5' if len(files) == 0 else f'data_{len(files)}.h5'
    
    filepath = os.path.join(data_dir, filename)

    lockfile = filepath + ".lock"

    with open(lockfile, 'w') as lf:
        fcntl.fcntl(lf, fcntl.LOCK_EX)

        try:
            with h5py.File(filepath, 'a') as f:
                group = f.require_group(str(script_id))
                #subgroup = group.create_group(str(script_id))
                group.create_dataset('spec_id', data = spec_id)

                group.create_dataset('wave', data=wave)
                group.create_dataset('flux', data=flux)
                group.create_dataset('err', data=err)
                # subgroup.create_dataset('wave', data=wave)
                # subgroup.create_dataset('flux', data=flux)
                # subgroup.create_dataset('err', data=err)
                for key, param in fixed_parameters.items():
                    group.create_dataset(key, data=param)
                # params_group = subgroup.create_group("parameters")
                # for key, param in fixed_parameters.items():
                #     params_group.create_dataset(key, data=param)
                for key, val in measurements.items():
                    group.create_dataset(key, data=val)
                # measurement_group = subgroup.create_group("measurements")
                # for key, val in measurements.items():
                #     measurement_group.create_dataset(key, data=val)

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
                'params':{k : group['parameters'][k][()] for k in group['parameters']},
                'measurements':{k : group['measurements'][k][()] for k in group['measurements']}
            }
    return realizations

def results_table(force_overwrite: bool = False) -> Table:
    root = get_root_path()
    data_dir = os.path.join(root, 'output/data')
    filename = 'datatable.fits'
    filepath = os.path.join(data_dir, filename)

    if not os.path.exists(filepath):
        print('table not found, creating table...')
        create_table()

    elif force_overwrite:
        print('force overwrite, creating table...')
        create_table()
    
    t = Table.read(filepath)
    return t

def create_table() -> None:
    root = get_root_path()
    data_dir = os.path.join(root, 'output/data')
    filename = 'data.h5'
    filepath = os.path.join(data_dir, filename)

    column_names = [
        "spectrum_id", "realization_id", 
        "v_synth", "logn_synth", 
        "bd_synth", "cf_synth", "SNR",
        "v_cen", "v_cen_err", 
        "logn", "logn_err", 
        "bd", "bd_err",
        "cf", "cf_err",
        "ew", "p"
    ]

    store_dict = {name:[] for name in column_names}

    with h5py.File(filepath, 'r') as file:
        for spectrum_id in file.keys():
            realizations = {}
            for real_id in file[spectrum_id].keys():
                group = file[spectrum_id][real_id]

                realizations[real_id] = {
                'wave':group['wave'][:],
                'flux':group['flux'][:],
                'err':group['err'][:],
                'params':{k : group['parameters'][k][()] for k in group['parameters']},
                'measurements':{k : group['measurements'][k][()] for k in group['measurements']}
                }
            
            for real_id in realizations.keys():
                store_dict["spectrum_id"].append(int(spectrum_id))
                store_dict['realization_id'].append(int(real_id))

                realization = realizations[real_id]
                params = realization['params']
                store_dict['v_synth'].append(params['vsynth'])
                store_dict['logn_synth'].append(params['logn'])
                store_dict['bd_synth'].append(params['bd'])
                store_dict['cf_synth'].append(params['cf'])
                store_dict['SNR'].append(params['snr'])

                measurements = realization['measurements']
                store_dict['v_cen'].append(measurements['v'])
                store_dict['v_cen_err'].append(measurements['verr'])
                store_dict['logn'].append(measurements['logn'])
                store_dict['logn_err'].append(measurements['lognerr'])
                store_dict['bd'].append(measurements['bd'])
                store_dict['bd_err'].append(measurements['bderr'])
                store_dict['cf'].append(measurements['cf'])
                store_dict['cf_err'].append(measurements['cferr'])
                store_dict['ew'].append(measurements['ew'])
                store_dict['p'].append(measurements['p'])

    t = Table(store_dict)
    tablename = 'datatable.fits'
    tablepath = os.path.join(data_dir, tablename)
    t.write(tablepath, overwrite=True)
    print(f'data table saved to {tablepath}')

    altpath = os.path.expanduser('~/Repo/musenap/src/nai_analysis/synthetic_fits')
    alttablepath = os.path.join(altpath, tablename)
    t.write(alttablepath, overwrite=True)
    print(f'data table saved to {alttablepath}')

def collapse_realizations(realizations: dict) -> dict:
    collapsed = {}
    
    script_entry = next(iter(realizations.values()))
    measurements = script_entry['measurements']
    measurement_keys = list(measurements.keys())

    collapsed['params'] = script_entry['params']

    for mkey in measurement_keys:
        values = [d['measurements'][mkey] for d in realizations.values()]
        collapsed[mkey] = np.mean(values)
        collapsed[f"{mkey}_err"] = (np.min(values), np.max(values))
    return collapsed

def unpack_output() -> dict:
    root = get_root_path()
    data_dir = os.path.join(root, 'output/data')
    filename = 'data.h5'
    filepath = os.path.join(data_dir, filename)

    results = {}
    with h5py.File(filepath, 'r') as f:
        for spec_id in f.keys():
            realizations = load_spec(spec_id)
            collapsed = collapse_realizations(realizations)
            results[spec_id] = {
                'realizations':realizations,
                'collapsed':collapsed
            }
    
    return results

def structured_output() -> dict:
    results = unpack_output()
    dtype = [
        ('snr', float),
        ('vcen', float),
        ('v', float),
        ('v_min', float),
        ('v_max', float),
        ('v_err', float),
        ('v_err_min', float),
        ('v_err_max', float),
        ('p', float),
        ('p_min', float),
        ('p_max', float),
        ('ew', float),
        ('ew_min', float),
        ('ew_max', float),
        ('logn', float),
        ('logn_min', float),
        ('logn_max', float),
        ('bd', float),
        ('bd_min', float),
        ('bd_max', float),
        ('cf', float),
        ('cf_min', float),
        ('cf_max', float),
        ('bd_fix', float),
        ('logn_fix', float),
        ('cf_fix', float)
    ]

    data = np.zeros(len(results), dtype=dtype)
    for i, (key, subdict) in enumerate(results.items()):
        d = subdict['collapsed']
        p = d['params']
        data[i] = (
            p['snr'], 
            p['vsynth'], 
            d['v'], 
            d['v_err'][0], 
            d['v_err'][1],
            d['verr'],
            d['verr_err'][0],
            d['verr_err'][1],
            d['p'],
            d['p_err'][0],
            d['p_err'][1],
            d['ew'],
            d['ew_err'][0],
            d['ew_err'][1],
            d['logn'],
            d['logn_err'][0],
            d['logn_err'][1],
            d['bd'],
            d['bd_err'][0],
            d['bd_err'][1],
            d['cf'],
            d['cf_err'][0],
            d['cf_err'][1],
            p['bd'],
            p['logn'],
            p['cf']
        )
    return data