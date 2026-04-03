from src.model.equiv_width import measure_equiv_width
from src.fitter.model_fitter import model_fitter
from src.io.file_handler import write_output

import numpy as np

def fit_spectra(script_id: int, spectra: dict[int, dict[str, object]]) -> None:
    lamred = 5897.5581
    logN = 14.5
    bD = 40.0
    Cf = 0.5
    theta_guess = lamred, logN, bD, Cf

    c = 2.998e5
    
    results = {}
    for i, (specnum, subdict) in enumerate(spectra.items()):
        print(f'\rScript{script_id}: Fitting Spec {i+1}/{len(spectra)}', end='', flush=True)
        xspec = subdict['spec']
        params = subdict['params']

        datfit = model_fitter(xspec, theta_guess)
        datfit.mcmc()

        ew = measure_equiv_width(xspec['flux'], xspec['wave'])

        lamred_mcmc, logN_mcmc, bD_mcmc, Cf_mcmc = datfit.theta_percentiles
        lamrest = 5897.5581
        lamerr = np.mean(lamred_mcmc[1:])

        velocity = ((lamred_mcmc[0] / lamrest) - 1) * c
        verr = lamerr * c / lamrest
        theta = np.array([param[0] for param in datfit.theta_percentiles])
        lamred, logn, bd, cf = theta

        theta_error = np.array(np.mean(param[1:]) for param in datfit.theta_percentiles)
        lamred_err, logn_err, bd_err, cf_err = theta_error

        samples = datfit.samples
        lambda_samples = samples[:,1000:,0].flatten()
        p = np.abs(np.sign(velocity) * np.sum(np.sign(velocity) * (lamrest - lambda_samples) < 0) / lambda_samples.size)

        percentiles = datfit.theta_percentiles

        result = {'ew':ew, 'v':velocity, 'verr':verr, 'p':p, 
                  'lambda':lamred, 'lambdaerr':lamred_err, 'logn':logn, 'lognerr':logn_err,
                  'bd':bd, 'bderr':bd_err, 'cf':cf, 'cferr':cf_err}

        results[specnum] = result #{"result":result, "params":params}

        write_output(specnum, script_id, xspec['wave'], xspec['flux'], xspec['err'],
                     params, result)
    print()