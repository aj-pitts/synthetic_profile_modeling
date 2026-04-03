from matplotlib.axes import Axes
from matplotlib.figure import Figure

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
import matplotlib.cm as cm
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgba

from src.model.model_nai import model_NaI
from src.model.velres import get_velres
from src.util.defaults import get_root_path
from src.config.measurement_config import MEASUREMENT_CONFIG

from typing import Optional


import os

def setup_figure(nspec: int) -> tuple[Figure, list[Axes]]:
    path = os.path.dirname(os.path.abspath(__file__))
    plt.style.use(os.path.join(path, 'figures.mplstyle'))

    base_w, base_h = plt.rcParams['figure.figsize']
    fig = plt.figure(figsize=(base_w * nspec, base_h * nspec))
    gs = GridSpec(nspec, nspec, figure=fig)

    ax_group = fig.add_subplot(gs[:,:])
    ax_group.set_xlabel('Flux', labelpad=15)
    ax_group.set_ylabel('Wavelength', labelpad=25)
    ax_group.tick_params(which = 'both', labelcolor='none', top=False, bottom=False, 
                        left=False, right=False)
    ax_group.set_frame_on(False)
    ax_group.patch.set_alpha(0)
    ax_group.set_zorder(-1)
    ax_group.set_xticks([])
    ax_group.set_yticks([])
    for spine in ax_group.spines.values():
        spine.set_visible(False)

    axes = []
    for i in range(nspec):
        for j in range(nspec):
            ax = fig.add_subplot(gs[i,j])
            axes.append(ax)
    return fig, axes

def plot_spec(
        xspec: dict, theta: tuple, ax: Axes,
        v: float,
        v_fit: float,
        xlabel: bool = True,
        ylabel: bool = True,
        text: Optional[str] = None,
        ) -> None:
    path = os.path.dirname(os.path.abspath(__file__))
    plt.style.use(os.path.join(path, 'figures.mplstyle'))

    flux = xspec['flux']
    wave = xspec['wave']

    ax.plot(wave, flux, 'k', drawstyle='steps-mid')
    ax.set_xlim(5870, 5920)
    ax.vlines([5891.5833, 5897.5581], 0, 2, linestyles='dotted', colors='dimgrey', linewidths=0.8)
    ax.vlines([v], 0, 2, colors='k', linestyles='dashed', linewidths=1.1)
    ax.vlines([v_fit], 0, 2, colors='magenta', linestyles='dashed', linewidths=1.1)

    w = (wave >= 5880) & (wave <= 5910)
    ymin = np.floor(np.min(flux[w]) * 10) / 10
    ax.set_ylim(ymin, 1.2)

    if text is not None:
        ax.text(0.05, 0.95, text, transform=ax.transAxes, ha='left', va='top')

    if not xlabel:
        ax.set_xticklabels([])
    if not ylabel:
        ax.set_yticklabels([])

    vres = get_velres(0, wave)
    mod = model_NaI(theta, vres, wave)

    ax.plot(mod['modwv'], mod['modflx'], color='b')


def plot_grids(
        modspec: dict,
        results: dict,
        max_rows: int = 10,
        max_cols: int = 10
) -> None:
    nspec = len(modspec)
    plots_per_fig = max_rows * max_cols
    nfigs = int(np.ceil(nspec / plots_per_fig))

    xspec_list = np.array(list(modspec.values()))
    spec_nums = np.array(list(modspec.keys()))

    count = 0
    for fig_num in range(nfigs):
        specs = xspec_list[count:count+plots_per_fig]
        nums = spec_nums[count:count+plots_per_fig]

        fig, axes = setup_figure(plots_per_fig)
        for i, (num,spec) in enumerate(zip(nums, specs)):
            nrow = i // max_cols
            ncol = i % max_cols

            params = spec['params']
            spectrum = spec['spec']
            snr, vcen, logn, bd, cf = params

            results_sub = results[num]
            result = results_sub['result']

            theta = result['theta']
            vfit = result['v']
            ax = axes[i]

            ylabel = True if ncol == 0 else False
            xlabel = True if nrow == max_rows - 1 else False

            s = rf"$S/N = {int(snr)}$"
            plot_spec(spectrum, theta, ax=ax, v=vcen, v_fit=vfit, xlabel=xlabel, ylabel=ylabel, text=s)

        rootpath = get_root_path()
        results_directory = os.path.join(rootpath, 'results')
        fig.savefig(os.path.join(results_directory, f"plots_{fig_num}.pdf"), bbox_inches='tight')
        count += plots_per_fig

def plot_results(results: dict[int, dict[str, object]]) -> None:
    path = os.path.dirname(os.path.abspath(__file__))
    plt.style.use(os.path.join(path, 'figures.mplstyle'))

    nrow = 1
    ncol = 2
    base_w, base_h = plt.rcParams['figure.figsize']
    fig = plt.figure(figsize=(base_w * ncol, base_h * nrow))
    gs = GridSpec(nrow, ncol, figure=fig)

    ax_left = fig.add_subplot(gs[0,0])
    ax_right = fig.add_subplot(gs[0,1])

    snrs = []
    dvs = []
    verrs = []
    ps = []

    for specnum, subdict in results.items():
        params = subdict['params']
        snr, vcen, logn, bd, cf = params

        res = subdict['result']
        vfit = res['v']
        verr = res['verr']
        p = res['p']
        ew = res['ew']

        snrs.append(snr)
        dvs.append((abs(vcen - vfit)))
        verrs.append(verr)
        ps.append(p)
    
    snrs = np.array(snrs)
    dvs = np.array(dvs)
    verrs = np.array(verrs)
    ps = np.array(ps)

    mean_verrs = []
    std_verrs = []

    mean_dvs = []
    std_dvs = []

    unique_snr = np.unique(snrs)
    for sn in unique_snr:
        w = snrs == sn
        mean_verrs.append(np.mean(verrs[w]))
        std_verrs.append(np.std(verrs[w]))

        mean_dvs.append(np.mean(dvs[w]))
        std_dvs.append(np.std(dvs[w]))

    ax_left.errorbar(unique_snr, mean_verrs, yerr=std_verrs, fmt='o', linestyle='none')
    ax_left.set_ylabel(r'$\sigma_{v_{\mathrm{cen}}}\ \left( \mathrm{km\ s^{1}} \right)$')
    ax_left.set_xlabel(r'$S/N$')

    ax_right.errorbar(unique_snr, mean_dvs, yerr=std_dvs, fmt='o', linestyle='none')
    ax_right.set_ylabel(r'$\Delta v\ \left( \mathrm{km\ s^{1}} \right)$')
    ax_right.set_xlabel(r'$S/N$')

    rootdir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    resdir = os.path.join(rootdir, 'results')
    fname = os.path.join(resdir, 'results.pdf')

    fig.savefig(fname, bbox_inches='tight')

def plot_results_2(results: dict[int, dict[str, object]]) -> None:
    def random_v() -> float:
        return np.sign(np.random.choice([-1, 1])) * (np.random.rand()+np.random.rand())

    path = os.path.dirname(os.path.abspath(__file__))
    plt.style.use(os.path.join(path, 'figures.mplstyle'))
    rootdir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    resdir = os.path.join(rootdir, 'results')

    dtype = [
        ('snr', float),
        ('vfit', float),
        ('verr', float),
        ('p', float),
        ('ew', float),
        ('vcen', float),
        ('logn', float),
        ('bd', float),
        ('cf', float),
    ]
    data = np.zeros(len(results), dtype=dtype)

    for i, (specnum, subdict) in enumerate(results.items()):
        params = subdict['params']
        snr, vcen, logn, bd, cf = params
        res = subdict['result']

        data[i] = (snr, res['v'], res['verr'], res['p'], res['ew'], vcen, logn, bd, cf)


    unique_logn = np.unique(data['logn'])
    unique_bd = np.unique(data['bd'])
    unique_cf = np.unique(data['cf'])

    hex = ['#e41a1c', "#004cff", '#ff7f00', "#00ce03", '#984ea3', '#a65628', '#f781bf']
    colors = hex[:len(unique_bd)]#cm.Set1(np.linspace(0, 1, len(unique_bd)))
    markers = ['o', '^', 'D', 'v', 'P', '*'][:len(unique_logn)]
    markersizes = np.linspace(5, 30, len(unique_cf))

    bd_map   = {v: c for v, c in zip(unique_bd, colors)}
    logn_map = {v: m for v, m in zip(unique_logn, markers)}
    cf_map   = {v: s for v, s in zip(unique_cf, markersizes)}

    for sn in np.unique(data['snr']):
        config = MEASUREMENT_CONFIG["SQUARE0.6"]
        for snlims, ewlim in config.items():
            if sn <= snlims[1] and sn > snlims[0]:
                ewlimit = ewlim
                break
        nrow = 3
        ncol = 1
        base_w, base_h = plt.rcParams['figure.figsize']
        fig = plt.figure(figsize=(base_w * ncol * 1.33, base_h * nrow * .5))
        gs = GridSpec(nrow, ncol, figure=fig, hspace=0)

        ax_top = fig.add_subplot(gs[0,0])
        ax_mid = fig.add_subplot(gs[1,0])
        ax_bott = fig.add_subplot(gs[2,0])

        ax_top.set_title(rf"$S/N = {int(sn)}$")
        ax_top.set_ylabel(r'$P$')
        ax_top.set_xticklabels([])

        ax_mid.set_ylabel(r"$\sigma_{v_{\mathrm{cen}}}\ \left( \mathrm{km\ s^{1}} \right)$")
        ax_mid.set_yscale('log')
        ax_mid.set_xticklabels([])

        ax_bott.set_ylabel(r"$\mathrm{EW}\ \left( \mathrm{\AA} \right)$")
        ax_bott.set_xlabel(r"$v_{\mathrm{synth}}\ \left( \mathrm{km\ s^{1}} \right)$")


        w = sn == data['snr']
        rows = data[w]

        for row in rows:

            rgba = to_rgba(bd_map[row['bd']])
            dimgray = (0.412, 0.412, 0.412, 1.0)

            color = rgba if row['ew'] >= ewlimit else dimgray

            ax_top.scatter(
                row['vcen']+random_v(), row['p'],
                s=cf_map[row['cf']],
                marker=logn_map[row['logn']],
                facecolors=(*color[:3], 0.5),
                edgecolors=(*color[:3], 1)
            )

            ax_mid.scatter(
                row['vcen']+random_v(), row['verr'],
                s=cf_map[row['cf']],
                marker=logn_map[row['logn']],
                facecolors=(*color[:3], 0.5),
                edgecolors=(*color[:3], 1)
            )

            ax_bott.scatter(
                row['vcen']+random_v(), row['ew'],
                s=cf_map[row['cf']],
                marker=logn_map[row['logn']],
                facecolors=(*color[:3], 0.5),
                edgecolors=(*color[:3], 1)
            )

        bd_handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=bd_map[v],
            markersize=8, label=rf'$b_D = {v}\ \left( \mathrm{{km\ s^{{-1}}}} \right)$')
        for v in unique_bd
        ]

        logn_handles = [
            Line2D([0], [0], marker=logn_map[v], color='k', linestyle='none',
                markersize=8, label=rf'$\mathrm{{log}}\, N = {v:.1f}\, (\mathrm{{cm}}^{{-2}})$')
            for v in unique_logn
        ]

        cf_handles = [
            Line2D([0], [0], marker='o', color='k', linestyle='none',
                markersize=np.sqrt(cf_map[v]), label=rf'$C_f$ = {v:.1f}')
            for v in unique_cf
        ]

        # Add each group with a title
        leg1 = ax_top.legend(handles=bd_handles, bbox_to_anchor=(1.00, .5), loc='upper left', frameon=False)
        leg2 = ax_mid.legend(handles=logn_handles, bbox_to_anchor=(1.00, .5), loc='upper left', frameon=False)
        leg3 = ax_bott.legend(handles=cf_handles, bbox_to_anchor=(1.00, .5), loc='upper left', frameon=False)

        # ax_bott.add_artist(leg1)
        # ax_bott.add_artist(leg2)

        fname = os.path.join(resdir, f'results_sn{int(sn)}.pdf')
        fig.savefig(fname, bbox_inches='tight')


