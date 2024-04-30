# Import common imports
from common_imports import *
import pathlib
script_directory = pathlib.Path(__file__).parent.resolve()

print(script_directory)

def VoodooAnalyser(
        date_str,
        site_meta,
        input_files,
        liquid_threshold,
        n_lwp_smoothing=60,
        do_QL_plot=True,
        do_csv=True,
):

    data = {}
    for key, value in input_files.items():
        data[key] = xr.open_dataset(value, decode_times=False)


    # set rain/precip flag
    is_raining = data['cat_cloudnet']['rain_detected'].values
    is_precipitating = np.mean(data['cat_cloudnet']['Z'][:, :5].values, axis=1) > 5
    is_no_lwp_ret = is_raining + is_precipitating


    liquid_masks = fetch_liquid_masks(
        data['cls_cloudnet']['target_classification'].values, 
        data['cls_voodoo']['target_classification'].values,
        data['cls_cloudnet']['detection_status'].values, 
        data['cls_cloudnet']['height'].values, 
        data['cat_voodoo']['liquid_prob'].values, 
        data['cat_cloudnet']['rain_detected'].values, 
        data['cat_cloudnet']['v'].values, 
        liquid_threshold=liquid_threshold
        )


    T, p, q = interpolate_T_p_q(
        data['cat_cloudnet']['time'].values,
        data['cat_cloudnet']['height'].values, 
        data['cat_cloudnet']['model_time'].values, 
        data['cat_cloudnet']['model_height'].values, 
        data['cat_cloudnet']['temperature'].values,
        data['cat_cloudnet']['pressure'].values, 
        data['cat_cloudnet']['q'].values,
        )

    
    llt_dict, lwp_dict = compute_llt_lwp(
        data['cat_cloudnet']['height'].values, 
        T, 
        p, 
        data['cat_cloudnet']['lwp'].values * 1000., 
        liquid_masks, 
        n_smoothing=n_lwp_smoothing, 
        idk_factor=1.5
        )
    valid_lwp = (0.0 < lwp_dict['mwr_s']) * (lwp_dict['mwr_s'] < 1000.0)


    correlations = correlation_llt_lwp(
        llt_dict, 
        lwp_dict, 
        ~is_no_lwp_ret, 
        liquid_masks.keys()
        )
    

    voodoo_status = fetch_voodoo_status(
        data['cat_cloudnet']['height'].values,
        liquid_masks, 
        )
    
    eval_metrics = fetch_evaluation_metrics(
        voodoo_status
        )


    if do_csv:
        create_csv_file(
            data['cat_cloudnet']['height'].values,
            llt_dict,
            lwp_dict,
            liquid_masks,
            f'{script_directory}/validation_csv/',
            date_str,
            site_meta['name']
        )


    if do_QL_plot:

        def ql_prediction(lwp_max=500):

            ts, rg = data['cat_cloudnet']['time'].values, data['cat_cloudnet']['height'].values * 0.001

            def make_rainflag(ax, ypos=-0.2):
                raining = np.full(is_no_lwp_ret.size, ypos)
                _m0 = np.argwhere(~is_no_lwp_ret * valid_lwp)
                _m1 = np.argwhere(is_no_lwp_ret + ~valid_lwp)
                ax.scatter(ts[_m0], raining[_m0], marker='|', color='green', alpha=0.75)
                ax.scatter(ts[_m1], raining[_m1], marker='|', color='red', alpha=0.75)

            def make_cbar(ax, p, ticks=None, ticklabels=None, label=None):
                ax.set(xlim=[0, 24], ylim=[-0.2, 12], xlabel='Time [UTC]', ylabel='Height [km]')
                cbar = fig.colorbar(
                    p,
                    cax=inset_axes(ax, width="50%", height="5%", loc='upper left'),
                    fraction=0.05,
                    pad=0.05,
                    orientation="horizontal",
                    extend='min'
                )

                if ticks is not None:
                    if type(ticks) == int:
                        cbar.set_ticks(np.arange(0, ticks) + 0.5)
                        cbar.ax.set_xticklabels(np.arange(0, ticks), fontsize=7)
                    else:
                        cbar.set_ticks(ticks)
                if ticklabels is not None:
                    cbar.ax.set_xticklabels(ticklabels, fontsize=7)
                if label is not None:
                    cbar.ax.set_ylabel(label, labelpad=-170, rotation=0, y=-0.5)
                return cbar

            def add_lwp(ax, lw=1.1, al=0.75):
                # ax_right = ax.twinx()

                ax.plot(ts, lwp_dict['mwr'], color='royalblue', linewidth=lw / 2, alpha=al / 2)
                ax.bar(ts, lwp_dict['mwr_s'], linestyle='-', width=0.01, color='royalblue', alpha=0.4, label=r'LWP$_\mathrm{MWR}$')

                ax.plot(
                    ts, lwp_dict['Cloudnet_s'], linestyle='-', c='black', linewidth=lw, alpha=al,
                    label=rf'LWP$_\mathrm{{Cloudnet}}$, $r^2 = {correlations["Cloudnetcorr(LWP)-s"]:.2f}$'
                )
                ax.plot(
                    ts, lwp_dict['Voodoo_s'], linestyle='-', c='red', linewidth=lw, alpha=al,
                    label=rf'LWP$_\mathrm{{VOODOO}}$, $r^2 = {correlations["Voodoocorr(LWP)-s"]:.2f}$'
                )
                ax.plot(
                    ts, lwp_dict['Combination_s'], linestyle='-', c='orange', linewidth=lw, alpha=al,
                    label=rf'LWP$_\mathrm{{Combination}}$, $r^2 = {correlations["Combinationcorr(LLT)-s"]:.2f}$'
                )

                ax.set(ylabel=r'LWP [g$\,$m$^{-2}$]', xlim=[0, 24], ylim=[-25, lwp_max])
                leg0 = ax.legend(loc='upper right', bbox_to_anchor=(0.99, 0.99), prop={'size': 8}, ncol=2)
                leg0.get_frame().set_alpha(None)
                leg0.get_frame().set_facecolor((1, 1, 1, 0.9))

                return ax

            def fetch_cmaps():
                from matplotlib import cm
                from matplotlib.colors import ListedColormap

                viridis = cm.get_cmap('viridis', 256)
                newcolors = viridis(np.linspace(0, 1, 256))
                newcolors[:1, :] = np.array([220 / 256, 220 / 256, 220 / 256, 1])
                viridis_new = ListedColormap(newcolors)

                colors = np.array([
                    [255, 255, 255, 255],
                    [0, 0, 0, 15],
                    [70, 74, 185, 255],
                    [0, 0, 0, 35],
                    [108, 255, 236, 255],
                    [220, 20, 60, 255],  # [180, 55, 87, 255],
                    [255, 165, 0, 155],
                ]) / 255
                V_status = ListedColormap(tuple(colors), "colors5")

                return viridis_new, V_status

            voodoo_probability_cmap, voodoo_status_cmap = fetch_cmaps()
            p = liquid_threshold
            liq_cbh = first_occurrence_indices(liquid_masks['Cloudnet'])
            liq_cbh = np.ma.masked_where(liq_cbh == None, liq_cbh)

            vars = ['$Z_e$', '$v_D$', r'$\beta_\mathrm{att}$', r'$P_\mathrm{liq}$', r'$v_\sigma$', 'Cloudnet', 'Cloudnet + Voodoo', 'Voodoo status']
            units = ['[dBZ]', r'[m$\,$s$^{-1}$]', r'[sr$^{-1}\,$m$^{-1}$]', '[1]', r'[m$\,$s$^{-1}$]', '', '', '']

            with plt.style.context(['science', 'ieee']):
                fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(14, 7))
                # 1. row
                p0 = ax[0, 0].pcolormesh(ts, rg, data['cat_cloudnet']['Z'].values.T, cmap='jet', vmin=-50, vmax=20)
                p1 = ax[0, 1].pcolormesh(ts, rg, data['cat_cloudnet']['v'].values.T, cmap='jet', vmin=-5, vmax=2)
                p2 = ax[0, 2].pcolormesh(ts, rg, data['cat_cloudnet']['beta'].values.T, cmap='jet', norm=mpl.colors.LogNorm(vmin=1.0e-7, vmax=1.0e-4))

                # 2. row
                lprob = data['cat_voodoo']['liquid_prob'].values
                #lprob[lprob < liquid_threshold] = 0.01
                lprob = np.ma.masked_where(~liquid_masks['cloud_mask'], lprob)
                p3 = ax[1, 0].pcolormesh(ts, rg, lprob.T, cmap=voodoo_probability_cmap, vmin=p, vmax=1)
                #ax[1, 0].scatter(ts, rg[liq_cbh] * 0.001, marker='*', color='red', alpha=0.55, s=1.0e-3)
                p4 = ax[1, 1].pcolormesh(ts, rg, data['cat_cloudnet']['v_sigma'].values.T, cmap='jet', norm=mpl.colors.LogNorm(vmin=1.0e-2, vmax=1.0))
                ax[1, 2] = add_lwp(ax[1, 2])
                make_rainflag(ax[1, 2], ypos=-25)

                # 3. row
                class_cmap = get_cloudnet_cmap()[1]
                p5 = ax[2, 0].pcolormesh(ts, rg, data['cls_cloudnet']['target_classification'].values.T, cmap=class_cmap, vmin=0, vmax=11)
                p6 = ax[2, 1].pcolormesh(ts, rg, data['cls_voodoo']['target_classification'].values.T, cmap=class_cmap, vmin=0, vmax=11)
                p7 = ax[2, 2].pcolormesh(ts, rg, voodoo_status.T, cmap=voodoo_status_cmap, vmin=0, vmax=7)
                make_cbar(ax[2, 2], p7, ticks=np.arange(0.5, 7.5), ticklabels=["clear\nsky", "no-CD", "CD", "TN", "TP", "FP", "FN"])
                make_rainflag(ax[2, 2])

                # scores
                ax[2, 0].text(
                    0.1, -4,
                    str([f'{key} = {val:.2f}' for key, val in eval_metrics.items()]),
                    dict(size=14)
                )
                # case meta
                ax[0, 0].text(0.1, 13, f'{date_str}  --  {site_meta["name"]}', dict(size=14))

                list_pmesh = [
                    [0, 0, 0, 1, 1, 2, 2],
                    [0, 1, 2, 0, 1, 0, 1],
                    [p0, p1, p2, p3, p4, p5, p6],
                    [None, None, None, None, None, 12, 12],
                    vars,
                    units
                ]
                for i, j, pmi, ticks, var, unit in zip(*list_pmesh):
                    make_cbar(ax[i, j], pmi, ticks=ticks, label=f'{var} {unit}')
                    make_rainflag(ax[i, j])

                    cont = ax[i, j].contour(
                        data['cat_cloudnet']['model_time'].values,
                        data['cat_cloudnet']['model_height'].values * 1.0e-3,
                        (data['cat_cloudnet']['temperature'].values - 273.15).T,
                        levels=[-38, -25, -15, -10, -5, 0, 5],
                        linestyles='dashed',
                        linewidths=[0.5],
                        colors=['black'],
                        alpha=1
                    )
                    clabels = ax[i, j].clabel(cont, inline=1, fmt='%1.1f°C', fontsize=7)
                    [txt.set_bbox(dict(facecolor='white', edgecolor='none', pad=0.75, alpha=0.7)) for txt in clabels]

                fig.subplots_adjust(bottom=0.1, right=0.95, top=0.95, left=0.05, hspace=0.11, wspace=0.15)
            return fig, ax

        fig, ax = ql_prediction()
        fig_name = f'{script_directory}/plots/{date_str}-{site_meta["name"]}-QL.png'
        plt.show()
        fig.savefig(fig_name, facecolor='white', dpi=400)
        print(f' saved  {fig_name}')



if __name__ == '__main__':
    ''' Main program for testing
    '''
    t0 = time.time()


    date = "20230217"
    

    # Required Input Files
    input_files = {
        'cls_cloudnet': f'{script_directory}/sample_data/{date}_classification.nc',
        'cat_cloudnet': f'{script_directory}/sample_data/{date}_categorize.nc',
        'cls_voodoo':   f'{script_directory}/sample_data/{date}_classification_voodoo.nc',
        'cat_voodoo':   f'{script_directory}/sample_data/{date}_categorize_voodoo.nc',
    }

    
    #site_meta = {'name': 'Punta-Arenas', 'altitude': 9}
    #site_meta = {'name': 'Leipzig, LIM', 'altitude': 117}
    site_meta = {'name': 'Eriswil',      'altitude': 923}


    VoodooAnalyser(
        date,
        site_meta=site_meta,
        input_files=input_files,
        liquid_threshold=0.5,
        n_lwp_smoothing=40,  # in sec
        do_QL_plot=True,
        do_csv=True,
    )
