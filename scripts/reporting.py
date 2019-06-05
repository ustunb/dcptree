import os
import re
import dill
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from dcptree.paths import *

# settings
pd.set_option('display.max_columns', 30)
np.set_printoptions(precision = 2)
np.set_printoptions(suppress = True)
sns.set()
sns.set_style("ticks")

def set_share_axes(axs, target=None, sharex=False, sharey=False, droplabelx = False, droplabely = False):

    if target is None:
        target = axs.flat[0]

    # Manage share using grouper objects
    for ax in axs.flat:
        if sharex:
            target._shared_x_axes.join(target, ax)
        if sharey:
            target._shared_y_axes.join(target, ax)

    # Turn off x tick labels and offset text for all but the bottom row
    if droplabelx and sharex and axs.ndim > 1:
        for ax in axs[:-1,:].flat:
            ax.xaxis.set_tick_params(which='both', labelbottom=False, labeltop=False)
            ax.xaxis.offsetText.set_visible(False)

    # Turn off y tick labels and offset text for all but the left most column
    if droplabely and sharey and axs.ndim > 1:
        for ax in axs[:,1:].flat:
            ax.yaxis.set_tick_params(which='both', labelleft=False, labelright=False)
            ax.yaxis.offsetText.set_visible(False)


HEATMAP_COLORMAPS = {
    'pvalue': sns.diverging_palette(h_neg = 10, h_pos = 240, l = 50, center = 'light'),
    #'pvalue': sns.light_palette("red", reverse = True),
    'error': sns.light_palette("seagreen", reverse=True),
    'error_gap': sns.diverging_palette(h_neg = 10, h_pos = 240, l = 50, center = 'light'),
    'error_relgap': sns.diverging_palette(h_neg = 10, h_pos = 240, l = 50, center = 'light'),
    }


def plot_group_heatmap(data, groups, p, ax, cmap, metric_type = 'error_gap', stat_field = 'train'):

    title_dict = {
        'error': 'Error',
        'error_gap': 'Error Gap',
        'pvalue': 'Pr(Envyfree)',
        'dcp_root_error': 'Pooled Error',
        'dcp_root_error_gap': 'Rationality Gap',
        'dcp_root_pvalue': 'Pr(Rational)',
        'dcp_root_error_relgap': 'Rationality Gap (Relative)',
        }

    metric_title = title_dict.get(metric_type, metric_type.capitalize())

    if 'dcp_root_' in metric_type:
        M = p.group_decoupling_stats(metric_type = metric_type[9:], parent_type = 'root', drop_missing = True,
                                     data = data, groups = groups, stat_field = stat_field)
        M = M[:, None]

    elif metric_type in ['error', 'error_gap', 'error_relgap', 'pvalue']:
        M = p.group_switch_stats(metric_type, drop_missing = True, data = data, groups = groups,
                                 stat_field = stat_field)


    annot_kws = {"size": 9}

    if metric_type in ['error_gap', 'dcp_root_error_gap']:

        cmap_center = 0.0
        fmt = '.1%'
        vmin, vmax = -0.25, 0.25
        sns.heatmap(M, linewidths = 4.0, cmap = cmap, center = cmap_center, cbar = True,
                    cbar_kws = {'orientation': 'vertical'}, annot = True, fmt = fmt, annot_kws = annot_kws, ax = ax,
                    vmin = vmin, vmax = vmax)

    else:

        if metric_type in ['error', 'dcp_root_error']:
            cmap_center = 0.0
            vmin, vmax = 0.0, 0.5
            fmt = '.1%'

        elif 'pvalue' in metric_type:
            cmap_center = 0.1
            vmin, vmax = 0.0, 1.0
            fmt = '.2'

        elif 'error_relgap' in metric_type:
            cmap_center = 0.0
            vmin, vmax = -0.5, 0.5
            fmt = '.1%'

        sns.heatmap(M, linewidths = 4.0, cmap = cmap, center = cmap_center, cbar = True,
                    cbar_kws = {'orientation': 'vertical'},
                    annot = True, fmt = fmt, annot_kws = annot_kws,
                    ax = ax, vmin = vmin, vmax = vmax)

    if stat_field == 'validation':
        plot_title = '%s (Test Set)' % metric_title
    else:
        plot_title = '%s (Training Set)' % metric_title

    ax.yaxis.set_label_text('Group')
    # ax.xaxis.set_label_text('Pooled')
    ax.set_title(plot_title)
    ax.tick_params(axis = 'both', which = 'both', bottom = False, left = False, top = False, right = False)

    # ax.set_yticklabels(ax.get_yticklabels(), rotation = 0)
    return ax


def tree_preference_report(data, groups, p, report_title = None, rationality_report = False, colormaps = HEATMAP_COLORMAPS):

    fig_metric_types = ['error', 'error_gap', 'pvalue']
    fig_stat_fields = ['train', 'validation']
    n_groups = len(p.split_names)

    plot_height = 1.5 * 3.0
    plot_width = 1.5 * 4.0
    if n_groups >= 16:
        plot_height *= np.ceil(n_groups / 12)
        plot_width *= np.ceil(n_groups / 12)

    if rationality_report:
        fig_metric_types = ['dcp_root_%s' % mt for mt in fig_metric_types]
        plot_width = 1.5 * 1.5

    # table with the names of the groups
    n_fig_rows = len(fig_stat_fields)
    n_fig_cols = len(fig_metric_types)
    fig, axs = plt.subplots(nrows = n_fig_rows, ncols = n_fig_cols,
                            sharex = True, sharey = True,
                            figsize = (n_fig_cols * plot_width, n_fig_rows * plot_height))


    # preferences
    for i, sf in enumerate(fig_stat_fields):
        for j, mt in enumerate(fig_metric_types):
            cmap = colormaps.get(mt, colormaps.get(mt[9:]))
            plot_group_heatmap(data, groups, p, ax = axs[i, j], cmap = cmap, metric_type = mt, stat_field = sf)

    if report_title is not None:
        fig.suptitle(report_title)
        fig.subplots_adjust(top = 0.88)

    return fig, axs


def plot_group_heatmap_pretty(data, groups, p, ax, cmap, metric_type = 'error_gap', stat_field = 'train'):

    title_dict = {
        'error': 'Error',
        'error_gap': 'Error Gap',
        'pvalue': 'Preference Violation',
        'dcp_root_error': 'Pooled Error',
        'dcp_root_error_gap': 'Rationality Gap',
        'dcp_root_pvalue': 'Preference Violation',
        'dcp_root_error_relgap': 'Rationality Gap (Relative)',
        }

    metric_title = title_dict.get(metric_type, metric_type.capitalize())
    if 'dcp_root_' in metric_type:
        M = p.group_decoupling_stats(metric_type = metric_type[9:], parent_type = 'root', drop_missing = True,
                                     data = data, groups = groups, stat_field = stat_field)
        M = M[:, None]

    elif metric_type in ['error', 'error_gap', 'error_relgap', 'pvalue']:
        M = p.group_switch_stats(metric_type, drop_missing = True, data = data, groups = groups,
                                 stat_field = stat_field)



    if metric_type in ['error_gap', 'dcp_root_error_gap']:
        cmap_center = 0.0
        fmt = '.1f'
        vmin, vmax = -25, 25
        sns.heatmap(M * 100, linewidths = 4.0, cmap = cmap, center = cmap_center, cbar = True,
                    cbar_kws = {'orientation': 'vertical', 'format':'%.0f%%'},
                    annot = True, fmt = fmt, annot_kws = {"size": 9}, ax = ax,
                    vmin = vmin, vmax = vmax, square= True)

    elif metric_type in ['pvalue', 'dcp_root_pvalue']:

        if 'dcp_root_' in metric_type:
            labels = p.group_decoupling_stats(metric_type = 'error_gap', parent_type = 'root', drop_missing = True, data = data, groups = groups, stat_field = stat_field)
            gaps = labels[:, None]
        else:
            gaps = p.group_switch_stats(metric_type = 'error_gap', drop_missing = True, data = data, groups = groups, stat_field = stat_field)

        cmap_center = 0.0
        vmin, vmax = np.log(1e-3/(1-1e-3)), np.log((1.0-1e-3)/(1e-3))
        M = np.log(M / (1.0 - M))

        sns.heatmap(M, linewidths = 4.0, cmap = cmap,
                    center = cmap_center, cbar = True,
                    cbar_kws = {'orientation': 'vertical', 'label': 'p-value'},
                    annot = gaps, fmt = '.0%', annot_kws = {"size": 9},
                    ax = ax, square = True, vmin = vmin, vmax = vmax)

        cbar = ax.collections[0].colorbar
        pvalues = [0.01, 0.05, 0.5, 0.95, 0.99]
        tick_labels = ['%1.2f' % p for p in pvalues]
        tick_values = [np.log(p / (1.0 - p)) for p in np.array(pvalues)]
        cbar.set_ticks(tick_values)
        cbar.set_ticklabels(tick_labels)

    else:

        fmt = '.1f'
        cmap_center = 0.0
        vmin, vmax = 0, 50
        if metric_type in ['error', 'dcp_root_error']:
            vmin, vmax = 0, 50

        elif 'error_relgap' in metric_type:
            vmin, vmax = -50, 50

        sns.heatmap(M * 100, linewidths = 4.0, cmap = cmap, center = cmap_center, cbar = True,
                    cbar_kws = {'orientation': 'vertical', 'format':'%.0f%%'},
                    annot = True, fmt = fmt, annot_kws = {"size": 9},
                    ax = ax, vmin = vmin, vmax = vmax, square=True)

    if stat_field == 'validation':
        plot_title = '%s - Test' % metric_title
    else:
        plot_title = '%s - Train' % metric_title

    ax.yaxis.set_label_text('Group')
    ax.xaxis.set_label_text('Classifier')
    ax.set_title(plot_title)
    ax.tick_params(axis = 'both', which = 'both', bottom = False, left = False, top = False, right = False)
    ax.set_xticklabels(['%1.0f' % (k + 1.0) for k in range(M.shape[0])])
    # ax.set_yticklabels(ax.get_yticklabels(), rotation = 0)
    return ax



def tree_preference_report_pretty(data, groups, p, report_title = None, rationality_report = False, colormaps = HEATMAP_COLORMAPS):

    fig_metric_types = ['error', 'error_gap', 'pvalue']
    fig_stat_fields = ['train', 'validation']
    plot_height = 2.0 * 3.0
    plot_width = 2.0 * 4.0

    if rationality_report:
        fig_metric_types = ['dcp_root_%s' % mt for mt in fig_metric_types]
        plot_width = 2.0 * 2.5

    #sharex = True, sharey = True,
    # table with the names of the groups
    n_fig_rows = len(fig_stat_fields)
    n_fig_cols = len(fig_metric_types)
    fig, axs = plt.subplots(nrows = n_fig_rows, ncols = n_fig_cols,
                            figsize = (n_fig_cols * plot_width, n_fig_rows * plot_height))

    if report_title is not None:
        fig.suptitle(report_title)
        fig.subplots_adjust(top = 0.9)

    # preferences
    for i, sf in enumerate(fig_stat_fields):
        for j, mt in enumerate(fig_metric_types):
            cmap = colormaps.get(mt, colormaps.get(mt[9:]))
            plot_group_heatmap_pretty(data, groups, p, ax = axs[i, j], cmap = cmap, metric_type = mt, stat_field = sf)
            yticks = axs[i, j].yaxis.get_ticklabels()
            new_ticks = []
            for k, t in enumerate(yticks):
                t.set_text(str(int(t.get_text())+1))
                new_ticks.append(t)
            axs[i, j].yaxis.set_ticklabels(new_ticks)

    return fig, axs



#### IO

def list_results_files(dir_name, all_data_names = None, all_method_names = None, balanced = False):
    """
    returns the names of all files that match prespecified criteria
    :param dir_name:
    :param all_data_names:
    :param all_method_names:
    :param balanced:
    :return:
    """

    all_results_files = [f for f in os.listdir(dir_name) if re.match(r'.*_results.pickle', f)]

    # filter out bugs
    file_list, bug_files = [], []
    for f in all_results_files:
        try:
            file_name = '%s/%s' % (dir_name, f)
            with open(file_name, 'rb') as infile:
                results = dill.load(infile)
            file_list.append(f)
        except EOFError:
            print('file loading error for: %s' % f)
            bug_files.append(f)
            pass

    all_results_files = file_list

    # list results for datasets
    if all_data_names is not None:
        assert isinstance(all_data_names, list)
        all_data_names = list(set(all_data_names))
        raw_file_list = []
        for data_name in all_data_names:
            data_match = '%s.*_results.pickle' % data_name
            raw_file_list += [f for f in all_results_files if re.match(data_match, f)]

        # remove balanced files
        if not balanced:
            balanced_files = []
            for data_name in all_data_names:
                balanced_match = '%s_bl.*_results.pickle' % data_name
                balanced_files += [f for f in raw_file_list if re.match(balanced_match, f)]
            raw_file_list = set(raw_file_list).difference(balanced_files)

        all_results_files = raw_file_list

    # filter by method
    if all_method_names is not None:
        assert isinstance(all_data_names, list)
        all_method_names = list(set(all_method_names))
        raw_file_list = []
        for method_name in all_method_names:
            method_match = '.*%s.*_results.pickle' % method_name
            raw_file_list += [f for f in all_results_files if re.match(method_match, f)]
        all_results_files = raw_file_list

    all_results_files = ['%s/%s' % (dir_name, f) for f in all_results_files]
    return all_results_files

