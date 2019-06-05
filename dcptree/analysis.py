import numpy as np
import itertools
from sklearn.utils import resample
from collections import OrderedDict

STAT_FIELD_NAMES = {'train', 'test', 'validation'}

GROUP_INDEX_NAMES = {
        'train': 'indices',
        'validation': 'indices_validation',
        'test': 'indices_test'
    }

def to_group_data(data, groups, stat_field = 'train'):

    assert stat_field in STAT_FIELD_NAMES
    if stat_field in ('validation', 'test'):
        xf = 'X_%s' % stat_field
        yf = 'Y_%s' % stat_field
    else:
        xf, yf, idxf = 'X', 'Y', 'indices'

    group_names, group_values = groups_to_group_data(groups, stat_field = stat_field)
    return data[xf], data[yf], group_names, group_values


def groups_to_group_data(groups, stat_field = 'train'):

    assert stat_field in GROUP_INDEX_NAMES

    index_field = str(GROUP_INDEX_NAMES[stat_field])
    group_names = []
    group_values = []

    for group_name, group_info in groups.items():
        assert index_field in group_info
        group_names.append(group_name)
        vals = np.array(group_info['labels'][group_info[index_field]], dtype = np.str_)
        vals = vals.reshape(len(vals), 1)
        group_values.append(vals)

    group_values = np.hstack(group_values)
    return group_names, group_values


def check_group_representation(groups):

    train_names, train_values = groups_to_group_data(groups, stat_field = 'train')
    train_labels = np.unique(train_values, axis = 0)

    for sf in ['validation', 'test']:
        sample_names, sample_values = groups_to_group_data(groups, stat_field = sf)
        assert set(sample_names) == set(train_names)
        if sample_values.shape[0] > 0:
            sample_labels = np.unique(sample_values, axis = 0)
            assert np.array_equal(sample_labels, train_labels)

    return


def groups_to_splits(groups, stat_field = 'train', drop_missing = True):

    if drop_missing:
        group_names, group_values = groups_to_group_data(groups, stat_field)
        group_values = np.unique(group_values, axis = 0).tolist()
        splits = [tuple(zip(group_names, v)) for v in group_values]

    else:
        splits = [[(k, l) for l in v['labels']] for k, v in groups.items()]
        splits = list(itertools.product(*splits))

    return splits


def error_rate(y, yhat):
    return np.mean(np.not_equal(y, yhat))


def leaf_stats(leaf, data, groups, stat_field = 'train'):

    assert stat_field in STAT_FIELD_NAMES
    if stat_field == 'test':
        xf, yf = 'X_test', 'Y_test'
    elif stat_field == 'validation':
        xf, yf = 'X_validation', 'Y_validation'
    else:
        xf, yf = 'X', 'Y'

    group_names, group_values = groups_to_group_data(groups, stat_field)
    idx = leaf.contains(group_names, group_values)

    X = data[xf][idx, :]
    Y = data[yf][idx]

    error_leaf = error_rate(y = Y, yhat = leaf.predict(X))
    error_pooled = error_rate(y = Y, yhat = leaf.root.predict(X))

    switch_info = [(s.name, error_rate(y = Y, yhat = s.predict(X))) for s in leaf.other_leaves]
    best_switch_idx = np.argmin([x[1] for x in switch_info])
    error_best_switch_name = switch_info[best_switch_idx][0]
    error_best_switch = switch_info[best_switch_idx][1]

    data_related_stats = {
        #
        'decouple': (error_pooled >= error_leaf) and (error_best_switch >= error_leaf),
        'rational': error_pooled >= error_leaf,
        'envyfree': error_best_switch >= error_leaf,
        #
        'envyfree_gap': error_best_switch - error_leaf,
        'envyfree_relgap': (error_best_switch - error_leaf) / error_leaf,
        'decoupling_gap': (error_pooled - error_leaf),
        'decoupling_relgap': (error_pooled - error_leaf) / error_leaf,
        #
        'n': X.shape[0],
        'n_pos': np.sum(Y > 0),
        'n_neg': np.sum(Y <= 0),
        #
        'error': error_leaf,
        'error_pooled': error_pooled,
        'error_best_switch': error_best_switch,
        'error_best_switch_name': error_best_switch_name,
        #
        'switch_stats': dict(switch_info),
        }

    leaf_stats = OrderedDict()
    leaf_stats.update({'name': leaf.name})
    old_keys = list(data_related_stats.keys())
    for k in old_keys:
        new_key = '%s_%s' % (stat_field, k)
        leaf_stats[new_key] = data_related_stats.pop(k)

    return leaf_stats


def stratified_resample(data, groups, stat_field = 'validation', p = 0.8, min_samples = 10):
    """
    :param data:
    :param groups:
    :param stat_field:
    :param p:
    :param min_samples:
    :return:
    """
    assert isinstance(p, float) and 0.0 < p <= 1.0
    assert isinstance(min_samples, int) and min_samples > 0

    if stat_field in ('validation', 'test'):
        xf = 'X_%s' % stat_field
        yf = 'Y_%s' % stat_field
        idxf = 'indices_%s' % stat_field
    else:
        xf, yf, idxf = 'X', 'Y', 'indices'

    X = data[xf]
    Y = data[yf]
    group_names, Z = groups_to_group_data(groups, stat_field)

    if p >= 1.0:
        return X, Y, group_names, Z

    n_samples = len(Y)
    n_groups = len(group_names)
    strata_values = np.reshape([g[idxf] for g in groups.values()], (n_samples, n_groups))
    strata_values = np.insert(strata_values, 0, Y, axis = 1)
    _, strata = np.unique(strata_values, axis = 0, return_inverse = True)
    n_strata = len(np.unique(strata))

    SX, SY, SZ = [], [], []
    for s in range(n_strata):
        idx = s == strata
        n_samples = int(np.floor(p * np.sum(idx)))
        if n_samples >= min_samples:
            x, y, z = resample(X[idx, :], Y[idx], Z[idx, :], n_samples = n_samples)
        else:
            x, y, z = X[idx, :], Y[idx], Z[idx, :]
        SX.append(x)
        SY.append(y)
        SZ.append(z)

    SX = np.vstack(SX)
    SY = np.concatenate(SY)
    SZ = np.vstack(SZ) if n_groups > 1 else np.concatenate(SZ)

    return SX, SY, group_names, SZ


def get_group_split_stats(group_models, group_info, X, Y, node_idx = None):
    """
    returns statistics to split group
    :param group_models:
    :param group_info:
    :param X:
    :param Y:
    :param node_idx:
    :return:
    """

    if node_idx is None:
        node_idx = np.ones(X.shape[0], dtype = int)

    indices = group_info['indices']
    labels = group_info['labels']
    n_labels = len(labels)

    stats = {}

    # compute matrix of errors by using classifiers from other group
    error_matrix = compute_switch_error_matrix(labels, indices, group_models, X, Y, node_idx)

    # get error for group a by using by classifier from group a
    stats['group_names'] = list(labels)
    stats['group_errors'] = np.diag(error_matrix)

    # compute bound
    stats['error_matrix'] = error_matrix

    # get best case error for group a by using by classifier any other group
    other_errors = np.array(error_matrix)
    np.fill_diagonal(other_errors, np.nan)

    # get error statistics for group a to switch classifiers
    stats['switch_errors'] = np.nanmin(other_errors, axis = 1)
    stats['switch_gain'] = stats['group_errors'] - stats['switch_errors']
    stats['fair_decoupling'] = np.less_equal(stats['switch_gain'], 0.0)
    stats['switch_groups'] = [labels[i] for i in np.flatnonzero(np.greater(stats['switch_gain'], 0.0))]

    # get switch counts
    stats['bound_values'] = []
    stats['n_pos'] = []
    stats['n_neg'] = []
    for i in range(n_labels):
        group_idx = np.logical_and(node_idx, indices == i)
        pos_ind = Y[group_idx] == 1
        neg_ind = Y[group_idx] == -1
        stats['n_pos'].append(np.sum(pos_ind))
        stats['n_neg'].append(np.sum(neg_ind))

    stats['n_pos'] = np.array(stats['n_pos'])
    stats['n_neg'] = np.array(stats['n_neg'])
    stats['n'] = stats['n_pos'] + stats['n_neg']
    return stats


def compute_switch_error_matrix(labels, indices, group_models, X, Y, node_idx = None):

    if node_idx is None:
        node_idx = np.ones(X.shape[0], dtype = int)

    n_labels = len(labels)
    error_matrix = np.empty(shape = (n_labels, n_labels))
    error_matrix.fill(np.nan)

    for i in range(n_labels):
        idx = np.logical_and(node_idx, indices == i)
        true_values = Y[idx]
        for j, model_label in enumerate(labels):
            model = group_models[model_label]
            predictions = model.predict(X[idx, :]).flatten()
            mistakes = np.not_equal(true_values, predictions)
            error_matrix[i, j] = np.mean(mistakes)

    return error_matrix



