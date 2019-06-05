import warnings
import numpy as np
import pandas as pd
from copy import deepcopy
from imblearn.over_sampling import RandomOverSampler
from dcptree.data import check_data, has_validation_set, has_test_set, has_intercept, add_intercept, remove_intercept, remove_variable, add_variable, get_partition_names, get_index_of, get_partition_indices
from dcptree.data_processing import convert_categorical_to_rules


def as_group_info(values, values_validation = [], values_test = []):

    labels, indices = np.unique(values, return_inverse = True)

    group_info = {
        'ids': np.arange(len(labels), dtype = 'int'),
        'labels': labels,
        'indices': indices,
        'indices_test': np.searchsorted(labels, values_test),
        'indices_validation': np.searchsorted(labels, values_validation),
        }

    return group_info


def check_group_info(group_info):

    assert 'ids' in group_info
    assert 'labels' in group_info
    assert 'indices' in group_info
    assert 'indices_test' in group_info
    assert 'indices_validation' in group_info

    assert np.all(group_info['ids'] == np.arange(len(group_info['labels']), dtype = int))
    assert np.all(group_info['indices'] >= 0), "index values should be all positive"

    assert len(group_info['labels']) == len(set(group_info['indices'])), "labels should be unique values of indices"
    assert set(group_info['indices_test']).issubset(group_info['ids'])
    assert set(group_info['indices_validation']).issubset(group_info['ids'])

    return True


def check_groups(groups, data = None):

    assert type(groups) is dict
    if len(groups) == 0:
        return True

    group_names = list(groups.keys())
    n_samples = len(groups[group_names[0]]['indices'])
    n_samples_validation = len(groups[group_names[0]]['indices_validation'])
    n_samples_test = len(groups[group_names[0]]['indices_test'])

    for group_info in groups.values():

        assert check_group_info(group_info)
        assert len(group_info['indices']) == n_samples
        assert len(group_info['indices_test']) == n_samples_test
        assert len(group_info['indices_validation']) == n_samples_validation

    if data is not None:

        assert check_data(data)
        assert n_samples == data['X'].shape[0]

        if n_samples_test > 0:
            assert has_test_set(data)
            assert n_samples_test == data['X_test'].shape[0]

        if n_samples_validation > 0:
            assert has_validation_set(data)
            assert n_samples_validation == data['X_validation'].shape[0]

        for g in group_names:
            assert g not in data['variable_names']
            assert g not in data['variable_types']
            assert g not in data['variable_orderings']
            assert g not in data['partitions']

    return True


def split_groups_from_data(data, group_names = None):

    data = deepcopy(data)
    groups = {}

    # quick return if no partition
    if 'partitions' not in data:
        data['partitions'] = []

    if len(data['partitions']) == 0:
        return data, groups

    # set defaults when omitted
    if group_names is None:
        group_names = get_partition_names(data)

    # make sure that each group is listed as a partition
    for g in group_names:
        assert g in data['partitions']

    # check data object
    found_intercept = has_intercept(data)
    found_validation_set = has_validation_set(data)
    found_test_set = has_test_set(data)

    # drop intercept
    if found_intercept:
        data = remove_intercept(data)

    for g in group_names:

        j = get_index_of(data, g)
        values = data['X'][:, j],

        if found_validation_set:
            validation_values = data['X_validation'][:, j]
        else:
            validation_values = []

        if found_test_set:
            test_values = data['X_test'][:, j]
        else:
            test_values = []

        groups[g] = as_group_info(values, validation_values, test_values)
        data = remove_variable(data, g)

    # add intercept if it was removed
    if found_intercept:
        data = add_intercept(data)

    assert check_groups(groups, data)
    return data, groups


def add_group_to_data(data, groups, group_name):

    idx = 0 + has_intercept(data)
    group = groups[group_name]
    labels = group['labels']
    values = np.array(labels[group['indices']])
    validation_values = None
    test_values = None

    if has_validation_set(data):
        validation_values = np.array(labels[group['indices_validation']])

    if has_test_set(data):
        test_values = np.array(labels[group['indices_test']])

    data = add_variable(data,
                        name = group_name,
                        idx = idx,
                        variable_type = 'categorical',
                        is_partition = True,
                        values = values,
                        test_values = test_values,
                        validation_values = validation_values)

    # remove variable from groups
    groups.pop(group_name)
    assert check_groups(groups, data)
    return data, groups


def get_variable_mode(data, name):
    idx = data['variable_names'].index(name)
    labels, counts = np.unique(data['X'][:, idx], return_counts = True)
    top_group = labels[np.argmax(counts)]
    return top_group


def convert_remaining_groups_to_rules(data):

    remaining_groups = list(data['partitions'])
    for name in remaining_groups:
        top_group = get_variable_mode(data, name)
        data = convert_group_to_rules(data, name, baseline_labels = [top_group])

    assert check_data(data)
    return data


def convert_group_to_rules(data, name, baseline_labels = None):

    assert data['variable_types'][name] == 'categorical'

    # convert variable from categorical to rules
    idx = data['variable_names'].index(name)
    labels = np.unique(data['X'][:, idx])
    conversion_dict = {k: [k] for k in labels}

    if baseline_labels is not None:

        if type(baseline_labels) is not list:
            baseline_labels = [baseline_labels]

        for g in baseline_labels:
            conversion_dict.pop(g)

    data = convert_categorical_to_rules(data, name, conversion_dict, prepend_name = True)
    assert check_data(data)
    return data


def get_group_outcome_strata(data, group_names = None):
    _, groups = split_groups_from_data(data, group_names)
    strata_values = np.vstack([g['indices'] for g in groups.values()]).transpose()
    strata_values = np.insert(strata_values, 0, data['Y'], axis = 1)
    _, strata = np.unique(strata_values, axis = 0, return_inverse = True)
    return strata


def oversample_by_group(data, **kwargs):

    data, groups = split_groups_from_data(data)

    # get names/labels/values
    group_names = []
    group_labels = []
    group_values = []
    for n, g in groups.items():
        group_names.append(n)
        group_values.append(g['indices'])
        group_labels.append(g['labels'][g['indices']])
    group_values = np.transpose(np.vstack(group_values))
    group_labels = np.transpose(np.vstack(group_labels))

    # get unique ids for each combination of group attributes
    _, profile_idx = np.unique(group_values, axis = 0, return_inverse = True)
    profile_labels = range(0, np.max(profile_idx) + 1)

    # oversample labels
    ros = RandomOverSampler(**kwargs)
    X = np.array(data['X'])
    Y = np.array(data['Y'])
    X_res = []
    Y_res = []
    G_res = []
    assert np.isin((-1, 1), Y).all()

    for i in profile_labels:
        row_idx = np.isin(profile_idx, i)
        profile_values = group_labels[row_idx, :][0]
        Xg = X[row_idx, :]
        Yg = Y[row_idx]
        if np.isin((-1, 1), Yg).all():
            Xs, Ys = ros.fit_sample(Xg, Yg)
            X_res.append(Xs)
            Y_res.append(Ys)
            G_res.append(np.tile(profile_values, (len(Ys), 1)))
        else:
            profile_name = ''.join(['%s' % s for s in profile_values])
            warnings.warn('missing + and - labels for group %s' % profile_name)
            X_res.append(Xg)
            Y_res.append(Yg)
            G_res.append(np.tile(profile_values, (len(Yg), 1)))

    G_res = np.vstack(G_res)
    X_res = np.vstack(X_res)
    Y_res = np.concatenate(Y_res)

    data['X'] = X_res
    data['Y'] = Y_res
    data['sample_weights'] = np.ones_like(data['Y'], dtype = float)

    for j, name in enumerate(group_names):
        data = add_variable(data,
                            name = name,
                            variable_type = 'categorical',
                            is_partition = True,
                            values = G_res[:, j])


    assert check_data(data)
    return data


def get_group_count_table(data, stat_field = 'train'):
    """
    :param data:
    :return: pandas data frame containing unique groups
    """
    if stat_field == 'train':
        xf, yf = 'X', 'Y'
    else:
        xf = 'X_%s' % stat_field
        yf = 'Y_%s' % stat_field

    partitions = data['partitions']
    if len(partitions) > 0:
        idx = get_partition_indices(data)
        df = pd.DataFrame(data[xf][:, idx])
        df.columns = partitions
        df['Y'] = data[yf] > 0
        df['Y'].astype(dtype = int)
        count_df = df.groupby(by = data['partitions']).agg(['size', 'sum']).reset_index()
    else:
        df = pd.DataFrame(np.ones_like(data[xf][:, 0]))
        df.columns = 'all'
        df['Y'] = data[yf] > 0
        df['Y'].astype(dtype = int)
        count_df = df.groupby(by = 'all').agg(['size', 'sum']).reset_index()

    return count_df

