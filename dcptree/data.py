import warnings
import itertools
import numpy as np
from copy import deepcopy
from dcptree.helper_functions import print_log

OUTCOME_NAME = 'Y'
POSITIVE_LABEL = '+1'
NEGATIVE_LABEL = '-1'

MISSING_NAME = '(Missing)'
INTERCEPT_NAME = '(Intercept)'
INTERCEPT_VAR_TYPE = 'numeric'
INTERCEPT_IDX = 0
VALID_VAR_TYPES = {'numeric', 'boolean', 'ordinal', 'categorical'}

STAT_TYPE_NAMES = {'train', 'validation', 'test'}
FORMAT_NAME_DEFAULT = 'standard'
FORMAT_NAME_RULES = 'rules'
FORMAT_NAME_DCP = 'dcptree'
NUMERIC_FIELDS_NAMES = {'X', 'Y', 'sample_weights',
                        'X_validation','Y_validation', 'sample_weights_validation',
                        'X_test','Y_test','sample_weights_test'}


# NOTES on naming
# functions starting with "check_" functions are assertions
# functions starting with "validate_" make assertions + simple corrections

def check_data(data, ready_for_training = False):
    """
    makes sure that 'data' contains training data that is suitable for binary classification problems
    throws AssertionError if

    'data' is a dictionary that must contain:

     - 'X' N x P matrix of features (numpy.ndarray) with a column of 1s for the '(Intercept)'
     - 'Y' N x 1 vector of labels (+1/-1) (numpy.ndarray)
     - 'variable_names' list of strings containing the names of each feature (list)

     data can also contain:

     - 'outcome_name' string containing the name of the output (optional)
     - 'sample_weights' N x 1 vector of sample weights, must all be positive

    Returns
    -------
    True if data passes checks

    """
    assert isinstance(data, dict),\
        "data should be a dict"

    # check training fields

    assert 'X' in data, \
        "data should contain X matrix"

    assert 'Y' in data, \
        "data should contain Y matrix"

    assert 'variable_names' in data, \
        "data should contain variable_names"

    if has_intercept(data):
        assert check_intercept(data)

    if 'outcome_name' in data:
        assert isinstance(data['outcome_name'], (str, np.str_)), \
            "outcome_name should be a str"

    if 'variable_types' in data:
        assert isinstance(data['variable_types'], dict)
        assert set(data['variable_types'].keys()) == set(data['variable_names'])
        assert set(data['variable_types'].values()).issubset(VALID_VAR_TYPES)

    if 'variable_orderings' in data:
        assert 'variable_types' in data
        assert isinstance(data['variable_orderings'], dict)
        variable_orderings = data['variable_orderings']
    else:
        variable_orderings = dict()


    for xf, yf, sw in [('X', 'Y', 'sample_weights'),
                   ('X_test', 'Y_test', 'sample_weights_test'),
                   ('X_validation', 'Y_validation', 'sample_weights_validation')]:

        if xf in data and yf in data:

            assert check_data_required_fields(X = data[xf],
                                              Y = data[yf],
                                              variable_names = data['variable_names'],
                                              ready_for_training = ready_for_training)

            if 'variable_types' in data:
                assert check_data_variable_types(variable_types = data['variable_types'],
                                                 variable_names = data['variable_names'],
                                                 variable_orderings = variable_orderings,
                                                 X = data[xf],
                                                 ready_for_training = ready_for_training)

        if sw in data:

            assert xf in data and yf in data,\
                'found sample weights for non-existent set' % xf

            assert isinstance(data[sw], np.ndarray),\
                'data[%s] is not of np.array' % sw

            assert np.can_cast(data[sw], np.float, casting = 'safe'),\
                'data[%s] cannot be cast to float' % sw

            n_weights = data[sw].shape[0]
            n_points = data[xf].shape[0]

            assert n_weights == n_points, \
                'data[%s] should contain %d elements (found %d)' % (sw, n_weights, n_points)

            assert data[sw].ndim == 1 or data[sw].shape[1] >= 1, \
                '%data[%s] should have at least 1 column' % sw

            assert np.all(data[sw] >= 0.0), \
                '%data[%s] contains negative entries' % sw

    if 'format' in data:

        if data['format'] == FORMAT_NAME_DCP:

            assert 'partitions' in data
            assert isinstance(data['partitions'], list)
            for p in data['partitions']:
                assert p in data['variable_names']

        if data['format'] == FORMAT_NAME_RULES:

            assert data['X'].dtype == 'bool'
            assert 'feature_groups' in data
            assert 'feature_names' in data
            assert 'feature_types' in data
            assert 'feature_orderings' in data
            assert set(data['variable_types'].values()) == {'boolean'}
            assert set(data['feature_groups']) == set(data['feature_names'])
            assert INTERCEPT_NAME not in data['variable_names']
            assert INTERCEPT_NAME not in data['feature_names']

    return True


def check_data_required_fields(X, Y, variable_names, ready_for_training = False, **args):
    """
        makes sure that 'data' contains training data that is suitable for binary classification problems
        throws AssertionError if

        'data' is a dictionary that must contain:

         - 'X' N x P matrix of features (numpy.ndarray) with a column of 1s for the '(Intercept)'
         - 'Y' N x 1 vector of labels (+1/-1) (numpy.ndarray)
         - 'variable_names' list of strings containing the names of each feature (list)

         data can also contain:

         - 'outcome_name' string containing the name of the output (optional)
         - 'sample_weights' N x 1 vector of sample weights, must all be positive

        Returns
        -------
        True if data passes checks

        """
    # type checks

    assert type(X) is np.ndarray, \
        "type(X) should be numpy.ndarray"

    assert type(Y) is np.ndarray, \
        "type(Y) should be numpy.ndarray"

    assert type(variable_names) is list, \
        "variable_names should be a list"

    assert len(variable_names) == len(set(variable_names)), \
        'variable_names is not unique'

    # if it's ready for training then it should be numeric
    if ready_for_training:
        assert np.can_cast(X.dtype, np.float, casting = 'safe')
        assert np.can_cast(Y.dtype, np.float, casting = 'safe')

    # labels values
    assert np.all(np.isin(Y, (-1, 1))), \
        'Y must be binary for all i'

    if all(Y == 1):
        warnings.warn('Y does not contain any positive examples. need Y[i] = +1 for at least 1 i.')

    if all(Y == -1):
        warnings.warn('Y does not contain any negative examples. need Y[i] = -1 for at least 1 i.')

    # sizes and uniqueness
    n_variables = len(variable_names)
    n, d = X.shape

    assert n > 0, \
        'X matrix must have at least 1 row'

    assert d > 0, \
        'X matrix must have at least 1 column'

    assert len(Y) == n, \
        'dimension mismatch. Y must contain as many entries as X. Need len(Y) = N.'

    assert n_variables == d, \
        'len(variable_names) should be same as # of cols in X'

    return True


def check_data_variable_types(variable_types, variable_names, X, variable_orderings = dict(), ready_for_training = False):

    # check feature matrix
    for name in variable_names:

        j = variable_names.index(name)
        t = variable_types[name]

        vals = np.array(list(X[:, j]))

        if t == 'numeric':
            assert np.all(~np.isnan(vals)), '%s has nan entries' % name
            assert np.all(~np.isinf(vals)), '%s has inf entries' % name
        elif t == 'boolean':
            assert np.all(~np.isnan(vals)), '%s has nan entries' % name
            assert np.all(~np.isinf(vals)), '%s has inf entries' % name
            assert np.all((vals == True) | (vals == False))
        elif t == 'categorical':
            levels = np.unique(vals)
            assert np.array(list(levels)).dtype.char in ('U', 'S')
            assert len(levels) >= 1
            if len(levels) == 1:
                warnings.warn('%s = %r for all entries' % (name, levels))
        elif t == 'ordinal':
            levels = set(np.unique(vals))
            assert name in variable_orderings
            assert levels == set(variable_orderings[name])
            assert len(levels) >= 1
            if len(levels) == 1:
                warnings.warn('%s = %r for all entries' % (name, levels))
        else:
            raise ValueError('invalid variable_type %s for variable %s' % (t, name))

    if ready_for_training:
        for name, t in variable_types.items():
            assert (t == 'boolean') or (t == 'numeric'), "type of %s is %s: must be 'numeric' or 'boolean' if ready for training)" % (name, t)

    return True


def set_defaults_for_data(data):

    data.setdefault('format', FORMAT_NAME_DEFAULT)
    data.setdefault('outcome_name', OUTCOME_NAME)
    data.setdefault('outcome_label_positive', POSITIVE_LABEL)
    data.setdefault('outcome_label_negative', NEGATIVE_LABEL)

    for xf, yf, sw in [('X', 'Y', 'sample_weights'),
                       ('X_test', 'Y_test', 'sample_weights_test'),
                       ('X_validation', 'Y_validation', 'sample_weights_validation')]:

        if xf and yf in data:

            n_points = data[xf]
            data[yf] = data[yf].flatten()
            if sw in data:
                if data[sw].ndim > 1 and data[sw].shape == (n_points, 1):
                    data[sw] = data[sw].flatten()
            else:
                data[sw] = np.ones(n_points, dtype = np.float)

    return data


def cast_numeric_fields(data, cast_type = np.float):

    for field in NUMERIC_FIELDS_NAMES:
        if field in data:
            data[field] = data[field].astype(cast_type)

    return data


#### views  ####

def variable_summary(data, name):

    if type(name) is list:
        s_list = [variable_summary(data, n) for n in name]
        return '\n'.join(s_list)

    vtype = data['variable_types'][name]
    if vtype == 'ordinal':
        ordering = data['variable_orderings'][name]
    else:
        ordering = 'N/A'

    if name in data['partitions']:
        in_partition = 'True'
    else:
        in_partition = 'False'

    idx = data['variable_names'].index(name)
    n_samples = float(data['X'].shape[0])
    distinct_values, counts = np.unique(data['X'][:, idx], return_counts = True)
    props = np.array(counts, dtype = float) / float(n_samples)

    dist_text = ['(VALUE, COUNT, PROP)']
    for v, n, p in zip(distinct_values, counts, props):
        dist_text += ['(%s, %d, %1.1f%%)' % (v, n, 100.0 * p)]
    dist_text = '\n'.join(dist_text)

    if vtype == 'numeric':
        min_str = '%1.2f' % distinct_values[0]
        max_str = '%1.2f' % distinct_values[-1]
    else:
        min_str = str(distinct_values[0])
        max_str = str(distinct_values[-1])

    spacing = '\t\t'
    s = ['=' * 60]
    s += ['name\t:%s%s' % (spacing, name)]
    s += ['type\t:%s%s' % (spacing, vtype)]
    s += ['ordering:%s%s' % (spacing, ordering)]
    s += ['partition:%s%s' % (spacing, in_partition)]
    s += ['min value:%s%s' % (spacing, min_str)]
    s += ['max value:%s%s' % (spacing, max_str)]
    s += ['n_distinct:%s%d' % (spacing, len(distinct_values))]
    s += ['-' * 30]
    s += [dist_text]
    s += ['\n' + ('=' * 60)]
    s = '\n'.join(s)
    return s


def has_test_set(data):
    return 'X_test' in data and 'Y_test' in data


def has_validation_set(data):
    return 'X_validation' in data and 'Y_validation' in data


def has_sample_weights(data):
    return 'sample_weights' in data and np.any(data['sample_weights'] != 1.0)


#### add / remove variables ####

def rename_variable(data, name, new_name):
    assert name in data['variable_names']
    idx = data['variable_names'].index(name)

    data['variable_names'][idx] = new_name
    data['variable_types'][new_name] = data['variable_types'].pop(name)

    if name in data['variable_orderings']:
        data['variable_orderings'][new_name] = data['variable_orderings'].pop(name)

    if 'partitions' in data and name in data['partitions']:
        part_idx = data['partitions'].index(name)
        data['partitions'][part_idx] = new_name

    return data


def add_variable(data, values, name, variable_type, idx = None, is_partition = False, variable_ordering = None, test_values = None, validation_values = None):


    assert type(name) is str, \
        'name must be str'

    assert len(name) > 0, \
        'name must have at least 1 character'

    assert name not in data['variable_names'], \
        'data already contains a variable with name %s already exists' % name

    assert type(variable_type) is str, \
        'variable_type must be str'

    assert variable_type in VALID_VAR_TYPES, \
        'invalid variable type %s' % variable_type

    assert (not is_partition) or (variable_type == 'categorical'), \
        'partition variables must be categorical'

    n_variables = len(data['variable_names'])
    idx = n_variables if idx is None else int(idx)
    assert idx in range(n_variables + 1)

    if variable_type == 'ordinal':
        assert type(variable_ordering) is list
        assert len(variable_ordering) >= 1
        label_set = set(variable_ordering)

    # add values first
    for vals, field in [(values, 'X'), (test_values, 'X_test'), (validation_values, 'X_validation')]:

        if field in data:

            assert vals is not None
            vals = np.array(vals).flatten()
            assert len(vals) in (1, data[field].shape[0]), \
                'invalid shape for %s values (must be scalar or array with same length)' % field

            if variable_type == 'ordinal':
                assert set(vals).issubset(label_set)
            data[field] = np.insert(arr = data[field], values = vals, obj = idx, axis = 1)

    data['variable_names'].insert(idx, name)

    if 'variable_types' in data:
        data['variable_types'][name] = variable_type

    if 'partitions' in data and is_partition:
        data['partitions'].append(name)

    if 'variable_orderings' in data and variable_type == 'ordinal':
        data['variable_orderings'][name] = list(variable_ordering)

    assert check_data(data)
    return data


def remove_variable(data, name):

    #check that name exists (will throw ValueError otherwise)
    data = deepcopy(data)
    idx = data['variable_names'].index(name)

    # remove entries from feature matrix
    data['X'] = np.delete(data['X'], idx, axis = 1)

    #remove fields
    data['variable_names'].remove(name)

    if 'variable_types' in data:
        data['variable_types'].pop(name)

    if 'variable_orderings' in data and name in data['variable_orderings']:
        data['variable_orderings'].pop(name)

    if 'partitions' in data and name in data['partitions']:
        data['partitions'].remove(name)

    if has_test_set(data):
        data['X_test'] = np.delete(data['X_test'], idx, axis = 1)

    if has_validation_set(data):
        data['X_validation'] = np.delete(data['X_validation'], idx, axis = 1)

    return data

#### add /remove intercept ####


def get_intercept_index(data):
    try:
        return data['variable_names'].index(INTERCEPT_NAME)
    except ValueError:
        return -1


def has_intercept(data):
    idx = get_intercept_index(data)
    if idx == -1:
        return False
    else:
        assert check_intercept(data)
        return True


def check_intercept(data):

    idx = np.flatnonzero(np.array([INTERCEPT_NAME == v for v in data['variable_names']]))

    if len(idx) > 0:

        assert len(idx) == 1, \
            "X has multiple columns named %s" % INTERCEPT_NAME

        assert np.all(data['X'][:, idx] == 1, axis = 0), \
            "found %s at column %d but X[:, %d] != 1.0" % (INTERCEPT_NAME, idx, idx)

        if 'X_test' in data and 'Y_test' in data:
            assert np.all(data['X_test'][:, idx] == 1, axis = 0), \
                "found %s at column %d but X_test[:, %d] != 1.0" % (INTERCEPT_NAME, idx, idx)

        if 'X_validation' in data and 'Y_validation' in data:
            assert np.all(data['X_validation'][:, idx] == 1, axis = 0), \
                "found %s at column %d but X_validation[:, %d] != 1.0" % (INTERCEPT_NAME, idx, idx)

    return True


def add_intercept(data, idx = INTERCEPT_IDX):

    if not has_intercept(data):

        data = add_variable(data,
                            name = INTERCEPT_NAME,
                            variable_type = INTERCEPT_VAR_TYPE,
                            idx = idx,
                            values = 1.00,
                            test_values = 1.00,
                            validation_values = 1.00)

        assert check_intercept(data)

    return data


def remove_intercept(data):
    if has_intercept(data):
        return remove_variable(data, INTERCEPT_NAME)
    else:
        return data


#### names and indices ####

def get_index_of(data, names):
    if type(names) is list:
        return list(map(lambda n: get_index_of(data, n), names))
    else:
        return data['variable_names'].index(names)


def get_variable_names(data, include_partitions = False, include_intercept = False):

    var_names = list(data['variable_names'])

    if not include_intercept and has_intercept(data):
        var_names.pop(get_intercept_index(data))

    if not include_partitions and 'partitions' in data:
        var_names = [n for n in var_names if n not in data['partitions']]

    return var_names


def get_variable_indices(data, include_partitions = False, include_intercept = False):
    var_names = get_variable_names(data, include_partitions, include_intercept)
    idx = np.array([data['variable_names'].index(n) for n in var_names], dtype = int)
    return idx


def get_partition_indices(data):
    partition_idx = np.array([data['variable_names'].index(n) for n in data['partitions']])
    return partition_idx


def get_partition_names(data):
    return list(data['partitions'])


#### redundancy in X ###


def check_full_rank(data):

    variable_idx = get_variable_indices(data, include_intercept = False, include_partitions = False)
    n_variables = len(variable_idx)

    X = np.array(data['X'][:, variable_idx], dtype = np.float_)
    assert n_variables == np.linalg.matrix_rank(X), 'X contains redundant features'

    if has_test_set(data):
        X_test = np.array(data['X_test'][: variable_idx], dtype = np.float_)
        assert n_variables == np.linalg.matrix_rank(X_test), 'X_test contains redundant features'

    if has_validation_set(data):
        X_validation = np.array(data['X_validation'][: variable_idx], dtype = np.float_)
        assert n_variables == np.linalg.matrix_rank(X_validation), 'X_validation contains redundant features'

    return True


def drop_trivial_variables(data):
    """
    drops unusuable features (due to trivial features etc)
    :param data:
    :return:
    """
    trivial_idx = np.all(data['X'] == data['X'][0, :], axis=0)

    if any(trivial_idx):

        drop_idx = np.flatnonzero(trivial_idx)
        variables_to_drop = [data['variable_names'][k] for k in drop_idx]
        for var_name in variables_to_drop:
            data = remove_variable(data, var_name)

    return data


def list_duplicate_variables(X):
    """

    :param X:
    :return:
    """
    d = X.shape[1]
    duplicates = []
    for (j, k) in itertools.combinations(range(d), 2):
        if np.array_equal(X[:, j], X[:, k]):
            duplicates.append((j, k))

    return duplicates


#### testing


def get_common_row_indices(A, B):
    """
    A and B need to only contain unique rows
    :param A:
    :param B:
    :return:
    """

    nrows, ncols = A.shape

    dtype = {
        'names': ['f{}'.format(i) for i in range(ncols)],
        'formats': ncols * [A.dtype]
        }

    common_rows = np.intersect1d(A.view(dtype), B.view(dtype))
    if common_rows.shape[0] == 0:
        common_idx = np.empty((0, 2), dtype = np.dtype('int'))
    else:
        # common_rows = common_rows.view(A.dtype).reshape(-1, ncols)
        a_rows_idx = np.flatnonzero(np.isin(A.view(dtype), common_rows))
        b_rows_idx = np.flatnonzero(np.isin(B.view(dtype), common_rows))
        common_idx = np.column_stack((a_rows_idx, b_rows_idx))

    return common_idx


def sample_test_data(data, max_features = 2, n_pos = 50, n_neg = 50, n_conflict = 10, remove_duplicates = True):

    assert check_data(data)

    # minimal sanity checks for sizes
    pos_ind = data['Y'] == 1
    neg_ind = ~pos_ind
    assert n_pos > 0
    assert n_neg > 0
    assert max_features > 0
    assert n_pos <= np.sum(pos_ind)
    assert n_neg <= np.sum(neg_ind)
    assert n_conflict <= min(n_pos, n_neg)
    max_features = min(data['X'].shape[1], max_features)

    # drop features (including intercept)
    feature_idx = get_variable_indices(data, include_intercept = False, include_partitions = False)
    feature_idx = feature_idx[np.arange(max_features)]
    if has_intercept(data):
        intercept_idx = get_intercept_index(data)
        feature_idx = np.insert(feature_idx, INTERCEPT_IDX, intercept_idx)

    variable_names = [data['variable_names'][j] for j in feature_idx]
    X = data['X'][:, feature_idx]
    Y = data['Y']
    pos_ind = Y == 1
    neg_ind = ~pos_ind
    X_pos = X[pos_ind, ]
    X_neg = X[neg_ind, ]

    # drop duplicates
    if remove_duplicates:

        XY = np.hstack((X, Y.reshape(X.shape[0], 1)))
        XY = np.unique(XY, axis = 0)
        X = XY[:, range(XY.shape[1]-1)]
        Y = XY[:, XY.shape[1]-1]

        pos_ind = Y == 1
        neg_ind = ~pos_ind
        X_pos = X[pos_ind,]
        X_neg = X[neg_ind,]

        # adjust sample sizes
        if n_pos > X_pos.shape[0]:
            print_log("only %d points with y[i] = +1 " % X_pos.shape[0])
            print_log("setting n_pos = %d" % X_pos.shape[0])
            n_pos = min(n_pos, X_pos.shape[0])

        if n_neg > X_neg.shape[0]:
            print_log("only %d points with y[i] = -1 " % X_neg.shape[0])
            print_log("setting n_neg = %d" % X_neg.shape[0])
            n_neg = min(n_neg, X_neg.shape[0])

        if n_conflict > n_pos:
            print_log("setting n_conflict = %d" % min(n_pos, X_neg.shape[0]))
            n_conflict = min(n_conflict, n_pos)

    # ensure that we have at exactly n_conflict rows with identical features but opposite labels
    if n_conflict > 0:

        common_idx = get_common_row_indices(X_pos, X_neg)
        n_common = common_idx.shape[0]

        # TODO: adjust code so that it can drop negative points to satisfy desired sample sizes
        # TODO: adjust code so that it can duplicate negative points to satisfy desired sample sizes
        if n_common > n_conflict:

            if X_neg.shape[0] > X_pos.shape[0]:

                # drop negative points
                neg_idx_to_drop = common_idx[n_conflict:, 1]
                X_neg = np.delete(X_neg, neg_idx_to_drop, axis = 0)

                if n_neg > X_neg.shape[0]:
                    print_log("only %d points with y[i] = -1 " % X_neg.shape[0])
                    print_log("setting n_neg = %d" % X_neg.shape[0])
                    n_neg = min(n_neg, X_neg.shape[0])

            else:

                pos_idx_to_drop = common_idx[n_conflict:, 0]
                X_pos = np.delete(X_pos, pos_idx_to_drop, axis = 0)

                if n_pos > X_pos.shape[0]:
                    print_log("only %d points with y[i] = +1 " % X_pos.shape[0])
                    print_log("setting n_pos = %d" % X_pos.shape[0])
                    n_pos = min(n_pos, X_pos.shape[0])

        elif n_common < n_conflict:

            #add n_conflict - n_common points from X_pos to X_neg
            n_required = n_conflict - n_common
            all_pos_idx = np.arange(X_pos.shape[0])
            distinct_pos_idx = np.delete(all_pos_idx, common_idx[:,0])
            assert len(distinct_pos_idx) >= n_required

            pos_idx_to_flip = distinct_pos_idx[0:n_required]
            X_neg = np.vstack((X_neg, X_pos[pos_idx_to_flip, :]))

        common_idx = get_common_row_indices(X_pos, X_neg)
        assert common_idx.shape[0] == n_conflict

    # choose samples
    X_pos = X_pos[0:n_pos, :]
    X_neg = X_neg[0:n_neg, :]
    Y_pos = np.repeat(1.0, n_pos).reshape((n_pos, 1))
    Y_neg = np.repeat(-1.0, n_neg).reshape((n_neg, 1))

    X = np.vstack((X_pos, X_neg))
    Y = np.vstack((Y_pos, Y_neg)).flatten()

    # check properties of X and Y
    assert X.shape[0] == n_pos + n_neg
    assert Y.shape[0] == n_pos + n_neg
    assert np.sum(Y == 1) == n_pos
    assert np.sum(Y == -1) == n_neg
    assert len(variable_names) == X.shape[1]

    if has_intercept(data):
        assert X.shape[1] == max_features + 1
    else:
        assert X.shape[1] == max_features

    # create final dataset
    new_data = deepcopy(data)
    new_data['X'] = X
    new_data['Y'] = Y
    new_data['variable_names'] = variable_names
    new_data['sample_weights'] = np.ones_like(Y)

    if 'variable_types' in data:
        new_data['variable_types'] = {k:data['variable_types'][k] for k in variable_names}

    if has_test_set(data):
        new_data['X_test'] = data['X_test'][:, feature_idx]
        new_data['sample_weights_test'] = np.ones_like(data['Y_test'])

    if has_validation_set(data):
        new_data['X_validation'] = data['X_validation'][:, feature_idx]
        new_data['sample_weights_validation'] = np.ones_like(data['Y_validation'])

    # check output dataset
    assert check_data(new_data, ready_for_training = True)
    return new_data


def sample_test_data_with_partitions(data, n_samples = 1000, scramble = True, random_seed = None):
    """
    returns a sample of the data points so that the sampled data contains the same partitions as the original partitions
    """

    group_idx = get_partition_indices(data)
    n_total = data['X'].shape[0]
    n_samples = min(n_samples, n_total)

    Y = np.array(data['Y'])
    pos_ind = Y == 1
    neg_ind = ~pos_ind
    Y[pos_ind] = data['outcome_label_positive']
    Y[neg_ind] = data['outcome_label_negative']

    Y = np.array(Y, dtype = str).reshape((n_total,1))
    G = np.array(data['X'][:, group_idx], dtype = str)
    Z = np.append(G, Y, axis = 1)

    distinct_groups, distinct_idx, counts = np.unique(Z, axis = 0, return_inverse = True, return_counts = True)
    n_distinct_groups = len(distinct_groups)

    U = [np.flatnonzero(distinct_idx == u).tolist() for u in range(n_distinct_groups)]

    if scramble:
        if random_seed is not None:
            np.random.seed(random_seed)

        for g in range(n_distinct_groups):
            U[g] = np.random.permutation(U[g]).tolist()

    row_idx = []
    s = 0
    g = 0

    while s < n_samples:

        if len(U[g]) > 0:
            row_idx += [U[g].pop()]
            s += 1

        g += 1
        if g == n_distinct_groups:
            g = 0

    assert len(row_idx) == n_samples

    new_data = deepcopy(data)
    new_data['X'] = data['X'][row_idx, :]
    new_data['Y'] = data['Y'][row_idx]
    new_data['sample_weights'] = np.ones_like(new_data['Y'])

    if has_validation_set(data):
        new_data['X_validation'] = np.array(new_data['X_validation'])
        new_data['Y_validation'] = np.array(new_data['Y_validation'])
        new_data['sample_weights_validation'] = np.ones_like(data['Y_validation'])

    if has_test_set(data):
        new_data['X_test'] = np.array(new_data['X_test'])
        new_data['Y_test'] = np.array(new_data['Y_test'])
        new_data['sample_weights_test'] = np.ones_like(data['Y_test'])

    assert check_data(new_data, ready_for_training = False)
    return new_data
