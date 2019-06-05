import numpy as np
from .data import check_data, add_variable, remove_variable, has_test_set, has_validation_set
from .rule_naming import convert_rule_name_to_feature_category, SET_START, SET_SEPARATOR, SET_END, RELATION_NAMES, COMPLEMENT_NAMES
from .rule_mining import generate_rules_from_bin, Bin, parse_bin

BOOLEAN_FEATURE_LABEL_TRUE = "Yes"
BOOLEAN_FEATURE_LABEL_FALSE = "No"

CONVERSION_SETTINGS_NUMERIC_TO_RULES = {
    'bin_generation': 'equal_width',
    #options include:
    #'all'
    #'equal_width'
    #'percentile'
    #'custom'
    ##
    'custom_bins': ('[-inf,2]', '(2,10]', '(10,50], (50,90]'),
    # user-defined bins used when bin_generation = custom
    #
    'n_bins': 5,
    # if n_bins > np.unique(values) then all possible rules will be generated
    #
    'percentiles':[0.05, 0.1, 0.25, 0.5, 0.75, 0.90, 0.95],
    # percentiles (used if rule_generation = 'percentiles'
    'bin_type': 'auto',
    # auto/int/float
    'bin_trimming': 'rounding'
    }


#### categorical -> categorical


def collapse_categorical(data, name, collapse_dict):

    to_collapse = []
    for k in collapse_dict:
        to_collapse += collapse_dict[k]
    assert len(to_collapse) == len(set(to_collapse)), 'cannot collapse multiple variables to same variable'

    idx = data['variable_names'].index(name)
    values = data['X'][:, idx]
    new_values = np.array(values)

    labels = set(values)
    to_keep = list(labels.difference(set(to_collapse)))
    new_labels = to_keep + list(collapse_dict.keys())

    for k in collapse_dict:
        change_idx = np.isin(values, collapse_dict[k])
        new_values[change_idx] = k

    assert set(new_values) == set(new_labels), 'conversion error'
    data['X'][:, idx] = new_values

    if has_test_set(data):
        values = data['X_test'][:, idx]
        for k in collapse_dict:
            change_idx = np.isin(values, collapse_dict[k])
            new_values[change_idx] = k
        assert set(new_values).issubset(set(to_keep + list(collapse_dict.keys())))
        data['X_test'][:, idx] = new_values

    if has_validation_set(data):
        values = data['X_validation'][:, idx]
        for k in collapse_dict:
            change_idx = np.isin(values, collapse_dict[k])
            new_values[change_idx] = k
        assert set(new_values).issubset(set(to_keep + list(collapse_dict.keys())))
        data['X_validation'][:, idx] = new_values

    return data


####  from categorical, ordinal -> boolean


def convert_categorical_to_boolean(data, name, test_labels, invert = False):

    assert data['variable_types'][name] == 'categorical'

    idx = data['variable_names'].index(name)

    boolean_values = np.isin(data['X'][:, idx], test_labels, invert = invert)
    data['X'][:, idx] = boolean_values

    if has_test_set(data):
        boolean_values = np.isin(data['X_test'][:, idx], test_labels, invert = invert)
        data['X_test'][:, idx] = boolean_values

    if has_validation_set(data):
        boolean_values = np.isin(data['X_validation'][:, idx], test_labels, invert = invert)
        data['X_validation'][:, idx] = boolean_values

    data['variable_types'][name] = 'boolean'

    return data


def convert_categorical_to_rules(data, name, conversion_dict, prepend_name = True):

    assert data['variable_types'][name] == 'categorical'

    new_values = {
        'train': {},
        'validation': {},
        'test': {}
        }

    idx = data['variable_names'].index(name)
    values = data['X'][:, idx]

    found_test_set = has_test_set(data)
    found_validation_set = has_validation_set(data)

    for k, c in conversion_dict.items():
        new_values['train'][k] = np.isin(values, c)

        if found_validation_set:
            new_values['validation'][k] = np.isin(data['X_validation'][:, idx], c)
        else:
            new_values['validation'][k] = None

        if found_test_set:
            new_values['test'][k] = np.isin(data['X_test'][:, idx], c)
        else:
            new_values['test'][k] = None

    # remove categorical variable
    data = remove_variable(data, name)

    # add new variables
    new_names = list(new_values['train'].keys())
    if prepend_name:
        new_names = list(map(lambda n: get_name_for_categorical_indicator(name, labels = n), new_names))

    for var_name in new_names:
        data = add_variable(data,
                            name = var_name,
                            variable_type = 'boolean',
                            idx = idx,
                            values = new_values['train'][k],
                            test_values = new_values['test'][k],
                            validation_values = new_values['validation'][k])

    assert check_data(data)
    return data


def get_name_for_categorical_indicator(name, labels, complement = False):

    if type(labels) is not list:
        labels = [labels]

    assert len(labels) == len(set(labels)), 'labels must be distinct'

    if len(labels) > 1:
        relation_type = 'in'
        category_str = '%s%s%s' % (SET_START, SET_SEPARATOR.join(labels), SET_END)
    else:
        relation_type = 'is'
        category_str = labels[0]

    if complement:
        relation_str = COMPLEMENT_NAMES[relation_type]
    else:
        relation_str = RELATION_NAMES[relation_type]

    rule_name = '%s%s%s' % (name, relation_str, category_str)
    return rule_name


def convert_ordinal_to_rules(data, name, conversion_dict, prepend_name = True):

    assert data['variable_types'][name] == 'ordinal'

    # make sure that indicator sets are effectively bins
    orderings = data['variable_orderings'][name]
    ind_labels = conversion_dict.values()
    for labels in ind_labels:
        values = np.array([orderings.index(v) for v in labels])
        if len(values) > 1:
            assert np.all(values == np.arange(np.min(values), np.max(values) + 1, dtype = int))

    data['variable_orderings'].pop(name)
    data['variable_types'][name] = 'categorical'
    data = convert_categorical_to_rules(data, name, conversion_dict, prepend_name)
    return data


####  from boolean/numeric -> categorical


def convert_boolean_to_categorical(data, name):
    assert data['variable_types'][name] == 'boolean'

    idx = data['variable_names'].index(name)
    values = data['X'][:, idx]
    new_name, labels = boolean_to_categorical(name, values)
    data['X'][:, idx] = labels

    if has_test_set(data):
        values = data['X_test'][:, idx]
        _, labels = boolean_to_categorical(name, values)
        data['X_test'][:, idx] = labels

    data['variable_types'][name] = 'categorical'
    return data


def boolean_to_categorical(name, values):

    name = str(name)
    assert len(name) > 0

    true_idx = values == 0
    false_idx = values == 1

    assert np.all(np.logical_or(true_idx, false_idx))

    true_label = BOOLEAN_FEATURE_LABEL_TRUE
    false_label = BOOLEAN_FEATURE_LABEL_FALSE

    labels = np.repeat(true_label, len(values))
    labels[false_idx] = false_label

    return name, labels


def numeric_to_categorical(name, values, conversion_settings = CONVERSION_SETTINGS_NUMERIC_TO_RULES):
    rules = numeric_to_rules(name, values, conversion_settings)
    rule_names = list(rules.keys())
    labels = list(map(convert_rule_name_to_feature_category, rule_names))
    indices = np.repeat(-1, values.shape[0])
    for k in range(len(rule_names)):
        rule_idx = rules[rule_names[k]]
        indices[rule_idx] = k
    assert (np.all(indices >= 0))
    categories = np.array([labels[i] for i in indices])
    return name, categories


def numeric_to_rules(name, values, settings = CONVERSION_SETTINGS_NUMERIC_TO_RULES):

    #fill in missing settings with default values
    settings = {k:settings[k] if k in settings else CONVERSION_SETTINGS_NUMERIC_TO_RULES[k] for k in CONVERSION_SETTINGS_NUMERIC_TO_RULES}

    # determine bin type
    assert settings['bin_type'] in ('auto', 'int', 'float'), "invalid bin_type: '%s'" % settings['bin_type']
    if settings['bin_type'] == 'auto':
        bin_type = 'int' if is_integer(values) else 'float'
    else:
        bin_type = settings['bin_type']
    values = values.astype(bin_type)

    # value characteristics
    distinct_values = np.unique(values)
    min_value = distinct_values[0]
    max_value = distinct_values[-1]

    # generate rules
    rules = {}
    # custom generation
    if settings['bin_generation'] == 'custom':
        get_rules = lambda bin: generate_rules_from_bin(bin, name, values, return_complement = False)
        for s in settings['custom_bins']:
            rules.update(get_rules(bin = parse_bin(s, bin_type)))
        return rules

    # exhaustive bin generation
    if settings['bin_generation'] == 'all':
        get_rules = lambda bin: generate_rules_from_bin(bin, name, values, return_complement=False)
        for v in distinct_values:
            rules.update(get_rules(Bin(start=v, end=v, type=bin_type, closure='[]')))
        return rules

    # equal width
    if settings['bin_generation'] == 'equal_width':
        _, bin_edges = np.histogram(a = values, bins = settings['n_bins'])

    # percentiles
    if settings['bin_generation'] == 'percentiles':
        bin_edges = np.percentile(a = values, q = settings['percentiles'], interpolation = 'midpoint')

    get_rules = lambda bin: generate_rules_from_bin(bin, name, values, return_complement = False)

    n_edges = len(bin_edges)
    for i in range(n_edges - 1):
        if i == 0:
            b = Bin(start = min_value, end = bin_edges[i + 1], type = bin_type, closure = '[)')
        elif i == (n_edges - 2):
            b = Bin(start = bin_edges[i], end = max_value, type = bin_type, closure = '[]')
        else:
            b = Bin(start = bin_edges[i], end = bin_edges[i + 1], type = bin_type, closure = '[)')
        rules.update(get_rules(bin = b))
    return rules

