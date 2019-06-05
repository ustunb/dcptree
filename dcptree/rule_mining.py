import numpy as np
import numpy_indexed as npi
from itertools import tee, combinations
from dcptree.data import remove_intercept
from dcptree.rule_naming import *


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


DEFAULTS_BOOLEAN = {
    'remove_trivial_rules': True,
    'include_complements': True,
}

DEFAULTS_CATEGORICAL = {
    'remove_trivial_rules': True,
    'include_complements': True,
    #
    'rule_generation': 'limited',
    'custom_categories': '',
    'max_subset_size': 3,
    'min_support': 0,
    'hard_limit': 100,
    'use_not_for_complements_of_one_category_rules': True,
    'treat_two_category_variables_as_boolean': False,
}

DEFAULTS_NUMERIC = {
    'bin_generation': 'equal_width',
    #options include:
    #'all'
    #'equal_width'
    #'percentile'
    #'custom'
    #'auto', 'fd', 'doane', 'scott', 'rice', 'sturges', 'sqrt' (see np.histogram)
    ##
    'custom_bins': ('[-inf,2]', '(2,10]', '(10,50], (50,90]'),
    # user-defined bins used when bin_generation = custom
    #
    'n_bins': 5,
    # if n_bins > np.unique(values) then all possible rules will be generated
    #
    'percentiles':[0.05, 0.1, 0.25, 0.5, 0.75, 0.90, 0.95],
    # percentiles (used if rule_generation = 'percentiles'
    #
    'bin_type': 'auto',
    # auto/int/float
    #
    'nested_bins': True,
    # if True then the bins are nested, otherwise bins not nested
    #
    'include_complements': True,
    # we force include_complements = True when 'bin_generation' = all and nested_bins = True
    # we force include_complements = False when 'bin_generation' = all and nested_bins = False
    #
    'bin_trimming': 'rounding',
    # 'none': bins are not trimmed
    # 'rounding': bins are trimmed so that the edges points are rounded values up/down
    # 'nearest': bins are trimmed so that the end points represent the closest value in the data
    #
    'names_are_one_sided': True,
    # if True, then we set rule_name as 1[V >= t] when the bin is (-Inf,t] or (min(V),t]
    #
    'names_use_equals_at_edges': True,
    # if True, then we use 'eq' for the rule_name when the bin is at a single edge value such as
    # V in [v_min,v_min] where v_min = min(V)
    # V in [v_max,v_max] where v_min = max(V)
}

DEFAULTS_ORDINAL = DEFAULTS_NUMERIC
DEFAULTS_ORDINAL['bin_type'] = 'int'
DEFAULTS_ORDINAL['names_use_equals_at_edges'] = True

DEFAULTS_MAIN = {
    'boolean': DEFAULTS_BOOLEAN,
    'categorical': DEFAULTS_CATEGORICAL,
    'numeric': DEFAULTS_NUMERIC,
    'ordinal': DEFAULTS_ORDINAL,
    'include_complements': True,
    'remove_trivial_rules': True,
    'remove_duplicate_rules': True,
    'minimum_support': 0.0,
    'variable_specific': {},
}


#### MAIN FUNCTIONS FOR RULE GENERATION ####


def generate_rules(data, settings = {}):

    data = remove_intercept(data)
    settings = {k:settings[k] if k in settings else DEFAULTS_MAIN[k] for k in DEFAULTS_MAIN}
    n_values = data['X'].shape[0]
    coverage_min, coverage_max = 0, n_values
    coverage_threshold = np.ceil(n_values * float(settings['minimum_support']))
    coverage_threshold = min(max(coverage_threshold, coverage_min), coverage_max)

    rules = {}
    for var_name in data['variable_names']:

        print('generating rules for feature %s' % var_name)

        var_idx = data['variable_names'].index(var_name)
        var_type = data['variable_types'][var_name]

        if var_name in settings['variable_specific']:
            var_settings = settings['variable_specific'][var_name]
        else:
            var_settings = settings[var_type]

        if var_type == 'boolean':

            var_rules = generate_rules_from_boolean(name=var_name,
                                                    values=np.array(data['X'][:, var_idx]),
                                                    settings = var_settings)

        elif var_type == 'categorical':
            var_rules = generate_rules_from_categorical(name=var_name,
                                                        values=np.array(data['X'][:, var_idx]),
                                                        settings = var_settings)
        elif var_type == 'numeric':

            var_rules = generate_rules_from_numeric(name=var_name,
                                                    values=np.array(data['X'][:, var_idx]),
                                                    settings = var_settings)


        elif var_type == 'ordinal':

            var_rules = generate_rules_from_ordinal(name = var_name,
                                                    values = np.array(data['X'][:, var_idx]),
                                                    ordering = data['variable_orderings'][var_name],
                                                    settings = var_settings)
        else:
            raise ('unsupported variable type: %s' % var_type)

        print('generated %d rules for feature %s' % (len(var_rules), var_name))

        #post processing for each variable
        if settings['remove_trivial_rules']:
            n_before = len(var_rules)
            var_rules = remove_trivial_rules(var_rules)
            n_dropped = n_before - len(var_rules)
            if n_dropped > 0:
                print("\t dropped %d of %d rules because they were trivial" % (n_dropped, n_before))

        if settings['remove_duplicate_rules']:
            n_before = len(var_rules)
            var_rules = remove_duplicate_rules(var_rules)
            n_dropped = n_before - len(var_rules)
            if n_dropped > 0:
                print("\t dropped %d of %d rules because they were duplicates of other rules" % (n_dropped, n_before))

        if coverage_threshold > 0:
            n_before = len(var_rules)
            var_rules = remove_rules_with_low_coverage(var_rules, coverage_threshold)
            n_dropped = n_before - len(var_rules)
            if n_dropped > 0:
                print("\t dropped %d of %d rules because of low support" % (n_dropped, n_before))

        if settings['include_complements']:
            n_before = len(var_rules)
            var_rules = add_missing_complements(var_rules)
            n_added = len(var_rules) - n_before
            if n_added > 0:
                print("\t added %d rules (missing complements)" % n_added)

        if len(var_rules) > 0:
            rules[var_name] = var_rules
            print('adding %d rules to rule list for feature %s' % (len(var_rules), var_name))

        print("\n")

    return rules


def generate_rules_from_boolean(name, values, settings=DEFAULTS_BOOLEAN):

    settings = {k: settings[k] if k in settings else DEFAULTS_BOOLEAN[k] for k in DEFAULTS_BOOLEAN}
    distinct_values, distinct_counts = np.unique(values, return_counts= True)
    n_distinct_values = len(distinct_values)

    rules = {}
    #basic input validation
    if n_distinct_values == 1:
        return rules
    elif n_distinct_values != 2:
        raise ValueError('variable has too many values to be cast to boolean; recode as one-vs-all')

    # figure out value type
    try:
        float(values[0])
        value_type = 'bool'
        values = values.astype(np.bool)
    except:
        try:
            str(values[0])
            value_type = 'str'
        except:
            raise ValueError('incorrect value type (read: %r)' % type(values[0]))

    if value_type == 'str':
        rule_idx, comp_idx = np.argmax(distinct_counts), np.argmin(distinct_counts)
        rule_name, comp_name = distinct_values[rule_idx], distinct_values[comp_idx]
        values = np.in1d(values, rule_name)
        rule_name = '%s%s%s' % (name, RELATION_NAMES['is'], rule_name)
        comp_name = '%s%s%s' % (name, RELATION_NAMES['is'], comp_name)
    else:
        rule_name = name
        comp_name = get_complement_name(rule_name)

    #add complement to rule set
    rules[rule_name] = values
    if settings['include_complements']:
        rules[comp_name] = ~values

    return rules


def generate_rules_from_categorical(name, values, settings = DEFAULTS_CATEGORICAL):

    settings = {k:settings[k] if k in settings else DEFAULTS_CATEGORICAL[k] for k in DEFAULTS_CATEGORICAL}
    n_values = len(values)
    category_names, value_ids = np.unique(values, return_inverse=True)
    n_categories = len(category_names)
    category_ids = np.arange(n_categories)


    if settings['rule_generation'] == 'custom':
        #todo implement ability to generate custom subsets/supersets
        raise NotImplementedError()

    if settings['treat_two_category_variables_as_boolean'] and n_categories < 3:
        return generate_rules_from_boolean(name, values)

    if settings['rule_generation'] == 'all':
        max_subset_size = n_categories
        min_cover_size = 0
    else:
        max_subset_size = float(settings['max_subset_size'])  # if user has used int
        assert max_subset_size == float('inf') or (max_subset_size.is_integer() and max_subset_size >= 1)
        min_support = float(settings['min_support'])
        assert 0.0 <= min_support <= 1.0
        min_cover_size = int(np.ceil(min_support * n_values))

    if settings['remove_trivial_rules']:
        max_subset_size = min(max_subset_size, n_categories - 1)
        min_cover_size = min(min_cover_size, n_values - (n_categories-1))
        min_cover_size = max(min_cover_size, 1)
    else:
        max_subset_size = min(max_subset_size, n_categories)
        min_cover_size = min(min_cover_size, n_values)
        min_cover_size = max(min_cover_size, 0)

    max_subset_size = int(max_subset_size)
    min_cover_size = int(min_cover_size)

    #generate rules
    rules = {}
    cat_separator = "%s" % SET_SEPARATOR
    if settings['include_complements']:

        max_subset_size = int(min(max_subset_size, np.floor(n_categories/2)))
        for idx in category_ids:

            rule_cat = category_names.take(idx)
            rule_name = '%s%s%s' % (name, RELATION_NAMES['is'], rule_cat)
            if settings['use_not_for_complements_of_one_category_rules']:
                comp_name = '%s%s%s' % (name, COMPLEMENT_NAMES['is'], rule_cat)
            else:
                comp_cats = np.setdiff1d(category_names, rule_cat)
                comp_name = '%s%s%s%s%s' % (name, RELATION_NAMES['in'], SET_START, cat_separator.join(comp_cats), SET_END)

            rule_values = np.in1d(value_ids, idx)
            rules[rule_name] = rule_values
            rules[comp_name] = ~rule_values

        if max_subset_size >= 2:

            new_rules = {}
            for subset_size in range(2, max_subset_size + 1):
                for idx in combinations(category_ids, subset_size):

                    rule_cats = category_names.take(idx)
                    rule_name = cat_separator.join(rule_cats)

                    comp_cats = np.setdiff1d(category_names, rule_cats)
                    comp_name = cat_separator.join(comp_cats)

                    rule_values = np.in1d(value_ids, idx)
                    new_rules[rule_name] = rule_values
                    new_rules[comp_name] = ~rule_values

            #rename rules
            current_names = new_rules.keys()
            rule_header = '%s%s%s' % (name, RELATION_NAMES['in'], SET_START)
            for old_name in current_names:
                new_name = '%s%s%s' % (rule_header, old_name, SET_END)
                rules[new_name] = new_rules.pop(old_name)

    else:

        rules = {}
        for subset_size in range(1, max_subset_size+1):

            if subset_size == 1:
                rule_header = '%s%s' % (name, RELATION_NAMES['is'])
                rule_footer = ''
            else:
                rule_header = '%s%s%s' % (name, RELATION_NAMES['in'], SET_START)
                rule_footer = SET_END

            for idx in combinations(category_ids, subset_size):
                cat_name = cat_separator.join(category_names.take(idx))
                rule_name = '%s%s%s' % (rule_header, cat_name, rule_footer)
                rules[rule_name] = np.in1d(value_ids, idx)

    #filter rules by cover size
    if min_cover_size > 1:
        rule_names = rules.keys()
        for rule_name in rule_names:
            if np.sum(rules[rule_name]) < min_cover_size:
                rules.pop(rule_name, None)

    return rules


def generate_rules_from_ordinal(name, values, ordering, settings=DEFAULTS_ORDINAL):
    #values = integers between 1, n_levels representative of the ordering
    #levels = strings correspond to levels

    lvls_to_vals = {l:ordering.index(l) for l in ordering}
    vals_to_lvls = dict((v, k) for k, v in lvls_to_vals.iteritems())
    settings = {k:settings[k] if k in settings else DEFAULTS_ORDINAL[k] for k in DEFAULTS_ORDINAL}

    if settings['bin_generation'] == 'custom':
        numeric_bins = []
        for b in settings['custom_bins']:
            try:
                numeric_bins.append(ordinal_bin_to_numeric_bin(ordinal_bin=b, ordering=ordering))
            except:
                numeric_bins.append(b)
        settings['custom_bins'] = numeric_bins

    # if values are text then convert them to text
    try:
        values = values.astype('int')
    except ValueError:
        values = np.array([lvls_to_vals[v] for v in values], dtype='int')

    rules = generate_rules_from_numeric(name=name, values=values, settings=settings)

    # replace numeric values in rule names with values
    numeric_names = rules.keys()
    for numeric_name in numeric_names:
        ordinal_name = numeric_name_to_ordinal_name(numeric_name, vals_to_lvls)
        rules[ordinal_name] = rules.pop(numeric_name)

    return rules


def generate_rules_from_numeric(name, values, settings=DEFAULTS_NUMERIC):

    #fill in missing settings with default values
    settings = {k:settings[k] if k in settings else DEFAULTS_NUMERIC[k] for k in DEFAULTS_NUMERIC}

    # determine bin type
    assert settings['bin_type'] in ('auto', 'int', 'float'), "invalid bin_type: '%s'" % settings['bin_type']
    if settings['bin_type'] == 'auto':
        bin_type = 'int' if is_integer(values) else 'float'
    else:
        bin_type = settings['bin_type']
    values = values.astype(bin_type)

    # value characteristics
    distinct_values = np.unique(values)
    n_distinct = len(distinct_values)
    min_value = distinct_values[0]
    max_value = distinct_values[-1]
    lower_edge_value = float('-inf') if settings['names_are_one_sided'] else min_value
    upper_edge_value = float('inf') if settings['names_are_one_sided'] else max_value

    # override settings
    if settings['bin_generation'] in ('percentiles', 'equal_width'):
        if settings['bin_generation'] in 'percentiles':
            n_bins = len(np.unique(settings['percentiles']))
        elif settings['bin_generation'] == 'equal_width':
            n_bins = settings['n_bins']
        if n_bins > n_distinct:
            settings['bin_generation'] = 'all'

    # generate rules
    rules = {}
    get_rules = lambda bin: generate_rules_from_bin(bin,
                                                    name,
                                                    values,
                                                    return_complement=settings['include_complements'])
    # custom generation
    if settings['bin_generation'] == 'custom':
        for s in settings['custom_bins']:
            rules.update(get_rules(bin=parse_bin(s, bin_type)))
        return rules

    # exhaustive bin generation
    if settings['bin_generation'] == 'all':

        get_rules = lambda bin: generate_rules_from_bin(bin,
                                                        name,
                                                        values,
                                                        return_complement=False)

        if settings['nested_bins']:

            for (v, w) in pairwise(distinct_values):

                if v == min_value and settings['names_use_equals_at_edges']:
                    rules.update(get_rules(Bin(start=v, end=v, type=bin_type, closure='[]')))
                else:
                    rules.update(get_rules(Bin(start=lower_edge_value, end=v, type=bin_type, closure='[]')))

                if w == max_value and settings['names_use_equals_at_edges']:
                    rules.update(get_rules(Bin(start=w, end=w, type=bin_type, closure='[]')))
                else:
                    rules.update(get_rules(Bin(start=w, end=upper_edge_value, type=bin_type, closure='[]')))

        else:

            for v in distinct_values:
                rules.update(get_rules(Bin(start=v, end=v, type=bin_type, closure='[]')))

        return rules

    # data-driven bin generation
    if settings['bin_generation'] == 'equal_width':
        _, bin_edges = np.histogram(a=values, bins=settings['n_bins'])
    elif settings['bin_generation'] == 'percentiles':
        bin_edges = np.percentile(a=values, q=settings['percentiles'], interpolation='midpoint')
    else:
        _, bin_edges = np.histogram(a=values, bins=settings['bin_generation'])

    bin_edges = np.unique(bin_edges)
    n_bins = len(bin_edges)

    if n_bins == 1 and n_distinct >= 3:
        if np.all(bin_edges == min_value):
            bin_edges = np.array([np.min(values[values > min_value])])
        elif np.all(bin_edges == max_value):
            bin_edges = np.array([np.max(values[values < max_value])])

    if settings['bin_trimming'] == 'none':

        if settings['nested_bins']:
            for i in range(n_bins):
                if bin_edges[i] > min_value:
                    b = Bin(start=lower_edge_value, end=bin_edges[i], type=bin_type, closure='[)')
                    rules.update(get_rules(bin=b))

            b = Bin(start=bin_edges[n_bins-1], end=upper_edge_value, type=bin_type, closure='[]')
            rules.update(get_rules(bin=b))

        else:  # distinct_rules

            for i in range(n_bins):
                bin_closure = '[)' if i < (n_bins-1) else '[]'
                b = Bin(start=bin_edges[i], end=bin_edges[i+1], type=bin_type, closure=bin_closure)
                rules.update(get_rules(bin=b))

        return rules


    if settings['bin_trimming'] == 'nearest':
        edge_values = get_surrounding_points(search_values=bin_edges, point_set=distinct_values)
        bin_starts, bin_ends = zip(*edge_values)
        bin_starts, bin_ends = np.array(bin_starts), np.array(bin_ends)

    elif settings['bin_trimming'] == 'rounding':
        bin_starts = np.maximum(min_value, np.minimum(np.floor(bin_edges), bin_edges-1))
        bin_ends = np.minimum(max_value, np.ceil(bin_edges))

    if settings['nested_bins']:

        for k in range(n_bins):

            rule_bin = Bin(start=lower_edge_value, end=bin_starts[k], type=bin_type, closure='[]')
            rule_name = name + rule_bin.rule_string()
            rule_values = np.array(rule_bin.contains(values))
            rules[rule_name] = rule_values

            if settings['include_complements']:
                comp_bin = Bin(start=bin_ends[k], end=upper_edge_value, type=bin_type, closure='[]')
                comp_name = name + comp_bin.rule_string()
                comp_values = np.array(comp_bin.contains(values))
                rules[comp_name] = comp_values


    else:#if distinct bins
        # rules         = [17-31, 32-46, 47-60, 61-75, 76-90]
        # complements   = [32-90, not 32-46, not 47-60, not 61-75, 17-75]

        # first bin
        rule_bin = Bin(start=bin_starts[0], end=bin_ends[0], type=bin_type, closure='[]')
        rule_name = name + rule_bin.rule_string()
        rule_values = np.array(rule_bin.contains(values))
        rules[rule_name] = rule_values

        if settings['include_complements']:
            comp_bin = Bin(start = bin_starts[1], end = upper_edge_value, type = bin_type, closure = '[]')
            comp_name = name + comp_bin.rule_string()
            rules[comp_name] = ~rule_values

        # last bin
        rule_bin = Bin(start = bin_starts[n_bins-1], end=bin_ends[n_bins-1], type=bin_type, closure='[]')
        rule_name = name + rule_bin.rule_string()
        rule_values = np.array(rule_bin.contains(values))
        rules[rule_name] = rule_values

        if settings['include_complements']:
            comp_bin = Bin(start = lower_edge_value, end = bin_ends[n_bins-2], type=bin_type, closure='[]')
            comp_name = name + comp_bin.rule_string()
            rules[comp_name] = ~rule_values

        # middle bins
        for k in range(1, n_bins-2):
            rule_bin = Bin(start=bin_starts[k], end=bin_ends[k], type=bin_type, closure='[]')
            rule_name = name + rule_bin.rule_string()
            rule_values = np.array(rule_bin.contains(values))
            rules[rule_name] = rule_values

            if settings['include_complements']:
                comp_name = name + rule_bin.rule_string(for_complement=True)
                rules[comp_name] = ~rule_values

    return rules


#### HELPER FUNCTIONS ####


def generate_rules_from_bin(bin, name, values, return_complement=True):
    # todo add spec
    rule_name = name + bin.rule_string()
    rule_values = np.array(bin.contains(values))

    if return_complement:
        comp_name = name + bin.rule_string(for_complement=True)
        return (rule_name, rule_values), (comp_name, ~rule_values)
    else:
        return (rule_name, rule_values),


def get_surrounding_points(search_values, point_set):
    """
    #for each value p[i] in search_values, returns a pair of surrounding points from point_set
    the surrounding points are a tuplet of the form (lb[i], ub[i]) where
    - lb[i] < p[i] < ub[i] if p[i] is not in point_set, and p[i] is within range
    - lb[i] == p[i] == ub[i] if p[i] in point_set, p[i] < min(point_set), p[i] > max(point_set)
    :param search_values: set of points that need neighbors
    :param point_set: set of points that need  be sorted
    :return: list of points in point_set that surround search_values
    """
    # http://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array
    upper_indices = np.searchsorted(point_set, search_values, side="left")
    n_points = len(point_set)
    n_search = len(search_values)
    neighbors = []
    for i in range(n_search):
        idx = upper_indices[i]
        val = search_values[i]
        if idx == 0:
            n = (point_set[0], point_set[0])
        elif idx == n_points:
            n = (point_set[-1], point_set[-1])
        else:
            n = (point_set[idx-1], point_set[idx])
        neighbors.append(n)

    return neighbors


def list_trivial_rules(rules):
    trivial_rule_names = []
    for n in rules:
        if len(rules[n]) > 1:
            if rules[n].dtype == np.bool and np.all(rules[n][0] == rules[n][1:]):
                trivial_rule_names.append(n)
    return trivial_rule_names


def remove_trivial_rules(rules):
    to_remove = list_trivial_rules(rules)
    for n in to_remove:
        rules.pop(n, None)
    return rules


def list_equivalent_rules(rules, preferences = 'none'):
    """
    :param rules: dictionary containing rule_names (key) and rule_values (boolean array) 
    :return: list of lists. each inner lists contains the rule_names with the same rule_value, ordered 
    """

    rule_names, rule_values = zip(*rules.items())
    rule_values = np.vstack(rule_values)
    equivalent_values, duplicate_idx = npi.unique(rule_values, return_inverse=True)
    n_equivalent = len(equivalent_values)

    # create list of rules containing equivalence classes
    if preferences == 'none':
        equivalent_rules = [[]] * n_equivalent
        for j in range(n_equivalent):
            rule_idx = np.flatnonzero(j == duplicate_idx)
            equivalent_rules[j] = [rule_names[k] for k in rule_idx]
    else:
        # order rules from first to last
        raise NotImplementedError()

    return equivalent_rules, equivalent_values


def remove_duplicate_rules(rules, preferences = 'none'):
    if len(rules) < 2:
        return rules

    equivalent_map, equivalent_values = list_equivalent_rules(rules, preferences)
    for e in equivalent_map:
        n = len(e)
        if n > 0:
            for k in range(1, len(e)):
                rules.pop(e[k], None)
    return rules


def remove_rules_with_low_coverage(rules, coverage_threshold):
    if len(rules) == 0:
        return rules

    rules_to_remove = [r for r in rules.keys() if np.sum(rules[r]) < coverage_threshold]
    for r in rules_to_remove:
        rules.pop(r, None)

    return rules


def list_rules_without_complements(rules):
    """
    :param rules: no duplicates
    :return: 
    """
    if len(rules) < 2:
        return []

    rule_names, rule_values = zip(*rules.items())

    rules_to_check = range(len(rule_names))
    rules_without_complements = []
    while len(rules_to_check) > 0:

        idx = rules_to_check.pop()
        val = ~rule_values[idx]

        comp_idx = -1
        for j in rules_to_check:
            if np.array_equal(val, rule_values[j]):
                comp_idx = j
                break

        if comp_idx >= 0:
            rules_to_check.remove(comp_idx)
        else:
            rules_without_complements.append(idx)

    return rules_without_complements


def add_missing_complements(rules):
    """
    :param rules: no duplicates
    :return: 
    """
    if len(rules) == 0:
        return rules

    rule_names, rule_values = zip(*rules.items())

    rules_without_complements = list_rules_without_complements(rules)
    for idx in rules_without_complements:
        comp_name = get_complement_name(rule_names[idx])
        rules[comp_name] = ~rule_values[idx]

    return rules


### Fast Rule Functions ####


def add_rule_limits(data, type_limits):
    feature_names = data['feature_names']


def has_all_complements(values):

    n_rules = values.shape[1]

    # sets with odd number of elements cannot have all complements
    if n_rules % 2 > 0:
        return False

    search_idx = range(n_rules)

    #find a complement for each rule
    while n_rules > 0:

        k = search_idx.pop()
        v = values[:,k]

        found_complement = False
        for l in search_idx:
            if np.all(np.logical_xor(v, values[:, l])):
                found_complement = True
                break

        if found_complement:
            search_idx.remove(l)
            n_rules -= 2
        else:
            #rule k has no complement
            return False

    return True


def has_overlapping_rules(values):


    #simple/slower version: return np.all(np.sum(values, axis = 1) <= 1)
    n_rules = values.shape[1]
    if n_rules < 2:
        return False

    v = values[:,0]
    k = 1
    while k < n_rules:
        w = v & values[:, k]
        if np.any(w):
            return True
        else:
            v = v | values[:, k]
        k += 1

    return False


def list_complementary_pairs(values):

    n_rules = values.shape[1]
    complementary_pairs = []
    search_idx = range(n_rules)

    while n_rules > 0:
        k = search_idx.pop()
        v = values[:, k]

        found_complement = False
        for l in search_idx:
            if np.all(np.logical_xor(v, values[:,l])):
                found_complement = True
                break

        if found_complement:
            complementary_pairs.append((k, l))
            search_idx.remove(l)

            n_rules -= 2

    return complementary_pairs


def split_complementary_pairs(values, complementary_pairs = None):

    #make defensive copy
    if complementary_pairs is None:
        rule_pairs = list_complementary_pairs(values)
    else:
        rule_pairs = list(complementary_pairs)

    n_pts, n_rules = values.shape

    rule_idx, comp_idx = [], []
    [k, l] = rule_pairs.pop()

    #split first pair based on which one has more non-zero entries
    if np.sum(values[:, k]) < np.ceil(n_pts/2):
        rule_idx.append(k)
        comp_idx.append(l)
    else:
        rule_idx.append(l)
        comp_idx.append(k)

    #split remaining pairs based on their overlap
    rule_values = values[:, rule_idx[0]]
    n_to_split = len(rule_pairs)
    while n_to_split > 0:
        [k, l] = rule_pairs.pop()
        overlap = rule_values & values[:,k]
        if np.any(overlap):
            rule_idx.append(l)
            comp_idx.append(k)
        else:
            rule_idx.append(k)
            comp_idx.append(l)

        rule_values = rule_values | values[:,rule_idx[-1]]
        n_to_split -=1

    return rule_idx, comp_idx


