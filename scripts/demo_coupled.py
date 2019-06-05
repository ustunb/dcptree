import dill
from dcptree.paths import *
from dcptree.data import *
from dcptree.data_io import load_processed_data
from dcptree.cross_validation import split_data_by_cvindices
from dcptree.group_helper import *
from dcptree.analysis import groups_to_splits, groups_to_group_data
from dcptree.classification_models import *
from dcptree.decoupled_set import *
from dcptree.baselines import CoupledRiskMinimizer

#### user dashboard ####
info = {
    'data_name': 'adult',
    'fold_id': 'K04N01',
    'max_runtime': 30,
    'random_seed': 2338
    }



def train_decoupled_models(info):

    # script variables
    data_file = info.get('data_file')
    fold_id = info.get('fold_id')

    # load data
    data, cvindices = load_processed_data(data_file)
    data = split_data_by_cvindices(data, cvindices, fold_id = fold_id, fold_num = 1, fold_num_test = -1)
    selected_groups = data['partitions']
    data, groups = split_groups_from_data(data = data, group_names = selected_groups)
    data = convert_remaining_groups_to_rules(data)
    data = cast_numeric_fields(data)

    splits = groups_to_splits(groups, drop_missing = True)
    group_names, group_values = groups_to_group_data(groups, stat_field = 'train')
    split_values, group_indicators = np.unique(group_values, axis = 0, return_inverse = True)

    if info['method_name'] == 'dcp_svm':

        training_handle = lambda data: train_svm(data, settings = None, normalize_variables = False)

    else: #info['method_name'] == 'dcp_lr':

        training_handle = lambda data: train_logreg(data, settings = None, normalize_variables = False)

    if info['attr_id'] == 'all':
        S = group_indicators
        sensitive_splits = split_values
    elif info['attr_id'] == 'none':
        S = np.zeros_like(group_indicators)
    else:
        matched_attr = np.flatnonzero([info['attr_id'].lower() == g.lower() for g in group_names])
        sensitive_splits, S = np.unique(group_values[:, matched_attr], axis = 0, return_inverse = True)

    # train pooled classifier
    pooled_model = training_handle({'X': data['X'], 'Y': data['Y'], 'variable_names': data['variable_names']})

    # train decoupled classifiers
    model_ids = np.unique(S)
    groups_to_models = {}
    model_dict = {}

    if info['attr_id'] == 'none':

        model_dict[0] = deepcopy(pooled_model)
        for k, s in enumerate(splits):
            groups_to_models[s] = 0


    elif info['attr_id'] == 'all':

        for group_id in model_ids:
            idx = S == group_id
            split = tuple(zip(group_names, sensitive_splits[group_id]))
            groups_to_models[split] = group_id
            model = training_handle({'X': data['X'][idx,:], 'Y': data['Y'][idx], 'variable_names': data['variable_names']})
            model_dict[group_id] = model

    else:

        for model_id, s in enumerate(sensitive_splits):
            idx = S == model_id
            model_dict[model_id] = training_handle({'X': data['X'][idx,:], 'Y': data['Y'][idx], 'variable_names': data['variable_names']})
            assignment_idx = np.isin(split_values[:, matched_attr], s).flatten()
            matched_full_values = split_values[assignment_idx, :]
            for vals in matched_full_values:
                split = tuple([(g, z) for g, z in zip(group_names, vals)])
                groups_to_models[split] = model_id

    decoupled_models = [model_dict[k] for k in range(len(model_dict))]

    assert check_model_assignment(groups_to_models, groups, decoupled_models)
    clf_set = DecoupledClassifierSet(data = data,
                                     groups = groups,
                                     pooled_model = pooled_model,
                                     decoupled_models = decoupled_models,
                                     groups_to_models = groups_to_models)

    info.update({'clf_set': clf_set})

    return info


def train_coupled_model(info):

    # script variables
    data_file = info.get('data_file')
    fold_id = info.get('fold_id')
    method_name = info.get('method_name')

    # load data
    data, cvindices = load_processed_data(data_file)
    data = split_data_by_cvindices(data, cvindices, fold_id = fold_id, fold_num = 1, fold_num_test = -1)
    selected_groups = data['partitions']
    data, groups = split_groups_from_data(data = data, group_names = selected_groups)
    data = convert_remaining_groups_to_rules(data)
    data = cast_numeric_fields(data)

    splits = groups_to_splits(groups, drop_missing = True)
    group_names, group_values = groups_to_group_data(groups, stat_field = 'train')
    split_values, group_indicators = np.unique(group_values, axis = 0, return_inverse = True)

    if 'dccp' in info['method_name']:

        data = add_intercept(data)
        method_specs = method_name.split('_')
        pooled_model_type = method_specs[1]

        if method_specs[2] == 'svm':
            loss_function = CoupledRiskMinimizer.LOSS_SVM
        else:
            loss_function = CoupledRiskMinimizer.LOSS_LOGISTIC

        pooled_params = {
            'EPS': 1e-3,
            'cons_type': CoupledRiskMinimizer.CONSTRAINT_PARITY if pooled_model_type == 'parity' else CoupledRiskMinimizer.CONSTRAINT_NONE,
            }

        decoupled_params = cons_params = {
            'cons_type': CoupledRiskMinimizer.CONSTRAINT_PREFERED_BOTH,
            'tau': 0.1,
            'print_flag': True
            }

    if info['attr_id'] == 'all':
        S = group_indicators
        sensitive_splits = split_values
    elif info['attr_id'] == 'none':
        S = np.zeros_like(group_indicators)
    else:
        matched_attr = np.flatnonzero([info['attr_id'].lower() == g.lower() for g in group_names])
        sensitive_splits, S = np.unique(group_values[:, matched_attr], axis = 0, return_inverse = True)

    # train pooled classifier
    pooled_clf = CoupledRiskMinimizer(loss_function, lam = 1e-5, train_multiple = False, sparse_formulation = True)
    pooled_clf.fit(X = data['X'], y = data['Y'], x_sensitive = S, cons_params = pooled_params)
    pooled_model = pooled_clf.classifier()
    debug = {'pooled_clf': pooled_clf}

    # train preference based classifier
    groups_to_models = {}
    if info['attr_id'] == 'none':

        for s in splits:
            groups_to_models[tuple(s)] = 0

        decoupled_models = [deepcopy(pooled_model)]

    else:

        _, dist_dict = pooled_clf.get_distance_boundary(X = data['X'], x_sensitive = S)
        switch_margins = pooled_clf.switch_margins(dist_dict)
        lam = {k: 1e-3 for k in np.unique(S)}
        decoupled_params['s_val_to_cons_sum'] = switch_margins
        clf = CoupledRiskMinimizer(loss_function, lam = lam, train_multiple = True, sparse_formulation = True)
        clf.fit(X = data['X'], y = data['Y'], x_sensitive = S, cons_params = cons_params)
        group_models = clf.classifier()

        if info['attr_id'] == 'all':

            for group_id in group_models.keys():
                split = tuple(zip(group_names, sensitive_splits[group_id]))
                groups_to_models[split] = group_id

        else:

            for s in split_values:

                # determine split
                split = tuple([(g, z) for g, z in zip(group_names, s)])

                # find model
                sensitive_values = s[matched_attr]
                model_id = int(np.flatnonzero(sensitive_values == sensitive_splits))
                groups_to_models[split] = model_id

        # build classifier sets
        decoupled_models = [group_models[i] for i in range(len(group_models))]
        for k, w in clf.w.items():
            assert np.isclose(decoupled_models[k].coefficients, w[1:]).all()
            assert np.isclose(decoupled_models[k].intercept, w[0])

        debug.update({'clf': clf})


    assert check_model_assignment(groups_to_models, groups, decoupled_models)
    clf_set = DecoupledClassifierSet(data = data,
                                     groups = groups,
                                     pooled_model = pooled_model,
                                     decoupled_models = decoupled_models,
                                     groups_to_models = groups_to_models)


    info.update({
        'clf_set': clf_set,
        'debug': debug
        })

    return info



# test decoupled models
for attr_id in ['all', 'sex']:

    for method_name in ['dcp_lr', "dccp_blind_lr"]:

        # setup training parameters
        test_info = dict(info)
        test_info['attr_id'] = attr_id
        test_info['method_name'] = method_name
        test_info['data_file'] = '%s/%s_processed.pickle' % (data_dir, info['data_name'])
        test_info['results_file'] = '%s/%s_%s_%s_results.pickle' % (results_dir, test_info['data_name'], test_info['attr_id'], test_info['method_name'])

        # train method
        if 'dccp' in method_name:
            test_info = train_coupled_model(test_info)
        else:
            test_info = train_decoupled_models(test_info)

        # check classifier set
        clf_set = test_info['clf_set']
        n_models = len(set(clf_set.groups_to_models.values()))
        if attr_id == 'all':
            assert len(clf_set) == n_models
        elif attr_id == 'none':
            assert n_models == 1
        elif attr_id == 'sex':
            assert n_models == 2

        with open(test_info['results_file'], 'wb') as outfile:
            dill.dump(info, outfile, protocol = dill.HIGHEST_PROTOCOL)
            print('saved file: %s' % outfile)