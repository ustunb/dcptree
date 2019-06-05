from dcptree.data import *
from dcptree.data_io import load_processed_data
from dcptree.cross_validation import split_data_by_cvindices
from dcptree.group_helper import *
from dcptree.analysis import groups_to_splits
from dcptree.tree import *
from dcptree.baselines import CoupledRiskMinimizer
from dcptree.classification_models import *
from dcptree.zero_one_loss.mip import train_zero_one_linear_model
from dcptree.decoupled_set import *


DECOUPLED_TREE_METHODS = ['tree_01', 'tree_lr', 'tree_svm']
DECOUPLED_CLF_METHODS = ['dcp_svm', 'dcp_lr']
COUPLED_CLF_METHODS = ['dccp_blind_svm', 'dccp_blind_lr', 'dccp_parity_lr', 'dccp_parity_svm']
ONEHOT_CLF_METHODS = ['onehot_lr', 'onehot_svm']
ALL_METHOD_NAMES = DECOUPLED_TREE_METHODS + DECOUPLED_CLF_METHODS + COUPLED_CLF_METHODS + ONEHOT_CLF_METHODS

def train_decoupled_tree(info):

    assert info['method_name'] in DECOUPLED_TREE_METHODS
    data, cvindices = load_processed_data(info['data_file'])
    data = split_data_by_cvindices(data, cvindices, fold_id = info['fold_id'], fold_num = 1, fold_num_test = 2)
    data, groups = split_groups_from_data(data = data, group_names = data['partitions'])
    data = convert_remaining_groups_to_rules(data)
    data = cast_numeric_fields(data)

    # specify training handle
    if info['method_name'] == 'tree_01':
        data = add_intercept(data)
        training_handle = lambda data: train_zero_one_linear_model(data, time_limit = info['max_runtime'])
    elif info['method_name'] == 'tree_lr':
        training_handle = lambda data: train_logreg(data, settings = None, normalize_variables = False)
    else:
        training_handle = lambda data: train_svm(data, settings = None, normalize_variables = False)

    # train tree
    tree = DecouplingTree(data, groups, training_handle = training_handle)
    tree.grow(print_flag = True, log_flag = True)

    ### prune tree
    tree.prune(alpha = 0.1, test_correction = True, atomic_groups = True, data = data, groups = groups, stat_field = 'test')
    tree.render()

    ### transform to classifier tree
    clf_set = DecoupledClassifierSet(data = data,
                                     groups = groups,
                                     pooled_model = tree.root.model,
                                     decoupled_models = [l.model for l in tree.partition.leaves],
                                     groups_to_models = build_model_assignment_map(tree.partition))

    ### save results
    info.update({
        'clf_set': clf_set,
        'debug': {'tree': tree},
        })

    return info


def train_decoupled_models(info):

    assert info['method_name'] in DECOUPLED_CLF_METHODS
    data, cvindices = load_processed_data(info['data_file'])
    data = split_data_by_cvindices(data, cvindices, fold_id = info['fold_id'], fold_num = 1, fold_num_test = -1)
    data, groups = split_groups_from_data(data = data, group_names = data['partitions'])
    data = convert_remaining_groups_to_rules(data)
    data = cast_numeric_fields(data)

    # group stuff
    splits = groups_to_splits(groups, drop_missing = True)
    group_names, group_values = groups_to_group_data(groups, stat_field = 'train')
    split_values, group_indicators = np.unique(group_values, axis = 0, return_inverse = True)

    # set method
    if info['method_name'] == 'dcp_svm':
        training_handle = lambda data: train_svm(data, settings = None, normalize_variables = False)
    else: #info['method_name'] == 'dcp_lr':
        training_handle = lambda data: train_logreg(data, settings = None, normalize_variables = False)

    if info['attr_id'] == 'none':
        S = np.zeros_like(group_indicators)
    elif info['attr_id'] == 'all':
        S = group_indicators
        sensitive_splits = split_values
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

    info.update({
        'clf_set': clf_set,
        })

    return info


def train_coupled_model(info):

    assert info['method_name'] in COUPLED_CLF_METHODS
    data, cvindices = load_processed_data(info['data_file'])
    data = split_data_by_cvindices(data, cvindices, fold_id = info['fold_id'], fold_num = 1, fold_num_test = -1)
    data, groups = split_groups_from_data(data = data, group_names = data['partitions'])
    data = convert_remaining_groups_to_rules(data)
    data = cast_numeric_fields(data)

    # group stuff
    splits = groups_to_splits(groups, drop_missing = True)
    group_names, group_values = groups_to_group_data(groups, stat_field = 'train')
    split_values, group_indicators = np.unique(group_values, axis = 0, return_inverse = True)

    # add intercept
    data = add_intercept(data)
    method_specs = info['method_name'].split('_')
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


def train_onehot_model(info):
    """
    trains linear classifier with a one-hot encoding
    :param info:
    :return:
    """
    assert info['method_name'] in ONEHOT_CLF_METHODS
    data, cvindices = load_processed_data(info['data_file'])
    data = split_data_by_cvindices(data, cvindices, fold_id = info['fold_id'], fold_num = 1, fold_num_test = -1)
    data, groups = split_groups_from_data(data = data, group_names = data['partitions'])
    data = convert_remaining_groups_to_rules(data)
    data = cast_numeric_fields(data)

    # group stuff
    splits = groups_to_splits(groups, drop_missing = True)
    group_names, group_values = groups_to_group_data(groups, stat_field = 'train')
    split_values, group_indicators = np.unique(group_values, axis = 0, return_inverse = True)

    if info['method_name'] == 'onehot_svm':
        training_handle = lambda data: train_svm(data, settings = None, normalize_variables = False)
    else: #info['method_name'] == 'onehot_lr':
        training_handle = lambda data: train_logreg(data, settings = None, normalize_variables = False)

    if info['attr_id'] == 'none':
        S = np.zeros_like(group_indicators)
    elif info['attr_id'] == 'all':
        S = group_indicators
        training_splits = split_values
    else:
        matched_attr = np.flatnonzero([info['attr_id'].lower() == g.lower() for g in group_names])
        training_splits, S = np.unique(group_values[:, matched_attr], axis = 0, return_inverse = True)

    # train blind classifier
    pooled_model = training_handle({'X': data['X'], 'Y': data['Y'], 'variable_names': data['variable_names']})

    # train group-specific models
    model_dict = {}
    groups_to_models = {}

    if info['attr_id'] == 'none':

        model_dict[0] = deepcopy(pooled_model)
        for k, s in enumerate(splits):
            groups_to_models[s] = 0

    else:


        coefficient_idx = get_variable_indices(data, include_intercept = False)

        # find majority group
        model_ids, subgroup_counts = np.unique(S, return_counts = True)
        majority_group_id = np.argmax(subgroup_counts)
        minority_group_ids = np.delete(model_ids, majority_group_id)

        # create group indicator variables
        Z = np.vstack([np.isin(S, s) for s in model_ids]).transpose().astype('float')
        assert np.isclose(Z.sum(axis = 1), 1.0).all()
        Z = Z[:, minority_group_ids]
        Z_names = ['Subgroup_is_%s' % ''.join(training_splits[s]) for s in minority_group_ids]
        ZX = np.concatenate((Z, data['X']), axis = 1)
        ZX_names = Z_names + data['variable_names']

        # setup parameters for model object
        baseline_model = training_handle({'X': ZX, 'Y': data['Y'], 'variable_names': ZX_names})

        baseline_intercept = float(baseline_model._intercept)
        baseline_coefficients = np.array(baseline_model._coefficients).flatten()
        coef_idx = np.arange(len(ZX_names))
        Z_coef_idx = np.arange(len(Z_names))
        X_coef_idx = np.setdiff1d(coef_idx, Z_coef_idx)

        baseline_coefs = np.copy(baseline_coefficients[X_coef_idx])
        subgroup_coefs = np.copy(baseline_coefficients[Z_coef_idx])
        subgroup_coefs = np.insert(arr = subgroup_coefs, obj = majority_group_id, values = 0.0)

        if info['attr_id'] == 'all':
            subgroup_intercepts = np.array(subgroup_coefs)
        else:
            subgroup_intercepts = np.repeat(np.nan, len(splits))
            for model_id, s in enumerate(training_splits):
                matched_subgroups = np.flatnonzero(np.isin(split_values[:, matched_attr], s).flatten())
                if model_id == majority_group_id:
                    subgroup_intercepts[matched_subgroups] = 0.0
                else:
                    subgroup_intercepts[matched_subgroups] = subgroup_coefs[model_id]
            assert np.isfinite(subgroup_intercepts).all()
            assert np.any(subgroup_intercepts == 0.0)

        # create new models for each group
        for k, s in enumerate(splits):
            model_info = {
                'intercept': float(baseline_intercept + subgroup_intercepts[k]),
                'coefficients': np.copy(baseline_coefs),
                'coefficient_idx': coefficient_idx,
                }
            training_info = dict(baseline_model.training_info)
            training_info['split'] = s
            model = ClassificationModel(predict_handle = lambda X: np.sign(X.dot(model_info['coefficients']) + model_info['intercept']),
                                        model_type = ClassificationModel.LINEAR_MODEL_TYPE,
                                        model_info = model_info,
                                        training_info = training_info)
            model_dict[k] = model
            groups_to_models[s] = k

    decoupled_models = [model_dict[k] for k in range(len(model_dict))]
    assert check_model_assignment(groups_to_models, groups, decoupled_models)
    clf_set = DecoupledClassifierSet(data = data,
                                     groups = groups,
                                     pooled_model = pooled_model,
                                     decoupled_models = decoupled_models,
                                     groups_to_models = groups_to_models)

    info.update({'clf_set': clf_set})
    return info

