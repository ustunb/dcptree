import os
from dcptree.paths import *
from dcptree.data import *
from dcptree.data_io import load_processed_data
from dcptree.cross_validation import filter_data_to_fold
from dcptree.group_helper import *
from dcptree.classification_models import *
from dcptree.tree import *


#directories
data_name = 'adult'
format_label = 'envyfree'
random_seed = 1337
repo_dir = os.getcwd() + '/'
data_dir = repo_dir + 'data/'
selected_groups = ['Sex']

## load data
data_file = '%s%s_processed.pickle' % (data_dir, data_name)
data, cvindices = load_processed_data(data_file)

# filter to fold
data = filter_data_to_fold(data, cvindices, fold_id = 'K05N01', fold_num = 0, include_validation = True)

# remove selected groups
data, groups = split_groups_from_data(data = data, group_names = selected_groups)
data = convert_remaining_groups_to_rules(data)
data = cast_numeric_fields(data)
training_handle = lambda data: train_logreg(data, settings = None, normalize_variables = False)
#tree = EnvyFreeTree(data, groups, training_handle = training_handle, splitting_handle = split_test)
#tree.grow()

##### basic tests #####

def test_tree_inputs():
    assert check_data(data, ready_for_training = True)
    assert check_groups(groups, data)


def test_groups_at_root():

    tree = EnvyFreeTree(data, groups, training_handle = training_handle, splitting_handle = split_test)
    tree_groups = tree.groups
    assert type(tree_groups) is dict
    assert check_groups(tree_groups, data)

    for group_name, group_values in tree_groups.items():
        assert group_name in groups
        assert type(group_values) is dict
        assert np.array_equal(group_values['ids'], groups[group_name]['ids'])
        assert np.array_equal(group_values['labels'], groups[group_name]['labels'])
        assert np.array_equal(group_values['indices'], groups[group_name]['indices'])


def test_data_at_root():

    tree = EnvyFreeTree(data, groups, training_handle = training_handle, splitting_handle = split_test)
    assert not has_intercept(tree.data)
    assert not has_intercept(tree.filter_data(tree.node_indices))
    assert not has_intercept(tree.filter_data(tree.group_indices))


def test_root_names():

    tree = EnvyFreeTree(data, groups, training_handle = training_handle, splitting_handle = split_test)
    assert tree.group_id == EnvyFreeTree.ROOT_GROUP_ID
    assert tree.group_label == EnvyFreeTree.ROOT_GROUP_LABEL
    assert tree.group_name == EnvyFreeTree.ROOT_GROUP_NAME
    assert [tree.group_name] == tree.path_names
    assert [tree.group_label] == tree.path_labels
    np.all(tree.node_indices == tree.group_indices)


def test_initialization():

    tree = EnvyFreeTree(data, groups, training_handle = training_handle, splitting_handle = split_test)

    assert tree.is_leaf
    assert tree.is_root
    assert tree.depth == 0
    assert tree.height == 0

    assert tree.has_model == False
    assert tree.max_depth == len(tree.groups)
    assert tree.terminal == False
    assert tree.split_attempt == False
    assert set(tree.potential_split_names) == set(selected_groups)
    assert tree.halted_by_suboptimality == False
    assert tree.halted_by_infeasibility == False
    assert tree.halted == False

    assert tree.is_fully_grown == False


def test_set_max_depth():

    tree = EnvyFreeTree(data, groups, training_handle = training_handle, splitting_handle = split_test)

    # max depth should be n_groups by default
    assert tree.max_depth == len(groups)

    # test setting max depth
    for test_depth in range(len(groups)):
        tree.max_depth = test_depth
        assert tree.max_depth == test_depth

    # cannot set max_depth to greater than n_groups
    test_depth = len(groups) + 1
    tree.max_depth = test_depth
    assert tree.max_depth == len(groups)

    # max depth must be positive
    try:
        tree.max_depth = -1
    except AssertionError:
        assert True
    else:
        assert False

    # max depth must be integer
    try:
        tree.max_depth = 0.5
    except AssertionError:
        assert True
    else:
        assert False


def test_min_samples():

    tree = EnvyFreeTree(data, groups, training_handle = training_handle, splitting_handle = split_test)

    # max depth should be n_groups by default
    assert tree.min_samples == 1

    n_samples = data['X'].shape[0]
    tree.min_samples = n_samples - 1
    assert tree.min_samples == n_samples - 1

    tree.min_samples = n_samples
    assert tree.min_samples == n_samples

    tree.min_samples = n_samples + 1
    assert tree.min_samples == n_samples

    # must be positive
    try:
        tree.min_samples = -1
    except AssertionError:
        assert True
    else:
        assert False

    # must be integer
    try:
        tree.min_samples = 0.5
    except AssertionError:
        assert True
    else:
        assert False


def test_get_model_no_model():

    tree = EnvyFreeTree(data, groups, training_handle = training_handle, splitting_handle = split_test)

    assert not tree.has_model

    try:
        model = tree.model
    except:
        assert True
    else:
        assert False


def test_train_model():

    tree = EnvyFreeTree(data, groups, training_handle = training_handle, splitting_handle = split_test)

    assert tree.has_model == False
    tree.train_model()
    assert tree.has_model

    model = tree.model
    training_data = tree.filter_data(indices = tree.node_indices)

    coefs = model.model_info['coefficients']
    coefs = coefs.flatten()
    assert len(coefs) == training_data['X'].shape[1]
    assert np.all(np.isfinite(coefs))

    yhat = model.predict(training_data['X'])
    assert len(yhat) == len(training_data['Y'])
    assert np.all(np.isin(yhat,(-1,1)))


def test_branch():

    tree = EnvyFreeTree(data, groups, training_handle = training_handle, splitting_handle = split_test)

    assert tree.has_model == False
    tree.branch()
    assert tree.has_model == True

    assert len(tree.children) >= 2

    subgroup_name = [s.group_name for s in tree.children]
    assert len(set(subgroup_name)) == 1
    subgroup_name = set(subgroup_name).pop()

    assert subgroup_name in tree.groups

    for subtree in tree.children:

        assert subtree.parent == tree
        assert subtree.group_name in groups
        assert subtree.group_label in groups[subtree.group_name]['labels']
        assert subtree.has_model
        assert subtree.splittable


def test_branch_max_depth_0():

    tree = EnvyFreeTree(data, groups, training_handle = training_handle, splitting_handle = split_test)

    assert tree.has_model == False
    tree.max_depth = 0
    tree.branch()
    assert tree.has_model == True
    assert tree.is_fully_grown
    assert len(tree.children) == 0


def test_grow():

    tree = EnvyFreeTree(data, groups, training_handle = training_handle, splitting_handle = split_test)

    assert tree.has_model == False
    tree.grow()
    assert tree.has_model == True
    assert tree.is_fully_grown


    for leaf in tree.leaves:
        assert leaf.root == tree
        assert leaf.group_name in groups
        assert leaf.group_label in groups[leaf.group_name]['labels']
        assert leaf.has_model


def test_grow_max_depth_0():

    tree = EnvyFreeTree(data, groups, training_handle = training_handle, splitting_handle = split_test)

    assert tree.has_model == False
    tree.max_depth = 0
    tree.grow()
    assert tree.has_model == True
    assert tree.is_fully_grown
    assert len(tree.children) == 0


def test_grow_max_depth_1():

    tree = EnvyFreeTree(data, groups, training_handle = training_handle, splitting_handle = split_test)
    tree.max_depth = 1
    assert tree.has_model == False
    tree.grow()
    assert tree.has_model == True
    assert tree.is_fully_grown
    assert tree.height == 1


def test_grow_with_no_groups():
    tree = EnvyFreeTree(data, groups, training_handle = training_handle, splitting_handle = split_test)
    tree.grow()
    assert tree.is_fully_grown
    assert tree.has_model


def test_branch_then_grow():

    tree = EnvyFreeTree(data, groups, training_handle = training_handle, splitting_handle = split_test)

    assert tree.has_model == False
    tree.max_depth = 1
    tree.branch()
    assert tree.has_model == True
    assert tree.is_fully_grown
    tree.grow()

    assert tree.is_fully_grown

