from dcptree.data import *
from dcptree.data_io import load_processed_data
from dcptree.cross_validation import filter_data_to_fold
from dcptree.group_helper import *
from dcptree.classification_models import *
from dcptree.paths import *
from dcptree.tree import DecoupledTrainer

#directories
data_name = 'adult'
random_seed = 2338
selected_groups = ['Race', 'Sex']

## load data
data_file = '%s%s_processed.pickle' % (data_dir, data_name)
data, cvindices = load_processed_data(data_file)

# filter to fold
data = filter_data_to_fold(data, cvindices, fold_id = 'K05N01', fold_num = 0, include_validation = True)


# remove selected groups
data, groups = split_groups_from_data(data = data, group_names = selected_groups)
data = convert_remaining_groups_to_rules(data)
data = cast_numeric_fields(data)


def test_feasible_names():

    node_idx = np.ones(data['X'].shape[0], dtype = bool)
    sh = DecoupledTrainer(groups = groups, node_idx = node_idx)

    for name in groups:
        assert name in sh.feasible_names
        assert name not in sh.infeasible_names

    #marking one label as infeasible should make entire name infeasible
    for name in sh.names:

        infeasible_label = sh.labels(name)[0]

        sh.mark_as_infeasible(name, infeasible_label)
        assert name not in sh.feasible_names
        assert name in sh.infeasible_names
        assert name, infeasible_label in sh.infeasible_splits
        assert name, infeasible_label not in sh.feasible_splits

        all_labels = sh.labels(name)
        n_labels = len(all_labels)
        for n in range(1, n_labels):
            assert name, all_labels[n] in sh.feasible_splits
            assert name, all_labels[n] not in sh.infeasible_splits

        sh.mark_as_feasible(name, infeasible_label)
        assert name in sh.feasible_names
        assert name not in sh.infeasible_names
        assert name, infeasible_label not in sh.infeasible_splits
        assert name, infeasible_label in sh.feasible_splits


    for name, label in sh.splits:

        assert name, label in sh.feasible_splits
        assert name, label not in sh.infeasible_splits

        sh.mark_as_infeasible(name, label)
        assert name not in sh.feasible_names
        assert name in sh.infeasible_names
        assert name, label in sh.feasible_splits
        assert name, label not in sh.infeasible_splits


def test_splits():

    node_idx = np.ones(data['X'].shape[0], dtype = bool)
    sh = DecoupledTrainer(groups = groups, node_idx = node_idx)

    for name, label in sh.splits:

        assert name in sh.names
        assert label in sh.labels(name)

        assert name in groups
        assert label in groups[name]['labels']

        expected_id = list(groups[name]['labels']).index(label)
        expected_idx = expected_id == groups[name]['indices']

        assert np.array_equal(expected_idx, sh.split_indices(name, label))
        assert sh.has_model(name, label) == False
        assert sh.needs_model(name, label)

        try:
            sh.get_model(name, label)
        except ValueError:
            assert True
        except Exception:
            assert False


def test_split_indices():

    node_idx = np.random.randint(low = 0, high = 2, size = data['X'].shape[0])
    sh = DecoupledTrainer(groups = groups, node_idx = node_idx)

    for name, label in sh.splits:

        id = list(groups[name]['labels']).index(label)
        idx = id == groups[name]['indices']
        split_idx = node_idx & idx
        assert np.array_equal(split_idx, sh.split_indices(name, label))


def test_infeasible():

    node_idx = np.ones(data['X'].shape[0], dtype = bool)
    sh = DecoupledTrainer(groups = groups, node_idx = node_idx)
    inf_codes = list(DecoupledTrainer.INFEASIBILITY_CODES)

    for name, label in sh.splits:

        sh.is_infeasible(name, label) == False

        sh.mark_as_infeasible(name, label)
        assert sh.is_infeasible(name, label)
        assert sh.needs_model(name, label) == False

        assert sh.infeasibility_code(name, label) == 'unspecified'
        assert sh.infeasibility_reason(name, label) == sh.INFEASIBILITY_DICT[sh.infeasibility_code(name, label)]

        sh.mark_as_feasible(name, label)
        assert sh.is_infeasible(name, label) == False
        assert sh.needs_model(name, label)

        assert sh.infeasibility_code(name, label) is None
        assert sh.infeasibility_reason(name, label) is None

        for code in inf_codes:
            sh.mark_as_infeasible(name, label, code)
            assert sh.is_infeasible(name, label)
            assert sh.infeasibility_code(name, label) == code

        try:
            sh.get_model(name, label)
        except ValueError:
            assert True
        except Exception:
            assert False

    assert sh.finished()
    assert sh.next() == (None, None, None)


def test_iteration():

    node_idx = np.ones(data['X'].shape[0], dtype = bool)
    sh = DecoupledTrainer(groups = groups, node_idx = node_idx)
    test_model = 'test'

    assert not sh.finished()

    while not sh.finished():

        name, label, idx = sh.next()

        assert not sh.has_model(name, label)

        assert name in sh.names
        assert label in sh.labels(name)
        assert name in groups
        assert label in groups[name]['labels']

        assert sh.needs_model(name, label) == True
        assert sh.is_infeasible(name, label) == False

        sh.set_model(name, label, test_model)
        assert sh.has_model(name, label)
        assert sh.get_model(name, label) == test_model
        assert sh.needs_model(name, label) == False
        assert sh.is_infeasible(name, label) == False

    assert sh.finished()
    assert sh.next() == (None, None, None)


def test_sample_infeasibility():

    node_idx = np.ones(data['X'].shape[0], dtype = bool)
    sh = DecoupledTrainer(groups = groups, node_idx = node_idx)
    sh.set_sample_infeasibility(min_samples = 0, propagate_to_group = True)
    assert len(sh.infeasible_names) == 0

    for name in sh.names:

        all_labels = list(groups[name]['labels'])

        _, cnts = np.unique(groups[name]['indices'], return_counts = True)

        min_label = all_labels[np.argmin(cnts)]
        idx = sh.split_indices(name, min_label)
        n_samples = np.sum(idx)

        sh.set_sample_infeasibility(min_samples = n_samples, propagate_to_group = False)
        assert name not in sh.infeasible_names

        sh.set_sample_infeasibility(min_samples = n_samples + 1, propagate_to_group = False)
        print(sh.infeasible_names)
        assert name in sh.infeasible_names
        assert name, min_label in sh.infeasible_splits

        other_labels = list(set(all_labels).difference(min_label))
        for label in other_labels:
            assert name, label not in sh.infeasible_splits

        sh.set_sample_infeasibility(min_samples = n_samples + 1, propagate_to_group = True)
        for label in other_labels:
            assert name, label in sh.infeasible_splits

    assert sh.finished()
