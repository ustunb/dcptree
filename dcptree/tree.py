import itertools
import pandas as pd
from scipy.stats import binom
from copy import deepcopy
from anytree import NodeMixin, RenderTree
from inspect import getfullargspec
from dcptree.helper_functions import *
from dcptree.analysis import to_group_data
from dcptree.data import check_data, has_intercept
from dcptree.group_helper import check_groups
from dcptree.classification_models import ClassificationModel
from dcptree.debug import ipsh

#### Scoring and Selection ####

def exact_mcn_test(y, yhat1, yhat2, two_sided = False):
    """
    :param y: true
    :param yhat1:
    :param yhat2:
    :param two_sided:
    :return: value of the discrete McNemar Test
    """

    f1_correct = np.equal(y, yhat1)
    f2_correct = np.equal(y, yhat2)

    table = np.zeros(shape = (2, 2))
    for i in range(2):
        for j in range(2):
            table[i, j] = np.sum((f1_correct == i) & (f2_correct == j))

    b = table[0, 1] #f1 wrong and f2 right
    c = table[1, 0] #f1 right and f2 wrong
    n = b + c

    # envy-freeness requires that
    # f1 is correct more often than f2 <=> b < c
    #
    # We test
    #
    # H0: error(f1) = error(f2)
    # H1: error(f1) > error(f2)
    #
    # This requires assuming b /(b+c) ~ Bin(0.5)

    if two_sided:
        test_statistic = min(b, c)
        p = 2.0 * binom.cdf(k = min(b, c), n = b + c, p = 0.5)
    else:
        test_statistic = c
        p = binom.cdf(k = test_statistic, n = n, p = 0.5)

    return p, test_statistic


def preference_score(data, partition, pooled_model):
    """

    scores a partition based on the probability of at least 1 fair_decoupling violation

    :param data:
    :param partition:
    :return:

    """
    assert callable(pooled_model)
    assert isinstance(partition, (list, DecoupledLeafSet))
    n_parts = len(partition)
    assert n_parts >= 1
    if isinstance(partition, DecoupledLeafSet):
        assert partition.is_partition
        assert partition.has_models
        predictors, indices = partition.predictors, partition.indices
    else:
        predictors, indices = zip(*partition)
        assert np.all(np.sum(np.stack(indices), axis = 0) == 1)

    X, Y = data['X'], data['Y']

    error_matrix = np.empty(shape = (n_parts, n_parts))
    envy_free_scores = np.empty(shape = (n_parts, n_parts))
    rationality_scores = np.empty(shape = (n_parts))

    for i, idx in enumerate(indices):
        n_samples = np.sum(idx)
        Yg, Xg = Y[idx], X[idx, :]
        for j, h in enumerate(predictors):
            mistakes = np.not_equal(Yg, h(Xg))
            error_matrix[i, j] = np.mean(mistakes)

        # compute errors for group i
        self_error = error_matrix[i, i]
        pooled_error = np.mean(np.not_equal(Yg, pooled_model(Xg)))

        # compute score components for group i
        rationality_scores[i] = 0.5 * n_samples * np.square(self_error - pooled_error)
        envy_free_scores[i, :] = 0.5 * n_samples * np.power(self_error - error_matrix[i, :], 2.0)

    rationality_scores = 4.0 * np.exp(-rationality_scores)
    envy_free_scores = 4.0 * np.exp(-envy_free_scores)
    np.fill_diagonal(envy_free_scores, 0.0)
    score = np.sum(rationality_scores) + np.sum(envy_free_scores)
    if not np.isfinite(score):
        ipsh()

    return score


def minimize_score_over_partitions(partitions):
    """
    :param partitions: list of dicts containing 'score'
    :return: index of partition that minimizes the score
    """
    assert isinstance(partitions, list)
    min_score = float('inf')
    min_idx = -1
    for i, p in enumerate(partitions):
        if p['score'] < min_score:
            min_score = p['score']
            min_idx = i
    return min_idx


#### Classes ####

class DecoupledTrainer(object):


    INFEASIBILITY_DICT = {
        'unspecified': 'unspecified',
        'n_samples': 'not enough samples',
        'n_samples_fold': 'not enough samples for a fold',
        'envyfree_violation': 'label prefers the model of another label',
        'decoupling_violation': 'label prefers not to decouple',
        'group_violation': 'label belongs to a group with a different infeasible label',
        }

    INFEASIBILITY_CODES = set(INFEASIBILITY_DICT.keys())


    #### initialization ####

    def __init__(self, groups, node_idx, pos_idx, neg_idx):

        assert check_groups(groups)
        self._node_idx = np.array(node_idx, dtype = bool).flatten()
        self._pos_idx = np.array(pos_idx, dtype = bool).flatten()
        self._neg_idx = np.array(neg_idx, dtype = bool).flatten()

        self._n = np.sum(self._node_idx)
        self._n_pos = np.sum(self._pos_idx)
        self._n_neg = np.sum(self._neg_idx)

        self._min_samples = 2
        self._min_samples_pos = 1
        self._min_samples_neg = 1

        self._names = list(groups.keys())
        self._labels = {g: list(groups[g]['labels']) for g in groups}
        self._indices = {g: np.array(groups[g]['indices'], dtype = int) for g in groups}

        self._ids = self._new_split_dictionary()
        for name, labels in self._ids.items():
            for l in labels:
                self._ids[name][l] = self._labels[name].index(l)

        self._models = self._new_split_dictionary()
        self._training_order = self.splits
        self._infeasible_splits = {}
        self._check_rep()
        self.set_sample_infeasibility()


    def _new_split_dictionary(self):
        d = dict.fromkeys(self._names)
        for name in d.keys():
            d[name] = dict.fromkeys(self._labels[name])
        return d


    #### validation ####

    def _check_rep(self):
        names = set(self.names)
        n_groups = len(names)
        assert len(self._training_order) >= 2 * n_groups
        assert names == set(self._ids.keys())
        assert names == set(self._indices.keys())
        assert names == set(self._models.keys())
        assert np.all(self._node_idx == np.logical_or(self._pos_idx, self._neg_idx))


    def _check_name_and_label(self, name, label):
        assert name in self._names
        assert label in self._labels[name]
        return True


    def _check_infeasibility_code(self, code):
        assert code in self.INFEASIBILITY_CODES


    #### properties


    @property
    def names(self):
        return list(self._names)


    @property
    def splits(self):
        splits = []
        for name, labels in self._labels.items():
            for l in labels:
                splits.append((name, l))
        return splits


    def labels(self, name):
        return list(self._labels[name])


    def split_indices(self, name, label):
        return np.logical_and(self._node_idx, self._ids[name][label] == self._indices[name])

    def split_indices_pos(self, name, label):
        return np.logical_and(self._pos_idx, self.split_indices(name, label))

    def split_indices_neg(self, name, label):
        return np.logical_and(self._neg_idx, self.split_indices(name, label))

    @property
    def feasible_names(self):
        return [n for n in self.names if n not in self._infeasible_splits]


    @property
    def infeasible_names(self):
        return [n for n in self.names if n in self._infeasible_splits]


    @property
    def feasible_splits(self):
        return list(filter(lambda v: not self.is_infeasible(v[0], v[1]), self.splits))


    @property
    def infeasible_splits(self):
        return list(filter(lambda v: self.is_infeasible(v[0], v[1]), self.splits))


    @property
    def min_samples(self):
        return int(self._min_samples)


    @property
    def min_samples_pos(self):
        return int(self._min_samples_pos)


    @property
    def min_samples_neg(self):
        return int(self._min_samples_neg)


    @min_samples.setter
    def min_samples(self, n):
        assert is_integer(n), 'min samples must be integer'
        assert n >= 2, 'min samples must be >= 2'
        self._min_samples = int(n)


    @min_samples_pos.setter
    def min_samples_pos(self, n):
        assert is_integer(n), 'min + samples must be integer'
        assert n >= 1, 'min + samples must be >= 2'
        n = int(n)
        self._min_samples_pos = min(n, self._n_pos)
        self._min_samples = max(self._min_samples_neg + self._min_samples_pos, self._min_samples)


    @min_samples_neg.setter
    def min_samples_neg(self, n):
        assert is_integer(n), 'min - samples must be integer'
        assert n >= 1, 'min - samples must be >= 2'
        n = int(n)
        self._min_samples_neg = min(n, self._n_neg)
        self._min_samples = max(self._min_samples_neg + self._min_samples_pos, self._min_samples)


    #### model management ####

    def has_model(self, name, label):
        return self._models[name][label] is not None


    def get_model(self, name, label):
        if self.has_model(name, label):
            return deepcopy(self._models[name][label])
        else:
            raise ValueError('no model exists for %s = %s' % (name, label))


    def set_model(self, name, label, model):
        self._check_name_and_label(name, label)
        self._models[name][label] = deepcopy(model)


    def needs_model(self, name, label):
        if self.is_infeasible(name, label):
            return False
        else:
            return not self.has_model(name, label)


    def get_models_for_name(self, name, include_infeasible = True):

        assert name in self.names
        models = {}

        if include_infeasible:
            for label in self.labels(name):
                model_item = {label: deepcopy(self._models[name][label])}
                models.update(model_item)
        else:
            assert name in self.feasible_names
            for label in self.labels(name):
                model_item = {label: deepcopy(self._models[name][label])}
                models.update(model_item)

        return models


    def get_feasible_models(self, propagate_infeasibility = True):

        feasible_names = self.feasible_names
        if propagate_infeasibility:
            feasible_names = filter(lambda n: not self.has_infeasible_split(n), feasible_names)

        models = {}
        for name in feasible_names:
            models[name] = {}
            for label in self.labels(name):
                model = self.get_model(name, label)
                models[name].update({label: model})

        return models


    #### infeasibility ####


    def is_infeasible(self, name, label):
        return name in self._infeasible_splits and label in self._infeasible_splits[name]


    def has_infeasible_split(self, name):
        return name in self._infeasible_splits


    def mark_as_infeasible(self, name, label, code = 'unspecified'):

        self._check_name_and_label(name, label)

        if name in self._infeasible_splits:
            self._infeasible_splits[name].update({label: code})
        else:
            self._infeasible_splits[name] = {label: code}

        self._check_rep()


    def mark_as_feasible(self, name, label):

        if self.is_infeasible(name, label):
            self._infeasible_splits[name].pop(label)
            if len(self._infeasible_splits[name]) == 0:
                self._infeasible_splits.pop(name)

            self._check_rep()


    def infeasibility_code(self, name, label):
        if self.is_infeasible(name, label):
            return str(self._infeasible_splits[name][label])
        else:
            return None


    def infeasibility_reason(self, name, label):
        if self.is_infeasible(name, label):
            code = self._infeasible_splits[name][label]
            return str(self.INFEASIBILITY_DICT[code])
        else:
            return None


    #### infeasibility setting ####


    def set_sample_infeasibility(self, propagate_to_group = True):

        for name, label in self.feasible_splits:
            n_samples = np.sum(self.split_indices(name, label))
            n_pos = np.sum(self.split_indices_pos(name, label))
            n_neg = np.sum(self.split_indices_neg(name, label))
            if (n_samples < self.min_samples) or (n_pos < self.min_samples_pos) or (n_neg < self.min_samples_neg):
                self.mark_as_infeasible(name, label, 'n_samples')

        if propagate_to_group:
            self.propagate_infeasibility()

        self._check_rep()


    def propagate_infeasibility(self):

        for name in self.infeasible_names:

            all_labels = set(self.labels(name))
            infeasible_labels = set(map(lambda s: s[1], filter(lambda s: s[0] == name, self.infeasible_splits)))
            feasible_labels = all_labels.difference(infeasible_labels)

            for label in feasible_labels:
                self.mark_as_infeasible(name, label, 'group_violation')

        self._check_rep()


    #### training ####


    def next(self):
        name, label, idx = None, None, None
        for n, l in self._training_order:
            if self.needs_model(n, l):
                name = n
                label = l
                idx = self.split_indices(name, label)
                break
        return name, label, idx


    def finished(self):
        return not any(map(lambda v: self.needs_model(v[0], v[1]), self._training_order))


class DecouplingTree(NodeMixin):

    ROOT_GROUP_NAME = 'root'
    ROOT_GROUP_LABEL = 'all'
    ROOT_GROUP_ID = 0
    TRAINING_HANDLE_ARGS = ['data']

    def __init__(self, data = None, groups = None, parent = None, **kwargs):

        self._halted = False
        self._split_attempt = False
        self._partition = None

        if parent is None:

            # check inputs
            assert check_data(data, ready_for_training = True)
            assert check_groups(groups, data)

            # initialize data
            self._data = {
                'X': np.array(data['X']),
                'Y': np.array(data['Y']),
                'variable_names' : list(data['variable_names'])
                }

            self._groups = deepcopy(groups)
            self._sample_indices_pos = self.data['Y'] == 1
            self._sample_indices_neg = self.data['Y'] == -1

            # set handles
            assert 'training_handle' in kwargs
            self.training_handle = kwargs.get('training_handle')
            self.scoring_handle = kwargs.get('scoring_handle', preference_score)
            self.selection_handle = kwargs.get('selection_handle', minimize_score_over_partitions)

            self._group_name = str(DecouplingTree.ROOT_GROUP_NAME)
            self._group_label = str(DecouplingTree.ROOT_GROUP_LABEL)
            self._node_indices = np.ones(self.data['X'].shape[0], dtype = bool)
            self.max_depth = len(self.groups)

            self._branch_log = []
            self._branch_iteration = 0
            self._log_branching = False

            # min sample sizes
            d = data['X'].shape[1] - has_intercept(data)
            min_samples = kwargs.get('min_samples', 2 * d)
            self._min_samples = min_samples

            min_samples_pos, min_samples_neg = kwargs.get('min_samples_pos', d), kwargs.get('min_samples_neg', d)
            assert isinstance(min_samples_pos, int) and min_samples_pos > 0
            assert isinstance(min_samples_neg, int) and min_samples_neg > 0
            self._min_samples_pos = min_samples_pos
            self._min_samples_neg = min_samples_neg

        else:

            assert all(k in kwargs for k in ['model', 'group_name', 'group_label'])
            group_name, group_label = kwargs.get('group_name'), kwargs.get('group_label')
            assert group_name in parent.root.groups
            assert group_label in parent.root.groups[group_name]['labels']

            self.parent = parent
            self._group_name = str(group_name)
            self._group_label = str(group_label)
            group_id = int(list(self.groups[group_name]['labels']).index(group_label))
            group_indices = self.groups[group_name]['indices'] == group_id
            self._node_indices = np.logical_and(self.parent.node_indices, group_indices)

        # model
        self._model = kwargs.get('model')

        # sample sizes
        self._n = np.sum(self.node_indices)
        self._n_pos = np.sum(self.node_indices_pos)
        self._n_neg = np.sum(self.node_indices_neg)

        # initialize split helper object
        self.split_helper = DecoupledTrainer(self.potential_split_groups, self.node_indices, self.node_indices_pos, self.node_indices_neg)
        self.split_helper.min_samples = self.min_samples
        self.split_helper.min_samples_pos = self.min_samples_pos
        self.split_helper.min_samples_neg = self.min_samples_neg
        self.check_rep()


    def check_rep(self):

        assert callable(self.training_handle)
        assert callable(self.scoring_handle)
        assert callable(self.selection_handle)

        if self.has_model:
            assert isinstance(self.model, ClassificationModel)

        assert self.min_samples_pos >= 1
        assert self.min_samples_neg >= 1
        assert self.min_samples >= self.min_samples_pos + self.min_samples_neg
        assert self.n >= self.min_samples
        assert self.n == self.n_pos + self.n_neg

        assert self.root.height <= self.max_depth
        assert self.depth <= self.max_depth
        if self.depth == self.max_depth:
            assert self.terminal

        if self.split_attempt:
            assert not self.terminal
            assert self.has_model
            assert self.halted or len(self.children) >= 2

        if self.halted:
            assert not self.terminal
            assert self.split_attempt
            assert self.has_model


    def check_tree(self):
        """
        checks the integrity of the entire tree
        :return:
        """

        # check tree must be run from the root
        if not self.is_root:
            return self.root.check_tree()

        for d in self.descendants:

            assert d.has_model, \
                'child nodes must have a model'

            assert d.group_name in self.groups, \
                "group_name must belong to groups"

            assert d.group_name not in [a.group_name for a in d.ancestors], \
                'group was previously branched on'

            assert d.group_label in self.groups[d.group_name]['labels'], \
                "group_label must belong to groups[group_name]['labels']"

            assert d.group_name not in [a.group_name for a in d.descendants], \
                'group is branched on later'

            if len(d.children) > 0:
                assert len(d.children) >= 2
                assert d.child_indices.shape == (len(d.node_indices), len(d.leaves))
                assert np.all(np.sum(d.child_indices, axis = 1) == d.node_indices)

            assert len(d.siblings) >= 1

            assert all(map(lambda s: d.group_name == s.group_name, d.siblings)), \
                'group names for siblings are not identical'

            assert all(map(lambda s: d.group_label != s.group_label, d.siblings)), \
                'group labels for siblings are not distinct'

        return True


    #### printing ####


    def __str__(self):
        return self.info


    def __repr__(self):
        return self.info


    @property
    def name(self):
        if self.is_root:
            s = str(DecouplingTree.ROOT_GROUP_NAME)
        else:
            s = []
            for name, label in self.partition_splits:
                s.append('%s:%s' % (name, label))
            s = ' & '.join(s)
        return s


    @property
    def info(self):

        info = [
            'node: %s' % self.name,
            'n (+/-): %d (%d,%d)' % (self.n, self.n_pos, self.n_neg),
            'children: %d' % len(self.children),
            'terminal: %s' % ('True' if self.terminal else 'False'),
            'has_model: %s' % ('True' if self.has_model else 'False'),
            ]

        if self.split_attempt and self.halted:
            info.extend(['split: halted'])
        elif self.split_attempt:
            info.extend(['split: attempted'])
        else:
            info.extend(['split: no attempt'])

        if self.has_model:

            mistakes = self.mistakes()
            error_rate = np.mean(mistakes)
            error_count = int(np.sum(mistakes))
            model_info = [
                '-' * 20,
                'train error: %1.1f%% (%d/%d)' % (100.0 * error_rate, error_count, self.n)
                ]

            info.extend(model_info)

            if len(self.children) > 0:
                X, Y = self.data['X'], self.data['Y']
                for c in self.children:
                    idx = c.node_indices
                    mistakes = self.mistakes(X[idx, :], Y[idx])
                    error_rate = np.mean(mistakes)
                    error_count = int(np.sum(mistakes))
                    child_info = ['-' * 5,
                                  'child group: %s' % c.group_label,
                                  'training error: %1.1f%% (%d/%d)' % (100.0 * error_rate, error_count, c.n)
                                  ]
                    info.extend(child_info)
        info = [' ', '=' * 20] + info + ['=' * 20]
        return '\n'.join(info)


    def render(self):
        if self.is_root:
            print(RenderTree(self).by_attr("_render_lines"))


    @property
    def _render_lines(self):

        group_string = '%s:%s' % (self._group_name, self._group_label)

        partition = self.partition
        if partition is not None:
            if self in partition:
                group_string = '[%s]' % group_string
            elif any([d in partition for d in self.descendants]):
                group_string = '%s' % group_string
            else:
                group_string = '#### %s' % group_string

        lines = ['%s (n = %d)' % (group_string, self.n)]

        if self.has_model:
            mistakes = self.mistakes()
            model_string = '-error: %1.1f%% (%d)' % (100.0 * np.mean(mistakes), int(np.sum(mistakes)))
            lines.append(model_string)

        if not self.is_root:
            parents = list(self.ancestors)
            parents.reverse()
            for p in parents:
                mistakes = p.mistakes(self.X, self.y)
                parent_string = '-error@%s: %1.1f%% (%d)' % (p.name, 100.0 * np.mean(mistakes), int(np.sum(mistakes)))
                lines.append(parent_string)

        lines.append('\n')
        return lines


    #### properties defined at ROOT for entire TREE ####

    @property
    def data(self):
        if self.is_root:
            return self._data
        else:
            return self.root.data


    @property
    def groups(self):
        if self.is_root:
            return self._groups
        else:
            return self.root.groups


    @property
    def sample_indices_pos(self):
        if self.is_root:
            return self._sample_indices_pos
        else:
            return self.root.sample_indices_pos


    @property
    def sample_indices_neg(self):
        if self.is_root:
            return self._sample_indices_neg
        else:
            return self.root.sample_indices_neg


    @property
    def min_samples_pos(self):
        if self.is_root:
            return int(self._min_samples_pos)
        else:
            return self.root.min_samples_pos


    @min_samples_pos.setter
    def min_samples_pos(self, n):
        if self.is_root:
            assert is_integer(n), 'min samples must be integer'
            assert n >= 2, 'min samples must be at least 2'
            self._min_samples_pos = min(int(n), self.n_pos)
        else:
            self.root.min_samples_pos(self, n)


    @property
    def min_samples_neg(self):
        if self.is_root:
            return int(self._min_samples_neg)
        else:
            return self.root.min_samples_neg


    @min_samples_neg.setter
    def min_samples_neg(self, n):
        if self.is_root:
            assert is_integer(n), 'min samples must be integer'
            assert n >= 2, 'min samples must be at least 2'
            self._min_samples_neg = min(int(n), self.n_neg)
        else:
            self.root.min_samples_neg(self, n)


    @property
    def min_samples(self):
        if self.is_root:
            return int(self._min_samples)
        else:
            return self.root.min_samples


    @min_samples.setter
    def min_samples(self, n):
        if self.is_root:
            assert is_integer(n), 'min samples must be integer'
            assert n >= 2, 'min samples must be at least 2'
            self._min_samples = min(int(n), self.n)
            self._min_samples = max(self._min_samples, self._min_samples_pos + self._min_samples_neg)
        else:
            self.root.min_samples(self, n)


    @property
    def max_depth(self):
        """
        :return: maximum depth of the tree
        """
        if self.is_root:
            return int(self._max_depth)
        else:
            return self.root.max_depth


    @max_depth.setter
    def max_depth(self, d):
        if self.is_root:
            assert is_integer(d), 'max depth must be integer'
            assert d >= 0, 'max depth must be non-negative'
            self._max_depth = min(int(d), len(self.groups))
        else:
            self.root.max_depth(self, d)


    @property
    def training_handle(self):
        """
        :return: function handle used for training
        """
        if self.is_root:
            return self._training_handle
        else:
            return self.root.training_handle


    @training_handle.setter
    def training_handle(self, handle):
        if not self.is_root:
            self.root.training_handle = handle
        else:
            if self.split_attempt:
                raise AttributeError('cannot change training handle after tree has already been branched')
            assert callable(handle)
            spec = getfullargspec(handle)
            for arg in self.TRAINING_HANDLE_ARGS:
                assert arg in spec.args
            self._training_handle = handle


    @property
    def scoring_handle(self):
        """
        :return: function handle used to score different partitions of the data
        """
        if self.is_root:
            return self._scoring_handle
        else:
            return self.root._scoring_handle


    @scoring_handle.setter
    def scoring_handle(self, handle):
        if not self.is_root:
            self.root.scoring_handle = handle
        else:
            if self.split_attempt:
                raise AttributeError('cannot change scoring handle after tree has already been branched')
            assert callable(handle)
            # spec = getfullargspec(handle)
            # for arg in self.SCORING_HANDLE_ARGS:
            #     assert arg in spec.args
            self._scoring_handle = handle


    @property
    def selection_handle(self):
        """
        :return: function handle used to branch_node by choosing partition among multiple feasible partitions
        """
        if self.is_root:
            return self._selection_handle
        else:
            return self.root.selection_handle


    @selection_handle.setter
    def selection_handle(self, handle):
        if not self.is_root:
            self.root.selection_handle = handle
        else:
            if self.split_attempt:
                raise AttributeError('cannot change selection handle after tree has already been branched')
            assert callable(handle)
            # spec = getfullargspec(handle)
            # for arg in self.SELECTION_HANDLE_ARGS:
            #     assert arg in spec.args
            self._selection_handle = handle

    #### properties defined for NODE at INITIALIZATION ####

    @property
    def n(self):
        return int(self._n)


    @property
    def n_pos(self):
        return int(self._n_pos)


    @property
    def n_neg(self):
        return int(self._n_neg)


    @property
    def X(self):
        return np.array(self.data['X'][self._node_indices, :])


    @property
    def y(self):
        return np.array(self.data['Y'][self._node_indices])


    @property
    def group_name(self):
        return str(self._group_name)


    @property
    def group_label(self):
        return str(self._group_label)


    @property
    def node_indices(self):
        return self._node_indices


    @property
    def node_indices_pos(self):
        return np.logical_and(self.node_indices, self.root.sample_indices_pos)


    @property
    def node_indices_neg(self):
        return np.logical_and(self.node_indices, self.root.sample_indices_neg)


    @property
    def child_indices(self):
        if self.is_leaf:
            return self.node_indices
        else:
            return np.transpose(list(map(lambda l: l.node_indices, self.leaves)))


    @property
    def path_nodes(self):
        node_list = []
        node = self
        while True:
            node_list.append(node)
            if node.is_root:
                break
            else:
                node = node.parent
        return node_list


    #### TREE METHODS ####

    @property
    def leaves(self):
        if self.is_leaf:
            return [self]
        else:
            return list(filter(lambda n: n.is_leaf, self.descendants))


    @property
    def other_leaves(self):
        if self.is_leaf:
            return list(filter(lambda n: n != self, self.root.leaves))
        else:
            return self.leaves


    def subtrees(self):

        """
        returns list of all pointers to all possible subtrees
        :return:
        """

        assert self.is_root
        initial_tree = DecoupledLeafSet(leaf_set = self.leaves)
        subtree_list = [initial_tree]
        stack = [initial_tree]
        while stack:
            t = stack.pop()
            for s in t.collapsed_trees:
                if s not in subtree_list:
                    subtree_list.append(s)
                    stack.append(s)

        assert all(map(lambda s: s.has_models, subtree_list))
        assert all(map(lambda s: s.is_partition, subtree_list))

        return subtree_list


    #### paths ####
    @property
    def path_splits(self):
        return list(map(lambda n: (n.group_name, n.group_label), self.path))


    @property
    def path_labels(self):
        return list(map(lambda n: n.group_label, self.path))


    @property
    def path_names(self):
        return list(map(lambda n: n.group_name, self.path))


    @property
    def partition_splits(self):
        return list(map(lambda n: (n.group_name, n.group_label), self.path[1:]))


    #### model training at the node ####
    def filter_data(self, indices = None):

        if indices is None:
            return self.data
        else:
            return {
                'X': np.array(self.data['X'][indices, :]),
                'Y': np.array(self.data['Y'][indices]),
                'variable_names': list(self.data['variable_names'])
                }


    def contains(self, group_names, group_values):
        """
        return indices for samples at this node that match a given group
        :param group_names: names of the group attributes (e.g., AgeGroup)
        :param group_values: values of the group attributes (e.g., Old)
        :return: indices of sample points for individuals in this group (for this node)
        """
        if self.is_root:
            idx = np.ones(group_values.shape[0], dtype = bool)
        else:
            names, labels = zip(*self.partition_splits)
            common_name_idx = [group_names.index(n) for n in names]
            idx = np.all(group_values[:, common_name_idx] == labels, axis = 1)
        return idx


    @property
    def has_model(self):
        return self._model is not None


    def train_model(self, overwrite = False):
        if (not self.has_model) or overwrite:
            self._model = self.training_handle(data = self.filter_data(indices = self.node_indices))
            assert self.has_model


    @property
    def model(self):
        """
        :return: model at this node
        """
        if self.has_model:
            return deepcopy(self._model)
        else:
            raise ValueError("no model at node")


    def predict(self, X):
        """
        predict using the model at this node
        :param X:
        :return:
        """
        if self.has_model:
            return self.model.predict(X)
        else:
            raise ValueError("no model at node")


    def mistakes(self, X = None, y = None):

        if not self.has_model:
            raise ValueError("no model at node")

        if X is None and y is None:
            mistakes = np.not_equal(self.y, self.model.predict(self.X))
        else:
            assert X is not None and y is not None
            assert isinstance(X, np.ndarray)
            assert isinstance(y, np.ndarray)
            assert len(y) == X.shape[0]
            mistakes = np.not_equal(y.flatten(), self.model.predict(X))

        return mistakes


    def error(self, X = None, y = None):
        return np.mean(self.mistakes(X, y))


    #### groups that can be split on ####

    @property
    def potential_split_names(self):
        """
        returns names of groups that can still be branched on at this node
        """
        return set(self.root.groups.keys()).difference(self.path_names)


    @property
    def potential_split_groups(self):
        return {d: self.root.groups[d] for d in self.potential_split_names}


    @property
    def has_potential_splits(self):
        """
        :return: True if there exists any group that can be split on at this node
        """
        return len(self.potential_split_names) > 0


    @property
    def feasible_split_names(self):
        self.split_helper.propagate_infeasibility()
        return self.split_helper.feasible_names


    @property
    def feasible_split_groups(self):
        return {d: self.root.groups[d] for d in self.feasible_split_names}


    @property
    def has_feasible_splits(self):
        return len(self.split_helper.feasible_names) > 0


    #### branching state ####
    @property
    def split_attempt(self):
        return bool(self._split_attempt)


    @split_attempt.setter
    def split_attempt(self, flag):
        assert isinstance(flag, bool)
        if flag:
            assert self.has_model
        self._split_attempt = bool(flag)


    @property
    def halted(self):
        return self._halted


    @halted.setter
    def halted(self, flag):
        assert isinstance(flag, bool)
        if flag:
            assert self.has_model
            assert self.split_attempt
        self._halted = bool(flag)


    @property
    def is_fully_grown(self):
        return not any(map(lambda n: n.splittable, self.leaves))


    @property
    def splittable(self):
        return self.is_leaf and (not self.terminal) and (not self.halted)


    @property
    def terminal(self):
        return (self.depth == self.max_depth) or (not self.has_potential_splits)


    ##### branch logging ####

    @property
    def branch_iteration(self):
        return int(self._branch_iteration)


    @branch_iteration.setter
    def branch_iteration(self, k):
        assert isinstance(k, int)
        assert k > self._branch_iteration
        self._branch_iteration = k


    @property
    def log_branching(self):
        if self.is_root:
            return self._log_branching
        else:
            return self.root.log_branching


    @log_branching.setter
    def log_branching(self, flag):
        if self.is_root:
            assert isinstance(flag, bool)
            self._log_branching = flag
        else:
            self.root.log_branching = flag


    @property
    def branch_log(self):
        if self.is_root:
            return self._branch_log
        else:
            return self.root.branch_log


    ##### branching  ####

    def train_split_models(self):

        sh = self.split_helper
        sh.set_sample_infeasibility(propagate_to_group = True)

        while not sh.finished():
            name, label, idx = sh.next()
            model = self.training_handle(data = self.filter_data(indices = idx))
            self.split_helper.set_model(name, label, model)


    def branch(self):

        """ split one node after checking all possible splits in the current tree """
        assert self.is_root

        # for each leaf, and each split, train all models
        splittable_leaves = [l for l in self.leaves if l.splittable]
        for l in splittable_leaves:
            l.train_split_models()
            l.split_attempt = True
            l.halted = len(l.feasible_split_groups) == 0

        if self.is_fully_grown:
            return

        # recompute splittable since some leaves may have been halted
        splittable_leaves = [l for l in self.leaves if l.splittable]

        # score each split based on envy-freeness of the partition
        potential_splits = []
        pooled_model = self.root.predict
        for leaf in splittable_leaves:

            other_leaves = [l for l in self.leaves if l != leaf]
            for group_name, group_info in leaf.feasible_split_groups.items():

                split_leaves = []
                for i, label in zip(group_info['ids'], group_info['labels']):

                    split_leaf = CandidateNode(model = leaf.split_helper.get_model(group_name, label),
                                               indices = np.logical_and(leaf.node_indices, i == group_info['indices']),
                                               group_names = leaf.partition_names + [group_name],
                                               group_labels = leaf.partition_labels + [label],
                                               parent = self)

                    split_leaves.append(split_leaf)

                partition = DecoupledLeafSet(leaf_set = split_leaves + other_leaves, node_type = NodeMixin)
                partition.score = self.scoring_handle(data = self.data, partition = partition, pooled_model = pooled_model)

                split_info = {
                    'leaf': leaf,
                    'group_name': group_name,
                    'partition': partition,
                    'score': partition.score,
                    }

                potential_splits.append(split_info)

        # choose best tree
        best_idx = self.selection_handle(potential_splits)
        split_leaf = potential_splits[best_idx]['leaf']
        split_name = potential_splits[best_idx]['group_name']

        # add child nodes to tree
        for split_label in self.groups[split_name]['labels']:
            DecouplingTree(parent = split_leaf,
                           group_name = split_name,
                           group_label = split_label,
                           model = split_leaf.split_helper.get_model(split_name, split_label))

        # update history
        self.branch_iteration = self.branch_iteration + 1
        if self.log_branching:
            potential_splits[best_idx]['partition'].selected = True
            for s in potential_splits:
                p = s['partition']
                p.order = self.branch_iteration
                self._branch_log += [{'iteration': self.branch_iteration, 'score': s['partition'].score, 'partition': s['partition']}]


    def grow(self, print_flag = True, log_flag = True):
        """
        branch_node recursively until fully trained
        :return:
        """
        self.log_branching = log_flag
        self.train_model()

        while not self.is_fully_grown:
            self.branch()
            if print_flag:
                self.render()
            self.check_tree()

    ### pruning methods ####

    def prune(self, alpha = 0.1, test_correction = True, atomic_groups = True, objectives = ['group_gain_min', 'group_gain_max', 'group_gain_med'], **data_args):

        subtrees = [t for t in self.subtrees() if len(t) >= 2]
        stats = []
        if atomic_groups:

            for p in subtrees:

                keep_idx = np.greater(p.group_sample_size_stats(metric_type = 'n'), 0)
                group_sizes = p.group_sample_size_stats(metric_type = 'n', **data_args)
                group_weights = group_sizes / np.nansum(group_sizes)
                gain = p.group_decoupling_stats(metric_type = 'error_gap', parent_type = 'root', **data_args)
                gain = gain[keep_idx]
                group_weights = group_weights[keep_idx]
                pvals_gain = p.group_decoupling_stats(metric_type = 'pvalue', parent_type = 'root', **data_args)
                pvals_switch = p.group_switch_stats(metric_type = 'pvalue', **data_args)
                training_sizes = p.partition_sample_size_stats(parent_type = 'self', metric_type = 'n', **data_args)

                if test_correction:
                    n_tests = np.isfinite(pvals_switch).sum() + np.isfinite(pvals_gain).sum()
                else:
                    n_tests = 1.0

                stats.append({
                    'name': p.name,
                    'n_leaves': len(p),
                    'n_missing_data': np.sum(group_sizes == 0),
                    'alpha': alpha / n_tests,
                    'p_violation_min': np.minimum(np.nanmin(pvals_gain), np.nanmin(pvals_switch)),
                    'violations_switch': np.less_equal(pvals_gain, (alpha / n_tests)).sum(),
                    'violations_decouple': np.less_equal(pvals_gain, (alpha / n_tests)).sum(),
                    'weighted_gain': group_weights.dot(gain),
                    'group_gain_min': np.min(gain),
                    'group_gain_med': np.median(gain),
                    'group_gain_max': np.max(gain),
                    'sample_size_min': np.min(training_sizes),
                    'sample_size_med': np.median(training_sizes),
                    'sample_size_max': np.max(training_sizes),
                    })
        else:

            for p in subtrees:

                group_sizes = p.partition_sample_size_stats(parent_type = 'self', metric_type = 'n', **data_args)
                group_weights = group_sizes / np.nansum(group_sizes)
                gain = p.partition_decoupling_stats(metric_type = 'error_gap', parent_type = 'root', **data_args)
                pvals_gain = p.partition_decoupling_stats(metric_type = 'pvalue', parent_type = 'root', **data_args)
                pvals_switch = p.partition_stat_matrix(metric_type = 'pvalue', **data_args)

                if test_correction:
                    n_tests = np.isfinite(pvals_switch).sum() + np.isfinite(pvals_gain).sum()
                else:
                    n_tests = 1.0

                stats.append({
                    'name': p.name,
                    'n_leaves': len(p),
                    'n_missing_data': np.sum(group_sizes == 0),
                    'alpha': alpha / n_tests,
                    'p_violation_min': np.minimum(np.nanmin(pvals_gain), np.nanmin(pvals_switch)),
                    'violations_switch': np.less_equal(pvals_gain, (alpha / n_tests)).sum(),
                    'violations_decouple': np.less_equal(pvals_gain, (alpha / n_tests)).sum(),
                    'weighted_gain': group_weights.dot(gain),
                    'group_gain_min': np.min(gain),
                    'group_gain_med': np.median(gain),
                    'group_gain_max': np.max(gain),
                    'sample_size_min': np.min(group_sizes),
                    'sample_size_med': np.median(group_sizes),
                    'sample_size_max': np.max(group_sizes),
                    })

        df = pd.DataFrame(stats)
        df = df.query('p_violation_min >= alpha')
        df = df.sort_values(objectives, ascending = False)
        self.partition = subtrees[df.index[0]]
        self.render()
        return df


    @property
    def partition(self):
        if self.is_root:
            return self._partition
        else:
            return self.root.partition

    @partition.setter
    def partition(self, partition):
        if self.is_root:
            assert isinstance(partition, DecoupledLeafSet)
            assert partition.has_models
            assert partition.is_partition
            assert partition.node_type == DecouplingTree

            # check that we have a valid partition
            for l in partition.leaves:
                assert l in self.root.descendants

            self._partition = DecoupledLeafSet(partition.leaves)
        else:
            self.root.partition = partition

    ### todo remove

    def predict_at_partition(self, X, group_names = None, group_values = None):
        """
        predict using the leaves from this node

        :param X:
        :param group_names:
        :param group_values:
        :return:
        """
        if self.partition is None:
            leaves = self.leaves
        else:
            leaves = self._partition.leaves

        assert group_names is not None and group_values is not None
        assert all(map(lambda l: map(lambda n: n in group_names, l.partition_names), leaves))

        yhat = np.repeat(np.nan, X.shape[0])
        for l in leaves:
            idx = l.contains(group_names, group_values)
            yhat[idx] = l.predict(X[idx, ])
        assert np.all(np.isfinite(yhat))
        return yhat

    @property
    def partition_names(self):
        return list(map(lambda n: n.group_name, self.path[1:]))

    @property
    def partition_labels(self):
        return list(map(lambda n: n.group_label, self.path[1:]))


class CandidateNode(NodeMixin):

    def __init__(self, model, indices, group_names, group_labels, parent = None):

        self._group_names = group_names
        self._group_labels = group_labels
        self._model = model
        self._indices = indices

        assert parent is None or isinstance(parent, DecouplingTree)
        self._parent_node = parent


    def contains(self, group_names, group_values):
        common_names = [group_names.index(n) for n in self._group_names]
        return np.all(group_values[:, common_names] == self._group_labels, axis = 1)

    @property
    def node_indices(self):
        return np.array(self._indices)

    @property
    def path_nodes(self):
        node_list = [self] + self._parent_node.path_nodes
        return node_list

    def predict(self, X):
        return self._model.predict(X)

    @property
    def model(self):
        return self._model

    @property
    def has_model(self):
        return True

    @property
    def partition_splits(self):
        return [(n, l) for n,l in zip(self._group_names, self._group_labels)]

    @property
    def partition_names(self):
        return self._group_names

    @property
    def partition_labels(self):
        return self._group_labels

    @property
    def name(self):
        s = []
        for name, label in zip(self._group_names, self._group_labels):
            s.append('%s:%s' % (name, label))
        s = ' & '.join(s)
        return s


class DecoupledLeafSet(object):

    _CONTAINMENT_CODE_MISSING = -1
    _CONTAINMENT_CODE_SUBSET = 0
    _CONTAINMENT_CODE_EXACT = 1
    _COMPARISON_METRIC_TYPES = {'error', 'error_gap', 'mistakes', 'mistakes_gap', 'pvalue'}
    _PARENT_TYPES = ['root', 'next', 'best', 'self']

    def __init__(self, leaf_set, node_type = DecouplingTree):

        """
        Wrapper class to represent and manipulate subtrees from a tree
        Must be initialized with a list of leaves for a tree (leaf_set), or a valid tree object (tree)

        :param leaf_set: list containing the leaves of a tree (must be distinct)
        :param node_type: class of each node in the tree (either NodeMixin, or EnvyFreeTree)
        """
        assert isinstance(node_type, object)
        assert isinstance(leaf_set, list)
        self._score = float('nan')
        self._order = -1
        self._selected = False
        self._node_type = node_type
        self._leaf_set = list(leaf_set)
        assert self._check_rep()


    def __len__(self):
        return len(self._leaf_set)


    def __str__(self):
        s = ['EnvyFreePartition (%d parts)' % len(self)]
        s += ['%s' % l.name for l in self._leaf_set]
        return '\n'.join(s)


    def __repr__(self):
        return '{%s}' % ','.join('%s' % l.name for l in self._leaf_set)


    def __eq__(self, other):
        return set(self._leaf_set) == set(other.leaves)


    def __contains__(self, node):
        return any([node == l for l in self._leaf_set])


    def _check_rep(self):

        leaf_set = self._leaf_set
        assert isinstance(leaf_set, list)
        n = len(leaf_set)
        assert n > 0, 'leaf set must be non-empty'
        assert n == len(list(set(leaf_set))), 'leaf set is not unique'

        if self.node_type == DecouplingTree:

            for l in leaf_set:
                assert isinstance(l, self.node_type)

            if n == 1:
                assert leaf_set[0].is_root

        elif self.node_type == CandidateNode:

            for l in leaf_set:
                assert l.is_root
                assert isinstance(l, self.node_type)

            assert not self.attached

        elif self.node_type == NodeMixin:

            if n >= 2:
                for l in leaf_set:
                    if isinstance(l, CandidateNode):
                        assert l.is_root
                    elif isinstance(l, DecouplingTree):
                        assert not l.is_root

            assert not self.attached

        return True


    @property
    def name(self):
        return ','.join('%s' % l.name for l in self._leaf_set)

    
    
    @property
    def split_names(self):
        return [l.name for l in self._leaf_set]


    @property
    def score(self):
        return self._score


    @score.setter
    def score(self, s):
        assert isinstance(s, (float, int))
        assert np.isfinite(s)
        self._score = s


    @property
    def order(self):
        return self._order



    @order.setter
    def order(self, k):
        assert isinstance(k, int)
        assert k >= 0
        self._order = int(k)


    @property
    def node_type(self):
        return self._node_type


    @property
    def selected(self):
        return self._selected


    @selected.setter
    def selected(self, flag):
        assert isinstance(flag, bool)
        self._selected = True


    @property
    def attached(self):
        check_type = all([isinstance(l, DecouplingTree) for l in self._leaf_set])
        root_name = DecouplingTree.ROOT_GROUP_NAME
        if len(self) == 1:
            check_root = self._leaf_set[0].name == root_name
        else:
            check_root = all([l.root.name == root_name for l in self._leaf_set])
        return check_type and check_root


    @property
    def leaves(self):
        """
        :return: list containing pointers to the leaf nodes of the tree
        """
        return list(self._leaf_set)


    @property
    def indices(self):
        return [l.node_indices for l in self._leaf_set]


    @property
    def is_partition(self):
        """
        :return: True iff the current set of leaves represents a valid partition
        """
        return np.sum(list(map(lambda l: l.node_indices, self._leaf_set)), axis = 0).all()


    @property
    def has_models(self):
        """
        :return: True iff the current set of leaves contains fully trained models
        """
        return all(map(lambda l: l.has_model, self._leaf_set))


    @property
    def parent_nodes(self):
        return list({l.parent for l in self._leaf_set})


    @property
    def collapsable_nodes(self):
        nodes = []
        for p in self.parent_nodes:
            if set(p.children).issubset(self._leaf_set):
                nodes.append(p)
        return nodes


    def collapse(self, parent_node):
        """
        :param parent_node:
        :return:
        """
        assert isinstance(parent_node, self.node_type)
        assert parent_node in self.collapsable_nodes
        collapsed_leaf_set = [l for l in self._leaf_set if l.parent != parent_node]
        collapsed_leaf_set.append(parent_node)
        return DecoupledLeafSet(collapsed_leaf_set)


    @property
    def collapsed_trees(self):
        if len(self) >= 2:
            return [self.collapse(p) for p in self.collapsable_nodes]
        else:
            return self.leaves


    @property
    def predictors(self):
        return [l.predict for l in self._leaf_set]


    def predict(self, X, group_names, group_values):
        """
        predict using all of the leaves in this partition
        :param X:
        :param group_names:
        :param group_values:
        :return:
        """
        assert all(map(lambda l: map(lambda n: n in group_names, l.partition_names), self._leaf_set))
        yhat = np.repeat(np.nan, X.shape[0])
        for l in self._leaf_set:
            idx = l.contains(group_names, group_values)
            yhat[idx] = l.predict(X[idx, ])
        assert np.all(np.isfinite(yhat))
        return yhat


    #### MATCHING AND ANALYSIS

    @property
    def data(self):
        assert self.node_type == DecouplingTree
        return self._leaf_set[0].root.data


    @property
    def groups(self):
        assert self.node_type == DecouplingTree
        return self._leaf_set[0].root.groups


    def splits(self, groups = None):
        """
        :param groups: group data structure
        :return: list of splits, where each split = [(group_name, group_label)] for all possible subgroups
        """
        if groups is None:
            groups = self.groups
        splits = [[(k, l) for l in v['labels']] for k, v in groups.items()]
        return list(itertools.product(*splits))


    def _parse_data_args(self, **kwargs):
        """
        helper function to parse X, y, group_names, and group_values
        from keyword inputs to different methods

        :param kwargs:
        :return:
        """
        if len(kwargs) == 0:
            return to_group_data(data = self.data, groups = self.groups, stat_field = 'train')

        elif set(['X', 'y', 'group_names', 'group_values']).issubset(kwargs):
            return kwargs['X'], kwargs['y'], kwargs['group_names'], kwargs['group_values']

        elif set(['data', 'groups']).issubset(kwargs):
            return to_group_data(data = kwargs['data'], groups = kwargs['groups'], stat_field = kwargs.get('stat_field') or 'train')

        else:
            raise ValueError('unsupport input arguments')


    def _parse_split(self, *args, **kwargs):
        """
        helper function to parse a split from arguments or keyword arguments to different methods
        split(G, v) == (G[1] = v[1], G[2] = v[2]... ) <- set identifying a specific groups
        :param args:
        :param kwargs:
        :return: split as a set
        """

        assert len(args) == 0 or len(kwargs) == 0
        split = None

        if len(args) == 1:
            split = args[0]
        elif len(args) == 2:
            assert len(args[0]) == len(args[1])
            split = zip(args[0], args[1])
        elif 'split' in kwargs:
            split = kwargs['split']
        elif ['group_names', 'group_labels']:
            split = zip(kwargs['group_names'], kwargs['group_labels'])

        if split is None:
            raise ValueError('could not parse split from args or kwargs')

        split = set(split)

        return split


    def _drop_missing_splits(self, splits, group_values):
        new = []
        for s in splits:
            group_labels = [t[1] for t in s]
            if np.all(group_values == group_labels, axis = 1).any():
                new.append(s)
        return new


    def match(self, *args, **kwargs):
        """
        V[i] = True <-> leaf i contains samples from split
        :param split
        :param group_names:
        :param group_labels:
        :return: a matching vector V such that

        -1 if group is not contained in a leaf
         0 if group is contained along with other groups in a leaf
         1 if group has own leaf
         k if group is contained in k leaves
        """
        split = self._parse_split(*args, *kwargs)
        return [split.issuperset(l.partition_splits) for l in self._leaf_set]


    def matched_leaves(self, *args, **kwargs):
        """
        return True if this node contains any samples for a group with group_labels
        :param group_names: names of the group attributes (e.g., AgeGroup)
        :param group_values: matrix of group attributes (e.g., Old)
        :param group_labels: labels that must match
        :return: indices of sample points for individuals in this group (for this node)
        """
        return list(itertools.compress(self._leaf_set, self.match(*args, **kwargs)))


    def match_code(self, *args, **kwargs):
        """
        :param group_names:
        :param group_labels:
        :return:

        -1 if group is not contained in a leaf
         0 if group is contained along with other groups in a leaf
         1 if group has own leaf
         k if group is contained in k leaves
        """
        split = self._parse_split(*args, **kwargs)
        match_exact = [split == set(l.partition_splits) for l in self._leaf_set]
        if any(match_exact):
            assert np.sum(match_exact) == 1
            return self._CONTAINMENT_CODE_EXACT

        match_subset = [split.issuperset(l.partition_splits) for l in self._leaf_set]
        match_subset = np.logical_and(match_subset, np.logical_not(match_exact))
        n_matches = np.sum(match_subset)
        if n_matches == 0:
            return self._CONTAINMENT_CODE_MISSING
        elif n_matches == 1:
            return self._CONTAINMENT_CODE_SUBSET
        else:
            return n_matches


    def partition_stat_matrix(self, metric_type = 'error', **kwargs):

        """
        return a matrix computing

        :param metric_type: 'error', 'error_gap', 'mistakes', 'mistake_gap', 'pvalue'

        :param kwargs: can be either:
            - data, groups
            - data, groups, stat_field
            - X, y, group_names, group_values

        :return: n_parts x n_parts matrix M of statistics where:

        - n_parts = # of leaves in partition
        - M[i, j] = value of metric on data in leaf i using model from leaf j
        """

        assert metric_type in DecoupledLeafSet._COMPARISON_METRIC_TYPES
        X, y, group_names, group_values = self._parse_data_args(**kwargs)
        n_parts = len(self)
        M = np.empty(shape = (n_parts, n_parts))

        # fast return for comparison based metrics
        is_root = len(self) == 1 and (self._leaf_set[0].name == DecouplingTree.ROOT_GROUP_NAME)
        if is_root and metric_type not in ['error', 'mistakes']:
            return np.array([np.nan])

        if metric_type in ['error', 'error_gap']:

            for i, leaf in enumerate(self._leaf_set):
                idx = leaf.contains(group_names, group_values)
                ys, Xs = y[idx], X[idx, :]
                for j, other in enumerate(self._leaf_set):
                    M[i, j] = np.not_equal(ys, other.predict(Xs)).mean()

        elif metric_type in ['mistakes', 'mistakes_gap']:

            for i, leaf in enumerate(self._leaf_set):
                idx = leaf.contains(group_names, group_values)
                ys, Xs = y[idx], X[idx, :]
                for j, other in enumerate(self._leaf_set):
                    M[i, j] = np.not_equal(ys, other.predict(Xs)).sum()

        elif metric_type == 'pvalue':

            for i, leaf in enumerate(self._leaf_set):
                idx = leaf.contains(group_names, group_values)
                ys, Xs = y[idx], X[idx, :]
                hs = leaf.predict(X[idx, :])
                for j, other in enumerate(self._leaf_set):
                    if i != j:
                        M[i, j], _ = exact_mcn_test(y = ys, yhat1 = hs, yhat2 = other.predict(Xs))

            np.fill_diagonal(M, np.nan)

        if '_gap' in metric_type:

            M = M - np.diag(M)[:, None]
            np.fill_diagonal(M, np.nan)

        return M


    def partition_sample_size_stats(self, metric_type = 'n', **kwargs):

        _, y, group_names, group_values = self._parse_data_args(**kwargs)
        S = np.zeros(len(self))

        for i, l in enumerate(self._leaf_set):
            idx = l.contains(group_names, group_values)
            if np.any(idx):
                if metric_type in ('n', 'p'):
                    S[i] = np.sum(idx)
                elif metric_type in ('n_pos', 'p_pos'):
                    S[i] = np.sum(y[idx] == 1)
                elif metric_type in ('n_neg', 'p_neg'):
                    S[i] = np.sum(y[idx] == -1)

        assert len(y) == np.sum(S)

        if metric_type in ('p', 'p_pos', 'p_neg'):
            S = S / len(y)

        return S


    def partition_decoupling_stats(self, metric_type = 'error_gap', parent_type = 'root', **kwargs):

        assert parent_type in DecoupledLeafSet._PARENT_TYPES
        X, y, group_names, group_values = self._parse_data_args(**kwargs)
        S = np.repeat(np.nan, len(self))

        # fast return for comparison based metrics
        no_parents = parent_type == 'self'
        is_root = len(self) == 1 and (self._leaf_set[0].name == DecouplingTree.ROOT_GROUP_NAME)
        no_comparison_node = is_root or no_parents
        if no_comparison_node and metric_type not in ['error', 'mistakes']:
            return S

        for i, l in enumerate(self._leaf_set):

            idx = l.contains(group_names, group_values)
            ys, Xs = y[idx], X[idx, :]
            predictions = [n.predict(Xs) for n in l.path_nodes]

            if metric_type == 'error':

                errors = [np.not_equal(ys, yhat).mean() for yhat in predictions]
                if parent_type == 'self':
                    metric_value = errors.pop(0)
                elif parent_type == 'root':
                    metric_value = errors.pop(-1)
                elif parent_type == 'next':
                    metric_value = errors.pop(1)
                elif parent_type == 'best':
                    metric_value = np.min(errors)

            elif metric_type in ['error_gap', 'error_relgap']:

                errors = [np.not_equal(ys, yhat).mean() for yhat in predictions]
                base_value = errors.pop(0)
                if parent_type == 'root':
                    comp_value = errors.pop(-1)
                elif parent_type == 'next':
                    comp_value = errors.pop(0)
                elif parent_type == 'best':
                    comp_value = np.min(errors)

                metric_value = comp_value - base_value
                if '_relgap' in metric_type:
                    metric_value = (comp_value - base_value) / base_value

            elif metric_type == 'pvalue':

                yhat1 = predictions.pop(0)
                pvals = [exact_mcn_test(ys, yhat1, yhat)[0] for yhat in predictions]
                if parent_type == 'root':
                    metric_value = pvals[-1]
                elif parent_type == 'next':
                    metric_value = pvals[0]
                elif parent_type == 'best':
                    metric_value = np.min(pvals)

            S[i] = metric_value

        return S


    def group_sample_size_stats(self, metric_type = 'n', drop_missing = False, **kwargs):
        _, y, group_names, group_values = self._parse_data_args(**kwargs)
        splits = self.splits(kwargs.get('groups'))
        if drop_missing:
            splits = self._drop_missing_splits(splits, group_values)

        S = np.repeat(np.nan, len(splits))
        for i, s in enumerate(splits):
            group_labels = [t[1] for t in s]
            idx = np.all(group_values == group_labels, axis = 1)
            if metric_type in ('n', 'p'):
                S[i] = np.sum(idx)
            elif metric_type in ('n_pos', 'p_pos'):
                S[i] = np.sum(y[idx] == 1)
            elif metric_type in ('n_neg', 'p_neg'):
                S[i] = np.sum(y[idx] == -1)
        if metric_type in ('p', 'p_pos', 'p_neg'):
            S = S / len(y)

        return S


    def group_switch_stats(self, metric_type = 'error_gap', drop_missing = False, **kwargs):

        X, y, group_names, group_values = self._parse_data_args(**kwargs)
        splits = self.splits(kwargs.get('groups'))
        if drop_missing:
            splits = self._drop_missing_splits(splits, group_values)

        n_groups, n_models = len(splits), len(self)
        M = np.tile(np.nan, [n_groups, len(self)])
        for i, s in enumerate(splits):
            group_labels = [t[1] for t in s]
            idx = np.all(group_values == group_labels, axis = 1)
            if any(idx):

                ys, Xs = y[idx], X[idx, :]
                assign_idx = np.array(self.match(s))
                match_idx = int(np.flatnonzero(assign_idx))
                other_idx = np.flatnonzero(np.logical_not(assign_idx))

                h = self._leaf_set[match_idx]
                ys_hat = h.predict(Xs)

                for k in other_idx:
                    yk_hat = self._leaf_set[int(k)].predict(Xs)

                    if metric_type == 'pvalue':
                        M[i, k] = exact_mcn_test(ys, ys_hat, yk_hat)[0]

                    elif metric_type == 'error':
                        M[i, k] = np.not_equal(ys, yk_hat).mean()

                    elif metric_type == 'error_gap':
                        M[i, k] = np.not_equal(ys, yk_hat).mean() - np.not_equal(ys, ys_hat).mean()

                    elif metric_type == 'error_relgap':
                        M[i, k] = (np.not_equal(ys, yk_hat).mean() - np.not_equal(ys, ys_hat).mean()) / np.not_equal(ys, ys_hat).mean()

                if metric_type == 'error':
                    M[i, match_idx] = np.not_equal(ys, ys_hat).mean()

                elif metric_type == 'error_relgap':
                    M[i, match_idx] = 0.0

        return M


    def group_decoupling_stats(self, metric_type = 'error_gap', parent_type = 'root',  drop_missing = False, **kwargs):

        assert parent_type in DecoupledLeafSet._PARENT_TYPES
        X, y, group_names, group_values = self._parse_data_args(**kwargs)
        splits = self.splits(kwargs.get('groups'))
        if drop_missing:
            splits = self._drop_missing_splits(splits, group_values)

        leaves = self._leaf_set
        S = np.repeat(np.nan, len(splits))

        if parent_type == 'self':

            if metric_type == 'error':
                for i, s in enumerate(splits):
                    group_labels = [t[1] for t in s]
                    idx = np.all(group_values == group_labels, axis = 1)
                    ys, Xs = y[idx], X[idx, :]
                    yhat = self.predict(Xs, group_names, group_values[idx, :])
                    S[i] = np.not_equal(ys, yhat).mean()

        elif parent_type == 'root':

            for i, s in enumerate(splits):

                group_labels = [t[1] for t in s]
                idx = np.all(group_values == group_labels, axis = 1)
                ys, Xs = y[idx], X[idx, :]

                yhat = self.predict(Xs, group_names, group_values[idx, :])
                yhat_root = leaves[0].root.predict(Xs)

                if metric_type == 'pvalue':
                    S[i], _ = exact_mcn_test(ys, yhat, yhat_root)

                elif metric_type in ['error']:
                    S[i] = np.not_equal(ys, yhat_root).mean()

                elif metric_type in ['error_gap', 'error_relgap']:
                    base_value = np.not_equal(ys, yhat).mean()
                    comp_value = np.not_equal(ys, yhat_root).mean()
                    metric_value = comp_value - base_value
                    if '_relgap' in metric_type:
                        metric_value = metric_value / base_value
                    S[i] = metric_value

        return S


