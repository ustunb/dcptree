import itertools
import numpy as np
from copy import deepcopy
from dcptree.analysis import to_group_data, groups_to_group_data
from dcptree.data import check_data, has_intercept
from dcptree.group_helper import check_groups
from dcptree.classification_models import ClassificationModel
from dcptree.tree import exact_mcn_test


def check_model_assignment(groups_to_models, groups, models):

    assert isinstance(groups_to_models, dict)
    assert isinstance(models, list)

    splits = [[(k, l) for l in v['labels']] for k, v in groups.items()]
    splits = set(itertools.product(*splits))
    assert set(groups_to_models.keys()).issubset(splits), 'mapper should include map every group in the data'


    model_indices = list(range(len(models)))
    assignment_indices = np.array(list(groups_to_models.values()))
    assert np.array_equal(np.unique(assignment_indices), model_indices), 'every model should cover at least one group'
    return True



def build_model_assignment_map(p):
    groups_to_models = {}
    group_labels, group_values = groups_to_group_data(p.groups, stat_field = 'train')
    split_values = np.unique(group_values, axis = 0)

    for vals in split_values:
        s = tuple([(g, z) for g, z in zip(group_labels, vals)])
        vals = vals[:, None].transpose()
        n_matches = 0
        for i, l in enumerate(p.leaves):
            if l.contains(group_labels, vals):
                groups_to_models[s] = i
                n_matches += 1

        assert n_matches == 1


    assert check_model_assignment(groups_to_models, p.groups, models = p.predictors)
    return groups_to_models



class DecoupledClassifierSet(object):


    def __init__(self, data, groups, pooled_model, decoupled_models, groups_to_models):

        # check inputs
        assert check_data(data, ready_for_training = True)
        assert check_groups(groups, data)

        # initialize data
        self._data = {
            'X': np.array(data['X']),
            'Y': np.array(data['Y']),
            'variable_names': list(data['variable_names'])
            }

        self._groups = deepcopy(groups)
        self._pooled_model = pooled_model
        self._decoupled_models = decoupled_models

        group_names, group_values = groups_to_group_data(groups)
        training_values = np.unique(group_values, axis = 0).tolist()
        training_splits = [tuple(zip(group_names, v)) for v in training_values]

        assert isinstance(groups_to_models, dict)
        assert set(training_splits) == set(groups_to_models.keys()), 'mapper should include map every group in the training data'
        assignment_idx = np.array(list(groups_to_models.values()))
        assert np.array_equal(np.unique(assignment_idx), np.arange(len(self))), 'every model should cover at least one group'

        models_to_groups = {k:[] for k in range(len(self))}
        for group_tuple, model_index in groups_to_models.items():
            group_value = [s[1] for s in group_tuple]
            assert len(group_value) == len(group_names)
            models_to_groups[model_index].append(group_value)

        self._splits = training_splits
        self.groups_to_models = groups_to_models
        self.models_to_groups = models_to_groups


    def __len__(self):
        return len(self._decoupled_models)



    def __repr__(self):


        info = [
            'DecoupledClassifierSet',
            '# group attributes: %d' % len(self._groups),
            '# groups: %d' % len(self.groups_to_models),
            '# models: %d' % len(self._decoupled_models),
            ]

        info = info + [', '.join(s) for s in self.split_names]
        return '\n'.join(info)


    @property
    def data(self):
        return self._data


    @property
    def groups(self):
        return self._groups


    @property
    def split_names(self):
        return [['%s = %s' % (a, b) for (a, b) in s] for s in self._splits]


    @property
    def pooled_model(self):
        return self._pooled_model


    @pooled_model.setter
    def pooled_model(self, clf):
        assert callable(clf)
        self._pooled_model = clf


    @property
    def decoupled_models(self):
        return [clf.predict for clf in self._decoupled_models]


    @decoupled_models.setter
    def decoupled_models(self, clf_set):
        assert len(clf_set) >= 2
        assert all([callable(clf) for clf in clf_set])
        self._decoupled_models = clf_set


    def assigned_indices(self, model_index, group_names, group_values):
        assignment_idx = np.repeat(False, group_values.shape[0])
        for s in self.models_to_groups[model_index]:
            assignment_idx = np.logical_or(assignment_idx, np.all(group_values == s, axis = 1))
        return assignment_idx


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


    def _drop_missing_splits(self, splits, group_values):
        new = []
        for s in splits:
            group_labels = [t[1] for t in s]
            if np.all(group_values == group_labels, axis = 1).any():
                new.append(s)
        return new



    def predict(self, X, group_names, group_values):
        """
        predict using all of the leaves in this partition
        :param X:
        :param group_names:
        :param group_values:
        :return:
        """
        yhat = np.repeat(np.nan, X.shape[0])
        for i, clf in enumerate(self._decoupled_models):
            idx = self.assigned_indices(i, group_names, group_values)
            yhat[idx] = clf.predict(X[idx, ])
        assert np.all(np.isfinite(yhat))
        return yhat


    def splits(self, groups = None):
        """
        :param groups: group data structure
        :return: list of splits, where each split = [(group_name, group_label)] for all possible subgroups
        """
        if groups is None:
            groups = self.groups
        splits = [[(k, l) for l in v['labels']] for k, v in groups.items()]
        return list(itertools.product(*splits))


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
        M = np.tile(np.nan, [n_groups, n_models])
        model_idx = np.arange(n_models)

        for i, s in enumerate(splits):

            group_labels = [t[1] for t in s]
            idx = np.all(group_values == group_labels, axis = 1)

            if any(idx):

                ys, Xs = y[idx], X[idx, :]
                match_idx = self.groups_to_models[s]
                other_idx = model_idx[model_idx != match_idx].tolist()

                h = self._decoupled_models[match_idx]
                ys_hat = h.predict(Xs)

                for k in other_idx:
                    yk_hat = self._decoupled_models[k].predict(Xs)

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


    def group_decoupling_stats(self, parent_type = 'root', metric_type = 'error_gap', drop_missing = False, **kwargs):

        X, y, group_names, group_values = self._parse_data_args(**kwargs)
        splits = self.splits(kwargs.get('groups'))
        if drop_missing:
            splits = self._drop_missing_splits(splits, group_values)

        leaves = self._decoupled_models
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

                yhat = self.predict(Xs, group_names, group_values[idx,:])
                yhat_root = self.pooled_model.predict(Xs)

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
        X, y, group_names, group_values = self._parse_data_args(**kwargs)
        n_models = len(self)
        M = np.empty(shape = (n_models, n_models))

        # fast return for comparison based metrics
        is_root = len(self) == 1
        if is_root and metric_type not in ['error', 'mistakes']:
            return np.array([np.nan])

        if metric_type in ['error', 'error_gap']:
            for i, leaf in enumerate(self._decoupled_models):
                idx = self.assigned_indices(i, group_names, group_values)
                ys, Xs = y[idx], X[idx, :]
                for j, other in enumerate(self._decoupled_models):
                    M[i, j] = np.not_equal(ys, other.predict(Xs)).mean()

        elif metric_type in ['mistakes', 'mistakes_gap']:
            for i, leaf in enumerate(self._decoupled_models):
                idx = self.assigned_indices(i, group_names, group_values)
                ys, Xs = y[idx], X[idx, :]
                for j, other in enumerate(self._decoupled_models):
                    M[i, j] = np.not_equal(ys, other.predict(Xs)).sum()

        elif metric_type == 'pvalue':

            for i, leaf in enumerate(self._decoupled_models):
                idx = self.assigned_indices(i, group_names, group_values)
                ys, Xs = y[idx], X[idx, :]
                hs = leaf.predict(X[idx, :])
                for j, other in enumerate(self._decoupled_models):
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

        for i, l in enumerate(self._decoupled_models):
            idx = self.assigned_indices(i, group_names, group_values)
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

        X, y, group_names, group_values = self._parse_data_args(**kwargs)
        S = np.repeat(np.nan, len(self))

        for i, l in enumerate(self._decoupled_models):

            idx = self.assigned_indices(i, group_names, group_values)
            ys, Xs = y[idx], X[idx, :]
            yhat = l.predict(Xs)
            yhat_pooled = self.pooled_model.predict(Xs)

            if metric_type == 'error':

                if parent_type == 'self':
                    metric_value = np.not_equal(ys, yhat).mean()
                elif parent_type == 'root':
                    metric_value = np.not_equal(ys, yhat_pooled).mean()

            elif metric_type in ['error_gap', 'error_relgap']:

                base_value = np.not_equal(ys, yhat).mean()
                if parent_type == 'root':
                    comp_value = np.not_equal(ys, yhat_pooled).mean()

                metric_value = comp_value - base_value
                if '_relgap' in metric_type:
                    metric_value = (comp_value - base_value) / base_value

            elif metric_type == 'pvalue':
                if parent_type == 'self':
                    metric_value = exact_mcn_test(ys, yhat, yhat)[0]
                elif parent_type == 'root':
                    metric_value = exact_mcn_test(ys, yhat, yhat_pooled)[0]

            S[i] = metric_value

        return S

