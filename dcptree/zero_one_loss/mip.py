from dcptree.data import remove_intercept, add_intercept, check_data_required_fields, get_intercept_index, get_variable_indices, get_common_row_indices
from dcptree.classification_models import ClassificationModel
from dcptree.helper_functions import *
from dcptree.zero_one_loss.cplex_helper import *
from copy import deepcopy

def train_zero_one_linear_model(data, time_limit = 60, settings = None, **kwargs):

    #filter settings
    settings = {} if settings is None else settings

    #fit intercept
    for k in ZeroOneLossMIP.SETTINGS.keys():
        if k in kwargs:
            settings[k] = deepcopy(kwargs[k])

    mip = ZeroOneLossMIP(data, settings)
    mip.solve(time_limit = time_limit)
    model = mip.get_classifier()

    return model


class ZeroOneLossMIP(object):

    _print_flag = True

    SETTINGS = {
        #
        'fit_intercept': True,
        'standardize_data': False,
        'compress_data': True,
        #
        'add_coefficient_sign_constraints': True,
        'add_constraints_for_conflicted_pairs': False,
        'use_cplex_indicators_for_mistakes': True,
        'use_cplex_indicators_for_signs': False,
        #
        'w_pos': 1.0,
        'margin': 0.0001,
        'total_l1_norm': 1.00,
        'add_l1_penalty': False,
        #
        'error_total_min': 0.0,
        'error_total_max': 1.0,
        'error_positive_min': 0.0,
        'error_positive_max': 1.0,
        'error_negative_min': 0.0,
        'error_negative_max': 1.0,
        }

    VAR_NAME_FMT = {
        'theta_pos': 'theta_pos_%d',
        'theta_neg': 'theta_neg_%d',
        'theta_sign': 'theta_sign_%d',
        'mistake_pos': 'mistake_pos_%d',
        'mistake_neg': 'mistake_neg_%d',
        }

    CON_NAME_FMT = {
        'mistake_pos': 'p',
        'mistake_neg': 'n',
        'sign_pos': 'def_sign_pos',
        'sign_neg': 'def_sign_neg',
        'norm_limit': 'norm_limit',
        'total_mistakes': 'def_total_mistakes',
        'total_mistakes_pos': 'def_total_mistakes_pos',
        'total_mistakes_neg': 'def_total_mistakes_neg',
        }


    _CPLEX_PARAMETERS = dict(DEFAULT_CPLEX_PARAMETERS)

    def __init__(self, data, settings = None, cpx_parameters = None, print_flag = None):

        # complete cplex parameters
        # todo write function handle for this pattern
        self._cpx_parameters = dict(self._CPLEX_PARAMETERS)
        if cpx_parameters is not None:
            cpx_parameters = dict(cpx_parameters)
            user_keys = set(self._cpx_parameters.keys()).intersection(cpx_parameters.keys())
            for k in user_keys:
                self._cpx_parameters[k] = cpx_parameters[k]


        # complete mip settings
        self._mip_settings = dict(self.SETTINGS)
        if settings is not None:
            settings = dict(settings)
            user_keys = set(self._mip_settings.keys()).intersection(settings.keys())
            for k in user_keys:
                self._mip_settings[k] = settings[k]

        # set print_flag
        self.print_flag = print_flag

        # initialize mip data
        self._mip_data = process_mip_data(data,
                                          standardize = self._mip_settings['standardize_data'],
                                          compress = self._mip_settings['compress_data'],
                                          fit_intercept = self._mip_settings['fit_intercept'])

        # build mip
        self._mip, self._mip_info, self._mip_data = build_mip(data = self._mip_data,
                                                              settings = self._mip_settings,
                                                              print_flag = self.print_flag)

        # overwrite settings
        self._mip_settings = self._mip_info.pop('settings')

        # extract indices of key parameters
        self._indices = self._mip_info.pop('indices')

        # set mip variables
        self.set_mip_parameters(param = self._cpx_parameters)

        self._n_coefficients = len(self._indices['theta_pos'])

        # initialize mip start pool
        self._start_pool = []


    @property
    def print_flag(self):
        return bool(self._print_flag)


    @print_flag.setter
    def print_flag(self, flag):
        if flag is None:
            self._print_flag = ZeroOneLossMIP._print_flag
        elif isinstance(flag, bool):
            self._print_flag = bool(flag)
        else:
            raise AttributeError('print_flag must be boolean or None')


    @property
    def n_coefficients(self):
        """
        :return: number of coefficients in ZeroOneLossMIP
        """
        return int(self._n_coefficients)


    @property
    def mip(self):
        """
        :return: handle to CPLEX mip object
        """
        return self._mip


    @property
    def variables(self):
        """
        :return: handle to CPLEX variables
        """
        return self._mip.variables


    @property
    def solution(self):
        """
        :return: handle to CPLEX solution
        """
        # todo add wrapper if solution does not exist
        return self._mip.solution


    @property
    def solution_info(self):
        """returns information associated with the current best solution for the mip"""
        return get_mip_stats(self._mip)


    @property
    def data(self):
        """

        :return: processed data used to build ZeroOneLossMIP
        """
        return dict(self._mip_data)


    @property
    def settings(self):
        """
        :return: dictionary of settings used for MIP Formulation
        """
        return dict(self._mip_settings)


    @property
    def mip_info(self):
        """
        :return: additional information about MIP formulation
        """
        return dict(self._mip_info)


    @property
    def indices(self):
        """
        :return: dictionary of indices of key variables in MIP Formulatio
        """
        return dict(self._indices)


    def solve(self, time_limit = None, node_limit = None, return_stats = False, return_incumbents = False):

        progress_info = None
        progress_incumbents = None
        attach_stats_callback = return_stats or return_incumbents

        if attach_stats_callback:
            self._add_stats_callback(store_solutions = return_incumbents)

        # update time limit
        if time_limit is not None:
            # TODO store time limit in settings
            self.set_time_limit(time_limit)

        if node_limit is not None:
            # TODO store node limit in settings
            self.set_node_limit(node_limit)

        # attach initial solutions from solution pool
        # todo provide a way to specify effort level for repair
        for sol in self._start_pool:
            self._mip = add_mip_start(cpx = self._mip, solution = sol)

        # solve
        self._mip.solve()

        # model = self.get_model_parameters(solution)
        # # model = (coefficients, threshold)

        if attach_stats_callback:
            progress_info, progress_incumbents = self._stats_callback.get_stats()
            # self._mip.unregister_callback(StatsCallback)

        return self.solution_info, progress_info, progress_incumbents


    def coefficients(self):
        coefs = np.array(self._mip.solution.get_values(self._mip_info['coefficient_idx']))
        coefs = coefs[0:self._n_coefficients] + coefs[self._n_coefficients:(2 * self._n_coefficients)]
        return coefs


    def get_classifier(self):

        coefs = self.coefficients()
        intercept_idx = np.array(self._mip_data['intercept_idx'])
        coefficient_idx = np.array(self._mip_data['coefficient_idx'])

        if self._mip_data['standardize_data']:

            mu = np.array(self._mip_data['X_shift']).flatten()
            sigma = np.array(self._mip_data['X_scale']).flatten()

            coefs = coefs / sigma
            total_shift = coefs.dot(mu)

            coefficients = coefs[coefficient_idx]

            if intercept_idx >= 0:
                intercept = coefs[intercept_idx] - total_shift
            else:
                intercept = -total_shift

        else:
            coefficients = coefs[coefficient_idx]
            if intercept_idx >= 0:
                intercept = coefs[intercept_idx]
            else:
                intercept = 0.0

        predict_handle = lambda X: np.sign(X[:, coefficient_idx].dot(coefficients) + intercept)

        # # setup parameters for model object
        model_info = {
            'intercept': intercept,
            'coefficients': coefficients,
            'coefficient_idx': coefficient_idx,
            }

        training_info = {'method_name': 'zero_one_loss_mip'}
        training_info.update(self.solution_info)
        training_info.update(self.settings)

        model = ClassificationModel(predict_handle = predict_handle,
                                    model_type = ClassificationModel.LINEAR_MODEL_TYPE,
                                    model_info = model_info,
                                    training_info = training_info)

        return model


    def mistake_indicators(self, coefs):

        scores_pos = self._mip_data['U_pos'].dot(coefs) - self._mip_info['margin_pos']
        scores_neg = self._mip_data['U_neg'].dot(coefs) - self._mip_info['margin_neg']
        mistakes_pos = np.not_equal(np.sign(scores_pos), 1.0)
        mistakes_neg = np.not_equal(np.sign(scores_neg), -1.0)

        close_to_margin_pos_idx = np.isclose(scores_pos, 0.0)
        if any(close_to_margin_pos_idx):
            mistakes_pos[close_to_margin_pos_idx] = 0.0

        close_to_margin_neg_idx = np.isclose(scores_neg, 0.0)
        if any(close_to_margin_neg_idx):
            mistakes_neg[close_to_margin_neg_idx] = 0.0

        return mistakes_pos, mistakes_neg


    def total_mistakes(self, coefs = None):
        if coefs is None:
            coefs = self.coefficients()
        mistakes_pos, mistakes_neg = self.mistake_indicators(coefs)
        total_mistakes = mistakes_pos.dot(self._mip_data['n_counts_pos']) + mistakes_neg.dot(self._mip_data['n_counts_neg'])
        return total_mistakes


    def as_coefficient_vector(self, solution):
        coefs = np.array(solution(self._mip_info['theta_pos'])) + np.array(solution(self._mip_info['theta_neg']))
        return coefs


    def as_solution_vector(self, coefs, coef_names = None):
        """

        :param coefs:
        :param coef_names:
        :return:
        """
        # construct theta
        if coef_names is None:
            theta = np.array(coefs, dtype = np.float_)
        else:
            var_idx = self._mip_info['variable_idx']
            n_variables = np.max(list(var_idx.values())) + 1
            theta = np.repeat(np.nan, n_variables)
            for j, name in enumerate(coef_names):
                theta[var_idx[name]] = coefs[j]

        #coefficients
        solution = {
            'theta_pos': np.maximum(theta, 0.0),
            'theta_neg': np.minimum(theta, 0.0),
            }

        # sign variables
        if self._mip_info['add_coefficient_sign_constraints']:
            solution['theta_sign'] = np.array(np.greater(theta, 0.0), dtype = np.float)

        # mistakes
        solution['mistakes_pos'], solution['mistakes_neg'] = self.mistake_indicators(coefs = theta)
        solution['total_mistakes_pos'] = solution['mistakes_pos'].dot(self._mip_data['n_counts_pos'])
        solution['total_mistakes_neg'] = solution['mistakes_neg'].dot(self._mip_data['n_counts_neg'])
        solution['total_mistakes'] = solution['total_mistakes_pos'] + solution['total_mistakes_neg']

        # convert solution dictionary
        mip_solution = np.repeat(np.nan, self._mip_info['n_vars'])
        for mip_var_name, mip_var_idx in self._indices.items():
            mip_solution[mip_var_idx] = solution[mip_var_name]

        assert np.all(np.isfinite(mip_solution))
        return mip_solution


    def set_mip_parameters(self, param = None):
        if param is not None:
            self._mip = set_mip_parameters(self._mip, param)


    def set_time_limit(self, time_limit = None):
        self._mip = set_mip_time_limit(self._mip, time_limit)


    def set_node_limit(self, node_limit = None):
        self._mip = set_mip_node_limit(self._mip, node_limit)


    def toggle_preprocessing(self, toggle = True):
        self._mip = toggle_mip_preprocessing(self._mip, toggle)


    def _add_stats_callback(self, store_solutions = False):

        if not hasattr(self, '_stats_callback'):
            sol_idx = self._mip_info['solution_idx']
            min_idx = min(sol_idx)
            max_idx = max(sol_idx)
            assert np.array_equal(np.array(sol_idx), np.arange(min_idx, max_idx + 1))

            cb = self._mip.register_callback(StatsCallback)
            cb.initialize(store_solutions, solution_start_idx = min_idx, solution_end_idx = max_idx)
            self._stats_callback = cb


    def add_to_start_pool(self, coefficients):

        if isinstance(coefficients, np.ndarray):
            assert 1 <= coefficients.ndim <= 2
            if coefficients.ndim  == 1:
                coefficients = coefficients.tolist()
            elif coefficients.ndim == 2:
                assert coefficients.shape[1] == self._n_coefficients
                coefficients = coefficients.tolist()

        assert isinstance(coefficients, list)

        if isinstance(coefficients[0], (list, np.ndarray)):
            for c in coefficients:
                assert isinstance(c, (list, np.ndarray))
                self._start_pool.append(self.as_solution_vector(c))
        else:
            self._start_pool.append(self.as_solution_vector(coefficients))


    def lp_relaxation(self):
        return get_lp_relaxation(self._mip)


    def polishing_mip(self, polish_after_solutions = 1, polish_after_time = float('inf'), time_limit = 60.0, display_flag = True):

        #todo clean this up
        polishing_mip = copy_cplex(cpx = self._mip)

        # display
        polishing_mip = set_cpx_display_options(cpx = polishing_mip,
                                                display_mip = display_flag,
                                                display_lp = display_flag,
                                                display_parameters = display_flag)

        # other parameters
        p = polishing_mip.parameters

        # general
        p.randomseed.set(0)
        p.parallel.set(1)
        p.threads.set(1)
        p.output.clonelog.set(0)

        # set polish start time
        if polish_after_time < p.mip.polishafter.time.max():
            p.mip.polishafter.time.set(float(polish_after_time))

        if polish_after_solutions < p.mip.polishafter.solutions.max():
            p.mip.polishafter.solutions.set(int(polish_after_solutions))

        # solution pool
        p.mip.pool.intensity.set(2)  # 0 auto; 1 normal; 2 more; 3 more with ; 4 all feasible solutions (set to 1-3)

        # MIP Strategy
        p.emphasis.mip.set(1)
        # p.mip.strategy.variableselect.set(0)
        # p.mip.strategy.nodeselect.set(2) #0: depth first, 1: best bound, 2 best-estimate, 3-best-estimate alternative
        # p.mip.strategy.bbinterval (for best bound search)
        p.mip.strategy.search.set(2)  # 1 for traditional B&C, 2 for dynamic search
        p.mip.strategy.probe.set(0)  # -1 for off;/ 0 for automatic
        p.mip.strategy.dive.set(2)  # 0 automatic;1 dive; 2 probing dive; 3 guided dive (set to 2 for probing)

        # Preprocessing
        p.preprocessing.symmetry.set(0)  # turn off symmetry breaking (there should not be symmetry in this model)
        p.preprocessing.boundstrength.set(0)  # -1 to turn off; 1 to turn on; 0 for CPLEX to choose

        # Cut Generation (No Cuts for Heuristic)
        p.mip.cuts.implied.set(-1)
        p.mip.cuts.localimplied.set(-1)  #
        p.mip.cuts.zerohalfcut.set(-1)  # -1 off; auto, 1 on, 2 aggreesive
        p.mip.cuts.mircut.set(-1)  # -1 off; 0 auto, 1 on, 2 aggressive
        p.mip.cuts.covers.set(-1)  # -1 off; 0 auto; 1-3 aggression level

        # General Heuristics
        # p.mip.strategy.heuristicfreq.set(100) #-1 for none, or # of nodes
        p.mip.strategy.rinsheur.set(0)  # RINS: -1 off; 0 auto; 0 for none; n >= as frequency
        p.mip.strategy.fpheur.set(-1)  # Feasibility Pump: -1: off; 0 auto; 1 to find feasible only; 2 to find feasible with good obj (use -1 or 2)
        p.mip.strategy.lbheur.set(0)  # Local Branching: 0 off; 1 on

        return polishing_mip



def process_mip_data(data, standardize = True, compress = True, fit_intercept = True):

    # TODO: only compute required quantities
    # TODO: switch standardization to scikit-learn
    # TODO: remove implied columns

    if 'format' in data and data['format'] is 'mip':
        return data

    assert check_data_required_fields(**data)

    # handle intercept
    if fit_intercept:
        data = add_intercept(data)
    else:
        data = remove_intercept(data)

    intercept_idx = get_intercept_index(data)  # returns -1 if no intercept
    coefficient_idx = get_variable_indices(data, include_intercept = False)

    # settings
    X = data['X'].astype('float')
    Y = data['Y'].astype('float')
    n_samples, n_variables = X.shape
    variable_names = data['variable_names']

    X_shift = np.zeros(n_variables)
    X_scale = np.ones(n_variables)
    if standardize:
        X_shift[coefficient_idx] = np.mean(X[:, coefficient_idx], axis = 0)
        X_scale[coefficient_idx] = np.std(X[:, coefficient_idx], axis = 0)
        X = (X - X_shift) / X_scale

    pos_ind = Y == 1
    neg_ind = ~pos_ind
    n_samples_pos = np.sum(pos_ind)
    n_samples_neg = n_samples - n_samples_pos
    X_pos, X_neg = X[pos_ind, ], X[neg_ind, ]

    if compress:

        U_pos, x_pos_to_u_pos_idx, u_pos_to_x_pos_idx, n_counts_pos = np.unique(X_pos,
                                                                                axis = 0,
                                                                                return_index = True,
                                                                                return_inverse = True,
                                                                                return_counts = True)

        U_neg, x_neg_to_u_neg_idx, u_neg_to_x_neg_idx, n_counts_neg = np.unique(X_neg,
                                                                                axis = 0,
                                                                                return_index = True,
                                                                                return_inverse = True,
                                                                                return_counts = True)

        assert np.all(U_pos == X_pos[x_pos_to_u_pos_idx,])
        assert np.all(U_neg == X_neg[x_neg_to_u_neg_idx,])

    else:

        U_pos = X_pos
        u_pos_to_x_pos_idx = np.arange(n_samples_pos)
        x_pos_to_u_pos_idx = np.arange(n_samples_pos)
        n_counts_pos = np.ones_like(x_pos_to_u_pos_idx)

        U_neg = X_neg
        u_neg_to_x_neg_idx = np.arange(n_samples_neg)
        x_neg_to_u_neg_idx = np.arange(n_samples_neg)
        n_counts_neg = np.ones_like(x_neg_to_u_neg_idx)


    x_to_u_pos_idx = np.flatnonzero(pos_ind)[x_pos_to_u_pos_idx]
    x_to_u_neg_idx = np.flatnonzero(neg_ind)[x_neg_to_u_neg_idx]

    # assert np.all(X_pos[x_pos_to_u_pos_idx,] == U_pos)
    # assert np.all(X_neg[x_neg_to_u_neg_idx,] == U_neg)
    # assert np.all(U_pos[u_pos_to_x_pos_idx,] == X_pos)
    # assert np.all(U_neg[u_neg_to_x_neg_idx,] == X_neg)
    # assert np.all(X[x_to_u_pos_idx,] == U_pos)
    # assert np.all(X[x_to_u_neg_idx,] == U_neg)
    # assert np.all(Y[x_to_u_pos_idx,] == 1)
    # assert np.all(Y[x_to_u_neg_idx,] == -1)

    conflicted_pairs = get_common_row_indices(U_pos, U_neg)

    mip_data = {
        'format': 'mip',
        'standardize_data': standardize,
        'compress_data': compress,
        'variable_names': variable_names,
        'intercept_idx': intercept_idx,
        'coefficient_idx': coefficient_idx,
        'Y': Y,
        'pos_ind': pos_ind,
        'neg_ind': neg_ind,
        'n_samples': n_samples,
        'n_variables': n_variables,
        'n_samples_pos': n_samples_pos,
        'n_samples_neg': n_samples_neg,
        'X_scale': X_scale,
        'X_shift': X_shift,
        'x_to_u_pos_idx': x_to_u_pos_idx,
        'x_to_u_neg_idx': x_to_u_neg_idx,
        'u_pos_to_x_pos_idx': u_pos_to_x_pos_idx,
        'u_neg_to_x_neg_idx': u_neg_to_x_neg_idx,
        'U_pos': U_pos,
        'U_neg': U_neg,
        'conflicted_pairs': conflicted_pairs,
        'n_counts_pos': n_counts_pos,
        'n_counts_neg': n_counts_neg,
        'n_points_pos': U_pos.shape[0],
        'n_points_neg': U_neg.shape[0],
        }

    return mip_data



def build_mip(data, settings = None, variable_name_fmt = None, constraint_name_fmt = None, print_flag = True):

    # set input variables
    assert data['format'] == 'mip'
    settings = dict() if settings is None else dict(settings)
    variable_name_fmt = dict(ZeroOneLossMIP.VAR_NAME_FMT) if variable_name_fmt is None else dict(variable_name_fmt)
    constraint_name_fmt = dict(ZeroOneLossMIP.CON_NAME_FMT) if constraint_name_fmt is None else dict(constraint_name_fmt)

    # completem mip settings
    default_settings = dict(ZeroOneLossMIP.SETTINGS)
    check_setting = lambda d, s, v: get_or_set_default(d, setting_name = s, default_value = v, type_check = True, print_flag = print_flag)
    for name, value in default_settings.items():
        settings = check_setting(settings, name, value)

    # class-based weights
    w_pos, w_neg = float(settings['w_pos']), 1.0

    # lengths
    n_samples, n_samples_pos, n_samples_neg = data['n_samples'], data['n_samples_pos'], data['n_samples_neg']
    n_counts_pos, n_counts_neg = data['n_counts_pos'], data['n_counts_neg']
    n_points_pos, n_points_neg = data['n_points_pos'], data['n_points_neg']
    variable_names = data['variable_names']
    n_variables = len(variable_names)

    # total positive error bounds
    error_pos_min = 0.0 if np.isnan(settings['error_positive_min']) else settings['error_positive_min']
    error_pos_max = 1.0 if np.isnan(settings['error_positive_max']) else settings['error_positive_max']
    mistakes_pos_min = max(0, np.ceil(n_samples_pos * error_pos_min))
    mistakes_pos_max = min(n_samples_pos, np.floor(n_samples_pos * error_pos_max))

    # total negative error bounds
    error_neg_min = 0.0 if np.isnan(settings['error_negative_min']) else settings['error_negative_min']
    error_neg_max = 1.0 if np.isnan(settings['error_negative_max']) else settings['error_negative_max']
    mistakes_neg_min = max(0, np.ceil(n_samples_neg * error_neg_min))
    mistakes_neg_max = min(n_samples_neg, np.floor(n_samples_neg * error_neg_max))

    # total error bounds
    error_total_min = 0.0 if np.isnan(settings['error_total_min']) else settings['error_total_min']
    error_total_max = 1.0 if np.isnan(settings['error_total_max']) else settings['error_total_max']
    mistakes_total_min = max(0, np.ceil(n_samples * error_total_min))
    mistakes_total_max = min(n_samples, np.floor(n_samples * error_total_max))

    # total mistakes
    if settings['add_constraints_for_conflicted_pairs']:
        conflict_counts_pos = data['n_counts_pos'][data['conflicted_pairs'][:, 0]]
        conflict_counts_neg = data['n_counts_neg'][data['conflicted_pairs'][:, 1]]
        min_mistakes_from_conflicts = np.minimum(conflict_counts_pos, conflict_counts_neg)
        assert np.all(min_mistakes_from_conflicts >= 1.0)
        mistakes_total_min = max(mistakes_total_min, np.sum(min_mistakes_from_conflicts))
        mistakes_total_max = min(mistakes_total_max, min(n_samples_pos, n_samples_neg))

    # sanity checks for error bounds
    assert 0 <= mistakes_pos_min <= mistakes_pos_max <= n_samples_pos
    assert 0 <= mistakes_neg_min <= mistakes_neg_max <= n_samples_neg
    assert 0 <= mistakes_total_min <= mistakes_total_max <= n_samples

    # if we add sign constraints then no need to penalize l1 penalty
    if settings['add_coefficient_sign_constraints'] and settings['add_l1_penalty']:
        print_log("L1 penalty is not required with coefficient sign constraints")
        print_log("changing settings['add_l1_penalty'] = False")

    # IP parameters
    total_l1_norm = float(settings['total_l1_norm'])
    margin = float(settings['margin'])
    l1_penalty = 0.5 * min(w_pos, w_neg) / total_l1_norm if settings['add_l1_penalty'] else 0.0
    l1_limit_constraint_sense = 'G' if settings['add_l1_penalty'] else 'E'

    assert total_l1_norm > 0.0
    assert 0.0 <= l1_penalty <= min(w_pos, w_neg)/total_l1_norm
    assert l1_limit_constraint_sense in ('E', 'G')
    assert np.all(np.isfinite(margin))

    # coefficient bounds
    theta_pos_lb, theta_pos_ub = np.zeros(n_variables), np.repeat(total_l1_norm, n_variables)
    theta_neg_lb, theta_neg_ub = np.repeat(-total_l1_norm, n_variables), np.zeros(n_variables)

    # margins
    if abs(margin) > 0.0:
        margin_pos = np.repeat(abs(margin), n_points_pos)
        margin_neg = np.repeat(abs(margin), n_points_neg)
    else:
        margin_pos = np.zeros(n_points_pos)
        margin_neg = np.zeros(n_points_neg)

    assert np.all(margin_pos >= 0.0)
    assert np.all(margin_neg >= 0.0)

    # big M constants
    if settings['use_cplex_indicators_for_mistakes']:

        M_pos = np.repeat(np.nan, n_points_pos)
        M_neg = np.repeat(np.nan, n_points_neg)

    else:
        # min_scores = compute_min_score_over_unit_l1_ball(data['U_pos'],
        #                                              theta_pos_lb, theta_pos_ub,
        #                                              theta_neg_lb, theta_neg_ub)
        #
        # max_scores = compute_max_score_over_unit_l1_ball(data['U_neg'],
        #                                                  theta_pos_lb, theta_pos_ub,
        #                                                  theta_neg_lb, theta_neg_ub)

        # M_pos = margin_pos - total_l1_norm * min_scores.flatten()
        # M_neg = margin_neg + total_l1_norm * max_scores.flatten()

        M_pos = margin_pos + (total_l1_norm * np.max(abs(data['U_pos']), axis = 1))
        M_pos = M_pos.reshape((n_points_pos, 1))
        assert np.all(M_pos > 0.0)

        M_neg = margin_neg + (total_l1_norm * np.max(abs(data['U_neg']), axis = 1))
        M_neg = M_neg.reshape((n_points_neg, 1))
        assert np.all(M_neg > 0.0)


    # build mip
    mip = Cplex()
    mip.objective.set_sense(mip.objective.sense.minimize)
    cons = mip.linear_constraints

    # Variable Vector:
    #
    # [theta_pos, theta_neg, sign, mistakes_pos, mistakes_neg, total_mistakes_pos, total_mistakes_neg, total_mistakes]
    #
    # where:
    # ----------------------------------------------------------------------------------------------------------------
    # name                  length              type        description
    # ----------------------------------------------------------------------------------------------------------------
    # theta_pos:            d x 1               real        positive components of weight vector
    # theta_neg:            d x 1               real        negative components of weight vector
    # theta_sign:           d x 1               binary      sign of weight vector. theta_sign[j] = 1 -> theta_pos > 0; theta_sign[j] = 0  -> theta_neg = 0.
    # mistakes_pos:         n_points_pos x 1    binary      mistakes on +ve points (mistake_pos[i] = 1 if mistake on i
    # mistakes_pos:         n_points_neg x 1    binary      mistakes on -ve points (mistake_neg[i] = 1 if mistake on i
    # total_mistakes_pos:   1 x 1               integer     sum(n_counts_pos * mistake_pos)
    # total_mistakes_neg:   1 x 1               integer     sum(n_counts_neg * mistake_neg)
    # total_mistakes:       1 x 1               integer     total_mistakes_pos + sumtotal_mistakes_neg

    # define variables
    print_vnames = lambda vfmt, vcnt: list(map(lambda v: vfmt % v, range(vcnt)))
    names = {
        'theta_pos': print_vnames(variable_name_fmt['theta_pos'], n_variables),
        'theta_neg': print_vnames(variable_name_fmt['theta_neg'], n_variables),
        'mistakes_pos': print_vnames(variable_name_fmt['mistake_pos'], n_points_pos),
        'mistakes_neg': print_vnames(variable_name_fmt['mistake_neg'], n_points_neg),
        'total_mistakes_pos': 'total_mistakes_pos',
        'total_mistakes_neg': 'total_mistakes_neg',
        'total_mistakes': 'total_mistakes',
        # 'objval': 'objval',
        }

    # coefficients
    add_variable(mip,
                 name = names['theta_pos'],
                 obj = np.repeat(l1_penalty, n_variables),
                 ub = theta_pos_ub,
                 lb = theta_pos_lb,
                 vtype = mip.variables.type.continuous)

    add_variable(mip,
                 name = names['theta_neg'],
                 obj = np.repeat(-l1_penalty, n_variables),
                 ub = theta_neg_ub,
                 lb = theta_neg_lb,
                 vtype = mip.variables.type.continuous)

    # mistake indicators
    add_variable(mip,
                 name = names['mistakes_pos'],
                 obj = w_pos * n_counts_pos,
                 ub = 1.0,
                 lb = 0.0,
                 vtype = mip.variables.type.binary)

    add_variable(mip,
                 name = names['mistakes_neg'],
                 obj = w_neg * n_counts_neg,
                 ub = 1.0,
                 lb = 0.0,
                 vtype = mip.variables.type.binary)

    # auxiliary variables
    add_variable(mip,
                 name = names['total_mistakes_pos'],
                 obj = 0.0,
                 ub = mistakes_pos_max,
                 lb = mistakes_pos_min,
                 vtype = mip.variables.type.integer)

    add_variable(mip,
                 name = names['total_mistakes_neg'],
                 obj = 0.0,
                 ub = mistakes_neg_max,
                 lb = mistakes_neg_min,
                 vtype = mip.variables.type.integer)

    add_variable(mip,
                 name = names['total_mistakes'],
                 obj = 0.0,
                 ub = mistakes_total_max,
                 lb = mistakes_total_min,
                 vtype = mip.variables.type.integer)

    #### Define Constraints

    if settings['use_cplex_indicators_for_mistakes']:
        #
        #  mip.indicator_constraints.add(indvar="x1",
        #                               complemented=0,
        #                               rhs=1.0,
        #                               sense="G",
        #                               lin_expr=cplex.SparsePair(ind=["x2"], val=[2.0]),
        #                               name="ind1")
        #
        # name : the name of the constraint.
        # indvar : name of variable that controls if constraint is active
        # complemented : 0 if "indvar = 1" -> constraint is active; 1 if "indvar = 0" -> "constraint is active"
        # lin_expr : SparsePair(ind,val) containing variable names and values
        # sense : the sense of the constraint, may be "L", "G", or "E": default is "E"
        # rhs : a float defining the righthand side of the constraint

        con_ind = names['theta_pos'] + names['theta_neg']

        pos_vals = np.hstack((data['U_pos'], data['U_pos'])).tolist()
        for i in range(n_points_pos):

            # if "z[i] = 0" -> "score[i] >= margin_pos[i]" is active
            mip.indicator_constraints.add(name = '%s_%d' % (constraint_name_fmt['mistake_pos'], i),
                                          indvar = names['mistakes_pos'][i],
                                          complemented = 1,
                                          lin_expr = SparsePair(ind = con_ind, val = pos_vals[i]),
                                          sense = 'G',
                                          rhs = abs(margin_pos[i]),
                                          indtype = 1)

        neg_vals = np.hstack((data['U_neg'], data['U_neg'])).tolist()
        for i in range(n_points_neg):

            # if "z[i] = 0" -> "score[i] <= margin_neg[i]" is active
            mip.indicator_constraints.add(name = '%s_%d' % (constraint_name_fmt['mistake_neg'], i),
                                          indvar = names['mistakes_neg'][i],
                                          complemented = 1,
                                          lin_expr = SparsePair(ind = con_ind, val = neg_vals[i]),
                                          sense = 'L',
                                          rhs = -abs(margin_neg[i]),
                                          indtype = 1)

    else:
        #
        # Big-M style indicator constraints
        #
        # positive points:
        #
        #  z_i   = 1[margin[i] >= score[i]]
        #       = 1[margin[i] - score[i] >= 0]
        #
        # -> M_i z_i >= margin[i] - score[i]
        # -> M_i z_i + score[i] >= margin[i]
        # -> M_i z_i + sum(theta_pos[j] * x[i,j]) + sum(theta_neg[j]*x[i,j]) >= 0
        #
        # where the big M constant is set as:
        #
        # M_i   = max  margin[i] - score[i]
        #       = margin[i] - min score[i]
        #       = margin[i] - min theta.dot(x[i]) st ||theta||_1 <= 1
        #       = margin[i] - min { min_j x[i,j] * theta_pos_ub[j], min_j x[i,j] * theta_neg_lb[j]}
        #
        # negative points:
        #
        # z_i   = 1[score[i] >= -margin[i]]
        #       = 1[margin[i] + score[i] >= 0]
        #
        # -> M_i z_i >= margin[i] + score[i]
        # -> M_i z_i - score[i] >= margin[i]
        # -> M_i z_i + sum(-x[i,j] * theta_pos[j]) + sum(-x[i,j] * theta_neg[j] >= margin[i]
        #
        # where the big M constant is set as:
        #
        #  M_i = max score[i]
        #     = max theta.dot(x[i]) st ||theta||_1 <= 1
        #     = max { max_j x[i,j] * theta_pos_ub[j], max_j x[i,j] * theta_neg_ub[j]}

        con_header = names['theta_pos'] + names['theta_neg']
        con_vals = np.hstack((data['U_pos'], data['U_pos'], M_pos)).tolist()
        for i in range(n_points_pos):
            con_ind = con_header + [names['mistakes_pos'][i]]
            con_val = con_vals[i]
            cons.add(names = ['%s_%d' % (constraint_name_fmt['mistake_pos'], i)],
                     lin_expr = [SparsePair(ind = con_ind, val = con_val)],
                     senses = ['G'],
                     rhs = [margin_pos[i]])

        con_vals = np.hstack((-data['U_neg'], -data['U_neg'], M_neg)).tolist()
        for i in range(n_points_neg):
            con_ind = con_header + [names['mistakes_neg'][i]]
            con_val = con_vals[i]
            cons.add(names = ['%s_%d' % (constraint_name_fmt['mistake_neg'], i)],
                     lin_expr = [SparsePair(ind = con_ind, val = con_val)],
                     senses = ['G'],
                     rhs = [margin_neg[i]])


    # limit L1 norm of coefficients
    # if add_L1_penalty = True  -> sum_j(theta_pos[j] - theta_neg[j]) >= total_l1_norm
    # if add_L1_penalty = False -> sum_j(theta_pos[j] - theta_neg[j]) = total_l1_norm
    con_ind = names['theta_pos'] + names['theta_neg']
    con_val = np.append(np.ones(n_variables), -np.ones(n_variables)).tolist()
    cons.add(names = [constraint_name_fmt['norm_limit']],
             lin_expr = [SparsePair(ind = con_ind, val = con_val)],
             senses = [l1_limit_constraint_sense],
             rhs = [total_l1_norm])

    # restrict theta_pos[j] > 0 or theta_neg[j] < 0
    if settings['add_coefficient_sign_constraints']:

        names['theta_sign'] = print_vnames(variable_name_fmt['theta_sign'], n_variables)

        add_variable(mip,
                     name = names['theta_sign'],
                     obj = 0.0,
                     ub = 1.0,
                     lb = 0.0,
                     vtype = mip.variables.type.binary)

        if settings['use_cplex_indicators_for_signs']:

            for j in range(n_variables):

                # "sign[j] = 1" -> "theta_neg = 0" is active
                mip.indicator_constraints.add(name = '%s_%d' % (constraint_name_fmt['sign_pos'], j),
                                              indvar = names['theta_sign'][j],
                                              complemented = 0,
                                              lin_expr = SparsePair(ind = [names['theta_neg'][j]], val = [1.0]),
                                              sense = 'E',
                                              rhs = 0.0,
                                              indtype = 1)

                # "sign[j] = 0" -> "theta_pos = 0" is active
                mip.indicator_constraints.add(name = '%s_%d' % (constraint_name_fmt['sign_neg'], j),
                                              indvar = names['theta_sign'][j],
                                              complemented = 1,
                                              lin_expr = SparsePair(ind = [names['theta_pos'][j]], val = [1.0]),
                                              sense = 'E',
                                              rhs = 0.0,
                                              indtype = 1)

        else:

            for j in range(n_variables):

                #if theta_pos[j] >= 0 then g[j] = 1
                #
                #T_pos[j] * g[j] >= theta_pos[j]
                #T_pos[j] * g[j] - theta_pos[j] >= 0
                T_pos = float(abs(theta_pos_ub[j]))
                con_name = '%s_%d' % (constraint_name_fmt['sign_pos'], j)
                con_ind = [names['theta_sign'][j], names['theta_pos'][j]]
                con_val = [T_pos, -1.0]
                cons.add(names = [con_name],
                         lin_expr = [SparsePair(ind = con_ind, val = con_val)],
                         senses = ['G'],
                         rhs = [0.0])

                # if theta_neg[j] < 0 then g[j] = 0
                # theta_neg[j] < 0 then 1 - g[j] = 1
                # -theta_neg[j] > 0 then 1 - g[j] > 0
                #
                # T_neg[j] * (1 - g[j]) > -theta_neg[j]
                # T_neg[j] - T_neg[j] * g[j] >= -theta_neg[j]
                # T_neg[j] >= T_neg[j] * g[j] - theta_neg[j]
                T_neg = float(abs(theta_neg_lb[j]))
                con_name = '%s_%d' % (constraint_name_fmt['sign_neg'], j)
                con_ind = [names['theta_sign'][j], names['theta_neg'][j]]
                con_val = [T_neg, -1.0]
                cons.add(names = [con_name],
                         lin_expr = [SparsePair(ind = con_ind, val = con_val)],
                         senses = ['L'],
                         rhs = [T_neg])


    # add constraints for points with identical features but opposite labels
    # z[i_pos] + z[i_neg] >= 1 for all x[i_pos] == x[i_neg]
    if settings['add_constraints_for_conflicted_pairs']:
        conflicted_pairs = tuple(data['conflicted_pairs'])
        for pi, ni in conflicted_pairs:
            cons.add(names = ['conflict_%d_%d' % (pi, ni)],
                     lin_expr = [SparsePair(ind = [names['mistakes_pos'][pi], names['mistakes_neg'][ni]],
                                                  val = [1.0, 1.0])],
                     senses = ['E'],
                     rhs = [1.0])

    # define auxiliary variable for total mistakes on positive points
    con_ind = [names['total_mistakes_pos']] + names['mistakes_pos']
    con_val = [-1.0] + data['n_counts_pos'].tolist()
    cons.add(names = [constraint_name_fmt['total_mistakes_pos']],
             lin_expr = [SparsePair(ind = con_ind, val = con_val)],
             senses = ['E'],
             rhs = [0.0])

    # define auxiliary variable for total mistakes on negative points
    con_ind = [names['total_mistakes_neg']] + names['mistakes_neg']
    con_val = [-1.0] + data['n_counts_neg'].tolist()
    cons.add(names = [constraint_name_fmt['total_mistakes_neg']],
             lin_expr = [SparsePair(ind = con_ind, val = con_val)],
             senses = ['E'],
             rhs = [0.0])

    # define auxiliary variable for total mistakes on all points
    con_ind = [names['total_mistakes'], names['total_mistakes_pos'], names['total_mistakes_neg']]
    con_val = [-1.0, 1.0, 1.0]
    cons.add(names = [constraint_name_fmt['total_mistakes']],
             lin_expr = [SparsePair(ind = con_ind, val = con_val)],
             senses = ['E'],
             rhs = [0.0])

    # collect information to validate solution
    variable_idx = {name: idx for idx, name in enumerate(variable_names)}

    indices = {k:mip.variables.get_indices(v) for k, v in names.items()}
    indices = {k:v if isinstance(v, list) else [v] for k, v in indices.items()}
    variable_types = {k:mip.variables.get_types(v) for k, v in names.items()}
    variable_types = {k:v if isinstance(v, list) else [v] for k, v in variable_types.items()}

    lower_bounds = {k:np.array(mip.variables.get_lower_bounds(names[k])) for k in names}
    upper_bounds = {k:np.array(mip.variables.get_upper_bounds(names[k])) for k in names}

    mip_info = {
        # build settings
        'settings': dict(settings),
        'add_coefficient_sign_constraints': settings['add_coefficient_sign_constraints'],
        'use_cplex_indicators_for_mistakes': settings['use_cplex_indicators_for_mistakes'],
        'use_cplex_indicators_for_signs': settings['use_cplex_indicators_for_signs'],
        #
        # mip properties
        'indices': indices,
        'n_vars': mip.variables.get_num(),
        'n_cons': mip.linear_constraints.get_num() + mip.indicator_constraints.get_num(),
        'var_names': names,
        'var_types': variable_types,
        'con_names': mip.linear_constraints.get_names(),
        'lower_bounds': lower_bounds,
        'upper_bounds': upper_bounds,
        'variable_idx': variable_idx,
        'coefficient_idx': mip.variables.get_indices(names['theta_pos'] + names['theta_neg']),
        #
        # optimization problem parameters
        'w_pos': w_pos,
        'w_neg': w_neg,
        'total_l1_norm': total_l1_norm,
        'l1_penalty': l1_penalty,
        'margin_pos': margin_pos,
        'margin_neg': margin_neg,
        'M_pos': M_pos.flatten(),
        'M_neg': M_neg.flatten(),
        }

    return mip, mip_info, data

