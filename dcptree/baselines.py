import numpy as np
import cvxpy as cp
import warnings
from dccp.problem import is_dccp
from dcptree.data import INTERCEPT_IDX
from dcptree.classification_models import ClassificationModel


class CoupledRiskMinimizer(object):
    """
    This object trains the preferential fairness classifiers from Zafar et al. 2017 (https://arxiv.org/abs/1707.00010)
    We have adapted their code, adding the following improvements:
    - Ability to define constraints for 3+ subgroups
    - Ability to use a sparse formulation to reduce runtime
    - Python 3 compatability
    It will produce the same results as the LinearClf class at:
    https://github.com/mbilalzafar/fair-classification/blob/master/fair_classification/linear_clf_pref_fairness.py
    """

    MU = 1.2
    TAU = 0.5
    EPS = 1e-4

    LOSS_LOGISTIC = 'logreg'
    LOSS_SVM = 'svm_linear'

    CONSTRAINT_NONE = -1
    CONSTRAINT_PARITY = 0
    CONSTRAINT_PREFERED_IMPACT = 1
    CONSTRAINT_PREFERED_TREATMENT = 2
    CONSTRAINT_PREFERED_BOTH = 3

    VALID_LOSS_FUNCTIONS = (LOSS_LOGISTIC, LOSS_SVM)
    VALID_PREFERED_CONSTRAINTS = (CONSTRAINT_PREFERED_IMPACT, CONSTRAINT_PREFERED_TREATMENT, CONSTRAINT_PREFERED_BOTH)
    VALID_CONSTRAINTS = (CONSTRAINT_NONE, CONSTRAINT_PARITY) + VALID_PREFERED_CONSTRAINTS


    def __init__(self, loss_function, lam = 0.0, train_multiple = False, sparse_formulation = True, random_state = 1234):

        """
            Model can be logistic regression or linear SVM in primal form

            We will define the lam parameter once and for all for a single object.
            For cross validating multiple models, we will write a function for doing that.

        """

        """ Setting default lam val and Making sure that lam is provided for each group """

        assert loss_function in self.VALID_LOSS_FUNCTIONS
        self.loss_function = loss_function

        assert isinstance(train_multiple, bool)
        self.train_multiple = train_multiple

        if isinstance(lam, dict):
            assert all([isinstance(v, (int, float)) for v in lam.values()])
            assert all([v >= 0.0 for v in lam.values()])
            assert train_multiple
            self.lam = {k: float(v) for k, v in lam.items()}
        else:
            assert isinstance(lam, (int, float))
            assert lam >= 0.0
            self.lam = float(lam)

        self.random_state = random_state
        self.sparse_formulation = sparse_formulation



    def fit(self, X, y, x_sensitive, **cons_params):

        """
            X: n x d array
            y: n length vector
            x_sensitive: n length vector
            cons_params will be a dictionary
            cons_params["tau"], cons_params["mu"] and cons_params["EPS"] are the solver related parameters. Check DCCP documentation for details
            cons_params["cons_type"] specified which type of constraint to apply
                - cons_type = -1: No constraint
                - cons_type = 0: Parity
                - cons_type = 1: Preferred impact
                - cons_type = 2: Preferred treatment
                - cons_type = 3: Preferred both

            cons_params["s_val_to_cons_sum"]: The ramp approximation -- only needed for cons_type 1 and 3
        """

        n, d = X.shape
        group_labels = set(x_sensitive)

        intercept_idx = INTERCEPT_IDX
        coefficient_idx = np.arange(d)
        coefficient_idx = coefficient_idx[coefficient_idx != intercept_idx]

        self._group_labels = group_labels
        self._intercept_idx = intercept_idx
        self._coefficient_idx = coefficient_idx
        self._n = n
        self._d = d

        if self.train_multiple:
            if isinstance(self.lam, float):
                self.lam = {z: float(self.lam) for z in group_labels}

            assert isinstance(self.lam, dict)
            assert group_labels == self.lam.keys()

        assert isinstance(cons_params, dict)
        cons_type = cons_params.get('cons_type', self.CONSTRAINT_NONE)
        assert cons_type in CoupledRiskMinimizer.VALID_CONSTRAINTS

        if self.loss_function == CoupledRiskMinimizer.LOSS_LOGISTIC:
            solver_settings = {
                'method': 'dccp',
                'verbose': cons_params.get('print_flag', False),
                'max_iters': cons_params.get('max_iters', 100),  # for CVXPY convex solver
                'max_iter': cons_params.get('max_iter', 50), # for the dccp. notice that DCCP hauristic runs the convex program iteratively until arriving at the solution
                'mu': cons_params.get('MU', self.MU),
                'tau': cons_params.get('TAU', self.TAU),
                'tau_max': 1e10,
                'feastol': cons_params.get('EPS', self.EPS),
                'abstol': cons_params.get('EPS', self.EPS),
                'reltol': cons_params.get('EPS', self.EPS),
                'feastol_inacc': cons_params.get('EPS', self.EPS),
                'abstol_inacc': cons_params.get('EPS', self.EPS),
                'reltol_inacc': cons_params.get('EPS', self.EPS),
                }
        else:
            solver_settings = {}

        ### setup optimization problem
        obj = 0
        np.random.seed(self.random_state)  # set the seed before initializing the values of w

        if self.train_multiple:
            w = {}
            for k in group_labels:
                idx = x_sensitive == k

                # setup coefficients and initialize as uniform distribution over [0,1]
                w[k] = cp.Variable(d)
                w[k].value = np.random.rand(d)

                # first term in w is the intercept, so no need to regularize that
                obj += cp.sum_squares(w[k][coefficient_idx]) * self.lam[k]

                # setup
                X_k, y_k = X[idx], y[idx]

                if self.sparse_formulation:
                    XY = np.concatenate((X_k, y_k[:, np.newaxis]), axis = 1)
                    UY, counts = np.unique(XY, return_counts = True, axis = 0)
                    pos_idx = np.greater(UY[:, -1], 0)
                    neg_idx = np.logical_not(pos_idx)
                    U = UY[:, 0:d]
                    obj_weights_pos = counts[pos_idx] / float(n)
                    Z_pos = -U[pos_idx, :]

                    obj_weights_neg = counts[neg_idx] / float(n)
                    Z_neg = U[neg_idx, :]

                    if self.loss_function == CoupledRiskMinimizer.LOSS_LOGISTIC:
                        obj += cp.sum(cp.multiply(obj_weights_pos, cp.logistic(Z_pos * w[k])))
                        obj += cp.sum(cp.multiply(obj_weights_neg, cp.logistic(Z_neg * w[k])))

                    elif self.loss_function == CoupledRiskMinimizer.LOSS_SVM:
                        obj += cp.sum(cp.multiply(obj_weights_pos, cp.pos(1.0 - Z_pos * w[k])))
                        obj += cp.sum(cp.multiply(obj_weights_neg, cp.pos(1.0 + Z_neg * w[k])))

                else:

                    if self.loss_function == CoupledRiskMinimizer.LOSS_LOGISTIC:
                        obj += cp.sum(cp.logistic(cp.multiply(-y_k, X_k * w[k]))) / float(n)

                    elif self.loss_function == CoupledRiskMinimizer.LOSS_SVM:
                        obj += cp.sum(cp.pos(1.0 - cp.multiply(y_k, X_k * w[k]))) / float(n)


                # notice that we are dividing by the length of the whole dataset, and not just of this sensitive group.
                # this way, the group that has more people contributes more to the loss

        else:
            w = cp.Variable(d)  # this is the weight vector
            w.value = np.random.rand(d)

            # regularizer -- first term in w is the intercept, so no need to regularize that
            obj += cp.sum_squares(w[1:]) * self.lam

            if self.loss_function == self.LOSS_LOGISTIC:
                obj += cp.sum(cp.logistic(cp.multiply(-y, X * w))) / float(n)

            elif self.loss_function == self.LOSS_SVM:
                obj += cp.sum(cp.maximum(0.0, 1.0 - cp.multiply(y, X * w))) / float(n)

        constraints = []
        if cons_type in self.VALID_PREFERED_CONSTRAINTS:
            constraints = self._stamp_preference_constraints(X, x_sensitive, w, cons_type, cons_params.get('s_val_to_cons_sum'))

        elif cons_type == self.CONSTRAINT_PARITY:
            constraints = self._stamp_disparate_impact_constraint(X, y, x_sensitive, w, cov_thresh = np.abs(0.0))

        prob = cp.Problem(cp.Minimize(obj), constraints)

        ### solve optimization problem
        if is_dccp(prob):
            print("solving disciplined convex-concave program (DCCP)")
        else:
            assert prob.is_dcp()
            print("solving disciplined convex program (DCP)")

        prob.solve(**solver_settings)

        print("solver stopped (status: %r)" % prob.status)
        if prob.status != cp.OPTIMAL:
            warnings.warn('solver did not recover optimal solution')

        # check that the fairness constraint is satisfied
        for f_c in constraints:
            if not f_c.value:
                warnings.warn("fairness constraint %r not satisfied!" % f_c)

        self._prob = prob


        # store results
        if self.train_multiple:
            coefs = {k: np.array(w[k].value).flatten() for k in group_labels}
            self.w = {k: np.array(v) for k, v in coefs.items()}
        else:
            coefs = np.array(w.value).flatten()
            self.w = np.array(coefs)


        return coefs




    def switch_margins(self, dist_dict):
        """
        computes the ramp function for each group to estimate the acceptance rate
        """
        group_labels = dist_dict.keys()
        switch_margins = {}
        for a in dist_dict.keys():
            switch_margins[a] = {}
            for b in dist_dict[a].keys():
                margins = dist_dict[a][b]
                switch_margins[a][b] = np.sum(np.maximum(0.0, margins)) / margins.shape[0]

        return switch_margins


    def decision_function(self, X, k = None):
        """ Predicts labels for all samples in X

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
        Returns

        k: the group whose decision boundary should be used.
        k = None means that we trained one clf for the whole dataset
        -------
        y : array of shape = [n_samples]
        """
        if k is None:
            score = np.dot(X, self.w)
        else:
            score = np.dot(X, self.w[k])

        return score


    def get_distance_boundary(self, X, x_sensitive):

        """
            returns two vals

            distance_boundary_arr:
                arr with distance to boundary, each groups owns w is applied on it
            distance_boundary_dict:
                dict of the form s_attr_group (points from group 0/1) -> w_group (boundary of group 0/1) -> distances for this group with this boundary

        """

        group_labels = set(x_sensitive)
        distances_boundary_dict = {}

        if isinstance(self.w, dict):

            distance_boundary_arr = np.repeat(np.nan, X.shape[0])
            for z in group_labels:
                idx = x_sensitive == z
                X_g = X[idx]
                # each group gets decision with their own boundary +  decision with both boundaries
                distance_boundary_arr[idx] = self.decision_function(X_g, z)
                distances_boundary_dict[z] = {k: self.decision_function(X_g, k) for k in self.w.keys()}

        else:
            # we have one model for the whole data
            # there is only one boundary, so the results with this_group and other_group boundaries are the same
            # apply same decision function for all the sensitive attrs because same w is trained for everyone:

            distance_boundary_arr = self.decision_function(X)
            for z in group_labels:
                idx = x_sensitive == z
                distances_boundary_dict[z] = {k: self.decision_function(X[idx]) for k in group_labels}

        return distance_boundary_arr, distances_boundary_dict


    def _stamp_disparate_impact_constraint(self, X, y, x_sensitive, w, cov_thresh):

        """
        Parity impact constraint
        """

        assert self.train_multiple == False # di cons is just for a single boundary clf
        assert cov_thresh >= 0.0  # covariance thresh has to be a small positive number

        constraints = []
        z_i_z_bar = x_sensitive - np.mean(x_sensitive)

        fx = X * w
        prod = cp.sum(cp.multiply(z_i_z_bar, fx)) / float(X.shape[0])
        constraints.append(prod <= cov_thresh)
        constraints.append(prod >= -cov_thresh)

        return constraints


    def _stamp_preference_constraints(self, X, x_sensitive, w, cons_type, s_val_to_cons_sum = None):

        """
            No need to pass s_val_to_cons_sum for preferred treatment (envy free) constraints
            # 1 - pref imp, 2 - EF, 3 - pref imp & EF
        """

        assert cons_type in self.VALID_PREFERED_CONSTRAINTS
        assert cons_type == self.CONSTRAINT_PREFERED_TREATMENT or s_val_to_cons_sum is not None
        assert set(x_sensitive) == w.keys()

        group_labels = set(x_sensitive)
        prod_dict = {}
        for z in group_labels:
            idx = x_sensitive == z
            Xz = X[idx, :]
            nz = float(Xz.shape[0])
            if self.sparse_formulation:
                Uz, mz = np.unique(Xz, axis = 0) #dropping duplicates so that we have fewer constraints
                Tz = Uz * mz[:, np.newaxis]
                prod_dict[z] = {o: cp.sum(cp.pos(Tz * wo)) / nz for o, wo in w.items()}
            else:
                prod_dict[z] = {o: cp.sum(cp.maximum(0.0, Xz * wo)) / nz for o, wo in w.items()}

        constraints = []
        if cons_type == self.CONSTRAINT_PREFERED_IMPACT:
            for z in group_labels:
                constraints.append(prod_dict[z][z] >= s_val_to_cons_sum[z][z])

        elif cons_type == self.CONSTRAINT_PREFERED_TREATMENT:
            for z in group_labels:
                other_groups = set(group_labels) - {z}
                for o in other_groups:
                    constraints.append(prod_dict[z][z] >= prod_dict[z][o])

        elif cons_type == self.CONSTRAINT_PREFERED_BOTH:
            for z in group_labels:
                constraints.append(prod_dict[z][z] >= s_val_to_cons_sum[z][z]) #preferred impact
                other_groups = set(group_labels) - {z}
                for o in other_groups:
                    constraints.append(prod_dict[z][z] >= prod_dict[z][o])

        return constraints

    @property
    def prob(self):
        return self._prob

    def classifier(self):
        if self.train_multiple:
            return {k: self._coefs_to_classifier(coefs) for k, coefs in self.w.items()}
        else:
            return self._coefs_to_classifier(self.w)


    def _coefs_to_classifier(self, coefs):
        coefs = np.array(coefs).flatten()
        coefficient_idx = np.array(self._coefficient_idx)
        intercept = float(coefs[self._intercept_idx])
        coefficients = coefs[self._coefficient_idx]
        clf = ClassificationModel(predict_handle = lambda X: np.sign(X[:, coefficient_idx].dot(coefficients) + intercept),
                                  model_type = ClassificationModel.LINEAR_MODEL_TYPE,
                                  model_info = {'intercept':intercept, 'coefficients': coefficients})


        return clf
