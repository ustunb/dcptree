from cplex import Cplex
from dcptree.zero_one_loss.mip import *

# Trivial Classifier

def coefs_of_trivial_classifier(data, total_l1_norm = 1.0):

    n_variables = len(data['variable_names'])
    intercept_idx = get_intercept_index(data)
    n_pos = np.sum(data['Y'] == 1)
    n_neg = data['Y'].shape[0] - n_pos

    majority_class_is_positive = n_pos >= n_neg

    # trivial classifier should predict majority class
    theta_trivial = np.zeros(n_variables, dtype = np.float_)
    if majority_class_is_positive:
        theta_trivial[intercept_idx] = 1.0
    else:
        theta_trivial[intercept_idx] = -1.0

    # normalize
    theta_trivial = total_l1_norm * (theta_trivial / abs(sum(theta_trivial)))

    return theta_trivial


# Get Coefs from Best 1D Subset

def coefs_from_best_1d_subset(data):

    # todo make this more efficient
    Y = data['Y']
    pos_ind = Y == 1
    neg_ind = ~pos_ind
    n = len(Y)
    n_pos = np.sum(pos_ind)
    n_neg = n - n_pos
    n_minority = min(n_pos, n_neg)

    n_variables = data['X'].shape[1]
    variable_names = data['variable_names']
    intercept_idx = get_intercept_index(data)
    feature_idx = np.arange(n_variables)
    feature_idx = np.delete(feature_idx, intercept_idx)

    pool = {}
    for j in feature_idx:

        X = data['X'][:, j]

        # get unique elements of x along with counts
        U, Iu, Ix, n_pdf = np.unique(X, return_counts = True, return_index = True, return_inverse = True)

        # u[1],u[2],...u[K]
        # predict yhat[i] = +1 if x[i] >= u[k] + eps
        # predict yhat[i] = -1 if x[i] <= u[k]
        # total_error_pos[k] = {y[i] == 1 and x[i] <= u[k]}
        # total_error_neg[k] = {y[i] == -1 and x[i] > u[k]}
        # {y[i] == -1 and x[i] > u[k]}
        # {x[i] > u[k] & y[i] == 1}
        n_distinct = len(U)
        total_error_pos = np.repeat(np.nan, n_distinct)
        total_error_neg = np.repeat(np.nan, n_distinct)
        for k in range(n_distinct):
            predict_pos = Ix > k
            predict_neg = ~predict_pos
            total_error_pos[k] = np.sum(pos_ind & predict_neg)
            total_error_neg[k] = np.sum(neg_ind & predict_pos)

        # add error for when we predict yhat = +1 if x_i >= min(u[k]))
        total_error_pos = np.insert(total_error_pos, 0, 0)
        total_error_neg = np.insert(total_error_neg, 0, n_neg)

        # compute minimal error
        total_error = total_error_pos + total_error_neg
        k_min = np.argmin(total_error)

        # R[0] = U[0] - eps
        # R[k] = U[k] + eps for k = 1,..., n_distinct

        # -w[0] / w[1] = R
        eps = 0.5 * np.min(np.diff(U))
        if k_min == 0:
            R = U[0] - eps
        else:
            R = U[k_min - 1] + eps

        #compute w[0], w[1] as solution to:
        # w[0] / w[1] = -R
        # abs(w[0]) + abs(w[1]) = total_l1_norm = 1
        C = 1.0 / (1.0 + abs(R))
        intercept = -R * C
        weight = C

        scores = intercept + (weight * X)
        coefs = np.array([intercept, weight])
        total_mistakes = np.sum(Y != np.sign(scores))

        if total_mistakes > n_minority:
            coefs = -coefs
            total_mistakes = n - total_mistakes

        # assert np.count_nonzero(scores) == n
        # assert total_mistakes == total_error[k_min]
        # assert np.isclose(np.sum(abs(coefs)), 1.0)

        pool[variable_names[j]] = {
            'coefs': np.array(coefs),
            'objval': total_mistakes,
            }

    vnames = list(pool.keys())
    best_idx = np.argmin([pool[k]['objval'] for k in vnames])
    feature_name = vnames[best_idx]
    feature_idx = variable_names.index(feature_name)

    coefs = pool[feature_name]['coefs']
    objval = pool[feature_name]['objval']

    theta = np.zeros(n_variables)
    theta[intercept_idx] = coefs[0]
    theta[feature_idx] = coefs[1]

    return theta, objval, pool


# Fast Heuristic via SVM Linear

def coefs_from_svm_linear(data, loss = "hinge", C = 1.0, max_iterations = 1e7, tolerance = 1e-8, total_l1_norm = 1.0, display_flag = False):

    from sklearn.svm import LinearSVC

    n_variables = data['X'].shape[1]
    intercept_idx = get_intercept_index(data)
    feature_idx = np.arange(n_variables)
    feature_idx = np.setdiff1d(feature_idx, intercept_idx)

    X = data['X'][:, feature_idx]
    y = data['Y']

    # Set classifier options.
    clf = LinearSVC(C = C,
                    class_weight = None,
                    dual = True,
                    fit_intercept = True,
                    intercept_scaling = 1,
                    loss = loss,
                    penalty = 'l2',
                    max_iter = max_iterations,
                    random_state = 0,
                    tol = tolerance,
                    verbose = display_flag)


    #Train the Model
    clf.fit(X, y)

    # Create Coefficient Vector
    coefs = np.zeros(n_variables)
    coefs[intercept_idx] = clf.intercept_
    coefs[feature_idx] = clf.coef_
    coefs = total_l1_norm * coefs/sum(abs(coefs))
    return coefs


# CPLEX Fast Polishing

def build_polishing_mip(cpx, polish_after_solutions = 1, polish_after_time = float('inf'), display_flag = True):

    # copy mip
    polishing_mip = Cplex(cpx)
    p = polishing_mip.parameters

    # display
    #p.mip.display.set(display_flag)

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
    #p.mip.strategy.variableselect.set(0)
    #p.mip.strategy.nodeselect.set(2) #0: depth first, 1: best bound, 2 best-estimate, 3-best-estimate alternative
    #p.mip.strategy.bbinterval (for best bound search)
    p.mip.strategy.search.set(2)  # 1 for traditional B&C, 2 for dynamic search
    p.mip.strategy.probe.set(0)  # -1 for off;/ 0 for automatic
    p.mip.strategy.dive.set(2)  # 0 automatic;1 dive; 2 probing dive; 3 guided dive (set to 2 for probing)

    #Preprocessing
    p.preprocessing.symmetry.set(0) #turn off symmetry breaking (there should not be symmetry in this model)
    p.preprocessing.boundstrength.set(0) #-1 to turn off; 1 to turn on; 0 for CPLEX to choose

    # Cut Generation (No Cuts for Heuristic)
    p.mip.cuts.implied.set(-1)
    p.mip.cuts.localimplied.set(-1) #
    p.mip.cuts.zerohalfcut.set(-1) #-1 off; auto, 1 on, 2 aggreesive
    p.mip.cuts.mircut.set(-1) #-1 off; 0 auto, 1 on, 2 aggressive
    p.mip.cuts.covers.set(-1) #-1 off; 0 auto; 1-3 aggression level

    # General Heuristics
    #p.mip.strategy.heuristicfreq.set(100) #-1 for none, or # of nodes
    p.mip.strategy.rinsheur.set(0) #RINS: -1 off; 0 auto; 0 for none; n >= as frequency
    p.mip.strategy.fpheur.set(-1) #Feasibility Pump: -1: off; 0 auto; 1 to find feasible only; 2 to find feasible with good obj (use -1 or 2)
    p.mip.strategy.lbheur.set(0) #Local Branching: 0 off; 1 on

    return polishing_mip
    # set time limit


def coefs_from_polishing_mip(cpx, cpx_indices, polish_after_solutions = 1, polish_after_time = float('inf'), time_limit = 60.0, display_flag = True):
    pmip = build_polishing_mip(cpx, polish_after_solutions, polish_after_time)
    pmip = set_mip_time_limit(pmip, time_limit)
    pmip.solve()
    #TODO if no solution exists return trivial solution
    coefs = get_coefficients(pmip, theta_pos_idx = cpx_indices['theta_pos'], theta_neg_idx = cpx_indices['theta_neg'])
    return coefs

