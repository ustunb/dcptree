from dcptree.zero_one_loss.tests.testing_helper_functions import *
from dcptree.zero_one_loss.heuristics import *
from dcptree.zero_one_loss.mip import *

#setup
random_seed = 2338
data_name = 'adult'
selected_groups = [] #['Race', 'Sex']

## load / process data data
data_file = '%s/%s_processed.pickle' % (data_dir, data_name)
data, cvindices = load_processed_data(data_file)

# remove selected groups
data, groups = split_groups_from_data(data = data, group_names = selected_groups)
data = convert_remaining_groups_to_rules(data)
data = add_intercept(data)
data = cast_numeric_fields(data)

# generate synthetic data
data = sample_test_data(data, max_features = 2, n_pos = 25, n_neg = 25, n_conflict = 5, remove_duplicates = True)
#data = sample_test_data(data, max_features = 4, n_pos = 100, n_neg = 100, n_conflict = 40, remove_duplicates = True)
#data = sample_test_data(data, max_features = 5, n_pos = 20, n_neg = 20, n_conflict = 5, remove_duplicates = True)

# Parameters
test_settings = {
    'compress_data': True,
    'standardize_data': False,
    'margin': 0.0001,
    'total_l1_norm': 1.00,
    'add_l1_penalty': False,
    'add_constraints_for_conflicted_pairs': True,
    'add_coefficient_sign_constraints': True,
    'use_cplex_indicators_for_mistakes': False,
    'use_cplex_indicators_for_signs': False,
    }

test_cpx_parameters = {
    #
    'display_cplex_progress': True,
    #set to True to show CPLEX progress in console
    #
    'n_cores': 1,
    # Number of CPU cores to use in B & B
    # May have to set n_cores = 1 in order to use certain control callbacks in CPLEX 12.7.0 and earlier
    #
    'randomseed': 0,
    # This parameter sets the random seed differently for diversity of solutions.
    # https://www.ibm.com/support/knowledgecenter/SSSA5P_12.8.0/ilog.odms.cplex.help/CPLEX/Parameters/topics/RandomSeed.html
    #
    'time_limit': 60,
    # runtime before stopping,
    #
    'node_limit': 9223372036800000000,
    # number of nodes to process before stopping,
    #
    'mipgap': np.finfo('float').eps,
    # Sets a relative tolerance on the gap between the best integer objective and the objective of the best node remaining.
    # https://www.ibm.com/support/knowledgecenter/SSSA5P_12.8.0/ilog.odms.cplex.help/CPLEX/Parameters/topics/EpGap.html
    #
    'absmipgap': np.finfo('float').eps,
    # Sets an absolute tolerance on the gap between the best integer objective and the objective of the best node remaining.
    # When this difference falls below the value of this parameter, the mixed integer optimization is stopped.
    # https://www.ibm.com/support/knowledgecenter/SSSA5P_12.8.0/ilog.odms.cplex.help/CPLEX/Parameters/topics/EpAGap.html
    #
    'objdifference': 0.0,
    # Used to update the cutoff each time a mixed integer solution is found. This value is subtracted from objective
    # value of the incumbent update, so that the solver ignore solutions that will not improve the incumbent by at
    # least this amount.
    # https://www.ibm.com/support/knowledgecenter/SSSA5P_12.8.0/ilog.odms.cplex.help/CPLEX/Parameters/topics/ObjDif.html#
    #
    'integrality_tolerance': 0.0,
    # specifies the amount by which an variable can differ from an integer and be considered integer feasible. 0 is OK
    # https://www.ibm.com/support/knowledgecenter/SSSA5P_12.8.0/ilog.odms.cplex.help/CPLEX/Parameters/topics/EpInt.html
    #
    'mipemphasis': 0,
    # Controls trade-offs between speed, feasibility, optimality, and moving bounds in MIP.
    # 0     =	Balance optimality and feasibility; default
    # 1	    =	Emphasize feasibility over optimality
    # 2	    =	Emphasize optimality over feasibility
    # 3 	=	Emphasize moving best bound
    # 4	    =	Emphasize finding hidden feasible solutions
    # https://www.ibm.com/support/knowledgecenter/SSSA5P_12.8.0/ilog.odms.cplex.help/CPLEX/Parameters/topics/MIPEmphasis.html
    #
    'bound_strengthening': -1,
    # Decides whether to apply bound strengthening in mixed integer programs (MIPs).
    # https://www.ibm.com/support/knowledgecenter/SSSA5P_12.8.0/ilog.odms.cplex.help/CPLEX/Parameters/topics/BndStrenInd.html
    # -1    = cplex chooses
    # 0     = no bound strengthening
    # 1     = bound strengthening
    #
    'cover_cuts': -1,
    # Decides whether or not cover cuts should be generated for the problem.
    # https://www.ibm.com/support/knowledgecenter/en/SSSA5P_12.8.0/ilog.odms.cplex.help/CPLEX/Parameters/topics/Covers.html
    # -1    = Do not generate cover cuts
    # 0	    = Automatic: let CPLEX choose
    # 1	    = Generate cover cuts moderately
    # 2	    = Generate cover cuts aggressively
    # 3     = Generate cover cuts very  aggressively
    #
    'zero_half_cuts': 0,
    # Decides whether or not to generate zero-half cuts for the problem. (set to off since these are not effective)
    # https://www.ibm.com/support/knowledgecenter/en/SSSA5P_12.8.0/ilog.odms.cplex.help/CPLEX/Parameters/topics/ZeroHalfCuts.html
    # -1    = Do not generate MIR cuts
    # 0	    = Automatic: let CPLEX choose
    # 1	    = Generate MIR cuts moderately
    # 2	    = Generate MIR cuts aggressively
    #
    'mir_cuts': 0,
    # Decides whether or not to generate mixed-integer rounding cuts for the problem. (set to off since these are not effective)
    # https://www.ibm.com/support/knowledgecenter/en/SSSA5P_12.8.0/ilog.odms.cplex.help/CPLEX/Parameters/topics/MIRCuts.html
    # -1    = Do not generate zero-half cuts
    # 0	    = Automatic: let CPLEX choose; default
    # 1	    = Generate zero-half cuts moderately
    # 2	    = Generate zero-half cuts aggressively
    #
    'implied_bound_cuts': 0,
    # Decides whether or not to generate valid implied bound cuts for the problem.
    # https://www.ibm.com/support/knowledgecenter/SSSA5P_12.8.0/ilog.odms.cplex.help/CPLEX/Parameters/topics/ImplBdLocal.html
    # -1    = Do not generate valid implied bound cuts
    # 0	    = Automatic: let CPLEX choose; default
    # 1	    = Generate valid implied bound cuts moderately
    # 2	    = Generate valid implied bound cuts aggressively
    # 3	    = Generate valid implied bound cuts very aggressively
    #
    'locally_implied_bound_cuts': 3,
    # Decides whether or not to generate locally valid implied bound cuts for the problem.
    # https://www.ibm.com/support/knowledgecenter/SSSA5P_12.8.0/ilog.odms.cplex.help/CPLEX/Parameters/topics/ImplBdLocal.html
    # -1    = Do not generate locally valid implied bound cuts
    # 0	    = Automatic: let CPLEX choose; default
    # 1	    = Generate locally valid implied bound cuts moderately
    # 2	    = Generate locally valid implied bound cuts aggressively
    # 3	    = Generate locally valid implied bound cuts very aggressively
    #
    'scale_parameters': 0,
    # Decides how to scale the problem matrix.
    # https://www.ibm.com/support/knowledgecenter/SSSA5P_12.8.0/ilog.odms.cplex.help/CPLEX/Parameters/topics/ScaInd.html
    # 0     = equilibration scaling
    # 1     = aggressive scaling
    # -1    = no scaling
    #
    'numerical_emphasis': 0,
    # Emphasizes precision in numerically unstable or difficult problems.
    # https://www.ibm.com/support/knowledgecenter/SSSA5P_12.8.0/ilog.odms.cplex.help/CPLEX/Parameters/topics/NumericalEmphasis.html
    # 0     = off
    # 1     = on
    #
    'poolsize': 100,
    # Limits the number of solutions kept in the solution pool
    # https://www.ibm.com/support/knowledgecenter/SSSA5P_12.8.0/ilog.odms.cplex.help/CPLEX/Parameters/topics/SolnPoolCapacity.html
    # number of feasible solutions to keep in solution pool
    #
    'poolrelgap': float('nan'),
    # Sets a relative tolerance on the objective value for the solutions in the solution pool.
    # https://www.ibm.com/support/knowledgecenter/SSSA5P_12.8.0/ilog.odms.cplex.help/CPLEX/Parameters/topics/SolnPoolGap.html
    #
    'poolreplace': 2,
    # Designates the strategy for replacing a solution in the solution pool when the solution pool has reached its capacity.
    # https://www.ibm.com/support/knowledgecenter/SSSA5P_12.8.0/ilog.odms.cplex.help/CPLEX/Parameters/topics/SolnPoolReplace.html
    # 0	= Replace the first solution (oldest) by the most recent solution; first in, first out; default
    # 1	= Replace the solution which has the worst objective
    # 2	= Replace solutions in order to build a set of diverse solutions
    #
    'repairtries': 20,
    # Limits the attempts to repair an infeasible MIP start.
    # https://www.ibm.com/support/knowledgecenter/SSSA5P_12.8.0/ilog.odms.cplex.help/CPLEX/Parameters/topics/RepairTries.html
    # -1	None: do not try to repair
    #  0	Automatic: let CPLEX choose; default
    #  N	Number of attempts
    #
    'nodefilesize': (120 * 1024) / 1,
    # size of the node file (for large scale problems)
    # if the B & B can no longer fit in memory, then CPLEX stores the B & B in a node file
    }

# Data Variables
intercept_idx = get_intercept_index(data)
n_variables = len(data['variable_names'])

# setup and solve MIP
mip = ZeroOneLossMIP(data = data, settings = test_settings, cpx_parameters = test_cpx_parameters)
mip.solve(time_limit = 10)

# Solution Components
indices = mip.indices
mip_info = mip.mip_info
mip_settings = mip.settings
mip_data = mip.data

#### Setup Test Variables
# Get Solutions
mip_stats = mip.solution_info
theta = mip.coefficients()
total_mistakes_ub = mip.total_mistakes()
total_mistakes_lb = mip_stats['lowerbound']

# Create Pool of Initial Solutions using Heuristics
theta_pool = {}
theta_pool['trivial'] = coefs_of_trivial_classifier(data)
theta_pool['best_subset'],_,_ = coefs_from_best_1d_subset(data)
theta_pool['svm_linear'] = coefs_from_svm_linear(data, loss = "hinge", C = 1.0, max_iterations = 1e7, tolerance = 1e-8, display_flag = False)
#theta_pool['polishing'] = coefs_from_polishing_mip(mip, indices, polish_after_solutions = 1, polish_after_time = float('inf'), time_limit = 60.0, display_flag = True)

### Helper Functions for Tests

def build_initialized_mip(coefs, preprocessing = False):
    initialized_mip = ZeroOneLossMIP(data = data, settings = test_settings, cpx_parameters = test_cpx_parameters)
    initialized_mip.toggle_preprocessing(toggle = preprocessing)
    initialized_mip.add_to_start_pool(coefs)
    return initialized_mip


### Test Conversion to Solution Vector

def test_to_mip_solution():
    coefs = np.array(mip.coefficients())
    solution = mip.as_solution_vector(coefs)
    compare_with_incumbent_solution(mip, solution)


def test_to_mip_solution_rearranged():

    coefs = np.array(mip.coefficients())
    coef_order = np.random.permutation(len(coefs) - 1)
    coef_order = np.insert(coef_order, 0, len(coefs) - 1)

    test_coefs = coefs[coef_order]
    test_coef_names = [mip_data['variable_names'][j] for j in coef_order]

    solution = mip.as_solution_vector(test_coefs, test_coef_names)
    compare_with_incumbent_solution(mip.mip, solution)


def test_initialization_with_mip_solution():

    # Create Initial Solution
    coefs = np.array(theta)
    initial_solution = mip.as_solution_vector(coefs)

    # Build Test MIP
    initialized_mip = build_initialized_mip(coefs = coefs)
    initialized_mip.solve(node_limit = 0)

    # Check Solution was Accepted
    diff_idx, diff_names = compare_with_incumbent_solution(initialized_mip, initial_solution)
    assert len(diff_idx) == 0, "mip solution differs from expected solution at\n: %s" % ('\n'.join(diff_names))


#### Test Initialization with Heuristic Solutions

for heuristic_name, heuristic_coefs in theta_pool.items():

    initialized_mip = build_initialized_mip(coefs = heuristic_coefs)
    heuristic_mistakes = initialized_mip.total_mistakes(heuristic_coefs)

    def test_heuristic_solution_format():
        assert len(heuristic_coefs) == len(data['variable_names'])

    def test_heuristic_is_suboptimal():
        assert total_mistakes_lb <= initialized_mip.total_mistakes(heuristic_coefs)

    def test_initialization_with_heuristic_solution():

        initialized_mip.solve(node_limit = 0)
        initialized_coefs = initialized_mip.coefficients()
        initialized_stats = initialized_mip.solution_info

        assert np.all(np.equal(initialized_coefs, heuristic_coefs))
        assert initialized_stats['objval'] == heuristic_mistakes

