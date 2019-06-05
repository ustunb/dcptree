from dcptree.zero_one_loss.tests.testing_helper_functions import *
from dcptree.zero_one_loss.mip import *

#setup
random_seed = 2338
data_name = 'adult'
selected_groups = ['Race', 'Sex']

## load / process data data
data_file = '%s/%s_processed.pickle' % (data_dir, data_name)
data, cvindices = load_processed_data(data_file)
#data = filter_data_to_fold(data, cvindices, fold_id = 'K05N01', fold_num = 0, include_validation = True)
#data = sample_test_data_with_partitions(data, n_samples = 10000, scramble = True, random_seed = 1337)

# remove selected groups
data, groups = split_groups_from_data(data = data, group_names = selected_groups)
data = convert_remaining_groups_to_rules(data)
data = add_intercept(data)
data = cast_numeric_fields(data)

#data = sample_test_data(data, max_features = 2, n_pos = 20, n_neg = 20, n_conflict = 5, remove_duplicates = True)
#data = sample_test_data(data, max_features = 4, n_pos = 100, n_neg = 100, n_conflict = 5, remove_duplicates = True)
data = sample_test_data(data, max_features = 5, n_pos = 20, n_neg = 20, n_conflict = 5, remove_duplicates = True)

#
test_settings = ['baseline', 'indicators_signs', 'indicators_signs_and_mistakes']

# Parameters
baseline_settings = {
    'compress_data': True,
    'standardize_data': False,
    'margin': 0.0001,
    'total_l1_norm': 1.00,
    'add_l1_penalty': False,
    'add_coefficient_sign_constraints': True,
    'use_cplex_indicators_for_mistakes': False,
    'use_cplex_indicators_for_signs': False,
    }

cplex_parameters = dict(DEFAULT_CPLEX_PARAMETERS)

all_mip_settings = {
    'baseline': dict(baseline_settings),
    'indicators_signs': dict(baseline_settings),
    'indicators_mistakes': dict(baseline_settings),
    'indicators_signs_and_mistakes': dict(baseline_settings),
    }

all_mip_settings['indicators_signs']['use_cplex_indicators_for_signs'] = True
all_mip_settings['indicators_mistakes']['use_cplex_indicators_for_mistakes'] = True
all_mip_settings['indicators_signs_and_mistakes']['use_cplex_indicators_for_signs'] = True
all_mip_settings['indicators_signs_and_mistakes']['use_cplex_indicators_for_mistakes'] = True

to_test = [k for k in all_mip_settings.keys() if k in test_settings]

# Solve MIP for All Settings
all_mip_stats = {}
for k in to_test:
    # Build MIP

    # setup and solve MIP
    mip = ZeroOneLossMIP(data = data, settings = all_mip_settings[k])
    mip.solve(time_limit = 10)

    # Store Stats
    all_mip_stats[k] = {
        'mip': mip,
        'indices': mip.indices,
        'mip_info': mip.mip_info,
        'mip_data': mip.data,
        'mip_stats': mip.solution_info,
        'mip_settings': all_mip_settings[k],
        }

# Print Stats
for k in all_mip_stats:

    print('\n%s\n\t\t\t%s\n%s' % ('-'*120, k, '-'*120))

    mip_stats = all_mip_stats[k]['mip_stats']
    pprint(mip_stats)

    mip = all_mip_stats[k]['mip']
    indices = all_mip_stats[k]['indices']
    theta_pos = np.array(mip.solution.get_values(indices['theta_pos']))
    theta_neg = np.array(mip.solution.get_values(indices['theta_neg']))
    theta = theta_pos + theta_neg

    print('solution:\t %s' % str(theta))
    sign = np.array(mip.solution.get_values(indices['theta_sign']))
    print('sign:\t\t %s' % str(sign))


