from dcptree.zero_one_loss.tests.testing_helper_functions import *
from dcptree.zero_one_loss.mip import *

# setup
data_name = 'adult'

## load / process data data
data_file = '%s/%s_processed.pickle' % (data_dir, data_name)
data, cvindices = load_processed_data(data_file)

# remove selected groups
data, groups = split_groups_from_data(data = data, group_names = data['partitions'])
data = add_intercept(data)
data = cast_numeric_fields(data)

mip_settings = {
    'compress_data': True,
    'standardize_data': True,
    'margin': 0.0001,
    'total_l1_norm': 1.00,
    'add_l1_penalty': False,
    'add_coefficient_sign_constraints': True,
    'use_cplex_indicators_for_mistakes': True,
    'use_cplex_indicators_for_signs': False,
    'add_constraints_for_conflicted_pairs': False,
    }

# setup and solve MIP
mip = ZeroOneLossMIP(data = data, settings = mip_settings)
mip.solve(time_limit = 10)
model = mip.get_classifier()
