import dill
from dcptree.paths import *
from dcptree.data import *
from dcptree.data_io import load_processed_data
from dcptree.cross_validation import split_data_by_cvindices
from dcptree.group_helper import *
from dcptree.tree import *
from dcptree.classification_models import *
from dcptree.zero_one_loss.mip import train_zero_one_linear_model
from dcptree.decoupled_set import *
from scripts.reporting import *

pd.set_option('display.max_columns', 30)

#### user dashboard ####

info = {
    'data_name': 'compas_violent_bl',
    'method_name': 'tree_01',
    'fold_id': 'K04N01',
    'attr_id': 'all',
    'max_runtime': 5,
    'random_seed': 2338
    }

# script variables
data_name = info.get('data_name')
method_name = info.get('method_name')
random_seed = info.get('random_seed')
fold_id = info.get('fold_id')

# setup files
data_file = '%s/%s_processed.pickle' % (data_dir, data_name)
results_file = '%s/%s_%s_%s_results.pickle' % (results_dir, data_name, info['attr_id'], info['method_name'])

# load data
data, cvindices = load_processed_data(data_file)
selected_groups = data['partitions']
data = split_data_by_cvindices(data, cvindices, fold_id = fold_id, fold_num = 1, fold_num_test = 2)
data, groups = split_groups_from_data(data = data, group_names = selected_groups)
data = convert_remaining_groups_to_rules(data)
data = cast_numeric_fields(data)

# specify method
if method_name == 'tree_01':
    data = add_intercept(data)
    training_handle = lambda data: train_zero_one_linear_model(data, time_limit = info['max_runtime'])
elif method_name == 'tree_lr':
    training_handle = lambda data: train_logreg(data, settings = None, normalize_variables = False)
elif method_name == 'tree_svm':
    training_handle = lambda data: train_svm(data, settings = None, normalize_variables = False)

# train tree
tree = DecouplingTree(data, groups, training_handle = training_handle)
tree.grow(print_flag = True, log_flag = True)

# prune tree
df, candidates = tree.prune(alpha = 0.05, test_correction = True, atomic_groups = True, data = data, groups = groups, stat_field = 'test')

# extra classifier set
clf_set = DecoupledClassifierSet(data = data,
                                 groups = groups,
                                 pooled_model = tree.root.model,
                                 decoupled_models = [l.model for l in tree.partition.leaves],
                                 groups_to_models = build_model_assignment_map(tree.partition))


# save results
info.update({
    'data_file': data_file,
    'results_file': results_file,
    'clf_set': clf_set,
    'debug': {'tree': tree, 'df': df, 'subtrees': candidates},
    })

with open(results_file, 'wb') as outfile:
    dill.dump(info, outfile, protocol = dill.HIGHEST_PROTOCOL, recurse = True)


with open(results_file, 'rb') as infile:
    results = dill.load(infile)

print('saved_file')

# # basic settings
# output_file_header = '%s_%s_%s' % (data_name, method_name, info['attr_id'])
# output_file_names = {
#     'group_envyfreeness': '%s/group_envy_%s.pdf' % (paper_dir, output_file_header),
#     'group_rationality': '%s/group_gain_%s.pdf' % (paper_dir, output_file_header),
#     }
#
# f, axs = tree_preference_report(data, groups, clf_set)
# f.show()
#
# g, axs = tree_preference_report(data, groups, clf_set, rationality_report = True)
# g.show()