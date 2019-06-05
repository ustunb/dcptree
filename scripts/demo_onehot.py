import dill
from dcptree.paths import *
from dcptree.data import *
from dcptree.cross_validation import split_data_by_cvindices
from dcptree.group_helper import *
from scripts.reporting import tree_preference_report
from scripts.training import *

#### user dashboard ####
info = {
    'data_name': 'adult',
    'fold_id': 'K04N01',
    'max_runtime': 30,
    'random_seed': 2338
    }

# setup training parameters
info['attr_id'] = 'maritalstatus'
info['method_name'] = 'onehot_lr'
info['data_file'] = '%s/%s_processed.pickle' % (data_dir, info['data_name'])
info['results_file'] = '%s/%s_%s_%s_results.pickle' % (results_dir, info['data_name'], info['attr_id'], info['method_name'])

# train method
results = train_onehot_model(info)

data_name = info['data_name']
data_file = '%s/%s' % (data_dir, Path(results['data_file']).name)
attr_id = results['attr_id']
method_name = results['method_name']

# basic settings
output_file_header = '%s_%s_%s' % (data_name, method_name, results['attr_id'])
output_file_names = {
    'group_envyfreeness': '%s/%s_envy.pdf' % (paper_dir, output_file_header),
    'group_rationality': '%s/%s_rationality.pdf' % (paper_dir, output_file_header),
    }

## setup data again
data, cvindices = load_processed_data(file_name = data_file)
if 'tree' in method_name:
    data = split_data_by_cvindices(data, cvindices, fold_id = results['fold_id'], fold_num = 1, fold_num_test = 2)
else:
    data = split_data_by_cvindices(data, cvindices, fold_id = results['fold_id'], fold_num = 1, fold_num_test = -1)

data, groups = split_groups_from_data(data = data, group_names = data['partitions'])
data = cast_numeric_fields(data)
clf_set = results['clf_set']

f, axs = tree_preference_report(data, groups, clf_set)
f.show()
#f.savefig(output_file_names['group_envyfreeness'])

g, axs = tree_preference_report(data, groups, clf_set, rationality_report = True)
g.show()
#g.savefig(output_file_names['group_rationality'])
