import os
import time
import numpy as np
import pandas as pd
from .data import check_data, set_defaults_for_data, add_intercept,  FORMAT_NAME_RULES, FORMAT_NAME_DCP
from .cross_validation import validate_folds, validate_cvindices, validate_fold_id, to_fold_id, is_inner_fold_id

RAW_DATA_OUTCOME_COL_IDX = 0
RAW_DATA_FILE_VALID_EXT = {'csv', 'data'}
RAW_HELPER_FILE_VALID_EXT = {'csv', 'helper'}
RAW_WEIGHTS_FILE_VALID_EXT = {'csv', 'weights'}
PROCESSED_DATA_FILE_RDATA = {'rdata'}
PROCESSED_DATA_FILE_PICKLE = {'p', 'pk', 'pickle'}
PROCESSED_DATA_FILE_VALID_EXT = PROCESSED_DATA_FILE_RDATA.union(PROCESSED_DATA_FILE_PICKLE)


#### reading raw data from disk

def load_raw_data_from_disk(dataset_file, helper_file = None, weights_file = None, include_intercept = False):
    """

    Parameters
    ----------
    dataset_file                    comma separated file ending in ".data" or "_data.csv"
                                    contains the training data, stored as a table with N+1 rows and d+1 columns
                                    column 1 contains the outcome variable entries; must be (-1,1) or (0,1)
                                    column 2 to d+1 contain the d input variables
                                    row 1 contains unique names for the outcome variable, and the input variables

    helper_file                     comma separated file
                                    contains additional information on each of the columns in dataset file:
                                    - header
                                    - is_outcome
                                    - is_partition #todo fill this out
                                    - type (boolean, categorical, numeric, ordinal)
                                    - ordering (for ordinal variables)

                                    if no file is provided or file does not exist, then the function will look for
                                     a file with the same name as dataset_file but that ends in either
                                    ".helper" (if dataset_file ends in ".data")
                                    "helper.csv" (if dataset_file ends in "_data.csv")

    weights_file                    comma separated file
                                    weights stored as a table with N rows and 1 column
                                    all sample weights must be non-negative

                                    if no file is provided or file does not exist, then the function will look for
                                     a file with the same name as dataset_file but that ends in either
                                    ".weights" (if dataset_file ends in ".data")
                                    "_weights.csv" (if dataset_file ends in "_data.csv")


    include_intercept               if True then an intercept is added to the X matrix


    Returns
    -------
    dictionary containing training data for a binary classification problem with fields

     - 'X' N x P matrix of features (numpy.ndarray) with a column of 1s for the '(Intercept)'
     - 'Y' N x 1 vector of labels (+1/-1) (numpy.ndarray)
     - 'variable_names' list of strings containing the names of each feature (list)
     - 'outcome_name' string containing the name of the output (optional)
     - 'sample_weights' N x 1 vector of sample weights, must all be positive

    """
    dataset_file, helper_file, weights_file = _validate_raw_data_files(dataset_file, helper_file, weights_file)
    has_helper_file = helper_file is not None and os.path.isfile(helper_file)
    has_weights_file = weights_file is not None and os.path.isfile(weights_file)

    # load data
    df = pd.read_csv(dataset_file, sep = ',')
    raw_data = df.values
    data_headers = list(df.columns.values)
    N = raw_data.shape[0]

    # load variable types and orderings
    if has_helper_file:
        helper_df = pd.read_csv(helper_file, sep=',')

        # get variable types
        outcome_column_idx = int(np.flatnonzero(helper_df['is_outcome'].values))
        variable_dict = helper_df[~helper_df['is_outcome']].set_index('header').to_dict()
        variable_types = variable_dict['type']

        # get ordering for variables
        ordering_dict = helper_df[helper_df['type'] == 'ordinal'].set_index('header').to_dict()
        variable_orderings = ordering_dict['ordering']
        for var_name in ordering_dict['ordering'].keys():
            variable_orderings[var_name] = variable_orderings[var_name].split('|')

        # get variable partitions
        if 'is_partition' in helper_df.columns:
            partitions = (helper_df['header'][helper_df['is_partition']]).values.tolist()

    else:
        raise NotImplementedError()
        outcome_column_idx = RAW_DATA_OUTCOME_COL_IDX
        partitions = []
        variable_orderings = {}
        variable_types = _infer_variable_types_from_data(raw_data)


    # load weights from disk
    if has_weights_file:
        sample_weights = pd.read_csv(weights_file, sep=',', header=None)
        sample_weights = sample_weights.values
        if len(sample_weights) == N + 1:
            sample_weights = sample_weights[1:]
        sample_weights = np.array(sample_weights, dtype = np.float).flatten()
    else:
        sample_weights = np.ones(N, dtype = np.float)

    # setup X and X_names
    X_col_idx = [j for j in range(raw_data.shape[1]) if j != outcome_column_idx]
    variable_names = [data_headers[j] for j in X_col_idx]
    X = raw_data[:, X_col_idx]

    # setup Y vector and Y_name
    Y = raw_data[:, outcome_column_idx]
    Y_name = data_headers[outcome_column_idx]
    Y[Y == 0] = -1
    Y = np.array(Y, dtype = int).flatten()

    #todo add this in as a required field
    meta = {
        'read_date': time.strftime("%d/%m/%y", time.localtime()),
        'dataset_file': dataset_file,
        'helper_file': helper_file,
        'weights_file': weights_file,
        }

    data = {
        'X': X,
        'Y': Y,
        'partitions': partitions,
        'sample_weights': sample_weights,
        'variable_names': variable_names,
        'variable_types': variable_types,
        'variable_orderings': variable_orderings,
        'outcome_name': Y_name,
        }

    # insert a column of ones to X for the intercept
    if include_intercept:
        data = add_intercept(data)

    # assert check_data(data)
    return data


def _validate_raw_data_files(dataset_file, helper_file = None, weights_file = None):

    if not os.path.isfile(dataset_file):
        raise IOError('could not find dataset_file: %s' % dataset_file)

    dataset_dir = os.path.dirname(dataset_file) + '/'
    file_header, _, file_extension = (os.path.basename(dataset_file)).rpartition('.')
    file_extension = file_extension.lower()
    assert file_extension in RAW_DATA_FILE_VALID_EXT

    #check for helper file
    if helper_file is not None:

        if not os.path.isfile(helper_file):
            raise IOError('could not find helper_file: %s' % helper_file)

    else:

        for check_ext in RAW_HELPER_FILE_VALID_EXT:

            if check_ext == 'csv':
                check_file = dataset_dir + file_header.rpartition('_data')[0] + '_helper.csv'
            else:
                check_file = dataset_dir + file_header + '.' + check_ext

            if os.path.isfile(check_file):
                helper_file = check_file
                break

    # check for weights file
    if weights_file is not None:

        if not os.path.isfile(weights_file):
            raise IOError('could not find weights_file: %s' % weights_file)

        assert os.path.isfile(weights_file), ("%s does not exist" % weights_file)

    else:

        for check_ext in RAW_WEIGHTS_FILE_VALID_EXT:
            if check_ext == 'csv':
                check_file = dataset_dir + file_header.rpartition('_data')[0] + '_weights.csv'
            else:
                check_file = dataset_dir + file_header + '.' + check_ext

            if os.path.isfile(check_file):
                weights_file = check_file
                break

    return dataset_file, helper_file, weights_file


def _infer_variable_types_from_data(raw_data):
    """
    infers variable types
    last column is outcome
    first column if the outcome variable
        can be (0,1) or (-1,1)
    other columns are the input variables
        numeric by default
        boolean if values are 0,1 or true/false
        categorical if entries are text
    """
    raise NotImplementedError()


#### saving data to disk


def save_data(file_name, data, cvindices = None, overwrite = False, stratified = True, check_save = True):

    if overwrite is False:
        if os.path.isfile(file_name):
            raise IOError('file %s already exist on disk' % file_name)

    try:
        file_extension = file_name.rsplit('.')[-1]
    except IndexError:
        raise ValueError('could not find extension in file_name (%r)', file_name)

    file_type = file_extension.lower()
    assert file_type in PROCESSED_DATA_FILE_VALID_EXT, \
        'unsupported extension %s\nsupported extensions: %s' % (file_type, ", ".join(PROCESSED_DATA_FILE_VALID_EXT))

    data = set_defaults_for_data(data)
    assert check_data(data)

    if cvindices is not None:
        cvindices = validate_cvindices(cvindices, stratified)

    if file_type in PROCESSED_DATA_FILE_RDATA:
        saved_file_flag = _save_data_as_rdata(file_name, data, cvindices)

    elif file_type in PROCESSED_DATA_FILE_PICKLE:
        saved_file_flag = _save_data_as_pickle(file_name, data, cvindices)

    assert os.path.isfile(file_name), 'could not locate saved file on disk'

    if check_save:
        loaded_data, loaded_cvindices = load_processed_data(file_name)
        assert np.all(loaded_data['X'] == data['X'])
        assert loaded_cvindices.keys() == cvindices.keys()

    return saved_file_flag


def _save_data_as_pickle(file_name, data, cvindices):
    try:
        import cPickle as pickle
    except:
        import pickle

    data = set_defaults_for_data(data)

    file_contents = {
        'data': data,
        'cvindices': cvindices
        }

    with open(file_name, 'wb') as outfile:
        pickle.dump(file_contents, outfile, protocol = pickle.HIGHEST_PROTOCOL)

    return True


def _save_data_as_rdata(file_name, data, cvindices):

    import rpy2.robjects as rn
    from .rpy2_helper import r_assign, r_save_to_disk
    from rpy2.robjects import pandas2ri
    data = set_defaults_for_data(data)
    assert check_data(data)

    fields_to_save = ["format", "Y", "sample_weights", "outcome_name", "variable_names", "variable_types", "variable_orderings"]

    if data['format'] == FORMAT_NAME_RULES:
        fields_to_save += ["feature_groups", "feature_names", "feature_types", "feature_orderings",
                           "feature_group_limits"]
    elif data['format'] == FORMAT_NAME_DCP:
        fields_to_save += ['partitions']

    try:

        for k in fields_to_save:
            r_assign(data[k], k)

    except:

        from dcptree.debug import ipsh
        ipsh()

    r_assign(cvindices, "cvindices")

    # feature matrix
    var_type_to_col_type = {'boolean': 'bool',
                            'categorical': 'str',
                            'numeric': 'float',
                            'ordinal': 'str',
                            }
    col_types = {n: var_type_to_col_type[data['variable_types'][n]] for n in data['variable_names']}

    pandas2ri.activate()

    X_df = pd.DataFrame(data = data['X'])
    X_df.columns = data['variable_names']
    X_df = X_df.astype(col_types)
    rn.r.assign('X', X_df)

    # test set
    has_test_set = ('X_test' in data) and ('Y_test' in data) and ('sample_weights_test' in data)
    if has_test_set:
        X_test_df = pd.DataFrame(data = data['X_test'])
        X_test_df.columns = data['variable_names']
        X_test_df = X_test_df.astype(col_types)
        rn.r.assign('X_test', pandas2ri.py2ri(X_test_df))
        r_assign(data['Y_test'], 'Y_test')
        r_assign(data['sample_weights_test'], 'sample_weights_test')
    else:
        rn.reval(
                """
                X_test = matrix(data=NA, nrow = 0, ncol = ncol(X));
                Y_test = matrix(data=NA, nrow = 0, ncol = 1);
                sample_weights_test = matrix(data=1.0, nrow = 0, ncol = 1);
                """
                )

    pandas2ri.deactivate()

    variables_to_save = fields_to_save + ["cvindices", "X", "X_test", "Y_test", "sample_weights_test"]
    r_save_to_disk(file_name, variables_to_save)
    return True


#### loading data from disk


def load_processed_data(file_name):

    assert os.path.isfile(file_name), \
        'file %s not found' % file_name

    try:
        file_extension = file_name.rsplit('.')[-1]
        file_type = file_extension.lower()
    except IndexError:
        raise ValueError('could not find extension in file_name (%r)', file_name)

    assert file_type in PROCESSED_DATA_FILE_VALID_EXT, \
        'unsupported file type; supported file types: %s' % ", ".join(PROCESSED_DATA_FILE_VALID_EXT)

    if file_type == 'rdata':
        data, cvindices = _load_processed_data_rdata(file_name)
    else:
        data, cvindices = _load_processed_data_pickle(file_name)

    assert check_data(data)
    data = set_defaults_for_data(data)
    cvindices = validate_cvindices(cvindices)
    return data, cvindices


def _load_processed_data_pickle(file_name):

    try:
        import cPickle as pickle
    except ImportError:
        import pickle

    with open(file_name, 'rb') as infile:
        file_contents = pickle.load(infile)

    assert 'data' in file_contents
    assert 'cvindices' in file_contents
    return file_contents['data'], file_contents['cvindices']


def _load_processed_data_rdata(file_name):

    import rpy2.robjects as rn

    rn.reval("data = new.env(); load('%s', data)" % file_name)
    r_data = rn.r.data
    data_fields = list(rn.r.data.keys())

    loaded_data = dict()


    for xf, yf, sw in [('X', 'Y', 'sample_weights'),
                       ('X_test', 'Y_test', 'sample_weights_test'),
                       ('X_validation', 'Y_validation', 'sample_weights_validation')]:

        if xf in data_fields and yf in data_fields and len(np.array(r_data[yf])) > 0:

            loaded_data[yf] = np.array(r_data[yf]).flatten()
            loaded_data[yf][loaded_data[yf] == 0] = -1
            loaded_data[xf] = np.array(r_data[xf])

            if loaded_data[xf].shape[1] == len(loaded_data[yf]):
                loaded_data[xf] = np.transpose(loaded_data[xf])

            if sw in data_fields:
                loaded_data[sw] = np.array(r_data[sw]).flatten()

    if 'variable_names' in data_fields:
        loaded_data['variable_names'] = np.array(rn.r.data['variable_names']).tolist()
    elif 'X_headers' in data_fields:
        loaded_data['variable_names'] = np.array(rn.r.data['X_headers']).tolist()
    elif 'X_header' in data_fields:
        loaded_data['variable_names'] = np.array(rn.r.data['X_header']).tolist()

    if 'outcome_name' in data_fields:
        loaded_data['outcome_name'] = np.array(r_data['outcome_name'])[0]
    elif 'Y_headers' in data_fields:
        loaded_data['outcome_name'] = np.array(r_data['Y_headers'])[0]
    elif 'Y_header' in data_fields:
        loaded_data['outcome_name'] = np.array(r_data['Y_header'])[0]

    if 'format' in data_fields:
        loaded_data['format'] = np.array(r_data['format'])[0]

    if 'partitions' in data_fields:
        loaded_data['partitions'] = np.array(rn.r.data['partitions']).tolist()

    cvindices = _load_cvindices_from_rdata(file_name)
    data = set_defaults_for_data(loaded_data)
    return data, cvindices


#### loading cvindices from processed file on disk ####


def load_cvindices_from_disk(fold_file):
    """
    Reads cross-validation indices from various file types including:
        - CSV file containing with exactly N data points
        - RData file containing cvindices object
        - mat file containing cvindices object
    :param fold_file:
    :return: dictionary containing folds
    keys have the form
    """
    # load fold indices from disk
    if not os.path.isfile(fold_file):
        raise IOError('could not find fold file on disk: %s' % fold_file)

    if fold_file.lower().ends_with('.csv'):
        folds = pd.read_csv(fold_file, sep=',', header=None)
        folds = validate_folds(folds=folds)
        fold_id = to_fold_id(total_folds = max(folds), replicate_idx = 1)
        cvindices = {fold_id: folds}

    if fold_file.lower().endswith('.rdata'):
        cvindices = _load_cvindices_from_rdata(data_file=fold_file)

    if fold_file.lower().endswith('.mat'):
        raise NotImplementedError()
        # import scipy.io as sio
        # cvindices = sio.loadmat(file_name = fold_file,
        #                         matlab_compatible=False,
        #                         chars_as_strings=True,
        #                         squeeze_me=True,
        #                         variable_names=['cvindices'],
        #                         verify_compressed_data_integrity=False)

    cvindices = validate_cvindices(cvindices)
    return cvindices


def _load_folds_from_rdata(data_file, fold_id):
    """
    (internal) returns folds from RData file in the pipeline
    :param data_file:
    :param fold_id:
    :param inner_fold_id:
    :return:
    """

    if os.path.isfile(data_file):
        file_extension = data_file.rsplit('.')[-1]
        assert file_extension.lower() == 'rdata', 'unsupported file extension: %r' % file_extension
    else:
        raise IOError('could not find data_file: %s' % data_file)

    fold_id = validate_fold_id(fold_id)
    r_variables = "data_file='%s'; fold_id='%s'" % (data_file, fold_id)

    import rpy2.robjects as rn
    from .rpy2_helper import r_clear

    if is_inner_fold_id(fold_id):
        r_cmd = """raw_data = new.env()
        load(data_file, envir=raw_data)
        folds = raw_data$cvindices[[fold_id]][,1]
        """
    else:
        r_cmd = """raw_data = new.env()
        load(data_file, envir=raw_data)
        folds = raw_data$cvindices[[substring(fold_id, 1, 3)]][, as.double(substr(fold_id, 5, 6))]
        """

    rn.reval(r_variables)
    rn.reval(r_cmd)
    folds = np.array(rn.r['folds'])
    folds = validate_folds(folds, fold_id)
    r_clear()
    return folds


def _load_cvindices_from_rdata(data_file):
    """
    (internal) cvindices object stored in a RData file in the pipeline
    :param data_file:
    """

    if not os.path.isfile(data_file):
        raise IOError('could not find data_file: %s' % data_file)

    import rpy2.robjects as rn
    from .rpy2_helper import r_clear

    r_variables = "data_file = '%s'" % data_file
    r_cmd = """
    raw_data = new.env()
    load(data_file, envir=raw_data)
    all_fold_ids = names(raw_data$cvindices)
    list2env(raw_data$cvindices, globalenv())
    remove(raw_data, cvindices);
    """
    rn.reval(r_variables)
    rn.reval(r_cmd)
    all_fold_ids = np.array(rn.r['all_fold_ids'])

    cvindices = {}
    max_fold_value = 0
    for key in all_fold_ids:
        try:
            folds = np.array(rn.r[key])
            if (folds.shape[0] == 1) or (folds.shape[1] == 1):
                folds = folds.flatten()
            max_fold_value = max(max_fold_value, np.max(folds))
            cvindices[key] = folds
        except Exception:
            pass

    #cast as unsigned integers to save storage space
    if max_fold_value < 255:
        storage_type = 'uint8'
    elif max_fold_value < 65535:
        storage_type = 'uint16'
    elif max_fold_value < 4294967295:
        storage_type = 'uint32'
    else:
        storage_type = 'uint64'

    for key in cvindices.keys():
        cvindices[key] = cvindices[key].astype(storage_type)

    #break down matrices to just folds
    all_keys = list(cvindices.keys())
    for key in all_keys:
        if key[0] == 'K' and len(key) == 3:
            fold_matrix = cvindices.pop(key)
            n_repeats = fold_matrix.shape[1]
            for r in range(n_repeats):
                folds = np.array(fold_matrix[:,r])
                folds = folds.flatten()
                folds = folds.astype(storage_type)
                fold_id = '%sN%02d' % (key, r)
                cvindices[fold_id] = folds

    # cleanup in the R environment just in case
    r_clear()
    return cvindices


