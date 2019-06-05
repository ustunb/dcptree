from copy import deepcopy
import numpy as np
from inspect import signature, getfullargspec
from dcptree.data import get_variable_indices, check_data_required_fields

SUPPORTED_METHOD_NAMES = {'svm_linear', 'logreg'}

DEFAULT_TRAINING_SETTINGS = {

    'svm_linear': {
        'fit_intercept': True,
        'intercept_scaling': 1.0,
        'class_weight': None,
        'loss': "hinge",
        'penalty': 'l2',
        'C': 1.0,
        'tol': 1e-4,
        'max_iter': 1e3,
        'dual': True,
        'random_state': None,
        'verbose': False
        },

    'logreg': {
        'fit_intercept': True,
        'intercept_scaling': 1.0,
        'class_weight': None,
        'penalty': 'l2',
        'dual': False,
        'C': 1.0,
        'tol': 1e-4,
        'solver': 'liblinear',
        'warm_start': False,
        'max_iter': 1e3,
        'random_state': None,
        'verbose': False,
        'n_jobs': 1
        },
    }


class ClassificationModel(object):

    LINEAR_MODEL_TYPE = 'linear'
    SUPPORTED_MODEL_TYPES = [LINEAR_MODEL_TYPE]

    def __init__(self, predict_handle, model_type, model_info, training_info = None):

        if training_info is None:
            training_info = dict()

        # check predict handle
        assert callable(predict_handle)
        spec = getfullargspec(predict_handle)
        assert 'X' in spec.args

        # check other fields
        assert isinstance(model_type, str)
        assert isinstance(model_info, dict)
        assert isinstance(training_info, dict)
        assert model_type in ClassificationModel.SUPPORTED_MODEL_TYPES, "unsupported model type"

        # initialize properties
        self.predict_handle = predict_handle
        self._model_type = str(model_type)
        self._model_info = deepcopy(model_info)
        self._training_info = deepcopy(training_info)

        if self._model_type is ClassificationModel.LINEAR_MODEL_TYPE:
            self._coefficients = np.array(self._model_info['coefficients'])
            self._intercept = np.array(self._model_info['intercept'])

        assert self.check_rep()


    def predict(self, X):
        return np.array(self.predict_handle(X)).flatten()


    @property
    def model_type(self):
        return str(self._model_type)


    @property
    def model_info(self):
        return deepcopy(self._model_info)

    @property
    def coefficients(self):
        if self._model_type is ClassificationModel.LINEAR_MODEL_TYPE:
            return np.array(self._model_info['coefficients']).flatten()

    @property
    def intercept(self):
        if self._model_type is ClassificationModel.LINEAR_MODEL_TYPE:
            return np.array(self._model_info['intercept'])

    @property
    def training_info(self):
        return deepcopy(self._training_info)


    def check_rep(self):
        assert callable(self.predict_handle)
        assert isinstance(self.model_type, str)
        assert isinstance(self.model_info, dict)
        assert isinstance(self.training_info, dict)
        return True


#### sklearn wrappers ####

def train_sklearn_linear_model(data, method_name, settings = None, normalize_variables = False, **kwargs):

    assert check_data_required_fields(ready_for_training = True, **data)
    assert method_name in SUPPORTED_METHOD_NAMES, 'method %s not supported' % method_name

    # import correct classifier from scikit learn
    if method_name == 'svm_linear':
        from sklearn.svm import LinearSVC
        Classifier = LinearSVC
    elif method_name == 'logreg':
        from sklearn.linear_model import LogisticRegression
        Classifier = LogisticRegression

    # set missing settings
    if settings is None:
        settings = dict(DEFAULT_TRAINING_SETTINGS[method_name])
    else:
        assert isinstance(settings, dict)
        settings = dict(settings)
        for name, default_value in DEFAULT_TRAINING_SETTINGS[method_name].items():
            settings.setdefault(name, default_value)

    # override settings with keyword arguments
    for k, v in kwargs.items():
        if k in settings:
            settings[k] = v

    # drop the intercept from the data if it exists
    coefficient_idx = get_variable_indices(data, include_intercept = False)
    X = data['X'][:, coefficient_idx]
    y = data['Y']

    # preprocess the data
    if normalize_variables:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler(copy = True, with_mean = True, with_std = True).fit(X)
        x_shift = np.array(scaler.mean_)
        x_scale = np.sqrt(scaler.var_)
        X = scaler.transform(X)
    else:
        x_shift = np.zeros(X.shape[1], dtype = float)
        x_scale = np.ones(X.shape[1], dtype = float)

    # extract classifier arguments from settings
    clf_args = dict()
    clf_argnames = list(signature(Classifier).parameters.keys())
    for k in clf_argnames:
        if k in settings and settings[k] is not None:
            clf_args[k] = settings[k]

    # fit classifier
    clf = Classifier(**clf_args)
    clf.fit(X, y)

    # store classifier parameters
    intercept = np.array(clf.intercept_) if settings['fit_intercept'] else 0.0
    coefficients = np.array(clf.coef_)

    # adjust coefficients for unnormalized data
    if normalize_variables:
        coefficients = coefficients * x_scale
        intercept = intercept + np.dot(coefficients, x_shift)

    # setup parameters for model object
    predict_handle = lambda X: np.sign(X[:, coefficient_idx].dot(coefficients.transpose()) + intercept)

    model_info = {
        'intercept': intercept,
        'coefficients': coefficients,
        'coefficient_idx': coefficient_idx,
        }

    training_info = {
        'method_name': method_name,
        'normalize_variables': normalize_variables,
        'x_shift': x_shift,
        'x_scale': x_scale,
        }

    training_info.update(settings)

    model = ClassificationModel(predict_handle = predict_handle,
                                model_type = ClassificationModel.LINEAR_MODEL_TYPE,
                                model_info = model_info,
                                training_info = training_info)

    return model


def train_svm(data, settings = None, normalize_variables = False, **kwargs):

    model = train_sklearn_linear_model(data,
                                        method_name = 'svm_linear',
                                        settings = settings,
                                        normalize_variables = normalize_variables,
                                        **kwargs)

    return model


def train_logreg(data, settings = None, normalize_variables = False, **kwargs):

    model = train_sklearn_linear_model(data,
                                      method_name = 'logreg',
                                      settings = settings,
                                      normalize_variables = normalize_variables,
                                      **kwargs)

    return model


