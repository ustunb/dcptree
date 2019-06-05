# Author: Berk Ustun | www.berkustun.com
# License: BSD 3 clause

import os
from dcptree.paths import *
from dcptree.data_io import load_processed_data, _load_folds_from_rdata, _load_cvindices_from_rdata
from dcptree.cross_validation import filter_data_to_fold, validate_fold_id, filter_cvindices

#load dataset
data_name = 'adult'
test_dir = repo_dir / 'dcptree/tests/'
dataset_file = data_dir + data_name + '/' + data_name + 'adult_processed.data'
weights_file = None
helper_file = None


def test_load_folds_from_rdata():

    fold_id = "K05N01"
    folds = _load_folds_from_rdata(data_file=dataset_file, fold_id=fold_id)
    assert len(folds) > 0

    fold_id = "K05N01_F01K05"
    folds = _load_folds_from_rdata(data_file=dataset_file, fold_id=fold_id)
    assert len(folds) > 0


def test_load_cvindices_from_rdata():

    cvindices = _load_cvindices_from_rdata(dataset_file)
    fold_ids = cvindices.keys()
    for key in fold_ids:
        assert len(key) > 3
        assert cvindices[key].ndim == 1


def test_filter_cvindices():

    cvindices = _load_cvindices_from_rdata(dataset_file)
    filtered = filter_cvindices(cvindices,
                                total_folds=[1, 2, 5, 10],
                                total_folds_inner_cv=[],
                                n_replicates=1)

    assert len(filtered) == 4
    expected_fold_ids = ("K01N01", "K02N01", "K05N01", "K10N01")
    for fold_id in expected_fold_ids:
        if fold_id in cvindices:
            assert fold_id in filtered


def test_validate_fold_id():

    valid_fold_ids = ("K01N01", " K01N02 ", "k10n30", "K10N01_F02K05")
    invalidate_fold_ids = ("K01", "K1000", "KK01N02", "K10N30 F02K01")
    for fold_id in valid_fold_ids:
        valid_id = validate_fold_id(fold_id)
        assert valid_id == fold_id.upper().strip()

    for fold_id in invalidate_fold_ids:
        try:
            valid_id = validate_fold_id(fold_id)
            assert False
        except Exception:
            assert True
