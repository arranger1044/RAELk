'''
  A script for building a predictor upon the representations learned with AutoEncoders with RAELk.
  This file is part of RAELk - Representation Learning with AutoEncoders in Keras

  Different learning settings provided:
  -  X  -> Y   (no embeddings)
  -  Ex -> Y
  -  X  -> Ey
  -  Ex -> Ey

  Copyright (C) 2017 University of Bari "Aldo Moro"
  Author: Antonio Vergari
  Email: antonio.vergari@uniba.it
  $Date: 2017-01-22$

  antonio vergari
  ***************************************************************************************************

  RAELk is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation (version 3).

  RAELk is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

  You should have received a copy of the GNU General Public License along with this program; if not,
see [http://www.gnu.org/licenses/](http://www.gnu.org/licenses/)

'''

import argparse

try:
    from time import perf_counter
except:
    from time import time
    perf_counter = time

from collections import defaultdict

import numpy

import datetime

import os

import sys

import logging

import pickle
import gzip

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import MultiTaskLasso
from sklearn.neighbors import KNeighborsClassifier
# from multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import neighbors
from sklearn import decomposition
from sklearn import manifold
from sklearn.multiclass import OneVsRestClassifier

from keras.models import load_model

from utils import load_cv_splits
from utils import load_train_val_test_splits
from utils import print_fold_splits_shapes
from utils import SPLIT_NAMES, SCORE_NAMES
from utils import compute_scores
from utils import compute_threshold


MAX_N_INSTANCES = 10000

PICKLE_SPLIT_EXT = 'pickle'
PREDS_PATH = 'preds'
PICKLE_SPLIT_EXT = 'pickle'
COMPRESSED_PICKLE_SPLIT_EXT = 'pklz'
FEATURE_FILE_EXT = 'features'
INFO_FILE_EXT = 'features.info'
SCOPE_FILE_EXT = 'scopes'
COMPRESSED_MODEL_EXT = 'model.pklz'


PREPROCESS_DICT = {
    'std-scl': StandardScaler,
    'min-max': MinMaxScaler,
    'l2-norm': Normalizer
}

CLASSIFIER_DICT = {
    'lr-l2-ovr-bal': lambda c:
    OneVsRestClassifier(LogisticRegression(C=c,
                                           penalty='l2',
                                           tol=0.0001,
                                           fit_intercept=True,
                                           class_weight='balanced',
                                           solver='liblinear')),
    # 'rr-l2-ovr-bal': lambda c:
    # MultiOutputRegressor(Ridge(alpha=c,
    #                            tol=0.0001, )),
    'rc-l2-ovr-bal': lambda c:
    OneVsRestClassifier(RidgeClassifier(alpha=c,
                                        tol=0.0001,
                                        fit_intercept=True,
                                        class_weight='balanced',)),

    'rr-l2-bal': lambda c: Ridge(alpha=c,
                                 tol=0.0001,
                                 fit_intercept=True),
    'mtl': lambda c: MultiTaskLasso(alpha=c,
                                    tol=0.0001,
                                    fit_intercept=True),
    'rf': lambda c: RandomForestRegressor(n_estimators=int(c),
                                          max_depth=4,
                                          # max_features='sqrt',
                                          ),

}


def decode_predictions(repr_preds, ae_decoder, threshold=0.5):

    preds = ae_decoder.predict(repr_preds)
    # print('preds', preds[:2])
    th_preds = compute_threshold(preds, threshold)
    # print('thresh', th_preds[:2])
    return th_preds


def decode_predictions_knn(preds, embeds, embeds_labels, **knn_wargs):
    knnc = KNeighborsClassifier(**knn_wargs)

    fit_start_t = perf_counter()
    knnc.fit(embeds, embeds_labels)
    fit_end_t = perf_counter()
    logging.info('\n\t\tKNN fitted in {} secs'.format(fit_end_t - fit_start_t))

    predict_start_t = perf_counter()
    dec_preds = knnc.predict(preds)
    predict_end_t = perf_counter()
    logging.info('\t\t\tprediction done in {} secs'.format(predict_end_t - predict_start_t))

    return dec_preds


def load_ae_decoder(model_path):

    logging.info('Loading AE model from {}'.format(model_path))

    #
    # loading with keras
    load_start_t = perf_counter()
    ae_decoder = load_model(model_path)
    load_end_t = perf_counter()
    logging.info('\tdone in {}'.format(load_end_t - load_start_t))

    return ae_decoder


parser = argparse.ArgumentParser()
parser.add_argument("dataset", type=str,
                    help='Specify a dataset file path')

parser.add_argument('--data-exts', type=str, nargs='+',
                    default=None,
                    help='Dataset split extensions')

parser.add_argument('--repr-x', type=str,
                    default=None,
                    help='Specify a learned representation for the X (file path)')

parser.add_argument('--repr-x-exts', type=str, nargs='+',
                    default=None,
                    help='Learned representations split extensions')

parser.add_argument('--repr-y', type=str,
                    default=None,
                    help='Specify a learned representation for the Y (file path)')

parser.add_argument('--repr-y-exts', type=str, nargs='+',
                    default=None,
                    help='Learned representations split extensions')

parser.add_argument('--dtype', type=str, nargs='?',
                    default='int32',
                    help='Loaded dataset type')

parser.add_argument('--repr-x-dtype', type=str, nargs='?',
                    default='float',
                    help='Loaded representation type')

parser.add_argument('--repr-y-dtype', type=str, nargs='?',
                    default='float',
                    help='Loaded representation type')

parser.add_argument('--decode-model', type=str, nargs='+',
                    help='AE decoder file path for decoding (or sequence of paths when --cv)')

parser.add_argument('--knn-decode', type=str, nargs='?',
                    help='Additional sklearn knn parameters in the for of a list' +
                    ' "[name1=val1,..,namek=valk]"')

parser.add_argument('--seed', type=int, nargs='?',
                    default=1337,
                    help='Seed for the random generator')

parser.add_argument('-o', '--output', type=str, nargs='?',
                    default='./exp/ae-ml-repr-x/',
                    help='Output dir path')

parser.add_argument('--scores', type=str, nargs='+',
                    default=['accuracy'],
                    help='Scores for the classifiers ("accuracy"|"hamming"|"exact")')

parser.add_argument('--preprocess', type=str, nargs='+',
                    default=[],
                    help='Algorithms to preprocess data')

parser.add_argument('--classifier', type=str, nargs='?',
                    default=None,
                    help='Parametrized version of the logistic regression')

#
# TODO: to generalize to different classifiers
parser.add_argument('--log-c', type=float, nargs='+',
                    default=[0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 5.0, 10.0],
                    help='logistic ')

parser.add_argument('--feature-inc', type=int, nargs='+',
                    default=None,
                    help='Considering features in batches')

parser.add_argument('--exp-name', type=str, nargs='?',
                    default=None,
                    help='Experiment name, if not present a date will be used')

parser.add_argument('--concat', action='store_true',
                    help='Whether to concatenate the new representation to the old dataset')

parser.add_argument('--expify', action='store_true',
                    help='Whether to exp transform the data')

parser.add_argument('--save-probs', action='store_true',
                    help='Whether to save predictions as probabilities')

parser.add_argument('--save-preds', action='store_true',
                    help='Whether to save predictions')

parser.add_argument('--save-model', action='store_true',
                    help='Whether to store the model file as a pickle file')

parser.add_argument('--x-orig', action='store_true',
                    help='Whether to evaluate only the original data')

parser.add_argument('-v', '--verbose', type=int, nargs='?',
                    default=1,
                    help='Verbosity level')

parser.add_argument('--cv', type=int,
                    default=None,
                    help='Folds for cross validation for model selection')

parser.add_argument('--save-text', action='store_true',
                    help='Saving the repr data to text as well')


#
# parsing the args
args = parser.parse_args()

decode = False
knn_sklearn_args = None

#
# setting verbosity level
if args.verbose == 1:
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
elif args.verbose == 2:
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

logging.info("Starting with arguments:\n%s", args)


if args.repr_y is not None:
    decode = True

    if not args.decode_model:
        raise ValueError('Missing model to decode data')

    if args.knn_decode is not None:
        sklearn_key_value_pairs = args.knn_decode.translate(
            {ord('['): '', ord(']'): ''}).split(',')
        knn_sklearn_args = {key.strip(): value.strip() for key, value in
                            [pair.strip().split('=')
                             for pair in sklearn_key_value_pairs]}
        if 'n_neighbors' in knn_sklearn_args:
            knn_sklearn_args['n_neighbors'] = int(knn_sklearn_args['n_neighbors'])
        if 'n_jobs' in knn_sklearn_args:
            knn_sklearn_args['n_jobs'] = int(knn_sklearn_args['n_jobs'])
        if 'p' in knn_sklearn_args:
            knn_sklearn_args['p'] = int(knn_sklearn_args['p'])
    else:
        knn_sklearn_args = {}
    logging.info('KNN sklearn args: {}'.format(knn_sklearn_args))


#
# loading the dataset splits
#
logging.info('Loading datasets: %s', args.dataset)
dataset_name = args.dataset.split('/')[-1]
#
# replacing  suffixes names
dataset_name = dataset_name.replace('.pklz', '')
dataset_name = dataset_name.replace('.pkl', '')
dataset_name = dataset_name.replace('.pickle', '')

fold_splits = None
repr_fold_x_splits = None
repr_fold_y_splits = None

train_ext = None
valid_ext = None
test_ext = None
repr_train_x_ext = None
repr_valid_x_ext = None
repr_test_x_ext = None
repr_train_y_ext = None
repr_valid_y_ext = None
repr_test_y_ext = None


if args.data_exts is not None:
    if len(args.data_exts) == 1:
        train_ext, = args.data_exts
    elif len(args.data_exts) == 2:
        train_ext, test_ext = args.data_exts
    elif len(args.data_exts) == 3:
        train_ext, valid_ext, test_ext = args.data_exts
    else:
        raise ValueError('Up to 3 data extenstions can be specified')

if args.repr_x_exts is not None:
    if len(args.repr_exts) == 1:
        repr_train_x_ext, = args.repr_x_exts
    elif len(args.repr_exts) == 2:
        repr_train_x_ext, repr_test_x_ext = args.repr_x_exts
    elif len(args.repr_exts) == 3:
        repr_train_x_ext, repr_valid_x_ext, repr_test_x_ext = args.repr_x_exts
    else:
        raise ValueError('Up to 3 repr data extenstions can be specified')

if args.repr_y_exts is not None:
    if len(args.repr_exts) == 1:
        repr_train_y_ext, = args.repr_y_exts
    elif len(args.repr_exts) == 2:
        repr_train_y_ext, repr_test_y_ext = args.repr_y_exts
    elif len(args.repr_exts) == 3:
        repr_train_y_ext, repr_valid_y_ext, repr_test_y_ext = args.repr_y_exts
    else:
        raise ValueError('Up to 3 repr data extenstions can be specified')


n_folds = args.cv if args.cv is not None else 1

#
# loading data and learned representations
if args.cv is not None:

    fold_splits = load_cv_splits(args.dataset,
                                 dataset_name,
                                 n_folds,
                                 train_ext=train_ext,
                                 valid_ext=valid_ext,
                                 test_ext=test_ext,
                                 dtype=args.dtype)

    if args.repr_x is not None:
        repr_fold_x_splits = load_cv_splits(args.repr_x,
                                            dataset_name,
                                            n_folds,
                                            x_only=True,
                                            train_ext=repr_train_x_ext,
                                            valid_ext=repr_valid_x_ext,
                                            test_ext=repr_test_x_ext,
                                            dtype=args.repr_x_dtype)
    else:
        repr_fold_x_splits = [[split[0] if split else None for split in fold]
                              for fold in fold_splits]

    if decode:
        repr_fold_y_splits = load_cv_splits(args.repr_y,
                                            dataset_name,
                                            n_folds,
                                            y_only=True,
                                            train_ext=repr_train_y_ext,
                                            valid_ext=repr_valid_y_ext,
                                            test_ext=repr_test_y_ext,
                                            dtype=args.repr_y_dtype)

else:
    fold_splits = load_train_val_test_splits(args.dataset,
                                             dataset_name,
                                             train_ext=train_ext,
                                             valid_ext=valid_ext,
                                             test_ext=test_ext,
                                             dtype=args.dtype)

    if args.repr_x is not None:
        repr_fold_x_splits = load_train_val_test_splits(args.repr_x,
                                                        dataset_name,
                                                        x_only=True,
                                                        train_ext=repr_train_x_ext,
                                                        valid_ext=repr_valid_x_ext,
                                                        test_ext=repr_test_x_ext,
                                                        dtype=args.repr_x_dtype)
    else:
        repr_fold_x_splits = [[split[0] if split else None for split in fold]
                              for fold in fold_splits]

    if decode:
        repr_fold_y_splits = load_train_val_test_splits(args.repr_y,
                                                        dataset_name,
                                                        y_only=True,
                                                        train_ext=repr_train_y_ext,
                                                        valid_ext=repr_valid_y_ext,
                                                        test_ext=repr_test_y_ext,
                                                        dtype=args.repr_y_dtype)


#
# printing
logging.info('Original folds')
print_fold_splits_shapes(fold_splits)
logging.info('Repr X folds')
print_fold_splits_shapes(repr_fold_x_splits)
if decode:
    logging.info('Repr Y folds')
    print_fold_splits_shapes(repr_fold_y_splits)


#
# Opening the file for test prediction
#
if args.exp_name:
    out_path = os.path.join(args.output, dataset_name + '_' + args.exp_name)
else:
    date_string = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_path = os.path.join(args.output, dataset_name + '_' + date_string)
out_log_path = os.path.join(out_path, 'exp.log')
os.makedirs(out_path, exist_ok=True)

logging.info('Opening log file {} ...'.format(out_log_path))

assert len(fold_splits) == len(repr_fold_x_splits)
if decode:
    assert len(fold_splits) == len(repr_fold_y_splits)

#
# shall we concatenate them? or just adding the labels?
# labelled_splits = None
# if args.eval_only_orig:
#     if decode:
#         logging.info('Classification only on original data (X -> Y\')\n')

#     else:
#         logging.info('Classification only on original data (X -> Y)\n')
#         labelled_splits = fold_splits
# else:
#     logging.info('Classification on the new representations (X\' -> Y)\n')
labelled_splits = []

for i in range(len(fold_splits)):

    repr_x_fold = repr_fold_x_splits[i]
    fold = fold_splits[i]
    repr_y_fold = None
    if decode:
        repr_y_fold = repr_fold_y_splits[i]

    labelled_fold = []

    for s, (repr_x, split) in enumerate(zip(repr_x_fold, fold)):

        split_x = None
        split_y = None
        repr_y = None

        if decode:
            repr_y = repr_y_fold[s]

        if split is not None:
            split_x, split_y = split

        if repr_x is not None:
            # print(repr_x)
            if args.concat:
                new_repr_x = numpy.concatenate((split_x, repr_x), axis=1)
                assert new_repr_x.shape[0] == split_x.shape[0]
                assert new_repr_x.shape[1] == split_x.shape[1] + repr_x.shape[1]
                logging.info('Concatenated representations: {} -> {}\n'.format(repr_x.shape,
                                                                               new_repr_x.shape))
                if decode:
                    labelled_fold.append([new_repr_x, repr_y])
                else:
                    labelled_fold.append([new_repr_x, split_y])

            else:
                if args.repr_x is None:
                    if decode:
                        logging.info('fold {}: {} (X |-> Y\') ({} |-> {})\n'.format(i,
                                                                                    SPLIT_NAMES[
                                                                                        s],
                                                                                    split_x.shape,
                                                                                    repr_y.shape))
                        labelled_fold.append([split_x, repr_y])
                    else:
                        logging.info('fold {}: {} (X |-> Y) ({} |-> {})\n'.format(i,
                                                                                  SPLIT_NAMES[s],
                                                                                  split_x.shape,
                                                                                  split_y.shape))
                        labelled_fold.append([split_x, split_y])
                else:
                    if decode:
                        logging.info('fold {}: {} (X\' |-> Y\') ({} |-> {})\n'.format(i,
                                                                                      SPLIT_NAMES[
                                                                                          s],
                                                                                      repr_x.shape,
                                                                                      repr_y.shape))
                        labelled_fold.append([repr_x, repr_y])
                    else:
                        logging.info('fold {}: {} (X\' |-> Y) ({} |-> {})\n'.format(i,
                                                                                    SPLIT_NAMES[s],
                                                                                    repr_x.shape,
                                                                                    split_y.shape))
                        labelled_fold.append([repr_x, split_y])
        else:
            labelled_fold.append(None)

    labelled_splits.append(labelled_fold)


if args.expify:
    logging.info('Turning into exponentials\n')
    for f in range(len(labelled_splits)):
        for i in range(len(labelled_splits[f])):
            if labelled_splits[f][i] is not None:
                labelled_splits[f][i][0] = numpy.exp(labelled_splits[f][i][0])
#
# preprocessing
if args.preprocess:
    raise ValueError('Preprocessing not implemented yet')
    # for prep in args.preprocess:
    #     preprocessor = PREPROCESS_DICT[prep]()
    #     logging.info('Preprocessing with {}:'.format(preprocessor))
    #     #
    #     # assuming the first split is the training set
    #     preprocessor.fit(labelled_splits[0][0])
    #     for i in range(len(labelled_splits)):
    #         labelled_splits[i][0] = preprocessor.transform(labelled_splits[i][0])

ae_fold_decoders = None
# feature_fold_infos = None
if decode:
    if not args.knn_decode:
        assert len(args.decode_model) == n_folds, args.decode_model
        # ae_fold_decoders = [load_ae_decoder(args.decode_model, model_suffix='decoder', fold=f)
        #                     for f in range(len(labelled_splits))]
        ae_fold_decoders = [load_ae_decoder(args.decode_model[f])
                            for f in range(len(labelled_splits))]


with open(out_log_path, 'w') as out_log:

    out_log.write("parameters:\n{0}\n".format(args))
    out_log.flush()

    if args.feature_inc:
        header_str = "\t\t\t{}\n\t\t\t{}\n".format('\t'.join(SPLIT_NAMES * len(args.scores)),
                                                   '\t'.join([SCORE_NAMES[s] for s in args.scores
                                                              for n in SPLIT_NAMES]))
        out_log.write('{}'.format(header_str))
        out_log.flush()

        min_feature = 0
        max_feature = labelled_splits[0][0][0].shape[1]

        increment = None
        if len(args.feature_inc) == 1:
            increment = args.feature_inc[0]
        elif len(args.feature_inc) == 2:
            min_feature = args.feature_inc[0]
            increment = args.feature_inc[1]
        elif len(args.feature_inc) == 3:
            min_feature = args.feature_inc[0]
            max_feature = args.feature_inc[1]
            increment = args.feature_inc[2]
        else:
            raise ValueError('More than three values specified for --feature-inc')

        increments_range = range(min_feature + increment, max_feature + increment, increment)
        n_increments = len(list(increments_range))
        n_params = len(args.log_c)
        n_folds = len(labelled_splits)
        n_splits = 3
        n_scores = len(args.scores)
        score_tensor = numpy.zeros((n_increments, n_params, n_folds, n_splits, n_scores))
        score_tensor[:] = None

        fold_preds = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        fold_probs = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

        for t, m in enumerate(increments_range):
            #
            # selecting subset features
            logging.info('Considering features {}:{}'.format(min_feature, m))

            sel_labelled_splits = []
            for f in range(len(labelled_splits)):

                sel_labelled_splits.append([])
                for i in range(len(labelled_splits[f])):

                    sel_labelled_splits[f].append((labelled_splits[f][i][0][:, min_feature:m],
                                                   labelled_splits[f][i][1]))

                for i in range(len(labelled_splits[f])):
                    logging.info('shapes {} {}'.format(sel_labelled_splits[f][i][0].shape,
                                                       sel_labelled_splits[f][i][1].shape))

            train_x, train_y = sel_labelled_splits[0][0]
            for p, c in enumerate(args.log_c):
                logging.info('C: {}'.format(c))

                for f in range(len(labelled_splits)):

                    # log_res = linear_model.LogisticRegression(C=c,
                    #                                           **LOGISTIC_MOD_DICT_PARAMS[args.classifier])
                    # clf = OneVsRestClassifier(log_res)
                    clf = CLASSIFIER_DICT[args.classifier](c)

                    train_x, train_y = sel_labelled_splits[f][0]
                    true_train_x, true_train_y = fold_splits[f][0]

                    #
                    # fitting
                    fit_s_t = perf_counter()
                    clf.fit(train_x, train_y)
                    fit_e_t = perf_counter()

                    logging.info('\tfold: {} ({})'.format(f, fit_e_t - fit_s_t))
                    #
                    # scoring
                    for i, split in enumerate(sel_labelled_splits[f]):
                        if split is not None:
                            split_x, split_y = split
                            _, true_split_y = fold_splits[f][i]

                            split_s_t = perf_counter()
                            # split_acc = log_res.score(split_x, split_y)
                            split_preds = clf.predict(split_x)
                            split_e_t = perf_counter()

                            if decode:
                                if args.knn_decode:
                                    split_preds = decode_predictions_knn(split_preds,
                                                                         train_y,
                                                                         true_train_y,
                                                                         **knn_sklearn_args)
                                else:
                                    split_preds = decode_predictions(split_preds,
                                                                     # feature_fold_infos[f],
                                                                     ae_fold_decoders[f])
                                assert split_preds.shape[0] == split_x.shape[0]
                                assert split_preds.shape[1] == split_y.shape[1]

                            fold_preds[t][p][i].append(split_preds)
                            if hasattr(clf, 'predict_proba'):
                                split_probs = clf.predict_proba(split_x)
                                fold_probs[t][p][i].append(split_probs)

                            for s, score_func in enumerate(args.scores):
                                split_score = compute_scores(true_split_y, split_preds, score_func)
                                score_tensor[t, p, f, i, s] = split_score

                            scores_str = '\t'.join(['{}:{:.6f}'.format(SCORE_NAMES[args.scores[s]],
                                                                       score_tensor[t, p, f, i, s])
                                                    for s in range(n_scores)])
                            logging.info('\t\t{}\t{}\t({})'.format(SPLIT_NAMES[i],
                                                                   scores_str,
                                                                   split_e_t - split_s_t))

                    #
                    # saving to file
                    scores_str = '\t'.join('{:.6f}'.format(score_tensor[t, p, f, i, s])
                                           for s in range(n_scores)
                                           for i in range(n_splits))
                    out_log.write('{}\t{}\t{}\t{}\n'.format(m, c, f, scores_str))
                    out_log.flush()

        #
        # computing statistics along folds
        fold_avg_score_tensor = score_tensor.mean(axis=2)
        fold_std_score_tensor = score_tensor.std(axis=2)

        logging.info('\n')
        out_log.write('\n')

        out_log.write('inc\tc\t{}\n'.format('\t'.join('avg-{0}-{1}\tstd-{0}-{1}'.format(SPLIT_NAMES[i],
                                                                                        SCORE_NAMES[args.scores[s]])
                                                      for s in range(n_scores)
                                                      for i in range(n_splits))))
        for m in range(n_increments):
            logging.info('{}'.format(list(increments_range)[m]))
            for p in range(n_params):
                logging.info('\t{}'.format(args.log_c[p]))
                for i in range(n_splits):
                    score_str = '\t'.join('{}:{:.6f} +/-{:.6f}'.format(SCORE_NAMES[args.scores[s]],
                                                                       fold_avg_score_tensor[
                                                                           m, p, i, s],
                                                                       fold_std_score_tensor[m, p, i, s])
                                          for s in range(n_scores))
                    logging.info('\t\t{}\t{}'.format(SPLIT_NAMES[i], score_str))
                out_log.write('{}\t{}\t{}\n'.format(list(increments_range)[m],
                                                    args.log_c[p],
                                                    '\t'.join('{:.6f}\t{:.6f}'.format(fold_avg_score_tensor[m, p, i, s],
                                                                                      fold_std_score_tensor[m, p, i, s])
                                                              for s in range(n_scores)
                                                              for i in range(n_splits))))
                out_log.flush()

        #
        # getting best parameters
        out_log.write('\n')
        logging.info('\n\tBest params: ->(best avg value)')
        best_split = 1 if not numpy.isnan(fold_avg_score_tensor[0, 0, 1, 0]) else 2
        eval_split = 2
        logging.info('\t\t(best split: {} score split: {})'.format(SPLIT_NAMES[best_split],
                                                                   SPLIT_NAMES[eval_split]))
        for m in range(n_increments):
            logging.info('{}'.format(list(increments_range)[m]))

            res_list = []

            for s in range(n_scores):
                best_p = numpy.argmax(fold_avg_score_tensor[m, :, best_split, s])
                res_list.append('\t{}:\t{}\t-> {} (+/-{})'.format(SCORE_NAMES[args.scores[s]],
                                                                  args.log_c[best_p],
                                                                  fold_avg_score_tensor[
                                                                      m, best_p, best_split, s],
                                                                  fold_std_score_tensor[m, best_p, eval_split, s]))

            res_str = '\n'.join(res_list)
            logging.info('\n{}'.format(res_str))

            out_log.write('{}\t\t{}\n'.format(list(increments_range)[m],
                                              '\t'.join(SCORE_NAMES[args.scores[s]]
                                                        for s in range(n_scores))))
            for p in range(n_params):
                scores_str = '\t'.join('{}'.format(fold_avg_score_tensor[m, p, best_split, s])
                                       for s in range(n_scores))
                out_log.write('\t{}\t{}\n'.format(args.log_c[p], scores_str))
                out_log.flush()

            out_log.write('\n')

        out_log.write('inc\tscore\tbest-c\tbest-avg\tbest-std\n')
        for m in range(n_increments):
            res_list_p = []
            for s in range(n_scores):
                best_p = numpy.argmax(fold_avg_score_tensor[m, :, best_split, s])
                res_list_p.append('{}\t{}\t{}\t{}\t{}'.format(list(increments_range)[m],
                                                              SCORE_NAMES[args.scores[s]],
                                                              args.log_c[best_p],
                                                              fold_avg_score_tensor[
                    m, best_p, eval_split, s],
                    fold_std_score_tensor[m, best_p, best_split, s]))
            res_str = '\n'.join(res_list_p)
            out_log.write('{}\n'.format(res_str))
            out_log.flush()

        #
        # saving predictions?
        if args.save_probs or args.save_preds:
            logging.info('\n')
            preds_path = os.path.join(out_path, PREDS_PATH)
            os.makedirs(preds_path, exist_ok=True)
            for m in range(n_increments):
                for s in range(n_scores):
                    best_p = numpy.argmax(fold_avg_score_tensor[m, :, best_split, s])
                    best_p_str = '{}'.format(args.log_c[best_p])

                    if args.save_preds:
                        for i in range(n_splits):
                            best_preds_list = fold_preds[m][best_p][i]
                            for f, best_preds in enumerate(best_preds_list):
                                preds_file_path = os.path.join(preds_path,
                                                               '{}-{}-f{}.{}.preds.gz'.format(list(increments_range)[m],
                                                                                              best_p_str,
                                                                                              f,
                                                                                              SPLIT_NAMES[i]))
                                logging.info('Dumping preds to {}'.format(preds_file_path))
                                numpy.savetxt(preds_file_path, best_preds, fmt='%d')

                    if args.save_probs:
                        for i in range(n_splits):
                            best_probs_list = fold_probs[m][best_p][i]
                            for f, best_probs in enumerate(best_probs_list):
                                probs_file_path = os.path.join(preds_path,
                                                               '{}-{}-f{}.{}.probs.gz'.format(list(increments_range)[m],
                                                                                              best_p_str,
                                                                                              f,
                                                                                              SPLIT_NAMES[i]))
                                logging.info('Dumping probs to {}'.format(probs_file_path))
                                numpy.savetxt(probs_file_path, best_probs)

    else:

        header_str = "\t\t{}\n\t\t{}\n".format('\t'.join(SPLIT_NAMES * len(args.scores)),
                                               '\t'.join([SCORE_NAMES[s] for s in args.scores
                                                          for n in SPLIT_NAMES]))
        out_log.write('{}'.format(header_str))
        out_log.flush()

        n_params = len(args.log_c)
        n_folds = len(labelled_splits)
        n_splits = 3
        n_scores = len(args.scores)
        score_tensor = numpy.zeros((n_params, n_folds, n_splits, n_scores))
        score_tensor[:] = None

        fold_models = defaultdict(list)
        for p, c in enumerate(args.log_c):
            logging.info('C: {}'.format(c))

            # log_res = linear_model.LogisticRegression(C=c,
            #                                           **LOGISTIC_MOD_DICT_PARAMS[args.classifier])

            #
            # for each fold
            for f, fold in enumerate(labelled_splits):

                # clf = OneVsRestClassifier(log_res)
                clf = CLASSIFIER_DICT[args.classifier](c)
                fold_models[p].append(clf)

                train_x, train_y = fold[0]
                true_train_x, true_train_y = fold_splits[f][0]

                #
                # fitting the classifier
                fit_s_t = perf_counter()
                clf.fit(train_x, train_y)
                fit_e_t = perf_counter()
                logging.info('fold {} ({})'.format(f, fit_e_t - fit_s_t))

                #
                # scoring
                for i, split in enumerate(labelled_splits[f]):

                    if split is not None:
                        split_x, split_y = split
                        _, true_split_y = fold_splits[f][i]

                        split_s_t = perf_counter()
                        split_preds = clf.predict(split_x)
                        split_e_t = perf_counter()

                        if decode:
                            if args.knn_decode:
                                split_preds = decode_predictions_knn(split_preds,
                                                                     train_y,
                                                                     true_train_y,
                                                                     **knn_sklearn_args)
                            else:
                                split_preds = decode_predictions(split_preds,
                                                                 # feature_fold_infos[f],
                                                                 ae_fold_decoders[f])
                            assert split_preds.shape[0] == split_x.shape[0]
                            assert split_preds.shape[1] == true_split_y.shape[1]

                        for s, score_func in enumerate(args.scores):
                            # print(true_split_y.shape, split_preds.shape)
                            split_score = compute_scores(true_split_y, split_preds, score_func)
                            score_tensor[p, f, i, s] = split_score

                        score_str = '\t'.join(['{}:{:.6f}'.format(SCORE_NAMES[args.scores[s]],
                                                                  score_tensor[p, f, i, s])
                                               for s in range(n_scores)])
                        logging.info('\t{}\t{}\t({})'.format(SPLIT_NAMES[i],
                                                             score_str,
                                                             split_e_t - split_s_t))

                    # else:
                    #     for score_func in args.scores:
                    #         scores[score_func].append(None)

                out_log.write('{}\t{}\t{}\n'.format(c, f,
                                                    '\t'.join('{:.6f}'.format(score_tensor[p, f, i, s])
                                                              for s in range(n_scores)
                                                              for i in range(n_splits))))
                out_log.flush()

        #
        # computing statistics along folds
        fold_avg_score_tensor = score_tensor.mean(axis=1)
        fold_std_score_tensor = score_tensor.std(axis=1)

        logging.info('\n')
        out_log.write('\n')

        for p in range(n_params):
            logging.info('{}'.format(args.log_c[p]))
            for i in range(n_splits):
                score_str = '\t'.join('{}:{:.6f} +/-{:.6f}'.format(SCORE_NAMES[args.scores[s]],
                                                                   fold_avg_score_tensor[p, i, s],
                                                                   fold_std_score_tensor[p, i, s])
                                      for s in range(n_scores))
                logging.info('{}\t{}'.format(SPLIT_NAMES[i], score_str))
            out_log.write('{}\t{}\n'.format(args.log_c[p],
                                            '\t'.join('{:.6f}\t{:.6f}'.format(fold_avg_score_tensor[p, i, s],
                                                                              fold_std_score_tensor[p, i, s])
                                                      for s in range(n_scores)
                                                      for i in range(n_splits))))
            out_log.flush()

        logging.info('\n')
        out_log.write('\n')

        #
        # getting best parameters
        logging.info('\n\tBest params: ->(best avg value)')
        best_split = 1 if not numpy.isnan(fold_avg_score_tensor[0, 1, 0]) else 2
        eval_split = 2
        logging.info('\t\t(best split: {} score split: {})'.format(SPLIT_NAMES[best_split],
                                                                   SPLIT_NAMES[eval_split]))

        res_list = []
        res_list_p = []
        for s in range(n_scores):
            best_p = numpy.argmax(fold_avg_score_tensor[:, best_split, s])
            # print(fold_avg_score_tensor[:, best_split, s])
            res_list.append('{}:\t{}\t-> {} (+/-{})'.format(SCORE_NAMES[args.scores[s]],
                                                            args.log_c[best_p],
                                                            fold_avg_score_tensor[
                                                                best_p, eval_split, s],
                                                            fold_std_score_tensor[best_p, best_split, s]))
            res_list_p.append('{}\t{}\t{}\t{}'.format(SCORE_NAMES[args.scores[s]],
                                                      args.log_c[best_p],
                                                      fold_avg_score_tensor[
                best_p, eval_split, s],
                fold_std_score_tensor[best_p, eval_split, s]))
        res_str = '\n'.join(res_list)
        logging.info('\n{}'.format(res_str))

        out_log.write('\t{}\n'.format('\t'.join(SCORE_NAMES[args.scores[s]]
                                                for s in range(n_scores))))
        for p in range(n_params):
            scores_str = '\t'.join('{}'.format(fold_avg_score_tensor[p, eval_split, s])
                                   for s in range(n_scores))
            out_log.write('{}\t{}\n'.format(args.log_c[p], scores_str))
            out_log.flush()

        out_log.write('\n')
        res_str = '\n'.join(res_list_p)
        out_log.write('{}\n'.format(res_str))
        out_log.flush()

        #
        # saving predictions?
        if args.save_probs or args.save_preds:
            logging.info('\n')
            preds_path = os.path.join(out_path, PREDS_PATH)
            os.makedirs(preds_path, exist_ok=True)

            for s in range(n_scores):
                best_p = numpy.argmax(fold_avg_score_tensor[:, best_split, s])
                best_p_str = '{}'.format(args.log_c[best_p])
                best_models = fold_models[best_p]

                if args.save_preds:
                    for i in range(n_splits):
                        for f, model in enumerate(best_models):
                            split = labelled_splits[f][i]
                            if split is not None:
                                x = split[0]
                                best_preds = model.predict(x)
                                preds_file_path = os.path.join(preds_path,
                                                               '{}-f{}.{}.preds.gz'.format(best_p_str,
                                                                                           f,
                                                                                           SPLIT_NAMES[i]))
                                logging.info('Dumping preds to {}'.format(preds_file_path))
                                numpy.savetxt(preds_file_path, best_preds, fmt='%d')

                if args.save_probs:
                    for i in range(n_splits):
                        for f, model in enumerate(best_models):
                            split = labelled_splits[f][i]
                            if split is not None and hasattr(model, 'predict_proba'):
                                x = split[0]
                                best_probs = model.predict_proba(x)
                                probs_file_path = os.path.join(preds_path,
                                                               '{}-f{}.{}.probs.gz'.format(best_p_str,
                                                                                           f,
                                                                                           SPLIT_NAMES[i]))
                                logging.info('Dumping probs to {}'.format(probs_file_path))
                                numpy.savetxt(probs_file_path, best_probs)
