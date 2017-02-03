import gzip
import pickle
import logging

import numpy

from sklearn.metrics import accuracy_score
from sklearn.metrics import hamming_loss
from sklearn.metrics import zero_one_loss
from sklearn.metrics import f1_score
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import average_precision_score


SPLIT_NAMES = ['train', 'valid', 'test']
SCORE_NAMES = {'accuracy': 'acc',
               'hamming': 'ham',
               'exact': 'exc',
               'jaccard': 'jac',
               'micro-f1': 'mif',
               'macro-f1': 'maf',
               'micro-auc-pr': 'mipr',
               'macro-auc-pr': 'mapr', }

#
#
# loading dataset routines


def load_cv_splits(dataset_path,
                   dataset_name,
                   n_folds,
                   train_ext=None, valid_ext=None, test_ext=None,
                   x_only=False,
                   y_only=False,
                   dtype='int32'):

    if x_only and y_only:
        raise ValueError('Both x and y only specified')

    logging.info('Expecting dataset into {} folds for {}'.format(n_folds, dataset_name))
    fold_splits = []

    if (train_ext is not None and test_ext is not None):
        #
        # NOTE: this applies only to x-only/y-only data files
        for i in range(n_folds):
            logging.info('Looking for train-test split {}'.format(i))

            train_path = '{}.{}.{}'.format(dataset_path, i, train_ext)
            logging.info('Loading training csv file {}'.format(train_path))
            train = numpy.loadtxt(train_path, dtype=dtype, delimiter=',')

            test_path = '{}.{}.{}'.format(dataset_path, i, test_ext)
            logging.info('Loading test csv file {}'.format(test_path))
            test = numpy.loadtxt(test_path, dtype=dtype, delimiter=',')

            assert train.shape[1] == test.shape[1]

            fold_splits.append((train, None, test))
    else:
        logging.info('Trying to load pickle file {}'.format(dataset_path))
        #
        # trying to load a pickle file containint k = n_splits
        # [((train_x,  train_y), (test_x, test_y))_1, ... ((train_x, train_y), (test_x, test_y))_k]

        fsplit = None
        if dataset_path.endswith('.pklz'):
            fsplit = gzip.open(dataset_path, 'rb')
        else:
            fsplit = open(dataset_path, 'rb')

        folds = pickle.load(fsplit)
        fsplit.close()

        assert len(folds) == n_folds

        for splits in folds:

            if len(splits) == 1:
                raise ValueError('Not expecting a fold made by a single split')
            elif len(splits) == 2:
                train_split, test_split = splits
                #
                # do they contain label information?
                if x_only and len(train_split) == 2 and len(test_split) == 2:
                    train_x, train_y = train_split
                    test_x, test_y = test_split
                    fold_splits.append((train_x, None, test_x))
                elif y_only and len(train_split) == 2 and len(test_split) == 2:
                    train_x, train_y = train_split
                    test_x, test_y = test_split
                    fold_splits.append((train_y, None, test_y))
                else:
                    fold_splits.append((train_split, None, test_split))
            elif len(splits) == 3:
                train_split, valid_split, test_split = splits
                if x_only and len(train_split) == 2 and len(test_split) == 2 and len(valid_split) == 2:
                    train_x, train_y = train_split
                    test_x, test_y = test_split
                    valid_x, valid_y = valid_split
                    fold_splits.append((train_x, valid_x, test_x))
                elif y_only and len(train_split) == 2 and len(test_split) == 2 and len(valid_split) == 2:
                    train_x, train_y = train_split
                    test_x, test_y = test_split
                    valid_x, valid_y = valid_split
                    fold_splits.append((train_y, valid_y, test_y))
                else:
                    fold_splits.append((train_split, valid_split, test_split))

    assert len(fold_splits) == n_folds
    # logging.info('Loaded folds for {}'.format(dataset_name))
    # for i, (train, valid, test) in enumerate(fold_splits):
    #     logging.info('\tfold:\t{} {} {} {} '.format(i, len(train), len(test), valid))
    #     if len(train) == 2 and len(test) == 2:
    #         logging.info('\t\ttrain x:\tsize: {}\ttrain y:\tsize: {}'.format(train[0].shape,
    #                                                                          train[1].shape))
    #         logging.info('\t\ttest:\tsize: {}\ttest:\tsize: {}'.format(test[0].shape,
    #                                                                    test[1].shape))
    #     else:
    #         logging.info('\t\ttrain:\tsize: {}'.format(train.shape))
    #         logging.info('\t\ttest:\tsize: {}'.format(test.shape))

    return fold_splits


def load_train_val_test_splits(dataset_path,
                               dataset_name,
                               train_ext=None, valid_ext=None, test_ext=None,
                               x_only=False,
                               y_only=False,
                               dtype='int32'):

    if x_only and y_only:
        raise ValueError('Both x and y only specified')

    logging.info('Looking for (train/valid/test) dataset splits: %s', dataset_path)

    if train_ext is not None:
        #
        # NOTE this works only with x-only data files
        train_path = '{}.{}'.format(dataset_path, train_ext)
        logging.info('Loading training csv file {}'.format(train_path))
        train = numpy.loadtxt(train_path, dtype='int32', delimiter=',')

        if valid_ext is not None:
            valid_path = '{}.{}'.format(dataset_path, valid_ext)
            logging.info('Loading valid csv file {}'.format(valid_path))
            valid = numpy.loadtxt(valid_path, dtype='int32', delimiter=',')
            assert train.shape[1] == valid.shape[1]

        if test_ext is not None:
            test_path = '{}.{}'.format(dataset_path, test_ext)
            logging.info('Loading test csv file {}'.format(test_path))
            test = numpy.loadtxt(test_path, dtype='int32', delimiter=',')
            assert train.shape[1] == test.shape[1]

    else:
        logging.info('Trying to load pickle file {}'.format(dataset_path))
        #
        # trying to load a pickle containing (train_x) | (train_x, test_x) |
        # (train_x, valid_x, test_x)
        fsplit = None
        if dataset_path.endswith('.pklz'):
            fsplit = gzip.open(dataset_path, 'rb')
        else:
            fsplit = open(dataset_path, 'rb')

        splits = pickle.load(fsplit)
        fsplit.close()

        if len(splits) == 1:
            logging.info('Only training set')
            train = splits
            if x_only and isinstance(train, tuple):
                logging.info('\tonly x')
                train = train[0]

        elif len(splits) == 2:
            logging.info('Found training and test set')
            train, test = splits

            if len(train) == 2 and len(test) == 2:
                assert train[0].shape[1] == test[0].shape[1]
                assert train[1].shape[1] == test[1].shape[1]
                assert train[0].shape[0] == train[1].shape[0]
                assert test[0].shape[0] == test[1].shape[0]
            else:
                assert train.shape[1] == test.shape[1]

            if x_only:
                logging.info('\tonly x')
                if isinstance(train, tuple) and isinstance(test, tuple):
                    train = train[0]
                    test = test[0]
                else:
                    raise ValueError('Cannot get x only for train and test splits')
            elif y_only:
                logging.info('\tonly y')
                if isinstance(train, tuple) and isinstance(test, tuple):
                    train = train[1]
                    test = test[1]
                else:
                    raise ValueError('Cannot get y only for train and test splits')

        elif len(splits) == 3:
            logging.info('Found training, validation and test set')
            train, valid, test = splits

            if len(train) == 2 and len(test) == 2 and len(valid) == 2:
                assert train[0].shape[1] == test[0].shape[1]
                assert train[0].shape[1] == valid[0].shape[1]
                if train[1].ndim > 1 and test[1].ndim > 1 and valid[1].ndim > 1:
                    assert train[1].shape[1] == test[1].shape[1]
                    assert train[1].shape[1] == valid[1].shape[1]
                assert train[0].shape[0] == train[1].shape[0]
                assert test[0].shape[0] == test[1].shape[0]
                assert valid[0].shape[0] == valid[1].shape[0]

                if x_only:
                    logging.info('\tonly x')
                    if isinstance(train, tuple) and \
                       isinstance(test, tuple) and \
                       isinstance(valid, tuple):

                        train = train[0]
                        valid = valid[0]
                        test = test[0]
                    else:
                        raise ValueError('Cannot get x only for train, valid and test splits')
                elif y_only:
                    logging.info('\tonly y')
                    if isinstance(train, tuple) and \
                       isinstance(test, tuple) and \
                       isinstance(valid, tuple):

                        train = train[1]
                        valid = valid[1]
                        test = test[1]
                    else:
                        raise ValueError('Cannot get y only for train, valid and test splits')
            else:
                assert train.shape[1] == test.shape[1]
                assert train.shape[1] == valid.shape[1]

        else:
            raise ValueError('More than 3 splits, check pkl file {}'.format(dataset_path))

    fold_splits = [(train, valid, test)]

    logging.info('Loaded dataset {}'.format(dataset_name))

    return fold_splits


def print_fold_splits_shapes(fold_splits):
    for f, fold in enumerate(fold_splits):
        logging.info('\tfold {}'.format(f))
        for s, split in enumerate(fold):
            if split is not None:
                split_name = SPLIT_NAMES[s]
                if len(split) == 2:
                    split_x, split_y = split
                    logging.info('\t\t{}\tx: {}\ty: {}'.format(split_name,
                                                               split_x.shape, split_y.shape))
                else:
                    logging.info('\t\t{}\t(x/y): {}'.format(split_name,
                                                            split.shape))


def compute_scores(y_true, y_preds, score='accuracy'):

    assert y_true.shape == y_preds.shape, (y_true.shape, y_preds.shape)

    if score == 'accuracy':
        return accuracy_score(y_true, y_preds)
    elif score == 'hamming':
        return 1 - hamming_loss(y_true, y_preds)
    elif score == 'exact':
        return 1 - zero_one_loss(y_true, y_preds)
    elif score == 'jaccard':
        return jaccard_similarity_score(y_true, y_preds)
    elif score == 'micro-f1':
        return f1_score(y_true, y_preds, average='micro')
    elif score == 'macro-f1':
        return f1_score(y_true, y_preds, average='macro')
    elif score == 'micro-auc-pr':
        return average_precision_score(y_true, y_preds, average='micro')
    elif score == 'macro-auc-pr':
        return average_precision_score(y_true, y_preds, average='macro')


def compute_threshold(data, threshold=0.5):
    return (data > threshold).astype(int)
