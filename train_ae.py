'''
  A script for training sparse [1],  denoisig [2], contractive [3] or variational [4] AEs using Keras.
  This file is part of RAELk - Representation Learning with AutoEncoders in Keras

  Copyright (C) 2017 University of Bari "Aldo Moro"
  Author: Antonio Vergari
  Email: antonio.vergari@uniba.it
  $Date: 2017-01-22$

  [1] Andrew NG "Lecture Notes on Sparse Autoencoders" (https://web.stanford.edu/class/cs294a/sparseAutoencoder_2011new.pdf)
  [2] Vincent et al. (2008) "Extracting and composing robust  features  with  denoising  autoencoders"
  [3] Rifai et al. (2011) "Contractive Auto-Encoders: Explicit Invariance During Feature Extraction"
  [4] Kingma et al. (2014) "Autoencoding Variational Bayes"

  antonio vergari
  ***************************************************************************************************
  
  RAELk is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation (version 3).

  RAELk is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more details.

  You should have received a copy of the GNU General Public License along with this program; if not,
see [http://www.gnu.org/licenses/](http://www.gnu.org/licenses/) 

'''

import sys
import argparse
import os
import logging
import datetime
import gzip
import pickle
import itertools
from time import perf_counter
from collections import defaultdict

import numpy

import keras
from keras.layers import Input, Dense, Lambda
from keras.layers.noise import GaussianNoise
from keras.models import Model
from keras.engine import Layer
from keras import backend as K
from keras import objectives
from keras import regularizers
# from keras.datasets import mnist

from utils import load_cv_splits
from utils import load_train_val_test_splits
from utils import print_fold_splits_shapes
from utils import SPLIT_NAMES, SCORE_NAMES
from utils import compute_scores
from utils import compute_threshold
from utils import plot_images_matrix

# from spn.linked.representation import decode_embeddings_mpn
# from spn.linked.representation import load_feature_info

MAX_N_INSTANCES = 10000

PICKLE_SPLIT_EXT = 'pickle'
PREDS_PATH = 'preds'
PICKLE_SPLIT_EXT = 'pickle'
COMPRESSED_PICKLE_SPLIT_EXT = 'pklz'
FEATURE_FILE_EXT = 'features'
INFO_FILE_EXT = 'features.info'
SCOPE_FILE_EXT = 'scopes'
COMPRESSED_MODEL_EXT = 'model.pklz'
RAND_SEED = 1337


class BernoulliNoise(Layer):

    """Apply Bernoulli noise (flipping input bits)
    As it is a regularization layer, it is only active at training time.
    # Arguments
        theta: float, Bernoulli param
    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
    # Output shape
        Same shape as input.
    """

    def __init__(self, theta, **kwargs):
        self.supports_masking = True
        self.theta = theta
        self.uses_learning_phase = True
        super(BernoulliNoise, self).__init__(**kwargs)

    def call(self, x, mask=None):

        bernoulli_noise = (K.random_uniform(shape=K.shape(x), low=0, high=1) > self.theta)
        noise_x = K.switch(bernoulli_noise, x, 1 - x)
        return K.in_train_phase(noise_x, x)

    def get_config(self):
        config = {'theta': self.theta}
        base_config = super(BernoulliNoise, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def even_batch_shuffled_data(data, batch_size, rand_gen=None):
    if rand_gen is None:
        rand_gen = numpy.random.RandomState(RAND_SEED)

    n_instances = data.shape[0]
    n_features = data.shape[1]

    rem_instances = batch_size - (n_instances % batch_size)
    random_ids = rand_gen.choice(n_instances, size=rem_instances)

    batch_even_data = numpy.concatenate([data, data[random_ids]], axis=0)

    assert batch_even_data.shape[0] % batch_size == 0, batch_even_data.shape[0]
    assert batch_even_data.shape[1] == n_features

    return batch_even_data


def tile_shuffled_data_epochs(data, batch_size, n_epochs, rand_gen=None):

    if rand_gen is None:
        rand_gen = numpy.random.RandomState(RAND_SEED)

    n_instances = data.shape[0]
    n_features = data.shape[1]

    epoch_tiles = []
    for i in range(n_epochs):

        rand_gen.shuffle(data)
        epoch_tiles.append(data)

    epoch_data = numpy.concatenate(epoch_tiles, axis=0)
    rem_instances = batch_size - (epoch_data.shape[0] % batch_size)
    random_ids = rand_gen.choice(n_instances, size=rem_instances)
    batch_even_data = numpy.concatenate([epoch_data, data[random_ids]], axis=0)

    assert batch_even_data.shape[0] % batch_size == 0, batch_even_data.shape[0]
    assert batch_even_data.shape[1] == n_features

    return batch_even_data


def tile_shuffled_data_epochs_(data, batch_size, n_epochs, rand_gen=None):

    data = numpy.array(data, copy=True)

    if rand_gen is None:
        rand_gen = numpy.random.RandomState(RAND_SEED)

    n_instances = data.shape[0]
    n_features = data.shape[1]
    rem_instances = batch_size - (n_instances % batch_size)

    epoch_tiles = []
    for i in range(n_epochs):

        rand_gen.shuffle(data)
        random_ids = rand_gen.choice(n_instances, size=rem_instances)
        batch_even_data = numpy.concatenate([data, data[random_ids]], axis=0)
        assert batch_even_data.shape[0] % batch_size == 0, batch_even_data.shape[0]
        assert batch_even_data.shape[1] == n_features

        epoch_tiles.append(batch_even_data)

    return numpy.concatenate(epoch_tiles, axis=0)


def build_ae(input_dim, hidden_dims, latent_dim,
             sparsity=None,
             contractive=None,
             denoising=None,
             variational=None,
             optimizer='adam',
             batch_size=100,
             epsilon_std=1.0):
    """
    Builds a k-hidden layers AE architecture in keras
    where k and their sizes is specified by hidden_dims, a sequence of integers, e.g. [500, 200]
    Note that k layers are used for the encoder and k for the decoder

    """

    x = Input(batch_shape=(None, input_dim))
    #
    # add noise
    if denoising is not None:
        # prev_input = GaussianNoise(sigma=denoising)(x)
        prev_input = BernoulliNoise(theta=denoising)(x)
    else:
        prev_input = x

    for i, hidden_dim in enumerate(hidden_dims):
        h = Dense(hidden_dim, activation='relu')(prev_input)
        prev_input = h

    repr = None
    z_mean, z_log_var, z = None, None, None

    if variational:
        z_mean = Dense(latent_dim)(h)
        z_log_var = Dense(latent_dim)(h)

        def sampling(args):
            z_mean, z_log_var = args
            epsilon = K.random_normal(shape=K.shape(z_mean), mean=0.,  std=epsilon_std)
            return z_mean + K.exp(z_log_var / 2) * epsilon

        z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
        repr = z_mean
        prev_input = z

    elif sparsity is not None:
        repr = Dense(latent_dim, activation='relu',
                     activity_regularizer=regularizers.activity_l1(sparsity),
                     name='repr')(prev_input)
        prev_input = repr

    else:
        repr = Dense(latent_dim, activation='relu',
                     name='repr')(prev_input)
        prev_input = repr

    #
    # saving the decoder to use it later
    decoder_hs = []
    for i, hidden_dim in enumerate(reversed(hidden_dims)):
        h_layer = Dense(hidden_dim, activation='relu')
        decoder_hs.append(h_layer)
        h = h_layer(prev_input)
        prev_input = h

    output_layer = Dense(input_dim, activation='sigmoid')
    x_hat = output_layer(prev_input)

    build_start_t = perf_counter()
    #
    # building whole  AE
    ae = Model(x, x_hat)

    def ae_loss(x, x_hat):
        xent_loss = input_dim * objectives.binary_crossentropy(x, x_hat)

        if variational:
            kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
            return xent_loss + kl_loss

        elif contractive:
            W = K.variable(value=ae.get_layer('repr').get_weights()[0])
            W = K.transpose(W)
            h = ae.get_layer('repr').output
            #
            # the grad of a sigmoid layer is:
            dh = h * (1 - h)
            #
            # TODO: can we make it more general with K.gradients?

            contractive_loss = contractive * K.sum(dh**2 * K.sum(W**2, axis=1), axis=1)
            return xent_loss + contractive_loss

        else:
            return xent_loss

    # optimizer = keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    ae.compile(optimizer=optimizer, loss=ae_loss)

    #
    # building only the encoder part
    encoder = Model(x, repr)

    #
    # building only the decoder part
    decoder_input = Input(shape=(latent_dim,))
    prev_input = decoder_input
    for h_l in decoder_hs:
        h = h_l(prev_input)
        prev_input = h

    _x_decoded = output_layer(prev_input)
    decoder = Model(decoder_input, _x_decoded)

    build_end_t = perf_counter()
    logging.info('AE Built in {}s\n\tmodel:{}\n\t:encoder:{}\n\tdecoder:{}'.format(build_end_t -
                                                                                   build_start_t,
                                                                                   ae,
                                                                                   encoder,
                                                                                   decoder))

    return ae, encoder, decoder


def decode_latent_repr(decoder,
                       train_repr,
                       valid_repr=None,
                       test_repr=None):

    valid_dec, test_dec = None, None
    train_dec = decoder.predict(train_repr)

    if valid_repr is not None:
        valid_dec = decoder.predict(valid_repr)

    if test_repr is not None:
        test_dec = decoder.predict(test_repr)

    return train_dec, valid_dec, test_dec


def train_ae(ae,
             x_train,
             x_valid,
             x_test,
             encoder=None,
             decoder=None,
             batch_size=100,
             n_epochs=50,
             make_even_batch=None,
             patience=50,
             show_rec=None,
             img_size=None,
             verbose=1):

    # x_train = x_train.astype('float32') / 255.
    # x_test = x_test.astype('float32') / 255.
    # x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    # x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

    n_train = x_train.shape[0]
    n_valid = x_valid.shape[0]
    n_test = x_test.shape[0]

    if make_even_batch == 'tile_shuffled_epochs':
        #
        # making just one big epoch
        train = tile_shuffled_data_epochs_(x_train, batch_size, n_epochs)
        valid = tile_shuffled_data_epochs_(x_valid, batch_size, n_epochs)
        test = tile_shuffled_data_epochs_(x_test, batch_size, n_epochs)
        n_epochs = 1
        logging.info('Tiled shuffled data {} {} {}'.format(train.shape,
                                                           valid.shape,
                                                           test.shape))
    else:
        train, valid, test = x_train, x_valid, x_test

    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                   min_delta=0,
                                                   patience=patience,
                                                   verbose=1,
                                                   mode='auto')

    callbacks = [early_stopping]
    if show_rec:
        n_images_to_show = 8

        def show_rec_images(epoch, logs):
            if epoch % show_rec == 0:
                output_path = 'rec.{}'.format(epoch)  # os.path.join('a')
                images = []
                #
                # append one validation image and its reconstruction
                for i in range(n_images_to_show):
                    img_i = valid[i]
                    img_i = img_i.reshape(-1, img_i.shape[0])
                    images.append(img_i)
                    enc_i = encoder.predict(img_i)
                    dec_i = decoder.predict(enc_i)
                    images.append(dec_i)

                plot_images_matrix(images,
                                   # m=n_square, n=n_square,
                                   img_size=img_size,
                                   w_space=0.0,
                                   h_space=0.0,
                                   output=output_path, show=False)

        epoch_show_rec_cb = keras.callbacks.LambdaCallback(
            on_epoch_end=show_rec_images)
        callbacks.append(epoch_show_rec_cb)

    ae.fit(train, train,
           shuffle=True,
           nb_epoch=n_epochs,
           verbose=verbose,
           batch_size=batch_size,
           validation_data=(valid, valid),
           callbacks=callbacks)

    #
    # eval loss
    train_loss = ae.evaluate(train, train, batch_size=batch_size)
    valid_loss = ae.evaluate(valid, valid, batch_size=batch_size)
    test_loss = ae.evaluate(test, test, batch_size=batch_size)
    logging.info('\n\tTrain loss:\t{}\n\tValid loss:\t{}\n\tTest loss:\t{}'.format(train_loss,
                                                                                   valid_loss,
                                                                                   test_loss))

    #
    # encode initial representations, optionally
    x_train_enc, x_valid_enc, x_test_enc = None, None, None
    x_train_dec, x_valid_dec, x_test_dec = None, None, None
    if encoder is not None:

        if make_even_batch == 'tile_shuffled_epochs':
            ebx_train = even_batch_shuffled_data(x_train, batch_size)
            ebx_valid = even_batch_shuffled_data(x_valid, batch_size)
            ebx_test = even_batch_shuffled_data(x_test, batch_size)
        else:
            ebx_train, ebx_valid, ebx_test = x_train, x_valid, x_test

        x_train_enc = encoder.predict(ebx_train, batch_size=batch_size)
        x_valid_enc = encoder.predict(ebx_valid, batch_size=batch_size)
        x_test_enc = encoder.predict(ebx_test, batch_size=batch_size)

        #
        # decode them, optionally
        if decoder is not None:

            x_train_dec, x_valid_dec, x_test_dec = decode_latent_repr(decoder,
                                                                      x_train_enc,
                                                                      x_valid_enc,
                                                                      x_test_enc)

    return (train_loss, valid_loss, test_loss), \
           (x_train_enc[:n_train], x_valid_enc[:n_valid], x_test_enc[:n_test]), \
           (x_train_dec[:n_train], x_valid_dec[:n_valid], x_test_dec[:n_test])


parser = argparse.ArgumentParser()
parser.add_argument("dataset", type=str,
                    help='Specify a dataset file path')

parser.add_argument('--data-exts', type=str, nargs='+',
                    default=None,
                    help='Dataset split extensions')

parser.add_argument('--dtype', type=str, nargs='?',
                    default='int32',
                    help='Loaded dataset type')

parser.add_argument('--seed', type=int, nargs='?',
                    default=1337,
                    help='Seed for the random generator')

parser.add_argument('--batch-size', type=int, nargs='+',
                    default=[100],
                    help='Mini batch size')

parser.add_argument('--hidden-units', type=eval, nargs='+',
                    default="[100]",
                    help='Number of hidden neurons in the encoder and decoder')

parser.add_argument('--latent-units', type=int, nargs='+',
                    default=[None],
                    help='Dimension of latent space')

parser.add_argument('--compress-factor', type=float, nargs='+',
                    default=[None],
                    help='Compression factor, determined the dimension of the latent space'
                    ' if not specified')

parser.add_argument('--sparsity', type=float, nargs='+',
                    default=[None],
                    help='Sparsity inducing L1 norm coefficient for repr layer (as in SAEs)')

parser.add_argument('--noise', type=float, nargs='+',
                    default=[None],
                    help='Bernoulli theta param for input level noise at training time (as in DAEs)')

parser.add_argument('--contractive', type=float, nargs='+',
                    default=[None],
                    help='Contractive penalty coefficient (as in CAEs)')

parser.add_argument('--variational', action='store_true',
                    help='Whether to use a variational autoencoder kind')

parser.add_argument('--n-epochs', type=int, nargs='+',
                    default=[50],
                    help='Number of epochs to train')

parser.add_argument('--patience', type=int, nargs='?',
                    default=50,
                    help='Number of epochs to wait if the valid loss is deacreasing')

parser.add_argument('--show-rec-after-iters', type=int, nargs='?',
                    default=None,
                    help='Display a reconstruction version (for images) after some iterates')

parser.add_argument('--img-size', type=int, nargs='+',
                    default=[],
                    help='For image data, specify the size of the image for reshaping purposes')

parser.add_argument('--eps-std', type=float, nargs='+',
                    default=[1.0],
                    help='Epsilon std (normal dist)')

parser.add_argument('-o', '--output', type=str, nargs='?',
                    default='./exp/ae/',
                    help='Output dir path')

parser.add_argument('--even-batch', type=str, nargs='+',
                    default=['tile_shuffled_epochs'],
                    help='Output dir path')

parser.add_argument('--optimizer', type=str, nargs='+',
                    default=['rmsprop'],
                    help='Keras optimizer')

parser.add_argument('--optimizer-params', type=str, nargs='+',
                    default=[None],
                    help='Keras optimizer parameters')

parser.add_argument('--scores', type=str, nargs='+',
                    default=[],
                    help='MLC scores to compute')

parser.add_argument('--save-model', action='store_true',
                    help='Whether to store the model file as a pickle file')

parser.add_argument('--save-enc', action='store_true',
                    help='Whether to store the encoded dataset splits')

parser.add_argument('--exp-name', type=str, nargs='?',
                    default=None,
                    help='Experiment name, if not present a date will be used')

parser.add_argument('-v', '--verbose', type=int, nargs='?',
                    default=1,
                    help='Verbosity level')

parser.add_argument('--x-only', action='store_true',
                    help='Whether to train to reconstruct the x')

parser.add_argument('--y-only', action='store_true',
                    help='Whether to train to reconstruct the y')

parser.add_argument('--cv', type=int,
                    default=None,
                    help='Folds for cross validation for model selection')


#
# parsing the args
args = parser.parse_args()

#
# setting verbosity level
if args.verbose == 1:
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
elif args.verbose == 2:
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

logging.info("Starting with arguments:\n%s", args)


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

train_ext = None
valid_ext = None
test_ext = None


if args.data_exts is not None:
    if len(args.data_exts) == 1:
        train_ext, = args.data_exts
    elif len(args.data_exts) == 2:
        train_ext, test_ext = args.data_exts
    elif len(args.data_exts) == 3:
        train_ext, valid_ext, test_ext = args.data_exts
    else:
        raise ValueError('Up to 3 data extenstions can be specified')


n_folds = args.cv if args.cv is not None else 1

x_only = None
y_only = None
if args.y_only:
    x_only = False
    y_only = True
else:
    x_only = True
    y_only = False

#
# loading data and learned representations
if args.cv is not None:

    fold_splits = load_cv_splits(args.dataset,
                                 dataset_name,
                                 n_folds,
                                 train_ext=train_ext,
                                 valid_ext=valid_ext,
                                 test_ext=test_ext,
                                 x_only=x_only,
                                 y_only=y_only,
                                 dtype=args.dtype)


else:
    fold_splits = load_train_val_test_splits(args.dataset,
                                             dataset_name,
                                             train_ext=train_ext,
                                             valid_ext=valid_ext,
                                             test_ext=test_ext,
                                             x_only=x_only,
                                             y_only=y_only,
                                             dtype=args.dtype)


#
# printing
logging.info('Original folds')
print_fold_splits_shapes(fold_splits)

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

n_features = fold_splits[0][0].shape[1]


preamble = ("hidden-units\tlatent-units\tcompress-factor\t" +
            "sparsity\tnoise\tcontractive\tvariational\t" +
            "eps-std\tn-epochs\t" +
            "\tbatch-size\toptimizer\tfold" +
            "\t".join("{}-{}".format(sp, s)
                      for s in args.scores
                      for sp in SPLIT_NAMES) + "\n")

with open(out_log_path, 'w') as out_log:

    out_log.write("parameters:\n{0}\n\n".format(args))
    out_log.write(preamble)
    out_log.flush()

    possible_configurations = itertools.product(args.hidden_units,
                                                args.latent_units,
                                                args.compress_factor,
                                                args.sparsity,
                                                args.noise,
                                                args.contractive,
                                                [args.variational],
                                                args.eps_std,
                                                args.n_epochs,
                                                args.batch_size,
                                                args.optimizer,
                                                args.optimizer_params)

    aggr_config_str = ""

    best_aes = None
    best_encoders = None
    best_decoders = None
    best_encoded_splits = None

    best_valid_loss = numpy.inf
    best_params = {}

    best_train_enc, best_valid_enc, best_test_enc = None, None, None

    for n_hiddens, n_latents, compress_factor, \
        sparsity, noise, contractive, variational, \
            eps_std, n_epochs, batch_size, \
            optimizer, optimizer_params in possible_configurations:

        #
        # reset seeds
        # TODO: is this enough for keras as well?
        numpy.random.seed(args.seed)

        if n_latents is None and compress_factor is not None:
            assert compress_factor > 0 and compress_factor < 1, compress_factor
            n_latents = int(n_features * compress_factor)
        elif n_latents is None:
            raise ValueError('Provide at least args.latent_units or args.compress_factor')

        config_str = 'n hiddens:\t{}\nn latents:\t{}\ncomp factor:\t{}\n' \
            'sparsity:\t{}\nnoise:\t{}\ncontractive:\t{}\nvariational:\t{}\n'\
            'eps std:\t{}\n' \
            'n epochs:\t{}\nbatch size:\t{}\noptimizer:\t{}\noptimizer params:\t{}\n'\
            .format(n_hiddens, n_latents, compress_factor,
                    sparsity, noise, contractive, variational,
                    eps_std, n_epochs, batch_size,
                    optimizer, optimizer_params)
        logging.info('\n\n<<<<<<<<<<<\n{}>>>>>>>>>>>>\n\n'.format(config_str))

        fold_train_scores = defaultdict(list)
        fold_valid_scores = defaultdict(list)
        fold_test_scores = defaultdict(list)

        fold_aes = []
        fold_encoders = []
        fold_decoders = []

        fold_encoded_splits = []

        #
        # iterating over folds
        for i, splits in enumerate(fold_splits):

            train, valid, test = splits

            logging.info('\n\n***** Processing fold {}/{} *****'.format(i + 1, len(fold_splits)))

            #
            # only x or y shall be considered
            assert len(train.shape) == 2, train.shape[1] == n_features
            assert len(valid.shape) == 2, valid.shape[1] == n_features
            assert len(test.shape) == 2, test.shape[1] == n_features

            logging.info('Building model...')
            #
            # building model
            ae, encoder, decoder = build_ae(input_dim=n_features,
                                            hidden_dims=n_hiddens,
                                            latent_dim=n_latents,
                                            sparsity=sparsity,
                                            contractive=contractive,
                                            variational=variational,
                                            denoising=noise,
                                            optimizer=optimizer,
                                            batch_size=batch_size,
                                            epsilon_std=eps_std)

            fold_aes.append(ae)
            fold_encoders.append(encoder)
            fold_decoders.append(decoder)

            #
            # training it
            train_start_t = perf_counter()
            losses, enc_splits, dec_splits = train_ae(ae,
                                                      train,
                                                      valid,
                                                      test,
                                                      encoder=encoder,
                                                      decoder=decoder,
                                                      batch_size=batch_size,
                                                      # n_epochs=args.n_epochs,
                                                      n_epochs=n_epochs,
                                                      patience=args.patience,
                                                      show_rec=args.show_rec_after_iters,
                                                      img_size=args.img_size,
                                                      verbose=args.verbose)
            train_end_t = perf_counter()
            logging.info('Train, enc, dec done in {} secs'.format(train_end_t - train_start_t))

            fold_encoded_splits.append(enc_splits)

            #
            # store losses
            train_loss, valid_loss, test_loss = losses
            fold_train_scores['loss'].append(train_loss)
            fold_valid_scores['loss'].append(valid_loss)
            fold_test_scores['loss'].append(test_loss)

            train_dec, valid_dec, test_dec = dec_splits

            #
            # more scores?
            train_preds = compute_threshold(train_dec, threshold=0.5)
            valid_preds = compute_threshold(valid_dec, threshold=0.5)
            test_preds = compute_threshold(test_dec, threshold=0.5)
            preds = [train_preds, valid_preds, test_preds]

            print(train_dec[:1])
            print(train_preds[:1])
            print(train[:1])

            for score in args.scores:

                train_score = compute_scores(train, train_preds, score=score)
                fold_train_scores[score].append(train_score)
                logging.info('\t\ttrain {}:\t{}'.format(score, train_score))
                valid_score = compute_scores(valid, valid_preds, score=score)
                fold_valid_scores[score].append(valid_score)
                logging.info('\t\tvalid {}:\t{}'.format(score, valid_score))
                test_score = compute_scores(test, test_preds, score=score)
                fold_test_scores[score].append(test_score)
                logging.info('\t\ttest {}:\t{}'.format(score, test_score))
                logging.info('')

            #
            # check shape correctness
            for s, e in zip(splits, enc_splits):
                assert s.shape[0] == e.shape[0], (s.shape[0], e.shape[0])

            for s, d in zip(splits, dec_splits):
                assert s.shape[0] == d.shape[0], (s.shape[0], d.shape[0])

            #
            # more prints?
            if args.verbose > 1:
                for j, s in enumerate(splits):
                    print("Encoded", enc_splits[j][:2])
                    print("Decoded", dec_splits[j][:2])
                    print("Predicted", preds[j][:2])
                    print("Original", splits[j][:2])

            #
            # save to log
            res_str = '{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}'.format(n_hiddens, n_latents,
                                                              compress_factor,
                                                              eps_std, n_epochs, batch_size,
                                                              optimizer, i)
            res_str += "\t{:.7f}\t{:.7f}\t{:.7f}".format(fold_train_scores['loss'][i],
                                                         fold_valid_scores['loss'][i],
                                                         fold_test_scores['loss'][i])
            res_str += "\t".join("\t{:.7f}\t{:.7f}\t{:.7f}".format(fold_train_scores[score][i],
                                                                   fold_valid_scores[score][i],
                                                                   fold_test_scores[score][i])
                                 for score in args.scores)

            out_log.write("{}\n".format(res_str))
            out_log.flush()

        #
        # updating aggr config str
        aggr_train_loss = numpy.mean(fold_train_scores['loss'])
        aggr_valid_loss = numpy.mean(fold_valid_scores['loss'])
        aggr_test_loss = numpy.mean(fold_test_scores['loss'])

        check_loss = None
        if valid is not None:
            check_loss = aggr_valid_loss
        else:
            check_loss = aggr_test_loss

        if check_loss < best_valid_loss:
            best_valid_loss = check_loss
            best_aes = fold_aes
            best_encoders = fold_encoders
            best_decoders = fold_decoders
            best_encoded_splits = fold_encoded_splits
            best_params['n_hiddens'] = n_hiddens
            best_params['n_latents'] = n_latents
            best_params['compress_factor'] = compress_factor
            best_params['eps_std'] = eps_std
            best_params['n_epochs'] = n_epochs
            best_params['batch_size'] = batch_size
            best_params['optimizer'] = optimizer

        aggr_config_str += "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}".format(n_hiddens, n_latents,
                                                                   compress_factor,
                                                                   eps_std, n_epochs,
                                                                   batch_size,
                                                                   optimizer, i)
        aggr_config_str += "\t{:.7f}\t{:.7f}\t{:.7f}".format(aggr_train_loss,
                                                             aggr_valid_loss,
                                                             aggr_test_loss)
        aggr_config_str += "\t".join("\t{:.7f}\t{:.7f}\t{:.7f}"
                                     .format(numpy.mean(fold_train_scores[score]),
                                             numpy.mean(fold_valid_scores[score]),
                                             numpy.mean(fold_test_scores[score]))
                                     for score in args.scores)
        aggr_config_str += "\n"

    #
    # logging aggr by fold str
    out_log.write('\n\n{}\n'.format("""hidden-units\tlatent-units\tcompress-factor\teps-std\tn-epochs""" +
                                    """\tbatch-size\toptimizer""" +
                                    "\t".join("{}-{}".format(sp, s)
                                              for s in args.scores for sp in SPLIT_NAMES) + "\n"))
    out_log.write('{}\n'.format(aggr_config_str))
    out_log.flush()

    #
    # logging best config
    best_state_str = ', '.join(['{}: {}'.format(k, best_params[k]) for k in sorted(best_params)])
    out_log.write("{}".format(best_state_str))
    out_log.flush()

    #
    # saving models
    if args.save_model:
        for i, (m, e, d) in enumerate(zip(best_aes, best_encoders, best_decoders)):
            m.save(os.path.join(out_path, 'ae.{}.model.{}'.format(dataset_name, i)))
            e.save(os.path.join(out_path, 'ae.{}.encoder.{}'.format(dataset_name, i)))
            d.save(os.path.join(out_path, 'ae.{}.decoder.{}'.format(dataset_name, i)))

    if args.save_enc:
        repr_save_path = os.path.join(out_path, 'ae.{}.repr-data.pklz'.format(dataset_name))
        with gzip.open(repr_save_path, 'wb') as f:
            print('Saving splits to {}'.format(repr_save_path))
            if n_folds > 1:
                pickle.dump(best_encoded_splits, f, protocol=4)
            elif n_folds == 1:
                pickle.dump(best_encoded_splits[0], f, protocol=4)

    grid_str = 'Grid search ended, best params: {}'.format(best_params)
    logging.info(grid_str)
    out_log.write(grid_str)
    out_log.flush()
