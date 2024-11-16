# importing numpy, pandas, and matplotlib
import numpy as np
import pandas as pd
from deepmicro.plot_tools import plot_loss

# importing sklearn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection
from sklearn import cluster
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

# importing keras
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier # scikit-learn wrapper for keras do not use that
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LambdaCallback
from tensorflow.keras.models import Model, load_model

# importing util libraries
import datetime
import time
import math
import os
import importlib

# importing custom library
import DNN_models
import exception_handle
from utils import parse_args
#fix np.random.seed for reproducibility in numpy processing
np.random.seed(7)



class Experience:
    def __init__(self,
                 args:dict,
                 seed:int):
        self.args = args
        self.seed = seed
        self.dm = None
        self.mode =

    def run(self):
        # create an object and load data_test
        if self.args.data is None and self.args.custom_data is None:
            print("[Error] Please specify an input file. (use -h option for help)")
            exit()
        elif self.args.data is not None:
            self.dm = DeepMicrobiome(data=self.args.data + '.txt', seed=self.seed, data_dir=self.args.data_dir)
            feature_string = ''
            data_string = str(self.args.data)
            if data_string.split('_')[0] == 'abundance':
                feature_string = "k__"
            if data_string.split('_')[0] == 'marker':
                feature_string = "gi|"
            self.dm.load_data(feature_string=feature_string, label_string='disease', label_dict=label_dict,
                              dtype=dtypeDict[self.args.dataType])
        elif self.args.custom_data is not None:
            if self.args.custom_data_labels is None:
                self.dm = DeepMicrobiome(data=self.args.custom_data, seed=self.seed, data_dir=self.args.data_dir)
                self.dm.load_custom_data(dtype=dtypeDict[self.args.dataType])
            else:
                self.dm = DeepMicrobiome(data=self.args.custom_data, seed=self.seed, data_dir=self.args.data_dir)
                self.dm.load_custom_data_with_labels(label_data=self.args.custom_data_labels, dtype=dtypeDict[self.args.dataType])
        else:
            exit()

        num_rl_required = self.args.pca + self.args.ae + self.args.rp + self.args.vae + self.args.cae
        if num_rl_required > 1:
            raise ValueError('No multiple dimensionality Reduction')

        self.dm.t_start = time.time()

        # Representation learning (Dimensionality reduction)
        if self.args.pca:
            self.dm.pca()
        if self.args.ae:
            self.dm.ae(dims=[int(i) for i in self.args.dims.split(',')], act=self.args.act, epochs=self.args.max_epochs, loss=self.args.aeloss,
                       latent_act=self.args.ae_lact, output_act=self.args.ae_oact, patience=self.args.patience, no_trn=self.args.no_trn)
        if self.args.vae:
            self.dm.vae(dims=[int(i) for i in self.args.dims.split(',')], act=self.args.act, epochs=self.args.max_epochs, loss=self.args.aeloss, output_act=self.args.ae_oact,
                        patience=25 if self.args.patience == 20 else self.args.patience, beta=self.args.vae_beta, warmup=self.args.vae_warmup, warmup_rate=self.args.vae_warmup_rate, no_trn=self.args.no_trn)
        if self.args.cae:
            self.dm.cae(dims=[int(i) for i in self.args.dims.split(',')], act=self.args.act, epochs=self.args.max_epochs, loss=self.args.aeloss, output_act=self.args.ae_oact,
                        patience=self.args.patience, rf_rate=self.args.rf_rate, st_rate=self.args.st_rate, no_trn=self.args.no_trn)
        if self.args.rp:
            self.dm.rp()

        if self.args.save_rep:
            if num_rl_required == 1:
                rep_file = self.dm.data_dir + "results/" + self.dm.prefix + self.dm.data + "_rep.csv"
                pd.DataFrame(self.dm.X_train).to_csv(rep_file, header=None, index=None)
                print("The learned representation of the training set has been saved in '{}'".format(rep_file))
            else:
                print("Warning: Command option '--save_rep' is not applied as no representation learning or dimensionality reduction has been conducted.")

        if self.args.no_clf or (self.args.data is None and self.args.custom_data_labels is None):
            print("Classification task has been skipped.")
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            importlib.reload(tensorflow.keras)

            if self.args.method == "svm":
                self.dm.classification(hyper_parameters=svm_hyper_parameters, method='svm', cv=self.args.numFolds,
                                       n_jobs=self.args.numJobs, scoring=self.args.scoring, cache_size=self.args.svm_cache)
            elif self.args.method == "rf":
                self.dm.classification(hyper_parameters=rf_hyper_parameters, method='rf', cv=self.args.numFolds,
                                       n_jobs=self.args.numJobs, scoring=self.args.scoring)
            elif self.args.method == "mlp":
                self.dm.classification(hyper_parameters=mlp_hyper_parameters, method='mlp', cv=self.args.numFolds,
                                       n_jobs=self.args.numJobs, scoring=self.args.scoring)
            elif self.args.method == "svm_rf":
                self.dm.classification(hyper_parameters=svm_hyper_parameters, method='svm', cv=self.args.numFolds,
                                       n_jobs=self.args.numJobs, scoring=self.args.scoring, cache_size=self.args.svm_cache)
                self.dm.classification(hyper_parameters=rf_hyper_parameters, method='rf', cv=self.args.numFolds,
                                       n_jobs=self.args.numJobs, scoring=self.args.scoring)
            else:
                self.dm.classification(hyper_parameters=svm_hyper_parameters, method='svm', cv=self.args.numFolds,
                                       n_jobs=self.args.numJobs, scoring=self.args.scoring, cache_size=self.args.svm_cache)
                self.dm.classification(hyper_parameters=rf_hyper_parameters, method='rf', cv=self.args.numFolds,
                                       n_jobs=self.args.numJobs, scoring=self.args.scoring)
                self.dm.classification(hyper_parameters=mlp_hyper_parameters, method='mlp', cv=self.args.numFolds,
                                       n_jobs=self.args.numJobs, scoring=self.args.scoring)

class DeepMicrobiome(object):
    def __init__(self, data, seed, data_dir):
        self.t_start = time.time()
        self.filename = str(data)
        self.data = self.filename.split('.')[0]
        self.seed = seed
        self.data_dir = data_dir
        self.prefix = ''
        self.representation_only = False
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.history = {}
        self.autoencoder = None
        self.encoder = None
        self.decoder = None
        self.vae = None
        self.cae = None

    def load_data(self, feature_string, label_string, label_dict, dtype=None):
        # read file
        filename = self.data_dir + "data_test/" + self.filename
        if os.path.isfile(filename):
            raw = pd.read_csv(filename, sep='\t', index_col=0, header=None)
        else:
            print("FileNotFoundError: File {} does not exist".format(filename))
            exit()

        # select rows having feature index identifier string
        X = raw.loc[raw.index.str.contains(feature_string, regex=False)].T

        # get class labels
        Y = raw.loc[label_string] #'disease'
        Y = Y.replace(label_dict)

        # train and test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X.values.astype(dtype), Y.values.astype('int'), test_size=0.2, random_state=self.seed, stratify=Y.values)
        self.print_data_shapes()

    def load_custom_data(self, dtype=None):
        # read file
        filename = self.data_dir + "data_test/" + self.filename
        if os.path.isfile(filename):
            raw = pd.read_csv(filename, sep=',', index_col=False, header=None)
        else:
            print("FileNotFoundError: File {} does not exist".format(filename))
            exit()

        # load data_test
        self.X_train = raw.values.astype(dtype)

        # put nothing or zeros for y_train, y_test, and X_test
        self.y_train = np.zeros(shape=(self.X_train.shape[0])).astype(dtype)
        self.X_test = np.zeros(shape=(1,self.X_train.shape[1])).astype(dtype)
        self.y_test = np.zeros(shape=(1,)).astype(dtype)
        self.print_data_shapes(train_only=True)

    def load_custom_data_with_labels(self, label_data, dtype=None):
        # read file
        filename = self.data_dir + "data_test/" + self.filename
        label_filename = self.data_dir + "data_test/" + label_data
        if os.path.isfile(filename) and os.path.isfile(label_filename):
            raw = pd.read_csv(filename, sep=',', index_col=False, header=None)
            label = pd.read_csv(label_filename, sep=',', index_col=False, header=None)
        else:
            if not os.path.isfile(filename):
                print("FileNotFoundError: File {} does not exist".format(filename))
            if not os.path.isfile(label_filename):
                print("FileNotFoundError: File {} does not exist".format(label_filename))
            exit()

        # label data_test validity check
        if not label.values.shape[1] > 1:
            label_flatten = label.values.reshape((label.values.shape[0]))
        else:
            print('FileSpecificationError: The label file contains more than 1 column.')
            exit()

        # train and test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(raw.values.astype(dtype),
                                                                                label_flatten.astype('int'), test_size=0.2,
                                                                                random_state=self.seed,
                                                                                stratify=label_flatten)
        self.print_data_shapes()


    #Principal Component Analysis
    def pca(self, ratio=0.99):
        # manipulating an experiment identifier in the output file
        self.prefix = self.prefix + 'PCA_'

        # PCA
        pca = PCA()
        pca.fit(self.X_train)
        n_comp = 0
        ratio_sum = 0.0

        for comp in pca.explained_variance_ratio_:
            ratio_sum += comp
            n_comp += 1
            if ratio_sum >= ratio:  # Selecting components explaining 99% of variance
                break

        pca = PCA(n_components=n_comp)
        pca.fit(self.X_train)

        X_train = pca.transform(self.X_train)
        X_test = pca.transform(self.X_test)

        # applying the eigenvectors to the whole training and the test set.
        self.X_train = X_train
        self.X_test = X_test
        self.print_data_shapes()

    #Gausian Random Projection
    def rp(self):
        # manipulating an experiment identifier in the output file
        self.prefix = self.prefix + 'RandP_'
        # GRP
        rf = GaussianRandomProjection(eps=0.5)
        rf.fit(self.X_train)

        # applying GRP to the whole training and the test set.
        self.X_train = rf.transform(self.X_train)
        self.X_test = rf.transform(self.X_test)
        self.print_data_shapes()

    #Shallow Autoencoder & Deep Autoencoder
    def ae(self,
           dims = [50],
           epochs= 2000,
           batch_size=100,
           verbose=2,
           loss='mean_squared_error',
           latent_act=False,
           output_act=False,
           act='relu',
           patience=20,
           val_rate=0.2,
           no_trn=False
           ):

        # manipulating an experiment identifier in the output file
        if patience != 20:
            self.prefix += 'p' + str(patience) + '_'

        # filename for temporary model checkpoint
        model_name = self.prefix + self.data + '.h5'

        # clean up model checkpoint before use
        if os.path.isfile(model_name):
            os.remove(model_name)

        # callbacks for each epoch
        callbacks = [EarlyStopping(monitor='val_loss', patience=patience, mode='min', verbose=1),
                     ModelCheckpoint(model_name, monitor='val_loss', mode='min', verbose=1, save_best_only=True)]

        # spliting the training set into the inner-train and the inner-test set (validation set)
        X_inner_train, X_inner_test, y_inner_train, y_inner_test = train_test_split(self.X_train, self.y_train, test_size=val_rate, random_state=self.seed, stratify=self.y_train)

        # insert input shape into dimension list
        dims.insert(0, X_inner_train.shape[1])

        # create autoencoder model
        self.autoencoder, self.encoder = DNN_models.autoencoder(dims, act=act, latent_act=latent_act, output_act=output_act)
        self.autoencoder.summary()

        if no_trn:
            return

        # compile model
        self.autoencoder.compile(optimizer='adam', loss=loss)

        # fit model
        self.history = self.autoencoder.fit(X_inner_train, X_inner_train, epochs=epochs, batch_size=batch_size, callbacks=callbacks,
                             verbose=verbose, validation_data=(X_inner_test, X_inner_test))
        # save loss progress
        self.save_loss_progress()

        # load best model
        self.autoencoder = load_model(model_name)
        layer_idx = int((len(self.autoencoder.layers) - 1) / 2)
        self.encoder = Model(self.autoencoder.layers[0].input, self.autoencoder.layers[layer_idx].output)

        # applying the learned encoder into the whole training and the test set.
        self.X_train = self.encoder.predict(self.X_train)
        self.X_test = self.encoder.predict(self.X_test)

    # Variational Autoencoder
    def vae(self, dims = [10], epochs=2000, batch_size=100, verbose=2, loss='mse', output_act=False, act='relu', patience=25, beta=1.0, warmup=True, warmup_rate=0.01, val_rate=0.2, no_trn=False):

        # manipulating an experiment identifier in the output file
        if patience != 25:
            self.prefix += 'p' + str(patience) + '_'
        if warmup:
            self.prefix += 'w' + str(warmup_rate) + '_'
        self.prefix += 'VAE'
        if loss == 'binary_crossentropy':
            self.prefix += 'b'
        if output_act:
            self.prefix += 'T'
        if beta != 1:
            self.prefix += 'B' + str(beta)
        self.prefix += str(dims).replace(", ", "-") + '_'
        if act == 'sigmoid':
            self.prefix += 'sig_'

        # filename for temporary model checkpoint
        model_name = self.prefix + self.data + '.h5'

        # clean up model checkpoint before use
        if os.path.isfile(model_name):
            os.remove(model_name)

        # callbacks for each epoch
        callbacks = [EarlyStopping(monitor='val_loss', patience=patience, mode='min', verbose=1),
                     ModelCheckpoint(model_name, monitor='val_loss', mode='min', verbose=1, save_best_only=True,save_weights_only=True)]

        # warm-up callback
        warm_up_cb = LambdaCallback(on_epoch_end=lambda epoch, logs: [warm_up(epoch)])  # , print(epoch), print(K.get_value(beta))])

        # warm-up implementation
        def warm_up(epoch):
            val = epoch * warmup_rate
            if val <= 1.0:
                K.set_value(beta, val)
        # add warm-up callback if requested
        if warmup:
            beta = K.variable(value=0.0)
            callbacks.append(warm_up_cb)

        # spliting the training set into the inner-train and the inner-test set (validation set)
        X_inner_train, X_inner_test, y_inner_train, y_inner_test = train_test_split(self.X_train, self.y_train,
                                                                                    test_size=val_rate,
                                                                                    random_state=self.seed,
                                                                                    stratify=self.y_train)

        # insert input shape into dimension list
        dims.insert(0, X_inner_train.shape[1])

        # create vae model
        self.vae, self.encoder, self.decoder = DNN_models.variational_ae(dims, act=act, recon_loss=loss, output_act=output_act, beta=beta)
        self.vae.summary()

        if no_trn:
            return

        # fit
        self.history = self.vae.fit(X_inner_train, epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=verbose, validation_data=(X_inner_test, None))

        # save loss progress
        self.save_loss_progress()

        # load best model
        self.vae.load_weights(model_name)
        self.encoder = self.vae.layers[1]

        # applying the learned encoder into the whole training and the test set.
        _, _, self.X_train = self.encoder.predict(self.X_train)
        _, _, self.X_test = self.encoder.predict(self.X_test)

    # Convolutional Autoencoder
    def cae(self, dims = [32], epochs=2000, batch_size=100, verbose=2, loss='mse', output_act=False, act='relu', patience=25, val_rate=0.2, rf_rate = 0.1, st_rate = 0.25, no_trn=False):

        # manipulating an experiment identifier in the output file
        self.prefix += 'CAE'
        if loss == 'binary_crossentropy':
            self.prefix += 'b'
        if output_act:
            self.prefix += 'T'
        self.prefix += str(dims).replace(", ", "-") + '_'
        if act == 'sigmoid':
            self.prefix += 'sig_'

        # filename for temporary model checkpoint
        model_name = self.prefix + self.data + '.h5'

        # clean up model checkpoint before use
        if os.path.isfile(model_name):
            os.remove(model_name)

        # callbacks for each epoch
        callbacks = [EarlyStopping(monitor='val_loss', patience=patience, mode='min', verbose=1),
                     ModelCheckpoint(model_name, monitor='val_loss', mode='min', verbose=1, save_best_only=True,save_weights_only=True)]


        # fill out blank
        onside_dim = int(math.sqrt(self.X_train.shape[1])) + 1
        enlarged_dim = onside_dim ** 2
        self.X_train = np.column_stack((self.X_train, np.zeros((self.X_train.shape[0], enlarged_dim - self.X_train.shape[1]))))
        self.X_test = np.column_stack((self.X_test, np.zeros((self.X_test.shape[0], enlarged_dim - self.X_test.shape[1]))))

        # reshape
        self.X_train = np.reshape(self.X_train, (len(self.X_train), onside_dim, onside_dim, 1))
        self.X_test = np.reshape(self.X_test, (len(self.X_test), onside_dim, onside_dim, 1))
        self.print_data_shapes()

        # spliting the training set into the inner-train and the inner-test set (validation set)
        X_inner_train, X_inner_test, y_inner_train, y_inner_test = train_test_split(self.X_train, self.y_train,
                                                                                    test_size=val_rate,
                                                                                    random_state=self.seed,
                                                                                    stratify=self.y_train)

        # insert input shape into dimension list
        dims.insert(0, (onside_dim, onside_dim, 1))

        # create cae model
        self.cae, self.encoder = DNN_models.conv_autoencoder(dims, act=act, output_act=output_act, rf_rate = rf_rate, st_rate = st_rate)
        self.cae.summary()
        if no_trn:
            return

        # compile
        self.cae.compile(optimizer='adam', loss=loss)

        # fit
        self.history = self.cae.fit(X_inner_train, X_inner_train, epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=verbose, validation_data=(X_inner_test, X_inner_test, None))

        # save loss progress
        self.save_loss_progress()

        # load best model
        self.cae.load_weights(model_name)
        if len(self.cae.layers) % 2 == 0:
            layer_idx = int((len(self.cae.layers) - 2) / 2)
        else:
            layer_idx = int((len(self.cae.layers) - 1) / 2)
        self.encoder = Model(self.cae.layers[0].input, self.cae.layers[layer_idx].output)

        # applying the learned encoder into the whole training and the test set.
        self.X_train = self.encoder.predict(self.X_train)
        self.X_test = self.encoder.predict(self.X_test)
        self.print_data_shapes()

    # Classification
    def ml_classification(self,
                          hyper_parameters,
                          method='svm',
                          cv=5,
                          scoring='roc_auc',
                          n_jobs=1,
                          cache_size=10000
                          ):
        clf_start_time = time.time()

        print("# Tuning hyper-parameters")
        print(self.X_train.shape, self.y_train.shape)

        # Support Vector Machine
        if method == 'svm':
            clf = GridSearchCV(SVC(probability=True, cache_size=cache_size), hyper_parameters, cv=StratifiedKFold(cv, shuffle=True), scoring=scoring, n_jobs=n_jobs, verbose=100, )
            clf.fit(self.X_train, self.y_train)

        # Random Forest
        if method == 'rf':
            clf = GridSearchCV(RandomForestClassifier(n_jobs=-1, random_state=0), hyper_parameters, cv=StratifiedKFold(cv, shuffle=True), scoring=scoring, n_jobs=n_jobs, verbose=100)
            clf.fit(self.X_train, self.y_train)


        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)

        # Evaluate performance of the best model on test set
        y_true, y_pred = self.y_test, clf.predict(self.X_test)
        y_prob = clf.predict_proba(self.X_test)

        # Performance Metrics: AUC, ACC, Recall, Precision, F1_score
        metrics = [round(roc_auc_score(y_true, y_prob[:, 1]), 4),
                   round(accuracy_score(y_true, y_pred), 4),
                   round(recall_score(y_true, y_pred), 4),
                   round(precision_score(y_true, y_pred), 4),
                   round(f1_score(y_true, y_pred), 4), ]

        # time stamp
        metrics.append(str(datetime.datetime.now()))

        # running time
        metrics.append(round( (time.time() - self.t_start), 2))

        # classification time
        metrics.append(round( (time.time() - clf_start_time), 2))

        # best hyper-parameter append
        metrics.append(str(clf.best_params_))

        # Write performance metrics as a file
        res = pd.DataFrame([metrics], index=[self.prefix + method])
        with open(self.data_dir + "results/" + self.data + "_result.txt", 'a') as f:
            res.to_csv(f, header=None)

        print('Accuracy metrics')
        print('AUC, ACC, Recall, Precision, F1_score, time-end, runtime(sec), classfication time(sec), best hyper-parameter')
        print(metrics)

    def print_data_shapes(self, train_only=False):
        print("X_train.shape: ", self.X_train.shape)
        if not train_only:
            print("y_train.shape: ", self.y_train.shape)
            print("X_test.shape: ", self.X_test.shape)
            print("y_test.shape: ", self.y_test.shape)

    # ploting loss progress over epochs
    def save_loss_progress(self):
        #print(self.history.history.keys())
        #print(type(self.history.history['loss']))
        #print(min(self.history.history['loss']))

        loss_collector, loss_max_at_the_end = self.save_loss_progress_ylim()

        # save loss progress - train and val loss only
        figure_name = self.prefix + self.data + '_' + str(self.seed)
        plot_loss(figure_title=figure_name,
                  loss_collector=loss_collector,
                  loss_max_at_the_end=loss_max_at_the_end,
                  history=self.history,
                  save_path_dir=self.data_dir,
                  save=True,
                  show=False
                  )

    # supporting loss plot
    def save_loss_progress_ylim(self):
        loss_collector = []
        loss_max_at_the_end = 0.0
        for hist in self.history.history:
            current = self.history.history[hist]
            loss_collector += current
            if current[-1] >= loss_max_at_the_end:
                loss_max_at_the_end = current[-1]
        return loss_collector, loss_max_at_the_end

if __name__ == '__main__':

    args = parse_args()
    # set labels for diseases and controls
    label_dict = get_label_dict()
    # hyper-parameter grids for classifiers
    rf_hyper_parameters = get_rf_hyper_parameters()

    svm_hyper_parameters = get_svm_hyper_parameters()
    mlp_hyper_parameters = get_mlp_hyper_parameters()

    # run experiments
    try:
        if args.repeat > 1:
            for i in range(args.repeat):
                experience = Experience(args, i)
                experience.run()
        else:
            experience = Experience(args, args.seed)
            experience.run()

    except OSError as error:
        exception_handle.log_exception(error)
